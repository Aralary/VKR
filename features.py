import re
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Автоматически выбираем GPU если доступна
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gpt2(model_name: str = "gpt2"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model.to(DEVICE)
    print(f"GPT-2 загружен на: {DEVICE}")
    return model, tokenizer


def compute_perplexity(
    text: str,
    model,
    tokenizer,
    max_length: int = 512
) -> float:
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    input_ids = encodings.input_ids.to(DEVICE)

    if input_ids.shape[1] < 2:
        # слишком короткий текст — возвращаем большое "штрафное" значение
        return 10000.0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    ppl = torch.exp(loss).item()
    # ограничиваем сверху, чтобы не было взрывных значений
    return min(ppl, 10000.0)


# ---------- Вспомогательные разбиения ----------

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_WORD_RE = re.compile(r'\b\w+\b')
_PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)


def split_sentences(text: str) -> list:
    sentences = _SENT_SPLIT_RE.split(text.strip())
    return [s for s in sentences if s.strip()]


def split_paragraphs(text: str) -> list:
    """
    Разбиваем по пустым строкам / двойным переводам строки.
    Для коротких SMS/сообщений почти всегда будет один "абзац".
    """
    paras = [p for p in re.split(r'\n\s*\n+', text.strip()) if p.strip()]
    return paras if paras else [text.strip()] if text.strip() else []


def tokenize_words(text: str) -> list:
    return _WORD_RE.findall(text)


# ---------- Признаки из таблицы 1.5 ----------

def compute_ttr(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_avg_word_length(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def compute_sentence_features(text: str) -> dict:
    """
    Возвращает:
      - avg_sent_len  (ASL)
      - std_sent_len  (σ_SL)
      - sent_count    (SC)
    """
    sentences = split_sentences(text)
    if not sentences:
        return {
            "avg_sent_len": 0.0,
            "std_sent_len": 0.0,
            "sent_count": 0
        }

    lengths = [len(s.split()) for s in sentences]
    return {
        "avg_sent_len": float(np.mean(lengths)),
        "std_sent_len": float(np.std(lengths)),
        "sent_count": len(sentences)
    }


def compute_punctuation_density(text: str) -> float:
    """
    PD(d) = |{c in d : c in P}| / |d|
    где P — множество знаков препинания.
    """
    if not text:
        return 0.0

    total_len = len(text)
    punct_count = len(_PUNCT_RE.findall(text))
    return punct_count / total_len if total_len > 0 else 0.0


def compute_avg_paragraph_length(text: str) -> float:
    """
    APL(d) = (1/K) * sum_k |W_k|
    где W_k — множество слов в k-м абзаце.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return 0.0

    lengths = [len(tokenize_words(p)) for p in paragraphs]
    if not lengths:
        return 0.0

    return float(np.mean(lengths))


def compute_long_word_ratio(text: str, threshold: int = 6) -> float:
    """
    LDW(d) = |{v in W(d) : |v| > threshold}| / |W(d)|
    по умолчанию threshold = 6 (как в таблице 1.5).
    """
    words = tokenize_words(text)
    if not words:
        return 0.0

    long_words = [w for w in words if len(w) > threshold]
    return len(long_words) / len(words)


# ---------- Итоговый вектор признаков (9 штук) ----------

def extract_features(text: str, model, tokenizer) -> np.ndarray:
    """
    Формирует вектор из 9 признаков:
      1) PPL
      2) TTR
      3) AWL
      4) ASL
      5) σ_SL
      6) SC
      7) PD
      8) APL
      9) LDW
    """
    ppl = compute_perplexity(text, model, tokenizer)
    ttr = compute_ttr(text)
    avg_word_len = compute_avg_word_length(text)
    sent_feats = compute_sentence_features(text)
    pd = compute_punctuation_density(text)
    apl = compute_avg_paragraph_length(text)
    ldw = compute_long_word_ratio(text)

    features = np.array([
        ppl,                         # 1
        ttr,                         # 2
        avg_word_len,               # 3
        sent_feats["avg_sent_len"], # 4
        sent_feats["std_sent_len"], # 5
        sent_feats["sent_count"],   # 6
        pd,                         # 7
        apl,                        # 8
        ldw                         # 9
    ], dtype=np.float64)

    # На всякий случай избавляемся от nan/inf
    features = np.nan_to_num(features, nan=0.0, posinf=10000.0, neginf=0.0)
    return features