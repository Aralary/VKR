import os
import json
import random
from typing import List, Dict

from datasets import load_dataset


BASE_DIR    = os.path.join("data", "security")
EN_RAW_DIR  = os.path.join(BASE_DIR, "en", "raw")
EN_PROC_DIR = os.path.join(BASE_DIR, "en", "processed")
RU_RAW_DIR  = os.path.join(BASE_DIR, "ru", "raw")
RU_PROC_DIR = os.path.join(BASE_DIR, "ru", "processed")

# Файл с AI-угрозами от генератора (из проекта VKR)
AI_THREATS_RU_PATH = os.path.join(RU_PROC_DIR, "ai_threats_ru.jsonl")
# Тот же файл копируется как EN (признаки стилометрические, не языковые)
AI_THREATS_EN_PATH = os.path.join(EN_RAW_DIR,  "ai_threats_en.jsonl")


def makedirs():
    for d in [EN_RAW_DIR, EN_PROC_DIR, RU_RAW_DIR, RU_PROC_DIR]:
        os.makedirs(d, exist_ok=True)


def save_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ─── Источники human-текстов ───────────────────────────────────────────────────

def prepare_en_sms():
    """
    ucirvine/sms_spam — ham (label=0) и spam (label=1).
    Нам нужен только ham как human-тексты для EN-детектора.
    """
    print("Загрузка EN SMS (ucirvine/sms_spam)...")
    ds = load_dataset("ucirvine/sms_spam", split="train")

    records = []
    for item in ds:
        text = item.get("sms", "") or item.get("text", "")
        raw_label = item.get("label", "")
        if not text:
            continue

        raw_str = str(raw_label).strip().lower()
        if raw_str in ("ham", "legitimate", "legit", "0"):
            label = 0
        elif raw_str in ("spam", "smishing", "phishing", "1"):
            label = 1
        else:
            continue

        records.append({"text": text.strip(), "label": label, "source": "sms_en"})

    out = os.path.join(EN_RAW_DIR, "sms_en.jsonl")
    save_jsonl(out, records)
    ham  = sum(1 for r in records if r["label"] == 0)
    spam = sum(1 for r in records if r["label"] == 1)
    print(f"EN SMS сохранены: {out} (ham={ham}, spam={spam})")


def prepare_en_emails():
    """
    luongnv89/phishing-email — легитимные (0) и фишинговые (1) письма.
    """
    print("Загрузка EN phishing emails (luongnv89/phishing-email)...")
    ds = load_dataset("luongnv89/phishing-email", split="train")

    if len(ds) == 0:
        print("⚠️ Email-датасет пустой.")
        return

    text_keys  = ["text", "email", "email_text", "body", "content", "message"]
    label_keys = ["label", "is_phishing", "class", "target", "phishing"]

    records = []
    for item in ds:
        text = next(
            (item[k].strip() for k in text_keys if isinstance(item.get(k), str) and item[k].strip()),
            ""
        )
        if not text:
            continue

        raw_label = next((item[k] for k in label_keys if k in item), None)
        if raw_label is None:
            continue

        if isinstance(raw_label, bool):
            label = 1 if raw_label else 0
        elif isinstance(raw_label, int) and raw_label in (0, 1):
            label = raw_label
        else:
            s = str(raw_label).strip().lower()
            if s in ("0", "ham", "legit", "legitimate", "safe", "benign", "non-phishing"):
                label = 0
            elif s in ("1", "spam", "phishing", "malicious", "attack"):
                label = 1
            else:
                continue

        records.append({"text": text, "label": label, "source": "email_en"})

    out = os.path.join(EN_RAW_DIR, "emails_en.jsonl")
    save_jsonl(out, records)
    print(f"EN Emails сохранены: {out} ({len(records)} писем)")


def prepare_ru_human_coat():
    """
    RussianNLP/coat (binary) — human-written тексты на русском.
    label=0 → human-written, label=1 → machine-generated.
    Берём только label=0 как human-корпус для RU-детектора.
    """
    print("Загрузка RU human-текстов (RussianNLP/coat, binary)...")
    ds = load_dataset("RussianNLP/coat", "binary", split="train")

    records = []
    for item in ds:
        text  = str(item.get("text", "")).strip()
        label = item.get("label", -1)
        if not text or int(label) != 0:
            continue
        records.append({"text": text, "label": 0, "source": "coat_human"})

    out = os.path.join(RU_RAW_DIR, "coat_human_ru.jsonl")
    save_jsonl(out, records)
    print(f"RU CoAT human-тексты сохранены: {out} ({len(records)} записей)")


# ─── Сборка итоговых корпусов ──────────────────────────────────────────────────

def _check_ai_threats(path: str, label_name: str) -> List[Dict]:
    """Загружает AI-угрозы и проверяет что файл существует."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл AI-угроз не найден: {path}\n"
            f"Скопируй output/ai_threats_ru.jsonl из проекта генератора:\n"
            f"  EN: → {AI_THREATS_EN_PATH}\n"
            f"  RU: → {AI_THREATS_RU_PATH}"
        )
    records = load_jsonl(path)
    ai = [r for r in records if r.get("label") == 1]
    print(f"  AI-угрозы ({label_name}): {len(ai)} записей из {path}")
    return ai


def build_security_en(max_per_class: int = 1600, seed: int = 42):
    """
    EN security корпус:
      label=0 (human): ham из sms_en.jsonl
      label=1 (AI):    ai_threats_en.jsonl
    """
    print("\nСборка security_en.jsonl...")
    random.seed(seed)

    # Human: только ham из SMS
    sms_path = os.path.join(EN_RAW_DIR, "sms_en.jsonl")
    if not os.path.exists(sms_path):
        print(f"⚠️ Не найден {sms_path} — сначала запусти prepare_en_sms()")
        return

    all_sms = load_jsonl(sms_path)
    human   = [r for r in all_sms if r["label"] == 0]

    # AI: из ai_threats_en.jsonl
    try:
        ai = _check_ai_threats(AI_THREATS_EN_PATH, "EN")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        return

    print(f"  Доступно: human={len(human)}, AI={len(ai)}")

    random.shuffle(human)
    random.shuffle(ai)
    n = min(len(human), len(ai), max_per_class)
    balanced = human[:n] + ai[:n]
    random.shuffle(balanced)

    out = os.path.join(EN_PROC_DIR, "security_en.jsonl")
    save_jsonl(out, balanced)
    print(f"  ✓ security_en.jsonl: {out} ({len(balanced)} записей, по {n} на класс)")


def build_security_ru(max_per_class: int = 1600, seed: int = 42):
    """
    RU security корпус:
      label=0 (human): coat_human_ru.jsonl (CoAT, human-written)
      label=1 (AI):    ai_threats_ru.jsonl
    """
    print("\nСборка security_ru.jsonl...")
    random.seed(seed)

    # Human: CoAT
    coat_path = os.path.join(RU_RAW_DIR, "coat_human_ru.jsonl")
    if not os.path.exists(coat_path):
        print(f"⚠️ Не найден {coat_path} — сначала запусти prepare_ru_human_coat()")
        return

    human = load_jsonl(coat_path)
    print(f"  Human (CoAT): {len(human)} записей")

    # AI: ai_threats_ru.jsonl
    try:
        ai = _check_ai_threats(AI_THREATS_RU_PATH, "RU")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        return

    print(f"  Доступно: human={len(human)}, AI={len(ai)}")

    random.shuffle(human)
    random.shuffle(ai)
    n = min(len(human), len(ai), max_per_class)
    balanced = human[:n] + ai[:n]
    random.shuffle(balanced)

    out = os.path.join(RU_PROC_DIR, "security_ru.jsonl")
    save_jsonl(out, balanced)
    print(f"  ✓ security_ru.jsonl: {out} ({len(balanced)} записей, по {n} на класс)")


def build_to_translate_ru(n_per_class: int = 500, seed: int = 42):
    """
    Опционально: подвыборка из security_en.jsonl для ручного перевода на RU.
    Нужна только если захочешь улучшить RU-корпус SMS-стилем.
    """
    src = os.path.join(EN_PROC_DIR, "security_en.jsonl")
    if not os.path.exists(src):
        print(f"⚠️ Не найден {src}")
        return

    records = load_jsonl(src)
    random.seed(seed)

    class0 = [r for r in records if r["label"] == 0]
    class1 = [r for r in records if r["label"] == 1]
    random.shuffle(class0)
    random.shuffle(class1)

    subset = class0[:n_per_class] + class1[:n_per_class]
    random.shuffle(subset)

    out = os.path.join(RU_RAW_DIR, "to_translate_ru.jsonl")
    save_jsonl(out, subset)
    print(f"\nПодвыборка для перевода: {out} ({len(subset)} записей)")
    print("→ Переведи 'text' на русский и сохрани как data/security/ru/processed/security_ru.jsonl")


# ─── Главная точка входа ───────────────────────────────────────────────────────

def main():
    makedirs()

    # 1. Human-корпуса
    prepare_en_sms()
    prepare_en_emails()
    prepare_ru_human_coat()

    # 2. Собрать EN и RU корпуса детектора
    build_security_en(max_per_class=1600)
    build_security_ru(max_per_class=1600)

    # 3. Опционально: подвыборка для перевода на RU
    build_to_translate_ru(n_per_class=500)

    print("\n✅ Готово!")
    print("  EN: data/security/en/processed/security_en.jsonl")
    print("  RU: data/security/ru/processed/security_ru.jsonl")


if __name__ == "__main__":
    main()