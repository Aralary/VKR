import os
import json
import requests

DATASET_PATH = "./hc3_dataset"
BASE_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main"

HC3_FILES = {
    "open_qa":     "open_qa.jsonl",
    "medicine":    "medicine.jsonl",
    "finance":     "finance.jsonl",
    "reddit_eli5": "reddit_eli5.jsonl",
    "wiki_csai":   "wiki_csai.jsonl",
}

SECURITY_EN_PATH = os.path.join("data", "security", "en", "processed", "security_en.jsonl")
SECURITY_RU_PATH = os.path.join("data", "security", "ru", "processed", "security_ru.jsonl")


def download_file(url: str, save_path: str):
    print(f"  Скачиваю: {os.path.basename(save_path)}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Готово: {os.path.basename(save_path)}")


def load_jsonl(filepath: str) -> tuple:
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            for answer in item.get("human_answers", []):
                if answer and answer.strip():
                    texts.append(answer.strip())
                    labels.append(0)
            for answer in item.get("chatgpt_answers", []):
                if answer and answer.strip():
                    texts.append(answer.strip())
                    labels.append(1)
    return texts, labels


def load_hc3(subset: str = "all") -> tuple:
    os.makedirs(DATASET_PATH, exist_ok=True)
    files_to_load = HC3_FILES if subset == "all" else {subset: HC3_FILES[subset]}

    all_texts, all_labels = [], []
    for name, filename in files_to_load.items():
        local_path = os.path.join(DATASET_PATH, filename)
        if not os.path.exists(local_path):
            print(f"[{name}] Файл не найден локально. Скачиваю...")
            download_file(f"{BASE_URL}/{filename}", local_path)
        else:
            print(f"[{name}] Файл найден локально. Пропускаю скачивание.")
        texts, labels = load_jsonl(local_path)
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  → загружено {len(texts)} текстов "
              f"(человек: {labels.count(0)}, ChatGPT: {labels.count(1)})")

    print(f"\nИтого HC3: {len(all_texts)} текстов "
          f"(человек: {all_labels.count(0)}, ChatGPT: {all_labels.count(1)})")
    return all_texts, all_labels


def load_cheat() -> tuple:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установи библиотеку: pip install datasets")

    print("Загрузка датасета CHEAT (научные абстракты)...")
    ds = load_dataset("acmc/cheat", split="train")

    texts, labels = [], []
    for item in ds:
        text = item.get("abstract", "").strip()
        label_raw = item.get("label", "").strip().lower()
        if not text:
            continue
        if label_raw == "init":
            labels.append(0)
        elif label_raw in ("generation", "polish", "fusion"):
            labels.append(1)
        else:
            continue
        texts.append(text)

    print(f"Итого CHEAT: {len(texts)} текстов "
          f"(человек: {labels.count(0)}, AI: {labels.count(1)})")
    return texts, labels


def load_coat() -> tuple:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установи библиотеку: pip install datasets")

    print("Загрузка датасета RussianNLP/CoAT...")
    ds = load_dataset("RussianNLP/coat", "binary", split="train")

    texts, labels = [], []
    for item in ds:
        text = item.get("text", "").strip()
        label = int(item.get("label", -1))
        if text and label in (0, 1):
            texts.append(text)
            labels.append(label)

    print(f"\nИтого CoAT: {len(texts)} текстов "
          f"(человек: {labels.count(0)}, AI: {labels.count(1)})")
    return texts, labels


def load_english() -> tuple:
    """HC3 + CHEAT объединённые для английского режима."""
    texts_hc3, labels_hc3 = load_hc3(subset="all")
    texts_cheat, labels_cheat = load_cheat()

    all_texts  = texts_hc3  + texts_cheat
    all_labels = labels_hc3 + labels_cheat

    print(f"\nИтого EN (HC3 + CHEAT): {len(all_texts)} текстов "
          f"(человек: {all_labels.count(0)}, AI: {all_labels.count(1)})")
    return all_texts, all_labels


def load_security_jsonl(lang: str = "en") -> tuple:
    """
    Загружает security_en.jsonl или security_ru.jsonl для обучения детектора угроз.
    label=0 — human-written, label=1 — AI-сгенерированный текст.
    """
    path = SECURITY_EN_PATH if lang == "en" else SECURITY_RU_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            f"Сначала запусти: python collect_dataset.py"
        )

    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text  = item.get("text", "").strip()
            label = item.get("label", -1)
            if text and label in (0, 1):
                texts.append(text)
                labels.append(int(label))

    print(f"\nИтого Security-{lang.upper()}: {len(texts)} текстов "
          f"(human: {labels.count(0)}, AI: {labels.count(1)})")
    return texts, labels