"""
Массовая генерация AI-угроз для обучения детектора (с батчингом).
Запуск: python generate_ai_threats.py --n 400 --lang ru --batch_size 8
"""

import json
import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.generation.generator import ThreatTextGenerator
from src.core.config import CATEGORY_CONFIG, MODEL_CONFIG, GENERATION_CONFIG, PATH_CONFIG
from src.utils import logger

TARGET_CATEGORIES = [
    "fraud_sms",
    "phishing_emails",
    "messenger_fraud",
    "fake_tech_notifications",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   default=400)
    parser.add_argument("--lang",       choices=["ru", "en"], default="ru")
    parser.add_argument("--output_dir", type=str,   default="output")
    parser.add_argument("--min_len",    type=int,   default=10)
    parser.add_argument("--max_len",    type=int,   default=1000)
    parser.add_argument("--batch_size", type=int,   default=8,
                        help="Кол-во текстов за один прогон модели (default: 8)")
    parser.add_argument("--seed",       type=int,   default=42)
    return parser.parse_args()


def build_prompts_for_batch(cat_key: str, categories: dict, batch_size: int) -> list[dict]:
    """Формирует batch_size разных инструкций для одной категории."""
    cat = categories[cat_key]
    items = []
    for _ in range(batch_size):
        template = random.choice(cat["instruction_templates"])
        _org   = random.choice(cat["orgs"])   if cat.get("orgs")   else ""
        _topic = random.choice(cat["topics"]) if cat.get("topics") else ""
        try:
            instruction = template.format(org=_org, topic=_topic)
        except KeyError:
            instruction = template.replace("{org}", _org).replace("{topic}", _topic)
        items.append({"instruction": instruction, "org": _org})
    return items


def generate_batch(generator: ThreatTextGenerator, batch_items: list[dict]) -> list[str]:
    """
    Генерирует батч текстов за один прогон модели.
    Формирует список промптов, прогоняет через tokenizer + model.generate.
    """
    cfg = GENERATION_CONFIG

    prompts = []
    for item in batch_items:
        instruction = item["instruction"]
        org = item.get("org", "")
        if org:
            prompt = f"<s>[INST] {instruction}\n\nОбразец: {org} [/INST]"
        else:
            prompt = f"<s>[INST] {instruction} [/INST]"
        prompts.append(prompt)

    tokenizer = generator.tokenizer
    model = generator.model

    # Токенизируем батч с padding
    tokenizer.padding_side = "left"  # для декодера важно padding слева
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    input_len = inputs["input_ids"].shape[-1]
    for output in outputs:
        generated_ids = output[input_len:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # Берём только первый абзац
        text = decoded.split("\n\n")[0].strip()
        results.append(text)

    return results


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ai_threats_{args.lang}.jsonl"

    generator = ThreatTextGenerator()
    generator.load_adapter()

    categories = CATEGORY_CONFIG.categories
    available = [c for c in TARGET_CATEGORIES if c in categories]

    logger.info(f"Категории: {available}")
    logger.info(f"Примеров на категорию: {args.n}, batch_size: {args.batch_size}")

    total = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for cat_key in available:
            cat_generated = 0
            cat_skipped = 0
            logger.info(f"\n── Категория: {cat_key} ──")
            t0 = time.time()

            while cat_generated < args.n:
                remaining = args.n - cat_generated
                current_batch = min(args.batch_size, remaining)

                try:
                    batch_items = build_prompts_for_batch(cat_key, categories, current_batch)
                    texts = generate_batch(generator, batch_items)
                except Exception as e:
                    logger.warning(f"Ошибка в батче: {e}")
                    time.sleep(1)
                    continue

                for i, text in enumerate(texts):
                    if not text or len(text) < args.min_len:
                        cat_skipped += 1
                        continue

                    text = text[:args.max_len]
                    record = {
                        "text": text,
                        "label": 1,
                        "lang": args.lang,
                        "source": cat_key,
                        "instruction": batch_items[i]["instruction"],
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    cat_generated += 1
                    total += 1

                elapsed = time.time() - t0
                speed = cat_generated / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  {cat_key}: {cat_generated}/{args.n} "
                    f"({speed:.1f} текстов/мин, пропущено: {cat_skipped})"
                )

            logger.info(f"  ✓ {cat_key}: готово {cat_generated}")

    logger.info(f"\n✅ Итого: {total} записей → {out_path}")

    # Статистика
    with open(out_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["source"], 0)
        by_cat[r["source"]] += 1
    print("\n── Итоговая статистика ──")
    for cat, cnt in by_cat.items():
        print(f"  {cat:<30}: {cnt}")
    print(f"  {'ИТОГО':<30}: {len(records)}")


if __name__ == "__main__":
    main()