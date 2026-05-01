import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.core.config import MODEL_CONFIG, GENERATION_CONFIG, PATH_CONFIG, CATEGORY_CONFIG
from src.utils import logger


class ThreatTextGenerator:
    """Генератор текстов угроз ИБ с LoRA-адаптером."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_adapter(self):
        """Загружает адаптер (однократно при инициализации)."""
        if self._loaded:
            return

        adapter_path = PATH_CONFIG.adapter_dir
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Адаптер не найден: {adapter_path}\n"
                "Сначала запустите обучение: python main.py --train"
            )

        logger.info("Загрузка адаптера IB Threats...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.use_cache = True
        self._loaded = True
        logger.info("✓ Адаптер загружен")

    def generate(self, instruction: str, input_text: str = "") -> str:
        """
        Генерирует текст угрозы по инструкции.

        Args:
            instruction: Инструкция (например, "Напиши фишинговое письмо от Сбербанка")
            input_text:  Опциональный контекст / образец

        Returns:
            Сгенерированный текст
        """
        self.load_adapter()

        cfg = GENERATION_CONFIG

        if input_text:
            prompt = f"<s>[INST] {instruction}\n\nОбразец: {input_text} [/INST]"
        else:
            prompt = f"<s>[INST] {instruction} [/INST]"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
                repetition_penalty=cfg.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]

        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        return decoded.split("\n\n")[0].strip()

    def generate_by_category(
        self,
        category_key: str,
        org: str = "",
        topic: str = "",
        n: int = 1,
    ) -> list:
        """
        Генерирует n текстов для заданной категории.

        Args:
            category_key: Ключ категории из CategoryConfig
            org:   Организация для подстановки в шаблон
            topic: Тема для подстановки в шаблон
            n:     Количество генераций
        """
        categories = CATEGORY_CONFIG.categories
        if category_key not in categories:
            raise ValueError(
                f"Неизвестная категория '{category_key}'. "
                f"Доступные: {list(categories.keys())}"
            )

        import random
        cat = categories[category_key]

        results = []
        for i in range(n):
            template = random.choice(cat["instruction_templates"])
            _org   = org   or (random.choice(cat["orgs"])   if cat.get("orgs")   else "")
            _topic = topic or (random.choice(cat["topics"]) if cat.get("topics") else "")

            try:
                instruction = template.format(org=_org, topic=_topic)
            except KeyError:
                instruction = template.replace("{org}", _org).replace("{topic}", _topic)

            input_text = _org if _org else ""

            logger.info(f"Генерация {i+1}/{n}: {instruction[:60]}...")
            text = self.generate(instruction, input_text=input_text)
            results.append({"instruction": instruction, "output": text})

        return results

    def save_results(self, results: list, filename: str = "generated.txt") -> Path:
        """Сохраняет результаты генерации в файл."""
        out_path = PATH_CONFIG.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(results, 1):
                f.write(f"{'='*60}\n")
                f.write(f"[{i}] {item['instruction']}\n")
                f.write(f"{'='*60}\n")
                f.write(item["output"] + "\n\n")
        logger.info(f"✓ Сохранено: {out_path}")
        return out_path