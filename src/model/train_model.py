import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from src.model.model_setup import ModelManager
from src.core.config import TRAINING_CONFIG, PATH_CONFIG
from src.utils import logger, print_section


class ThreatAdapterTrainer:
    @staticmethod
    def format_example(example: dict) -> str:
        """Форматирует пример в Mistral Instruct шаблон."""
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]

        if input_text:
            return (
                f"<s>[INST] {instruction}\n\n"
                f"Образец: {input_text} [/INST] {output}</s>"
            )

        return f"<s>[INST] {instruction} [/INST] {output}</s>"

    def train(self):
        print_section("ОБУЧЕНИЕ АДАПТЕРА IB THREATS")

        train_path = PATH_CONFIG.datasets_dir / "ib_threats_train.jsonl"
        val_path = PATH_CONFIG.datasets_dir / "ib_threats_val.jsonl"
        output_dir = PATH_CONFIG.adapter_dir

        if not train_path.exists():
            raise FileNotFoundError(
                f"Датасет не найден: {train_path}\n"
                "Сначала запустите: python -m src.main --prepare"
            )

        if not val_path.exists():
            raise FileNotFoundError(
                f"Валидационный датасет не найден: {val_path}\n"
                "Сначала запустите: python -m src.main --prepare"
            )

        logger.info(f"Train dataset: {train_path}")
        logger.info(f"Val dataset: {val_path}")
        logger.info(f"Adapter output dir: {output_dir}")

        model, tokenizer, _ = ModelManager.setup_lora_model()

        train_dataset = load_dataset(
            "json",
            data_files=str(train_path),
            split="train",
        )

        val_dataset = load_dataset(
            "json",
            data_files=str(val_path),
            split="train",
        )

        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

        def add_text_column(example: dict) -> dict:
            return {
                "text": ThreatAdapterTrainer.format_example(example)
            }


        train_dataset = train_dataset.map(
            add_text_column,
            remove_columns=train_dataset.column_names,
        )

        val_dataset = val_dataset.map(
            add_text_column,
            remove_columns=val_dataset.column_names,
        )

        logger.info(f"Formatted train: {len(train_dataset)} | Formatted val: {len(val_dataset)}")

        use_bf16 = (
            TRAINING_CONFIG.bf16
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

        training_args = SFTConfig(
            output_dir=str(output_dir),

            num_train_epochs=TRAINING_CONFIG.num_train_epochs,
            per_device_train_batch_size=TRAINING_CONFIG.per_device_train_batch_size,
            gradient_accumulation_steps=TRAINING_CONFIG.gradient_accumulation_steps,
            gradient_checkpointing=TRAINING_CONFIG.gradient_checkpointing,

            learning_rate=TRAINING_CONFIG.learning_rate,
            lr_scheduler_type=TRAINING_CONFIG.lr_scheduler_type,
            warmup_ratio=TRAINING_CONFIG.warmup_ratio,
            max_grad_norm=TRAINING_CONFIG.max_grad_norm,

            optim=TRAINING_CONFIG.optim,

            logging_steps=TRAINING_CONFIG.logging_steps,
            save_strategy=TRAINING_CONFIG.save_strategy,
            save_total_limit=TRAINING_CONFIG.save_total_limit,

            eval_strategy="epoch",

            bf16=use_bf16,
            fp16=not use_bf16,

            max_seq_length=TRAINING_CONFIG.max_seq_length,
            packing=False,

            report_to=TRAINING_CONFIG.report_to,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )

        logger.info("Запуск обучения...")
        trainer.train()

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        logger.info(f"✓ Адаптер сохранён: {output_dir}")

        del model
        del trainer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_section("ОБУЧЕНИЕ ЗАВЕРШЕНО")