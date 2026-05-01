import argparse
from src.utils import print_section


def main():
    parser = argparse.ArgumentParser(
        description="IB Threats Generator — генерация и распознавание текстов угроз ИБ"
    )
    parser.add_argument("--prepare", action="store_true",
                        help="Подготовить обучающий датасет")
    parser.add_argument("--train",   action="store_true",
                        help="Обучить LoRA-адаптер")
    parser.add_argument("--generate", action="store_true",
                        help="Запустить генерацию текстов")
    parser.add_argument("--validate", action="store_true",
                        help="Валидировать датасет")
    parser.add_argument("--category", type=str, default="phishing_emails",
                        help="Категория угрозы для генерации: disinformation_news, fake_tech_notifications, fraud_sms, messenger_fraud, phishing_emails")
    parser.add_argument("--org",   type=str, default="",
                        help="Организация для подстановки")
    parser.add_argument("--topic", type=str, default="",
                        help="Тема для подстановки")
    parser.add_argument("--n", type=int, default=3,
                        help="Количество текстов для генерации")
    args = parser.parse_args()

    if args.prepare:
        from src.data.data_preparation import ThreatDatasetGenerator
        ThreatDatasetGenerator().prepare_dataset()

    elif args.validate:
        from src.data.validate_dataset import validate_and_report
        from src.core.config import PATH_CONFIG
        for name in ["ib_threats_train", "ib_threats_val", "ib_threats_test"]:
            path = PATH_CONFIG.datasets_dir / f"{name}.jsonl"
            if path.exists():
                validate_and_report(path)

    elif args.train:
        print(">>> [1] Входим в --train ветку", flush=True)
        try:
            print(">>> [2] Импортируем ThreatAdapterTrainer...", flush=True)
            from src.model.train_model import ThreatAdapterTrainer

            print(">>> [3] Импорт успешен, запускаем train()...", flush=True)
            ThreatAdapterTrainer().train()

        except BaseException as e:
            import traceback
            print("[ОШИБКА]", flush=True)
            print(f"Тип: {type(e).__name__}", flush=True)
            print(f"Текст: {e}", flush=True)
            traceback.print_exc()
            raise

    elif args.generate:
        from src.generation.generator import ThreatTextGenerator
        gen = ThreatTextGenerator()
        results = gen.generate_by_category(
            args.category, org=args.org, topic=args.topic, n=args.n
        )
        out_path = gen.save_results(results, f"{args.category}_results.txt")
        print_section("ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
        print(f"Результаты: {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()