import argparse
from classifier import train_on_hc3
from evaluate import evaluate
from features import extract_features, load_gpt2


def parse_args():
    parser = argparse.ArgumentParser(description="AI Text Detector")
    parser.add_argument(
        "--lang",
        choices=["en", "ru", "en_sec", "ru_sec"],
        default="en",
        help=(
            "Язык и режим: "
            "'en' (HC3+CHEAT, GPT-2), "
            "'ru' (CoAT, ruGPT-3), "
            "'en_sec' (SMS+AI-угрозы, GPT-2), "
            "'ru_sec' (CoAT human+AI-угрозы, ruGPT-3)"
        )
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.lang == "ru":
        model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"
        model_path = "detector_ru.pkl"
        cache_path = "features_cache_ru.npz"
        subset     = "ru"
        lang_label = "Русский (CoAT + ruGPT-3)"

    elif args.lang == "en_sec":
        model_name = "gpt2"
        model_path = "detector_en_sec.pkl"
        cache_path = "features_cache_en_sec.npz"
        subset     = "en_sec"
        lang_label = "Английский — Security (SMS + AI-угрозы, GPT-2)"

    elif args.lang == "ru_sec":
        model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"
        model_path = "detector_ru_sec.pkl"
        cache_path = "features_cache_ru_sec.npz"
        subset     = "ru_sec"
        lang_label = "Русский — Security (CoAT human + AI-угрозы, ruGPT-3)"

    else:  # en
        model_name = "gpt2"
        model_path = "detector_en.pkl"
        cache_path = "features_cache_en.npz"
        subset     = "all"
        lang_label = "Английский (HC3 + GPT-2)"

    print(f"\n🌐 Режим: {lang_label}")

    pipeline, X_test, y_test = train_on_hc3(
        model_path=model_path,
        cache_path=cache_path,
        subset=subset,
        model_name=model_name,
    )

    evaluate(X_test, y_test, pipeline)

    print("\nЗагрузка языковой модели для классификации...")
    gpt2_model, tokenizer = load_gpt2(model_name)

    print("\n=== Интерактивный детектор AI-текста ===")
    print("Введите текст для анализа. Для выхода нажмите Ctrl+C.\n")

    while True:
        try:
            text = input(">>> Введите текст: ").strip()

            if not text:
                print("  ⚠️ Пустой ввод, попробуйте снова.\n")
                continue

            word_count = len(text.split())

            # Вариант 1: отказ от анализа при очень коротком тексте
            if word_count < 15:
                # Реально бессмысленно: 1-2 слова
                print(f"  ⛔ Слишком мало слов ({word_count}) — анализ невозможен.\n")
                continue

            if word_count < 40:
                print(f"  ℹ️ Короткий текст ({word_count} слов) — уверенность может быть снижена.")

            elif word_count > 400:
                print(f"  ℹ️ Текст длинный ({word_count} слов) — "
                    f"модель анализирует первые ~400 слов.")

            features = extract_features(text, gpt2_model, tokenizer)
            pred     = pipeline.predict([features])[0]
            proba    = pipeline.predict_proba([features])[0]
            ai_prob  = proba[1] * 100

            # Вариант 2: серая зона
            if ai_prob >= 70:
                label = "🤖 Синтетический (AI)"
            elif ai_prob <= 40:
                label = "🧑 Человеческий"
            else:
                label = "❓ Неопределённо (пограничный случай)"

            confidence = max(proba) * 100
            print(f"\n  Результат: {label}")
            print(f"  Уверенность: {confidence:.1f}%")
            print(f"  (AI: {ai_prob:.1f}% | Человек: {proba[0]*100:.1f}%)\n")

        except KeyboardInterrupt:
            print("\n\nВыход. До свидания!")
            break


if __name__ == "__main__":
    main()