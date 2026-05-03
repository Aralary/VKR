import os
import numpy as np
import joblib
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from features import extract_features, load_gpt2
from dataset import load_english, load_coat, load_security_jsonl


def build_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        ))
    ])


def train_on_hc3(
    model_path: str = "detector.pkl",
    subset: str = "all",
    test_size: float = 0.2,
    cache_path: str = "features_cache.npz",
    model_name: str = "gpt2",
) -> tuple:

    # 1. Загружаем/считаем признаки
    if os.path.exists(cache_path):
        print(f"Найден кэш признаков: {cache_path}, загружаю...")
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        print(f"Загружено {len(X)} текстов из кэша.")
    else:
        # Выбираем датасет по subset
        if subset == "ru":
            texts, labels = load_coat()
        elif subset in ("en_sec", "ru_sec"):
            lang = "ru" if subset == "ru_sec" else "en"
            texts, labels = load_security_jsonl(lang)
        else:
            texts, labels = load_english()

        gpt2_model, tokenizer = load_gpt2(model_name)
        print(f"Извлечение признаков из {len(texts)} текстов...")

        X = np.array([
            extract_features(t, gpt2_model, tokenizer)
            for t in tqdm(texts, desc="Признаки", unit="текст")
        ])
        y = np.array(labels)

        X = np.nan_to_num(X, nan=0.0, posinf=10000.0, neginf=0.0)
        bad = np.sum(~np.isfinite(X))
        if bad > 0:
            print(f"⚠️ Исправлено {bad} некорректных значений в признаках")

        np.savez(cache_path, X=X, y=y)
        print(f"Кэш признаков сохранён: {cache_path}")

    # 2. Разбиваем на train / test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    # 3. Подбор C по логарифмической сетке
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    alpha_acc = 0.3
    alpha_f1  = 0.5
    alpha_auc = 0.2

    best_score    = -np.inf
    best_C        = None
    best_pipeline = None

    print("\n=== Подбор гиперпараметра C (логистическая регрессия) ===")

    for C in C_values:
        pipeline = build_pipeline(C)
        pipeline.fit(X_train, y_train)

        y_val_pred  = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]

        acc  = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred)
        rec  = recall_score(y_val, y_val_pred)
        f1   = f1_score(y_val, y_val_pred)
        auc  = roc_auc_score(y_val, y_val_proba)

        f_general = alpha_acc * acc + alpha_f1 * f1 + alpha_auc * auc

        print(
            f"  C={C:7.3f} | "
            f"Acc={acc:.4f}, "
            f"Prec={prec:.4f}, "
            f"Rec={rec:.4f}, "
            f"F1={f1:.4f}, "
            f"AUC={auc:.4f} | "
            f"f(F)={f_general:.4f}"
        )

        if f_general > best_score:
            best_score    = f_general
            best_C        = C
            best_pipeline = pipeline

    print(f"\nЛучшее значение C по валидации: {best_C} (f(F)={best_score:.4f})")

    # 4. Переобучаем на полном train и сохраняем
    final_pipeline = build_pipeline(best_C)
    final_pipeline.fit(X_train_full, y_train_full)

    joblib.dump(final_pipeline, model_path)
    print(f"Модель сохранена: {model_path}")

    return final_pipeline, X_test, y_test


def predict(
    texts: list,
    model_path: str = "detector.pkl",
    model_name: str = "gpt2"
) -> np.ndarray:
    gpt2_model, tokenizer = load_gpt2(model_name)
    pipeline = joblib.load(model_path)

    X = np.array([
        extract_features(t, gpt2_model, tokenizer)
        for t in texts
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=10000.0, neginf=0.0)
    return pipeline.predict(X)