import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)


def evaluate(X_test: np.ndarray, y_test: np.ndarray, pipeline) -> dict:
    """
    Принимает готовый pipeline и тестовую выборку.
    Никаких повторных загрузок модели или признаков.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_true = np.array(y_test)

    metrics = {
        "F1-score":  f1_score(y_true, y_pred),
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "AUROC":     roc_auc_score(y_true, y_proba)
    }

    print("\n=== Оценка качества алгоритма на HC3 ===")
    for name, value in metrics.items():
        marker = " <-- основная метрика" if name == "F1-score" else ""
        print(f"  {name:<12}: {value:.4f}{marker}")

    print("\n=== Подробный отчёт ===")
    print(classification_report(
        y_true, y_pred,
        target_names=["Человеческий", "Синтетический"]
    ))

    return metrics
