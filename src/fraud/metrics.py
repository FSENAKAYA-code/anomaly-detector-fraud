"""Class-imbalance-aware evaluation helpers; scores do not establish fraud."""

from collections.abc import Sequence

from sklearn.metrics import precision_recall_fscore_support


def fraud_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("non-empty, equally sized label sequences are required")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
