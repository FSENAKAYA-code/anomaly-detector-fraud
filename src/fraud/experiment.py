"""Imbalance-aware baseline with a stratified validation and held-out test set."""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from fraud.splits import split_data


def run_fraud_experiment(X: Any, y: Any, *, random_state: int = 42) -> dict[str, float]:
    parts = split_data(X, y, random_state=random_state)
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(parts.X_train, parts.y_train)
    validation_predictions = model.predict(parts.X_validation)
    validation_f1 = f1_score(
        parts.y_validation, validation_predictions, average="binary", zero_division=0
    )
    import numpy as np

    X_final = np.concatenate([parts.X_train, parts.X_validation], axis=0)
    y_final = np.concatenate([parts.y_train, parts.y_validation], axis=0)
    model.fit(X_final, y_final)
    predictions = model.predict(parts.X_test)
    return {
        "validation_f1": float(validation_f1),
        "test_precision": float(
            precision_score(parts.y_test, predictions, zero_division=0)
        ),
        "test_recall": float(recall_score(parts.y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(parts.y_test, predictions, zero_division=0)),
    }
