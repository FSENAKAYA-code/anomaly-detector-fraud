"""Validation-thresholded unsupervised anomaly baselines."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM


def run_unsupervised_anomaly(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    algorithm: str = "isolation_forest",
    random_state: int = 42,
) -> dict[str, float]:
    """Fit only normal training rows and select an anomaly threshold on validation."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2 or labels.ndim != 1 or len(X) != len(labels):
        raise ValueError("X and labels must be aligned 2D features and 1D labels")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must use 0 for normal and 1 for anomaly")
    if algorithm == "isolation_forest":
        model = IsolationForest(contamination="auto", random_state=random_state)
    elif algorithm == "one_class_svm":
        model = OneClassSVM(nu=0.05, gamma="scale")
    else:
        raise ValueError("algorithm must be isolation_forest or one_class_svm")
    indices = np.arange(len(labels))
    train_validation, test = train_test_split(
        indices, test_size=0.2, random_state=random_state, stratify=labels
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=random_state,
        stratify=labels[train_validation],
    )
    normal_train = train[labels[train] == 0]
    if len(normal_train) < 5:
        raise ValueError("at least five normal training examples are required")
    model.fit(X[normal_train])
    validation_scores = -model.decision_function(X[validation])
    candidates = np.unique(validation_scores)
    threshold = max(
        candidates,
        key=lambda value: f1_score(
            labels[validation], validation_scores >= value, zero_division=0
        ),
    )
    validation_predictions = validation_scores >= threshold
    test_scores = -model.decision_function(X[test])
    test_predictions = test_scores >= threshold
    train_scores = -model.decision_function(X[normal_train])
    return {
        "train_loss_score_proxy": float(train_scores.mean()),
        "validation_loss_1_minus_f1": float(
            1 - f1_score(labels[validation], validation_predictions, zero_division=0)
        ),
        "test_precision": float(
            precision_score(labels[test], test_predictions, zero_division=0)
        ),
        "test_recall": float(recall_score(labels[test], test_predictions, zero_division=0)),
        "test_f1": float(f1_score(labels[test], test_predictions, zero_division=0)),
    }
