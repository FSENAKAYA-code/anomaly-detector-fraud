import numpy as np
import pytest
from sklearn.datasets import make_classification

from fraud.experiment import run_fraud_experiment
from fraud.metrics import fraud_metrics
from fraud.unsupervised import run_unsupervised_anomaly


def test_metrics_report_zero_for_no_positive_predictions():
    assert fraud_metrics([0, 1], [0, 0])["recall"] == 0

def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        fraud_metrics([], [])


def test_fraud_experiment_reports_validation_and_imbalance_metrics():
    X, y = make_classification(
        n_samples=300, n_features=8, weights=[0.9, 0.1], random_state=4
    )
    assert set(run_fraud_experiment(X, y)) == {
        "train_loss",
        "validation_loss",
        "validation_f1",
        "test_precision",
        "test_recall",
        "test_f1",
    }


def test_isolation_forest_selects_validation_threshold():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 4))
    y = np.zeros(300, dtype=int)
    y[-30:] = 1
    X[-30:] += 4
    metrics = run_unsupervised_anomaly(X, y)
    assert "validation_loss_1_minus_f1" in metrics
    assert "test_f1" in metrics
