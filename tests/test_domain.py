import pytest
from sklearn.datasets import make_classification

from fraud.experiment import run_fraud_experiment
from fraud.metrics import fraud_metrics


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
        "validation_f1", "test_precision", "test_recall", "test_f1"
    }
