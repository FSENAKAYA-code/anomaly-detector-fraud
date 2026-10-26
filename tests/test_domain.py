import pytest

from fraud.metrics import fraud_metrics


def test_metrics_report_zero_for_no_positive_predictions():
    assert fraud_metrics([0, 1], [0, 0])["recall"] == 0

def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        fraud_metrics([], [])
