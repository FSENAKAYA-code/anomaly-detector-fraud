"""Run an imbalanced synthetic fraud-classification demo."""

from sklearn.datasets import make_classification

from fraud.experiment import run_fraud_experiment
from fraud.unsupervised import run_unsupervised_anomaly


def main() -> None:
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_informative=6,
        weights=[0.97, 0.03],
        random_state=42,
    )
    print("class_weighted_classifier:", run_fraud_experiment(X, y, random_state=42))
    print("isolation_forest:", run_unsupervised_anomaly(X, y, random_state=42))
    print("one_class_svm:", run_unsupervised_anomaly(X, y, algorithm="one_class_svm"))


if __name__ == "__main__":
    main()
