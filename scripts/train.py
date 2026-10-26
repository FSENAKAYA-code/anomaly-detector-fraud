"""Run an imbalanced synthetic fraud-classification demo."""

from sklearn.datasets import make_classification

from fraud.experiment import run_fraud_experiment


def main() -> None:
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_informative=6,
        weights=[0.97, 0.03],
        random_state=42,
    )
    print(run_fraud_experiment(X, y, random_state=42))


if __name__ == "__main__":
    main()
