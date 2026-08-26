"""Script to search hyperparamaters for training BRISQUE."""

import json
from pathlib import Path

import numpy as np
from libsvm import svmutil
from numpy.typing import NDArray
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from colorization_metrics.metrics.brisque import FeaturesUsed
from colorization_metrics.metrics.train_brisque import (
    LIVE_IQA_DMOS_FILE,
    extract_features_from_dataset,
    train,
)
from colorization_metrics.utils import METRICS_MODELS_DIR


class BRISQUESVR(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        cost: float = 1.0,
        epsilon: float = 0.1,
        gamma: float = 0.05,
        probability_estimates: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        self.cost = cost
        self.epsilon = epsilon
        self.gamma = gamma
        self.probability_estimates = probability_estimates
        self.model = None

    def fit(self, x: list | tuple | NDArray, y: list | tuple | NDArray):
        self.model = svmutil.svm_train(
            y,
            x,
            f"-q -s 3 -t 2 -g {self.gamma} -c {self.cost} -p {self.epsilon} -b {int(self.probability_estimates)}",  # noqa: E501
        )
        return self

    def predict(self, x: list | tuple | NDArray) -> NDArray:
        dummy_labels = [0] * len(x)
        predictions, _, _ = svmutil.svm_predict(dummy_labels, x, self.model, "-q")
        return np.array(predictions)

    def get_params(self, deep: bool = True) -> dict[float, float, float, bool]:  # noqa: ARG002, FBT001, FBT002
        return {
            "cost": self.cost,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "probability_estimates": self.probability_estimates,
        }

    def set_params(self, **params: float | bool):
        for param, value in params.items():
            setattr(self, param, value)
        return self


def _sort_by_first_list(
    first_list: list[float], second_list: list[float]
) -> tuple[list[float], list[float]]:
    """Sorts two lists based on the values of the first list.

    Args:
        first_list: The primary list to sort by.
        second_list: The secondary list to sort based on the first list.

    Returns:
        Two lists sorted based on the values of the first list.

    Raises:
        ValueError: If the input lists do not have the same length.
    """
    if len(first_list) != len(second_list):
        msg = "Both lists should have same length."
        raise ValueError(msg)

    first_list_sorted = sorted(first_list)
    second_list_sorted = [
        y for _, y in sorted(zip(first_list, second_list, strict=False))
    ]

    return first_list_sorted, second_list_sorted


def main() -> None:
    # Data loading
    if not Path.exists(LIVE_IQA_DMOS_FILE):
        msg = f"The file '{LIVE_IQA_DMOS_FILE}' does not exist."
        raise FileNotFoundError(msg)
    with Path.open(LIVE_IQA_DMOS_FILE, encoding="utf8") as f:
        datas = json.load(f)
    y_dmos = np.array(datas["dmos"])

    for method in [
        FeaturesUsed.ORIGINAL.value,
        FeaturesUsed.RGB_CORRELATION.value,
        FeaturesUsed.RGB_ANALYSIS.value,
        FeaturesUsed.RGB_ALL.value,
        FeaturesUsed.CIELAB.value,
    ]:
        print("-- Method: " + method)  # noqa: T201
        features_file = Path(METRICS_MODELS_DIR) / f"BRISQUE_{method}_features.json"
        features_file.parent.mkdir(parents=True, exist_ok=True)

        if Path.exists(features_file):
            print(f"Load already computed features. Saved at {features_file}")  # noqa: T201
            with Path.open(features_file, encoding="utf8") as f:
                x_features = np.array(json.load(f))
        else:
            print(f"Compute and save features at {features_file}")  # noqa: T201
            x_features = extract_features_from_dataset(method)
            with Path.open(features_file, "w", encoding="utf8") as f:
                json.dump(x_features.tolist(), f, indent=4)

        # Split en train/test
        x_train, x_test, y_train, y_test = train_test_split(
            x_features, y_dmos, test_size=0.1, random_state=42
        )

        param_distributions = {
            "cost": [1024],
            "epsilon": loguniform(1e-1, 1e2),
            "gamma": [0.05],
            "probability_estimates": [True],
        }

        search = RandomizedSearchCV(
            BRISQUESVR(),
            param_distributions,
            n_iter=int(1e3),
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
            verbose=1,
            n_jobs=-1,
        )

        # Hyperparameters search on train set
        search.fit(x_train, y_train)

        print("Best hyperparameters:", search.best_params_)  # noqa: T201

        # Evaluation on test set
        y_pred = search.best_estimator_.predict(x_test)
        test_mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE: {test_mse:.4f}")  # noqa: T201

        # Final model training
        train(method=method, epsilon=search.best_params_["epsilon"])


if __name__ == "__main__":
    main()
