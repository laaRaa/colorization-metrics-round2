"""Estimate the Blind/Referenceless Image Spatial QUality Evaluator (BRISQUE) model.

This module implements the BRISQUE model estimation presented in [Mittal et al.](https://doi.org/10.1109/TIP.2012.2214050).
"""

import concurrent
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from libsvm import svmutil
from numpy.typing import NDArray
from skimage.io import imread
from skimage.util import img_as_ubyte
from tqdm import tqdm

from colorization_metrics.metrics.brisque import (
    FeaturesUsed,
    compute_brisque_features,
    normalize_features,
)
from colorization_metrics.utils import METRICS_MODELS_DIR

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


LIVE_IQA_DIR = Path("data/live_iqa")
"""Path to the LIVE IQA dataset."""
LIVE_IQA_IMAGES_DIR = LIVE_IQA_DIR / "sorted"
"""Path to the LIVE IQA images used in training."""
LIVE_IQA_DMOS_FILE = LIVE_IQA_DIR / "dmos_sorted.json"
"""Path to the DMOS human score."""


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def train(
    dmos_file: str | Path = LIVE_IQA_DMOS_FILE,
    model_path: str | Path = METRICS_MODELS_DIR,
    method: str = "original",
    gamma: float = 0.05,
    cost: float = 1024.0,
    epsilon: float = 0.1,
    probability_estimates: bool = True,
) -> None:
    """Train a Support Vector Regressor (SVR) model on BRISQUE features and DMOS values.

    This function performs SVR model training using BRISQUE features and DMOS values,
    and saves the trained model to the specified path.

    Args:
        dmos_file: Path to the NumPy file containing DMOS values.
        model_path: Directory path where the trained model will be saved.
        method: Name of the feature extraction method used.
        gamma: Kernel coefficient for the SVR model.
        cost: Regularization parameter for the SVR model.
        epsilon: Epsilon parameter for the SVR model.
        probability_estimates: Whether to enable probability estimates.
    """
    # Load DMOS values
    dmos_path = Path(dmos_file)
    if not dmos_path.exists():
        msg = f"The file '{dmos_file}' does not exist."
        raise FileNotFoundError(msg)
    with dmos_path.open(encoding="utf8") as f:
        datas = json.load(f)
    dmos = np.array(datas["dmos"])

    # Extract BRISQUE features
    features_file = Path(model_path) / f"BRISQUE_{method}_features.json"
    if features_file.exists():
        print("Load already computed features.")  # noqa: T201
        with features_file.open(encoding="utf8") as f:
            features = np.array(json.load(f))
    else:
        print("Compute and save features.")  # noqa: T201
        features = extract_features_from_dataset(method)
        with features_file.open("w", encoding="utf8") as f:
            json.dump(features.tolist(), f, indent=4)

    # Normalizes features
    model_name = "BRISQUE" + "_" + method
    limits_file = Path(model_path) / (model_name + "_range")
    if not limits_file.is_file():
        limits = np.array([np.min(features, axis=0), np.max(features, axis=0)])
        write_features_limits(limits_file, limits)
    normalized_features = normalize_features(features, limits_file)

    # Train the kept model on all dataset
    options = f"_gamma_{gamma}_cost_{cost}_epsilon_{epsilon}"
    regressor = svmutil.svm_train(
        dmos,
        normalized_features,
        f"-q -s 3 -t 2 -g {gamma} -c {cost} -p {epsilon} -b {int(probability_estimates)}",  # noqa: E501
    )
    model_file = Path(model_path) / model_name
    model_param_file = model_file.with_name(model_file.name + options)
    svmutil.svm_save_model(str(model_param_file), regressor)
    shutil.copy(model_param_file, model_file)


def write_features_limits(
    file_path: str | Path,
    feature_limits: NDArray[np.float64],
    expected_range: tuple[int, int] = (-1, 1),
) -> None:
    """Writes the feature limits to the given file.

    Args:
        file_path: Path to the output text file.
        feature_limits: Array of shape (2, nb_features) containing the feature limits.
        expected_range: Represents the expected range of features after normalization.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(file_path).open("w", encoding="utf8") as file:
        file.write("x\n")

        # Write the expected range
        file.write(f"{expected_range[0]} {expected_range[1]}\n")

        # Write the feature limits
        file.writelines(
            f"{i + 1} {feature_limits[0, i]} {feature_limits[1, i]}\n"
            for i in range(feature_limits.shape[1])
        )


def extract_features_from_dataset(
    method: str = FeaturesUsed.ORIGINAL.value,
    images_dir: str | Path = LIVE_IQA_IMAGES_DIR,
) -> NDArray[np.float64]:
    """Extract BRISQUE features from the dataset images using parallel processing.

    This function loads the images from the LIVE IQA dataset and computes the BRISQUE
    features for each image using multiple processes.

    Returns:
        Extracted BRISQUE features as a NumPy array of shape [img_nb, feat_nb].
    """
    image_paths = sorted(Path(images_dir).glob("*.bmp"))

    features = []
    indices = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_image = {
            executor.submit(_process_image_path, img_path, i, method): img_path
            for i, img_path in enumerate(image_paths)
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_image),
            total=len(image_paths),
            desc="Features extraction",
            ncols=90,
        ):
            result, ind = future.result()
            features.append(result)
            indices.append(ind)

    features = [feat for _, feat in sorted(zip(indices, features, strict=False))]
    return np.array(features, ndmin=2, dtype=np.float64)


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _process_image_path(
    image_path: str | Path, list_index: int, method: str
) -> NDArray[np.float64]:
    """Load an image and compute its BRISQUE features.

    Args:
        image_path: Path to the image file.
        list_index: Index of the image.
        method: Feature extraction method (e.g., 'original').

    Returns:
        The BRISQUE features extracted from the image and its index.
    """
    img = img_as_ubyte(imread(image_path))
    return compute_brisque_features(img, method), list_index


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the BRISQUE regressor on the LIVE IQA dataset."
    )
    parser.add_argument(
        "-m",
        "--method",
        default="original",
        required=False,
        type=str,
        help="Method to use (default : %(default)s).",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        default=0.05,
        required=False,
        type=float,
        help="Parameter gamma of the epsilon-SVR (default : %(default)s).",
    )
    parser.add_argument(
        "-c",
        "--cost",
        default=1024.0,
        required=False,
        type=float,
        help="Parameter C of the epsilon-SVR (default : %(default)s).",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        default=0.1,
        required=False,
        type=float,
        help="Parameter epsilon of the epsilon-SVR (default : %(default)s).",
    )

    args = parser.parse_args()

    sys.exit(
        train(
            method=args.method, gamma=args.gamma, cost=args.cost, epsilon=args.epsilon
        )
    )
