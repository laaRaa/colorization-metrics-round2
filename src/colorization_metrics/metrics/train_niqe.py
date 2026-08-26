"""Estimate the Natural Image Quality Evaluator (NIQE) model.

This module implements the NIQE model estimation presented in [Mittal et al.](https://doi.org/10.1109/LSP.2012.2227726).
"""

import concurrent
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import savemat
from skimage.io import imread
from skimage.util import img_as_ubyte
from tqdm import tqdm

from colorization_metrics.metrics.brisque import FeaturesUsed
from colorization_metrics.metrics.niqe import compute_niqe_features, nancov
from colorization_metrics.utils import METRICS_MODELS_DIR

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


PRISTINE_IMAGES_DIR = Path("data") / "pristine"
"""Path to the pristine dataset."""


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def estimate_model(
    model_path: str | Path = METRICS_MODELS_DIR / "NIQE", method: str = "original"
) -> None:
    """Train a Support Vector Regressor (SVR) model on BRISQUE features and DMOS values.

    This function performs multiple iterations of SVR model training using random
    train/test splits, evaluates the performance using the Spearman rank correlation
    coefficient, and saves the median performing model.

    Args:
        model_path: Path to the .mat file where to save NIQE model parameters.
        method: Name of the method used.
    """
    # Extract NIQE features
    features = _extract_features_from_dataset(method)

    # Estimate the model
    mu_training = np.nanmean(features, axis=0)
    cov_training = nancov(features)

    # Save the model parameters for each iteration
    model_file = model_path.with_name(model_path.name + f"_{method}.mat")
    _save_niqe_model(mu_training, cov_training, model_file)
    print(f"NIQE parameters saved to {model_file}")  # noqa: T201


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _extract_features_from_dataset(
    method: str = "original", dataset_dir: str | Path = PRISTINE_IMAGES_DIR
) -> NDArray[np.float64]:
    """Extract NIQE features from the dataset images.

    This function loads the images from the LIVE IQA dataset and computes the NIQE
    features for each image.

    Returns:
        Extracted NIQE features as a NumPy array of shape [img_block_nb, feat_nb].
    """
    return np.concatenate(
        extract_features_from_dataset_listed(method, dataset_dir), axis=0
    )


def extract_features_from_dataset_listed(
    method: str = FeaturesUsed.ORIGINAL.value,
    dataset_dir: str | Path = PRISTINE_IMAGES_DIR,
    compute_sharpness: bool = True,
) -> list[NDArray[np.float64]]:
    """Extract NIQE features from the dataset images.

    This function loads the images from the LIVE IQA dataset and computes the NIQE
    features for each image.

    Returns:
        Extracted NIQE features as a list of length img_nb
            of NumPy array of shape [block_nb, feat_nb].
    """
    image_paths = sorted(Path(dataset_dir).glob("*"))
    images = [img_as_ubyte(imread(image_path)) for image_path in image_paths]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_image = {
            executor.submit(
                partial(
                    compute_niqe_features,
                    method=method,
                    compute_sharpness=compute_sharpness,
                ),
                img,
            ): img
            for img in images
        }
        return [
            feat.result()
            for feat in tqdm(
                concurrent.futures.as_completed(future_to_image),
                total=len(images),
                desc="Features extraction",
                ncols=90,
            )
        ]


def _save_niqe_model(
    mu: NDArray[np.float64], cov: NDArray[np.float64], model_path: str | Path
) -> None:
    """Save NIQE model parameters (mean and covariance) to a .mat file.

    Args:
        mu: Mean vector of the MultiVariate Gaussian model.
        cov: Covariance matrice of the MultiVariate Gaussian model.
        model_path: Path to the .mat file where to save NIQE model parameters.
    """
    METRICS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"mu_prisparam": mu, "cov_prisparam": cov}
    savemat(model_path, data)


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        default=FeaturesUsed.ORIGINAL.value,
        type=str,
        choices=[m.value for m in FeaturesUsed],
        help="Feature selection method (default : %(default)s)",
    )

    args = parser.parse_args()
    estimate_model(method=args.method)
