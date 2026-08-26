"""Compute the Natural Image Quality Evaluator (NIQE) metric.

This module implements the NIQE metric presented in [Mittal et al.](https://doi.org/10.1109/LSP.2012.2227726).

It provides functions to compute NIQE features and scores for individual images or
all images in a directory. Model parameters (mean and covariance) for pristine
images are loaded from a `.mat` file, and Mahalanobis distance is used to compare
features of pristine and distorted images.

Usage:
    For an image:
    >>> niqe_score = compute_niqe("path_to_colored_image")

    For a video:
    >>> niqe_score = compute_niqe_dir("path_to_colored_frames_dir")
"""

import logging
from math import nan
from pathlib import Path

import numpy as np
from matpy import im2gray, imresize
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.linalg import pinv
from scipy.ndimage import convolve
from scipy.special import gamma as gamma_func
from scipy.special import rgamma as rgamma_func
from skimage.io import imread

from colorization_metrics.metrics.brisque import (
    AGGD_EPSILON,
    FeaturesUsed,
    compute_2d_gaussian_weights,
    compute_channel_products,
    compute_pairwise_products,
)
from colorization_metrics.utils import METRICS_MODELS_DIR, get_dir_imgs

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


MODEL_FILE = Path("data") / "modelparameters.mat"
"""Path to the MVG model for NIQE computation."""
SHARPNESS_THRESHOLD_PERCENTAGE = 0.75
"""Sharpness threshold used to keep only interesting patches in model estimation."""

# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------


class NoValidDataError(Exception):
    """Exception raised when the matrix contains only rows with NaN values."""

    def __init__(  # noqa: D107
        self,
        message: str = "The matrix contains only rows with NaN values.\
No valid data available for computation.",
    ) -> None:
        self.message = message
        super().__init__(self.message)


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def compute_niqe(
    image_path: str | Path,
    method: str | None = None,
    block_size: tuple[int, int] = (96, 96),
    overlap: tuple[int, int] = (0, 0),
) -> float:
    """Compute the NIQE score for an image.

    It computes the Equation 26 of our article.

    Args:
        image_path: Path to the image file.
        method: Name of the method used. Use original model if None.
        block_size: (height, width) of blocks to divide the image.
        overlap: (vertical, horizontal) overlap of blocks.

    Returns:
        Computed NIQE score for the image.
    """
    if method:
        model_file = METRICS_MODELS_DIR / ("NIQE_" + method)
    else:
        method = FeaturesUsed.ORIGINAL.value
        model_file = MODEL_FILE

    image = imread(image_path)

    features = compute_niqe_features(image, block_size, overlap, method)

    mu_model, cov_model = load_niqe_model(model_file)

    mu_image = np.nanmean(features, axis=0)
    try:
        cov_image = nancov(features)
    except NoValidDataError:
        return nan

    return compute_niqe_score(mu_image, cov_image, mu_model, cov_model)


def compute_niqe_dir(images_dir: str | Path, method: str | None = None) -> float:
    """Compute the average NIQE score for all images in a directory.

    It computes the Equation 26 of our article.

    Args:
        images_dir: Directory containing the images.
        method: Name of the method used. Use original model if None.

    Returns:
        Average NIQE score for all images in the directory.
    """
    image_list = get_dir_imgs(images_dir)
    if not image_list:
        msg = "No valid images found in the directory."
        raise ValueError(msg)

    total_score = 0.0
    for image_path in image_list:
        full_image_path = Path(images_dir) / image_path
        total_score += compute_niqe(full_image_path, method)

    return total_score / len(image_list)


def nancov(
    x: NDArray[np.float64], normalize_by_n: bool = False, pairwise: bool = False
) -> NDArray[np.float64]:
    """Compute the covariance of X, removing NaN values.

    Args:
        x: Input array or matrix with NaN values.
        normalize_by_n: If True, normalize by n instead of n-1.
        pairwise: If True, compute pairwise covariance.

    Returns:
        Covariance matrix or sample variance.
    """
    match x.ndim:
        case 1:
            # Remove NaN values
            clean_data = x[~np.isnan(x)]
            if len(clean_data) == 0:
                msg = "Only NaN values"
                raise NoValidDataError(msg)
            n = len(clean_data)
            mean_clean = np.mean(clean_data)
            var = np.sum((clean_data - mean_clean) ** 2)
            return var / (n - 1) if n > 1 else var  # Sample variance

        case 2:
            if pairwise:
                # Compute pairwise covariance
                n_cols = x.shape[1]
                cov_matrix = np.full((n_cols, n_cols), np.nan)
                for i in range(n_cols):
                    for j in range(n_cols):
                        # Get the non-NaN values for the current column pair
                        valid_mask = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
                        valid_data = x[valid_mask, [i, j]]
                        if valid_data.shape[0] > 1:
                            cov_matrix[i, j] = np.cov(
                                valid_data,
                                rowvar=False,
                                ddof=1 if not normalize_by_n else 0,
                            )[0, 1]
                return cov_matrix

            # Remove rows with any NaN values
            clean_data = x[~np.isnan(x).any(axis=1)]
            if clean_data.shape[0] == 0:
                msg = "Only NaN values"
                raise NoValidDataError(msg)

            n = clean_data.shape[0]
            # Remove the mean and calculate covariance
            mean_clean = np.mean(clean_data, axis=0)
            centered_data = clean_data - mean_clean
            cov_matrix = (centered_data.T @ centered_data) / (n - 1 if n > 1 else n)

            return cov_matrix if not normalize_by_n else cov_matrix * (n / (n - 1))

        case _:
            msg = "Input must be a 1D or 2D array."
            raise ValueError(msg)


def compute_niqe_features(
    image: NDArray[np.number],
    block_size: tuple[int, int] = (96, 96),
    overlap: tuple[int, int] = (0, 0),
    method: str = "original",
    compute_sharpness: bool = False,
) -> NDArray[np.float64]:
    """Compute NIQE features from the image by processing blocks.

    Args:
        image: Input image as a NumPy array.
        block_size: (height, width) of blocks to divide the image.
        overlap: (vertical, horizontal) overlap of blocks.
        method: Name of the method used for features computation.
        compute_sharpness: Filter the blocks features by their sharpness.

    Returns:
        Array of NIQE features.
    """
    features = []
    downscale_factor = 2

    height, width = image.shape[:2]
    new_height = height // block_size[0]
    new_width = width // block_size[1]
    image_cut = image[: new_height * block_size[0], : new_width * block_size[1], :]

    sharpness_list = []
    for scale in (0, 1):
        img_r = imresize(
            image_cut[:, :, 0].astype(np.float64), 1 / downscale_factor**scale
        )
        img_g = imresize(
            image_cut[:, :, 1].astype(np.float64), 1 / downscale_factor**scale
        )
        img_b = imresize(
            image_cut[:, :, 2].astype(np.float64), 1 / downscale_factor**scale
        )
        img_lum = imresize(
            im2gray(image_cut).astype(np.float64), 1 / downscale_factor**scale
        )
        normalized_r = _normalize_image_intensity(img_r)
        normalized_g = _normalize_image_intensity(img_g)
        normalized_b = _normalize_image_intensity(img_b)
        normalized_lum = _normalize_image_intensity(img_lum)
        sigma = _get_local_std(img_lum)
        img_features = []
        current_scale_factor = downscale_factor**scale
        image_height, image_width = img_lum.shape[:2]
        block_height_scaled = block_size[0] / current_scale_factor
        block_width_scaled = block_size[1] / current_scale_factor
        overlap_y_scaled = overlap[0] / current_scale_factor
        overlap_x_scaled = overlap[1] / current_scale_factor

        for y in range(
            0,
            int(image_height - block_height_scaled + 1),
            int(block_height_scaled - overlap_y_scaled),
        ):
            for x in range(
                0,
                int(image_width - block_width_scaled + 1),
                int(block_width_scaled - overlap_x_scaled),
            ):
                if scale == 0:
                    sharpness_list.append(
                        np.mean(
                            sigma[
                                int(y) : int(y + block_height_scaled),
                                int(x) : int(x + block_width_scaled),
                            ]
                        )
                    )
                block_l = normalized_lum[
                    int(y) : int(y + block_height_scaled),
                    int(x) : int(x + block_width_scaled),
                ]
                block_r = normalized_r[
                    int(y) : int(y + block_height_scaled),
                    int(x) : int(x + block_width_scaled),
                ]
                block_g = normalized_g[
                    int(y) : int(y + block_height_scaled),
                    int(x) : int(x + block_width_scaled),
                ]
                block_b = normalized_b[
                    int(y) : int(y + block_height_scaled),
                    int(x) : int(x + block_width_scaled),
                ]
                block_feature = _compute_block_features(
                    block_l, block_r, block_g, block_b, method
                )
                img_features.append(block_feature)

        features.append(np.array(img_features))
    if compute_sharpness:
        sharpness_threshold = SHARPNESS_THRESHOLD_PERCENTAGE * max(sharpness_list)
        mask = np.array(sharpness_list) > sharpness_threshold
        filtered_features = [scaled_features[mask] for scaled_features in features]
    else:
        filtered_features = features

    return np.concatenate(filtered_features, axis=1)


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def compute_niqe_score(
    mu_image: NDArray[np.float64],
    cov_image: NDArray[np.float64],
    mu_model: NDArray[np.float64],
    cov_model: NDArray[np.float64],
) -> float:
    """Computes the Naturalness Image Quality Evaluator (NIQE) score.

    Args:
        mu_image: Mean vector of the Multivariate Gaussian (MVG)
            model of the evaluated image.
        cov_image: Covariance matrix of the MVG model of the
            evaluated image.
        mu_model: Mean vector of the MVG model of the pristine
            dataset.
        cov_model: Covariance matrix of the MVG model of the
            pristine dataset.

    Returns:
        The computed NIQE score, which indicates the quality of the evaluated
            image relative to the pristine dataset. Lower scores typically indicate
            better quality.
    """
    invcov_param = pinv((cov_model + cov_image) / 2)
    return float(
        np.sqrt(
            np.transpose(mu_model - mu_image) @ invcov_param @ (mu_model - mu_image)
        )
    )


def load_niqe_model(
    model_file: str | Path = MODEL_FILE,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load NIQE model parameters (mean and covariance) from a .mat file.

    Args:
        model_file: Path to the .mat file containing NIQE model parameters.

    Returns:
        A tuple (mu_pristine, cov_pristine):
        - mu_pristine: Mean of the pristine model parameters.
        - cov_pristine: Covariance of the pristine model parameters.
    """
    data = loadmat(str(model_file))
    mu_pristine = data["mu_prisparam"].squeeze()
    cov_pristine = data["cov_prisparam"].squeeze()
    return mu_pristine, cov_pristine


def _normalize_image_intensity(
    image: NDArray[np.number], epsilon: float = 1.0
) -> NDArray[np.float64]:
    """Normalize the intensity of an image using local mean and standard deviation.

    Convolutions are done with border replication, the same way as it is done
    in the official MATLAB implementation with the `imfilter` function
    with `replicate` parameter.

    Args:
        image: Input image, either grayscale or RGB.
        epsilon: Small constant to avoid division by zero.

    Returns:
        The normalized grayscale image.
    """
    grayscale_image = im2gray(image).astype(np.float64)
    weights = compute_2d_gaussian_weights()
    local_mean = convolve(grayscale_image, weights, mode="nearest")
    local_std_dev = np.sqrt(
        abs(convolve(grayscale_image**2, weights, mode="nearest") - local_mean**2)
    )
    logger = logging.getLogger("NIQE normalization")
    logger.debug("Image:")
    logger.debug(grayscale_image.dtype)
    logger.debug(grayscale_image)
    logger.debug("Weights:")
    logger.debug(weights.dtype)
    logger.debug(weights)
    logger.debug("Mean:")
    logger.debug(local_mean.dtype)
    logger.debug(local_mean)
    logger.debug("Std:")
    logger.debug(local_std_dev.dtype)
    logger.debug(local_std_dev)

    return (grayscale_image - local_mean) / (local_std_dev + epsilon)


def _get_local_std(image: NDArray[np.number]) -> NDArray[np.float64]:
    """Compute the local standard deviation of an image.

    It computes the Equation 18 of our article. Convolutions are done with border
    replication, the same way as it is done in the official MATLAB implementation
    with the `imfilter` function with `replicate` parameter.

    Args:
        image: Input image, either grayscale or RGB.

    Returns:
        The local standard deviation.
    """
    grayscale_image = im2gray(image)
    weights = compute_2d_gaussian_weights()
    local_mean = convolve(grayscale_image, weights, mode="nearest")
    return np.sqrt(
        abs(convolve(grayscale_image**2, weights, mode="nearest") - local_mean**2)
    )


def _compute_aggd_parameters(
    pairwise_product: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Estimate AGGD parameters from pairwise product values.

    Args:
        pairwise_product: 1D array of pairwise product values.

    Returns:
        A tuple containing (nu, left_beta, right_beta).
    """
    nu_range = np.arange(0.2, 10, 0.001)
    ratio_gamma = (gamma_func(2 / nu_range) ** 2) * (
        rgamma_func(1 / nu_range) * rgamma_func(3 / nu_range)
    )

    left_std = (
        np.sqrt(np.mean(pairwise_product[pairwise_product < -AGGD_EPSILON] ** 2))
        if np.any(pairwise_product < -AGGD_EPSILON)
        else 0
    )
    right_std = (
        np.sqrt(np.mean(pairwise_product[pairwise_product >= AGGD_EPSILON] ** 2))
        if np.any(pairwise_product >= AGGD_EPSILON)
        else AGGD_EPSILON
    )
    if right_std == 0:
        right_std = AGGD_EPSILON
    asymmetry = left_std / right_std
    mean_abs = np.mean(np.abs(pairwise_product))
    mean_sq = np.mean(pairwise_product**2)
    if mean_sq == 0:
        mean_sq = AGGD_EPSILON

    ratio_est = mean_abs**2 / mean_sq
    ratio_est_norm = (ratio_est * (asymmetry**3 + 1) * (asymmetry + 1)) / (
        (asymmetry**2 + 1) ** 2
    )

    index = np.argmin(np.abs(ratio_gamma - ratio_est_norm))
    nu_est = nu_range[index]

    left_beta = left_std * np.sqrt(gamma_func(1 / nu_est) * rgamma_func(3 / nu_est))
    right_beta = right_std * np.sqrt(gamma_func(1 / nu_est) * rgamma_func(3 / nu_est))

    return nu_est, left_beta, right_beta


def _compute_block_features(
    block_l: NDArray[np.float64],
    block_r: NDArray[np.float64],
    block_g: NDArray[np.float64],
    block_b: NDArray[np.float64],
    method: str = "original",
) -> NDArray[np.float64]:
    """Compute features for a given block of the image.

    Args:
        block_l: Luminance block of the image.
        block_r: Red channel block of the image.
        block_g: Green channel block of the image.
        block_b: Blue channel block of the image.
        method: Name of the method used for features computation.

    Returns:
        1D array of NIQE features for the block.
    """
    block_features = []

    match method:
        case FeaturesUsed.ORIGINAL.value:
            nu, left_beta, right_beta = _compute_aggd_parameters(np.ravel(block_l, "F"))
            block_features.extend([nu, (left_beta + right_beta) / 2])

            for pairwise_product in compute_pairwise_products(block_l):
                nu, left_beta, right_beta = _compute_aggd_parameters(
                    np.ravel(pairwise_product, "F")
                )
                eta = (
                    (right_beta - left_beta) * gamma_func(2 / nu) * rgamma_func(1 / nu)
                )
                block_features.extend([nu, eta, left_beta, right_beta])
        case FeaturesUsed.RGB_CORRELATION.value:
            nu, left_beta, right_beta = _compute_aggd_parameters(np.ravel(block_l, "F"))
            block_features.extend([nu, (left_beta + right_beta) / 2])

            for pairwise_product in compute_pairwise_products(block_l):
                nu, left_beta, right_beta = _compute_aggd_parameters(
                    np.ravel(pairwise_product, "F")
                )
                eta = (
                    (right_beta - left_beta) * gamma_func(2 / nu) * rgamma_func(1 / nu)
                )
                block_features.extend([nu, eta, left_beta, right_beta])
            for pairwise_product in compute_channel_products(block_r, block_g, block_b):
                nu, left_beta, right_beta = _compute_aggd_parameters(
                    np.ravel(pairwise_product, "F")
                )
                eta = (
                    (right_beta - left_beta) * gamma_func(2 / nu) * rgamma_func(1 / nu)
                )
                block_features.extend([nu, eta, left_beta, right_beta])
        case FeaturesUsed.RGB_ANALYSIS.value:
            for block in (block_r, block_g, block_b):
                nu, left_beta, right_beta = _compute_aggd_parameters(
                    np.ravel(block, "F")
                )
                block_features.extend([nu, (left_beta + right_beta) / 2])

                for pairwise_product in compute_pairwise_products(block):
                    nu, left_beta, right_beta = _compute_aggd_parameters(
                        np.ravel(pairwise_product, "F")
                    )
                    eta = (
                        (right_beta - left_beta)
                        * gamma_func(2 / nu)
                        * rgamma_func(1 / nu)
                    )
                    block_features.extend([nu, eta, left_beta, right_beta])
        case FeaturesUsed.RGB_ALL.value:
            for block in (block_r, block_g, block_b):
                nu, left_beta, right_beta = _compute_aggd_parameters(
                    np.ravel(block, "F")
                )
                block_features.extend([nu, (left_beta + right_beta) / 2])

                for pairwise_product in compute_pairwise_products(block):
                    nu, left_beta, right_beta = _compute_aggd_parameters(
                        np.ravel(pairwise_product, "F")
                    )
                    eta = (
                        (right_beta - left_beta)
                        * gamma_func(2 / nu)
                        * rgamma_func(1 / nu)
                    )
                    block_features.extend([nu, eta, left_beta, right_beta])
            for pairwise_product in compute_channel_products(block_r, block_g, block_b):
                nu, left_beta, right_beta = _compute_aggd_parameters(
                    np.ravel(pairwise_product, "F")
                )
                eta = (
                    (right_beta - left_beta) * gamma_func(2 / nu) * rgamma_func(1 / nu)
                )
                block_features.extend([nu, eta, left_beta, right_beta])

    return np.array(block_features)


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute the NIQE score for an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "-m", default=None, required=False, type=str, help="Method to use"
    )

    args = parser.parse_args()

    print(compute_niqe(args.image_path, args.m))  # noqa: T201
