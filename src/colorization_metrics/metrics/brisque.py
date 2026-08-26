"""Compute the Blind/Referenceless Image Spatial QUality Evaluator (BRISQUE) metric.

This module implements the BRISQUE metric presented in [Mittal et al.](https://doi.org/10.1109/TIP.2012.2214050).

Usage:
    For an image:
    >>> brisque_score = compute_brisque("path_to_colored_image")

    For a video:
    >>> brisque_score = compute_brisque_dir("path_to_colored_frames_dir")
"""

import logging
from enum import Enum
from pathlib import Path

import numpy as np
from libsvm import svmutil
from matpy import im2gray, imresize
from numpy.typing import NDArray
from scipy.ndimage import convolve
from scipy.special import gamma as gamma_func
from skimage.color import rgb2lab
from skimage.io import imread
from skimage.util import img_as_ubyte

from colorization_metrics.utils import METRICS_MODELS_DIR, get_dir_imgs

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


MODEL_FILE = Path("data") / "allmodel"
"""Path for the MATLAB trained model for BRISQUE computation."""
RANGE_FILE = Path("data") / "allrange"
"""Path for the MATLAB features range for BRISQUE features normalization."""
RESIZE_FACTOR = 2
"""Downsize factor used in features computation."""
AGGD_EPSILON = 1e-10
"""Espilon to avoid values near 0 to interfer in AGGD computation."""


# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------


class FeaturesUsed(Enum):
    """Enumeration for different feature extraction methods.

    Attributes:
        ORIGINAL: Represents the original paper features extraction.
        RGB_CORRELATION: Represents the ORIGINAL method
            with additional RGB correlation feature extraction.
        RGB_FEATURES: Represents the original features computed,
            not on the grayscale version, but on each of the RGB channels.
        RGB_ALL: Represents the RGB_FEATURES method
            with additional RGB correlations features extraction.
        CIELAB: Same model as 'RGB_CORRELATION' but using CIELAB instead.

    The last three methods (RGB_FEATURES, RGB_ALL and CIELAB) are not validated in
    this article. RGB_FEATURES and RGB_ALL are explored in a French paper published
    at ORASIS, a French young researchers conference. More details are available in
    <https://hal.science/hal-05131218v1>.
    """

    ORIGINAL = "original"
    RGB_CORRELATION = "rgb_correl"
    RGB_ANALYSIS = "rgb_analysis"
    RGB_ALL = "rgb_all"
    CIELAB = "cielab"


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def compute_brisque(image_path: str | Path, method: str | None = None) -> float:
    """Compute the BRISQUE score for an image.

    Args:
        image_path: Path to the image file.
        method: Name of the method used. Use original model if None.

    Returns:
        Computed BRISQUE score for the image.
    """
    image = img_as_ubyte(imread(image_path))
    if method:
        model_file = METRICS_MODELS_DIR / ("BRISQUE_" + method)
        range_file = METRICS_MODELS_DIR / ("BRISQUE_" + method + "_range")
    else:
        method = FeaturesUsed.ORIGINAL.value
        model_file = MODEL_FILE
        range_file = RANGE_FILE
    # Computes Equations 19 and 24 parameters
    features = np.array(
        compute_brisque_features(image, method), ndmin=2, dtype=np.float64
    )
    normalized_features = normalize_features(features, range_file)
    model = svmutil.svm_load_model(str(model_file))
    prediction = svmutil.svm_predict([], normalized_features, model, "-q")
    return prediction[0][0]


def compute_brisque_dir(
    images_dir: str | Path,
    model_file: str | Path = MODEL_FILE,
    limits_file: str | Path = RANGE_FILE,
) -> float:
    """Compute the average BRISQUE score for all images in a directory.

    Args:
        images_dir: Directory containing the images.
        model_file: Path to the saved model.
        limits_file: Path to the saved range of LIVE IQA features.

    Returns:
        Average BRISQUE score for all images in the directory.
    """
    image_list = get_dir_imgs(images_dir)
    model = svmutil.svm_load_model(str(model_file))
    total_score = 0
    for image_path in image_list:
        image = img_as_ubyte(imread(Path(images_dir) / image_path))
        features = np.array(compute_brisque_features(image), ndmin=2, dtype=np.float64)
        normalized_features = normalize_features(features, limits_file)
        prediction = svmutil.svm_predict([], normalized_features, model, "-q")
        total_score += prediction[0][0]
    return total_score / len(image_list)


def compute_2d_gaussian_weights(
    size: int = 7, sigma: float = 7.0 / 6.0
) -> NDArray[np.float64]:
    """Compute a 2D Gaussian kernel.

    Args:
        size: The size of the Gaussian kernel.
        sigma: The standard deviation of the Gaussian distribution.

    Returns:
        2D array representing the Gaussian kernel weights.
    """
    radius = size // 2
    odd_center = size % 2
    x, y = np.meshgrid(
        np.arange(-radius, radius + odd_center), np.arange(-radius, radius + odd_center)
    )
    weights = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    weights /= np.sum(weights)
    return weights


def compute_pairwise_products(
    image: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Compute pairwise products of adjacent pixel values in different directions.

    It computes the Equations 20 to 23 of our article.

    Args:
        image: 2D array representing the image.

    Returns:
        Contains four 2D arrays of pairwise products:
            - horizontal_prod: Products of horizontally adjacent pixels.
            - vertical_prod: Products of vertically adjacent pixels.
            - main_diagonal_prod: Products along the main diagonal.
            - secondary_diagonal_prod: Products along the secondary diagonal.
    """
    used_shifts = np.array([[0, 1], [1, 0], [1, 1], [-1, 1]])
    pairwise_products = []
    for shift in used_shifts:
        image_shifted = np.roll(image, shift, axis=(0, 1))
        pairwise_products.append(image.ravel("F") * image_shifted.ravel("F"))
    return tuple(pairwise_products)


def compute_channel_products(
    channel_r: NDArray[np.float64],
    channel_g: NDArray[np.float64],
    channel_b: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute the pairwise products of pixel values between different color channels.

    This function calculates the products of pixel values between the red, green,
    and blue channels of an image.

    Args:
        channel_r: A 2D array representing the red channel of the image.
        channel_g: A 2D array representing the green channel of the image.
        channel_b: A 2D array representing the blue channel of the image.

    Returns:
        Contains three 1D arrays of pairwise products:
            - rg_product: Products of red and green channel pixels.
            - rb_product: Products of red and blue channel pixels.
            - gb_product: Products of green and blue channel pixels.
    """
    rg_product = channel_r.ravel("F") * channel_g.ravel("F")
    rb_product = channel_r.ravel("F") * channel_b.ravel("F")
    gb_product = channel_g.ravel("F") * channel_b.ravel("F")
    return rg_product, rb_product, gb_product


def compute_brisque_features(
    image: NDArray[np.float64], method: str = "original"
) -> list[float]:
    """Compute the BRISQUE features for an image.

    Args:
        image: Input image, either grayscale or RGB.
        method: Name of the feature extraction method used.

    Returns:
        List of computed BRISQUE features.
    """
    features: list[float] = []

    for scale in (0, 1):
        match method:
            case FeaturesUsed.ORIGINAL.value:
                normalized_lum = _normalize_image_intensity(
                    imresize(
                        im2gray(image).astype(np.float64), 1 / RESIZE_FACTOR**scale
                    )
                )
                # Computes Equation 19 parameters
                features.extend(_compute_ggd_parameters(np.ravel(normalized_lum, "F")))
                # Computes Equation 24 parameters
                for pairwise_product in compute_pairwise_products(normalized_lum):
                    features.extend(_compute_aggd_parameters(pairwise_product))
            case FeaturesUsed.RGB_CORRELATION.value:
                normalized_lum, normalized_r, normalized_g, normalized_b = (
                    _compute_normalized_rgb_image(image, scale)
                )
                features.extend(_compute_ggd_parameters(np.ravel(normalized_lum, "F")))
                for pairwise_product in compute_pairwise_products(normalized_lum):
                    features.extend(_compute_aggd_parameters(pairwise_product))
                for pairwise_product in compute_channel_products(
                    normalized_r, normalized_g, normalized_b
                ):
                    features.extend(_compute_aggd_parameters(pairwise_product))
            case FeaturesUsed.RGB_ANALYSIS.value:
                normalized_lum, normalized_r, normalized_g, normalized_b = (
                    _compute_normalized_rgb_image(image, scale)
                )
                for normalized_channel in (normalized_r, normalized_g, normalized_b):
                    features.extend(
                        _compute_ggd_parameters(np.ravel(normalized_channel, "F"))
                    )
                    for pairwise_product in compute_pairwise_products(
                        normalized_channel
                    ):
                        features.extend(_compute_aggd_parameters(pairwise_product))
            case FeaturesUsed.RGB_ALL.value:
                normalized_lum, normalized_r, normalized_g, normalized_b = (
                    _compute_normalized_rgb_image(image, scale)
                )
                for normalized_channel in (normalized_r, normalized_g, normalized_b):
                    features.extend(
                        _compute_ggd_parameters(np.ravel(normalized_channel, "F"))
                    )
                    for pairwise_product in compute_pairwise_products(
                        normalized_channel
                    ):
                        features.extend(_compute_aggd_parameters(pairwise_product))
                for pairwise_product in compute_channel_products(
                    normalized_r, normalized_g, normalized_b
                ):
                    features.extend(_compute_aggd_parameters(pairwise_product))
            case FeaturesUsed.CIELAB.value:
                normalized_l, normalized_a, normalized_b = (
                    _compute_normalized_lab_image(image, scale)
                )
                features.extend(_compute_ggd_parameters(np.ravel(normalized_l, "F")))
                for pairwise_product in compute_pairwise_products(normalized_l):
                    features.extend(_compute_aggd_parameters(pairwise_product))
                for pairwise_product in compute_channel_products(
                    normalized_l, normalized_a, normalized_b
                ):
                    features.extend(_compute_aggd_parameters(pairwise_product))

    _log_brisque_features(features)

    return features


def normalize_features(
    features: NDArray[np.float64], limits_file: str | Path = RANGE_FILE
) -> NDArray[np.float64]:
    """Normalizes feature vectors to the range specified in the range file.

    Args:
        features: A NumPy array of shape (x, 36) containing feature values.
        limits_file: Path to the saved range of LIVE IQA features.

    Returns:
        A NumPy array of the normalized features.
    """
    # Get the feature limits and the final range
    expected_range, limits = _read_features_limits(limits_file)

    # Ensure the features array has the correct shape
    if features.shape[1] != limits.shape[1]:
        msg = f"Expected feature array of shape (x, {limits.shape[1]}),\
            got {features.shape}."
        raise ValueError(msg)

    # Normalize using vectorized operations
    return expected_range[0] + (expected_range[1] - expected_range[0]) * (
        (features - limits[0, :]) / (limits[1, :] - limits[0, :])
    )


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _normalize_image_intensity(
    image: NDArray[np.float64], epsilon: float = 1.0
) -> NDArray[np.float64]:
    """Normalize the intensity of an image using local mean and standard deviation.

    It computes the Equation 16 of our article. Convolutions are done with zero-padding,
    the same way as it is done in the official MATLAB implementation with
    the `filter2` function with `same` parameter.

    Args:
        image: Input image, either grayscale or RGB.
        epsilon: Small constant to avoid division by zero.

    Returns:
        The normalized grayscale image.
    """
    grayscale_image = im2gray(image).astype(np.float64)
    weights = compute_2d_gaussian_weights()
    local_mean = convolve(grayscale_image, weights, mode="constant", cval=0.0)
    local_std_dev = np.sqrt(
        abs(
            convolve(grayscale_image**2, weights, mode="constant", cval=0.0)
            - local_mean**2
        )
    )

    logger = logging.getLogger("BRISQUE normalization")
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


def _compute_ggd_parameters(
    normalized_luminance: NDArray[np.float64],
) -> tuple[float, float]:
    """Estimate the parameters of a GGD.

    Estimate the parameters of a Generalized Gaussian Distribution (GGD)
        from normalized luminance data, as in Equation 19 of our article.

    Args:
        normalized_luminance: 1D array of normalized luminance values.

    Returns:
        The estimated shape parameter (alpha) and variance parameter (sigma) of the GGD.
    """
    alpha_range = np.arange(0.2, 10.0, 0.001)
    ratio_gamma = (gamma_func(1 / alpha_range) * gamma_func(3 / alpha_range)) / (
        gamma_func(2 / alpha_range) ** 2
    )

    sigma_sq = np.mean(normalized_luminance**2)
    mean_abs = np.mean(np.abs(normalized_luminance))
    if mean_abs == 0:
        mean_abs = AGGD_EPSILON
    rho = sigma_sq / mean_abs**2

    index = np.argmin(np.abs(rho - ratio_gamma))

    return alpha_range[index], sigma_sq


def _compute_aggd_parameters(
    pairwise_product: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Estimate the parameters of a AGGD.

    Estimate the parameters of an Asymmetric Generalized Gaussian Distribution (AGGD)
        from pairwise product values, as in Equation 24 of our article.

    Args:
        pairwise_product: 1D array of pairwise product values.

    Returns:
        The estimated shape parameter (nu), mean (eta), left standard deviation
            (left_std) and right standard deviation (right_std) of the AGGD.
    """
    nu_range = np.arange(0.2, 10.001, 0.001)
    ratio_gamma = (gamma_func(2 / nu_range) ** 2) / (
        gamma_func(1 / nu_range) * gamma_func(3 / nu_range)
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

    eta = (
        (right_std - left_std)
        * gamma_func(2 / nu_est)
        / np.sqrt(gamma_func(1 / nu_est) * gamma_func(3 / nu_est))
    )

    return nu_est, eta, left_std**2, right_std**2


def _read_features_limits(
    file_path: str | Path = RANGE_FILE,
) -> tuple[tuple[int, int], NDArray[np.float64]]:
    """Reads the given file and extracts feature limits.

    Args:
        file_path: Path to the input text file.

    Returns:
        Tuple containing:
        - A tuple representing the expected range of features after normalization.
        - A NumPy array of shape (36, 2) containing the feature limits.
    """
    with Path(file_path).open(encoding="utf8") as file:
        lines = file.readlines()

    # Recover the expected range (2nd line)
    range_line = lines[1].split()
    expected_range = tuple(map(int, range_line))

    # Recover the feature limits (starting from 3rd line)
    data = np.array(
        [[float(line.split()[1]), float(line.split()[2])] for line in lines[2:]],
        dtype=np.float64,
    )

    return expected_range, data.T


def _log_brisque_features(features: list[float]) -> None:
    """Log BRISQUE features computed for debug."""
    logger = logging.getLogger("BRISQUE Features")
    logger.debug("---- Features:")
    logger.debug(len(features))
    logger.debug("-- Image 1:1:")
    logger.debug("GGD:")
    logger.debug(features[:2])
    logger.debug("AGGD horizontal:")
    logger.debug(features[2:6])
    logger.debug("AGGD vertical:")
    logger.debug(features[6:10])
    logger.debug("AGGD D1:")
    logger.debug(features[10:14])
    logger.debug("AGGD D2:")
    logger.debug(features[14:18])
    logger.debug("-- Image 1:2:")
    logger.debug("GGD:")
    logger.debug(features[18:20])
    logger.debug("AGGD horizontal:")
    logger.debug(features[20:24])
    logger.debug("AGGD vertical:")
    logger.debug(features[24:28])
    logger.debug("AGGD D1:")
    logger.debug(features[28:32])
    logger.debug("AGGD D2:")
    logger.debug(features[32:])


def _compute_normalized_lab_image(
    image: NDArray[np.float64], scale: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert RGB image to CIELAB, resize and normalize each channel.

    Args:
        image: Input RGB image.
        scale: Downscaling exponent factor for resizing the image.

    Returns:
        Normalized L*, a*, and b* channels after resizing.
    """
    img_lab = rgb2lab(image)
    img_l = imresize(img_lab[:, :, 0].astype(np.float64), 1 / RESIZE_FACTOR**scale)
    img_a = imresize(img_lab[:, :, 1].astype(np.float64), 1 / RESIZE_FACTOR**scale)
    img_b = imresize(img_lab[:, :, 2].astype(np.float64), 1 / RESIZE_FACTOR**scale)
    normalized_l = _normalize_image_intensity(img_l)
    normalized_a = _normalize_image_intensity(img_a)
    normalized_b = _normalize_image_intensity(img_b)
    return normalized_l, normalized_a, normalized_b


def _compute_normalized_rgb_image(
    image: NDArray[np.float64], scale: int
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Resize and normalize grayscale and RGB channels of an image.

    Args:
        image: Input RGB image.
        scale: Downscaling exponent factor for resizing the image.

    Returns:
        Normalized grayscale, red, green, and blue channels after resizing.
    """
    img_lum = imresize(im2gray(image).astype(np.float64), 1 / RESIZE_FACTOR**scale)
    img_r = imresize(image[:, :, 0].astype(np.float64), 1 / RESIZE_FACTOR**scale)
    img_g = imresize(image[:, :, 1].astype(np.float64), 1 / RESIZE_FACTOR**scale)
    img_b = imresize(image[:, :, 2].astype(np.float64), 1 / RESIZE_FACTOR**scale)
    normalized_gray = _normalize_image_intensity(img_lum)
    normalized_r = _normalize_image_intensity(img_r)
    normalized_g = _normalize_image_intensity(img_g)
    normalized_b = _normalize_image_intensity(img_b)
    return normalized_gray, normalized_r, normalized_g, normalized_b


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the BRISQUE score for an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument(
        "-m", default=None, required=False, type=str, help="Method to use."
    )
    parser.add_argument(
        "-d", default=False, required=False, type=bool, help="Print debugs."
    )

    args = parser.parse_args()

    if args.d:
        logging.basicConfig(level=logging.DEBUG)
    print(compute_brisque(args.image_path, args.m))  # noqa: T201
