"""Compute the Temporally Consistent Video Colorfulness (CDC) metric.

This module implements the CDC metric presented in [Liu et al.](https://doi.org/10.48550/arXiv.2110.04562).
    More specifically, it provides functions for calculating
    the Jensen-Shannon (JS) divergence between two probabilistic distributions
    and the CDC score of a video for specific frame intervals.

Usage:
    For a video:
    >>> cdc_score = compute_cdc_dir("path_to_colored_frames_dir")

Note:
    The CDC score measures the temporal consistency of color distributions
        in video frames. It is calculated for specified intervals,
        and the `compute_cdc_dir` function allows for combining scores
        with different intervals and weights.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import histogram
from scipy.stats import entropy
from skimage.color import rgb2lab
from skimage.io import imread
from skimage.util import img_as_ubyte

from colorization_metrics.utils import ColorSpace, get_dir_imgs

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

RGB_COLOR_RANGE = (0, 255)
LAB_COLOR_RANGE = (-128, 127)

# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


# pylint: disable = dangerous-default-value
def compute_cdc_dir(
    directory: str | Path,
    interval: tuple[int] = (1, 2, 4),
    weight: tuple[float] = (1 / 3, 1 / 3, 1 / 3),
    color_space: ColorSpace = ColorSpace.RGB,
) -> float:
    """Compute the CDC metrics presented in [Liu et al.](https://doi.org/10.48550/arXiv.2110.04562).

    It computes the Equation 42 of our article.

    Args:
        directory: Directory containing all video frames.
        interval: Time intervals between frames.
        weight: Weights for each interval.
        color_space: The color space used for analysis.

    Returns:
        The CDC score of the video.
    """
    return sum(
        w * _compute_cdc(directory, i, color_space)
        for w, i in zip(weight, interval, strict=False)
    )


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _compute_js(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Compute the Jensen-Shannon (JS) divergence between two distributions.

    It computes the Equation 41 of our article.

    Args:
        p: One probabilistic distribution.
        q: Another probabilistic distribution.

    Returns:
        The JS divergence between p and q.
    """
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def _compute_normalized_histogram(
    directory: str | Path, color_space: ColorSpace = ColorSpace.RGB
) -> list[tuple[np.ndarray, ...]]:
    """Computes the normalized color histograms for a directory of images.

    Args:
        directory: The path to the directory containing the images.
        color_space: The color space for histogram calculation.

    Returns:
        A list of tuples, where each tuple contains normalized histograms
        as NumPy arrays for the specified color space.
    """
    image_list = get_dir_imgs(directory)
    histograms = []

    for image_name in image_list:
        image = img_as_ubyte(imread(Path(directory) / image_name))
        pixel_number = image.shape[0] * image.shape[1]

        match color_space:
            case ColorSpace.RGB:
                histogram_r = (
                    histogram(image[:, :, 0], *RGB_COLOR_RANGE, 256) / pixel_number
                )
                histogram_g = (
                    histogram(image[:, :, 1], *RGB_COLOR_RANGE, 256) / pixel_number
                )
                histogram_b = (
                    histogram(image[:, :, 2], *RGB_COLOR_RANGE, 256) / pixel_number
                )
                histograms.append((histogram_r, histogram_g, histogram_b))

            case ColorSpace.LAB:
                lab_image = rgb2lab(image)
                histogram_a = (
                    histogram(lab_image[:, :, 1], *LAB_COLOR_RANGE, 256) / pixel_number
                )
                histogram_b = (
                    histogram(lab_image[:, :, 2], *LAB_COLOR_RANGE, 256) / pixel_number
                )
                histograms.append((histogram_a, histogram_b))
            case _:
                msg = f"Unsupported color space: {color_space}"
                raise ValueError(msg)

    return histograms


def _compute_cdc(
    directory: str | Path, interval: int, color_space: ColorSpace = ColorSpace.RGB
) -> float:
    """Compute the CDC score of a video for a specific frame interval.

    It computes the Equation 40 of our article.

    Args:
        directory: Directory containing all video frames.
        interval: Time interval between frames.
        color_space: The color space used for analysis.

    Returns:
        The CDC score of the video for the specified interval.
    """
    cdc_interval = 0.0
    histograms = _compute_normalized_histogram(directory, color_space)
    image_nb = len(histograms)

    match color_space:
        case ColorSpace.RGB:
            for i in range(image_nb - interval):
                js_r = _compute_js(histograms[i][0], histograms[i + interval][0])
                js_g = _compute_js(histograms[i][1], histograms[i + interval][1])
                js_b = _compute_js(histograms[i][2], histograms[i + interval][2])
                cdc_interval += js_r + js_g + js_b
        case ColorSpace.LAB:
            for i in range(image_nb - interval):
                js_a = _compute_js(histograms[i][0], histograms[i + interval][0])
                js_b = _compute_js(histograms[i][1], histograms[i + interval][1])
                cdc_interval += js_a + js_b
        case _:
            msg = f"Unsupported color space: {color_space}"
            raise ValueError(msg)
    return (1 / (3 * (image_nb - interval))) * cdc_interval


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute the CDC score for an image.")
    parser.add_argument(
        "directory_path",
        type=str,
        help="Path to the directory containing the video frames.",
    )
    parser.add_argument(
        "-cs",
        "--color_space",
        default=ColorSpace.RGB.value,
        choices=ColorSpace.list(),
        type=str,
        help="Color space in which to compute the PSNR (default : %(default)s).",
    )

    args = parser.parse_args()
    print(compute_cdc_dir(args.directory_path, color_space=args.color_space))  # noqa: T201
