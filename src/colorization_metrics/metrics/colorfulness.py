"""Compute the colorfulness metric.

This module implements the colorfulness metric presented in [Hasler & Suesstrunk](https://doi.org/10.1117/12.477378).

Usage:
    >>> colorfulness_score = compute_colorfulness(
        "path_to_colored_frames_dir",
        metric = 3,
    )

Note:
    The colorfulness metric options are as follows:
    - Metric 1: Uses the mean and standard deviation of the 'a' and 'b' channel.
    - Metric 2: Uses the mean and standard deviation of the chroma values in Lab.
    - Metric 3: Utilizes the RG and YB color opposition channels.
"""

from enum import Enum
from pathlib import Path

from numpy import astype, float64, sqrt
from skimage.color import rgb2lab
from skimage.io import imread
from skimage.util import img_as_ubyte

from colorization_metrics.utils import get_dir_imgs

# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------


class ColorfulnessMetric(Enum):
    """Enumeration for selecting the colorfulness metric to compute.

    The different options correspond to different methods of measuring colorfulness:

    Attributes:
        AB_CHANNELS (1): Uses the a* and b* channels of the LAB color space
            for colorfulness computation.
        CHROMA_MEAN (2): Computes colorfulness based on chroma,
            a measure of color intensity.
        RG_YB_CHANNELS (3): Uses the (R-G) and (Y-B) channels
            for colorfulness calculation.
    """

    AB_CHANNELS = 1
    """First version of the colorfulness metric."""
    CHROMA_MEAN = 2
    """Second version of the colorfulness metric."""
    RG_YB_CHANNELS = 3
    """Third version of the colorfulness metric."""

    @classmethod
    def list(cls) -> list[int]:
        """Return a list of accepted metric versions.

        Returns:
            A list of metric versions as int.
        """
        return [version.value for version in cls]


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def compute_colorfulness(
    image_path: str | Path, metric: int = ColorfulnessMetric.RG_YB_CHANNELS.value
) -> float:
    """Compute the colorfulness metric presented in [Hasler & Suesstrunk](https://doi.org/10.1117/12.477378).

    Args:
        image_path: Path to the image to measure.
        metric: Selects the version of the colorfulness metric to use. Options are:
            - 1: Uses the a* and b* channels. (Equation 13 of our article)
            - 2: Computes colorfulness based on chroma. (Equation 14 of our article)
            - 3: Uses the (R-G) and (Y-B) channels. (Equation 15 of our article)

    Returns:
        Colorfulness score computed based on the selected metric.

    Raises:
        ValueError: If an unsupported metric value is provided.
    """
    img_rgb = img_as_ubyte(imread(image_path))
    img_lab = rgb2lab(img_rgb)
    std_ab = sqrt(img_lab[:, :, 1].std() ** 2 + img_lab[:, :, 2].std() ** 2)
    match metric:
        case ColorfulnessMetric.AB_CHANNELS.value:
            # Computes Equation 13
            mean_ab = sqrt(img_lab[:, :, 1].mean() ** 2 + img_lab[:, :, 2].mean() ** 2)
            color = std_ab + 0.37 * mean_ab

        case ColorfulnessMetric.CHROMA_MEAN.value:
            # Computes Equation 14
            mean_c = sqrt(img_lab[:, :, 1] ** 2 + img_lab[:, :, 2] ** 2).mean()
            color = std_ab + 0.94 * mean_c

        case ColorfulnessMetric.RG_YB_CHANNELS.value:
            # Computes Equation 15
            img_rgb = astype(img_rgb, float64)
            rg = img_rgb[:, :, 0] - img_rgb[:, :, 1]  # pylint: disable = unsubscriptable-object
            yb = 0.5 * (img_rgb[:, :, 0] + img_rgb[:, :, 1]) - img_rgb[:, :, 2]  # pylint: disable = unsubscriptable-object
            std_rgyb = sqrt(rg.std() ** 2 + yb.std() ** 2)
            mean_rgyb = sqrt(rg.mean() ** 2 + yb.mean() ** 2)
            color = std_rgyb + 0.3 * mean_rgyb

        case _:
            msg = f"Colorfulness metrics available are 1, 2 or 3.\
                    Got {metric} instead."
            raise ValueError(msg)
    return color


def compute_colorfulness_dir(
    directory: str | Path, metric: int = ColorfulnessMetric.RG_YB_CHANNELS.value
) -> float:
    """Compute the colorfulness metric presented in [Hasler & Suesstrunk](https://doi.org/10.1117/12.477378).

    Args:
        directory: Path to the directory containing the images to measure.
        metric: Selects the version of the colorfulness metric to use. Options are:
            - 1: Uses the a* and b* channels. (Equation 13 of our article)
            - 2: Computes colorfulness based on chroma. (Equation 14 of our article)
            - 3: Uses the (R-G) and (Y-B) channels. (Equation 15 of our article)

    Returns:
        Average colorfulness score for all images in the directory.

    Raises:
        FileNotFoundError: If the directory does not exist or contains no valid images.
    """
    # Get the list of images in the directory
    img_list = get_dir_imgs(directory)
    if not img_list:
        msg = f"No valid images found in directory: {directory}"
        raise FileNotFoundError(msg)

    # Compute the average colorfulness
    total_colorfulness = sum(
        compute_colorfulness(Path(directory) / img_name, metric)
        for img_name in img_list
    )

    return total_colorfulness / len(img_list)


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the Colorfulness measure for an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "-m",
        "--metric",
        default=ColorfulnessMetric.RG_YB_CHANNELS.value,
        choices=ColorfulnessMetric.list(),
        type=str,
        help="Version of the colorfulness metric to to compute (default : %(default)s).",  # noqa: E501
    )

    args = parser.parse_args()
    print(compute_colorfulness(args.image_path, args.metric))  # noqa: T201
