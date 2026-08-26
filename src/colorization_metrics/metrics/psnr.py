"""Compute Peak Signal-to-Noise Ratio (PSNR).

This module use the implementation in scikit-image of PSNR on RGB channels and
    on chrominance channels (a*,b*) in the CIELAB color space.

Usage:
    For an image:
    >>> psnr = compute_psnr_rgb("path_to_colored_image", "path_to_gt_image")
    >>> psnr = compute_psnr_lab("path_to_colored_image", "path_to_gt_image")

    For a video sequence:
    >>> psnr = compute_psnr_dir("path_to_colored_frames_dir", "path_to_gt_frames_dir")

Note:
    The color space options for `compute_psnr_dir` include "rgb" and "lab".
        The PSNR values are indicative of the quality of colorization.
"""

from pathlib import Path

from skimage.color import rgb2lab
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

from colorization_metrics.utils import (
    ColorSpace,
    assess_correct_pairing,
    get_dir_imgs,
    reshape_same_as,
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

AB_DATA_RANGE = 255
"""Data range for the a*b* chrominance channels in the CIELAB color space.

This value represents the possible range of values for the a* and b* channels
in the CIELAB color space, typically spanning from -100 to 100.
It is used when calculating the PSNR (Peak Signal-to-Noise Ratio) on
these channels, ensuring proper scaling of the error metric.
"""

# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def compute_psnr(
    colored_img: str | Path, gt_img: str | Path, color_space: str = ColorSpace.LAB.value
) -> float:
    """Compute the PSNR on a colorized image in a given color space.

    It computes the Equation 1 of our article with:
        - the MSE described in the Equation 2 for the RGB colorspace.
        - the MSE described in the Equation 3 for the CIELAB colorspace.

    Args:
        colored_img: Path to the colorized image.
        gt_img: Path to the ground truth image.
        color_space: Color space.
        pair_step: Constant index step between two paired files.

    Returns:
        The mean PSNR.
    """
    match color_space:
        case ColorSpace.RGB.value:
            # Computes Equation 1 using Equation 2
            return _compute_psnr_rgb(colored_img, gt_img)
        case ColorSpace.LAB.value:
            # Computes Equation 1 using Equation 3
            return _compute_psnr_lab(colored_img, gt_img)
        case _:
            msg = f"The selected color space {color_space} is not implemented."
            raise ValueError(msg)


def compute_psnr_dir(
    colored_dir: str | Path,
    gt_dir: str | Path,
    color_space: str = ColorSpace.LAB.value,
    pair_step: int = 0,
) -> float:
    """Compute the PSNR on a colorized video in a given color space.

    It computes the Equation 1 of our article with:
        - the MSE described in the Equation 2 for the RGB colorspace.
        - the MSE described in the Equation 3 for the CIELAB colorspace.
    The score of the video is computed as the mean of all frames PSNR,
        as specified in Section 2.3.

    Args:
        colored_dir: Directory containing the colored frames to measure.
        gt_dir: Directory containing the ground truth frames.
        color_space: Color space.
        pair_step: Constant index step between two paired files.

    Returns:
        The mean PSNR.
    """
    img_col_list = get_dir_imgs(colored_dir)
    img_gt_list = get_dir_imgs(gt_dir)
    assess_correct_pairing(img_col_list, img_gt_list, pair_step)
    mean_psnr = 0.0
    for img_col, img_gt in zip(img_col_list, img_gt_list, strict=False):
        match color_space:
            case ColorSpace.RGB.value:
                # Computes Equation 1 using Equation 2
                mean_psnr += _compute_psnr_rgb(
                    Path(colored_dir) / img_col, Path(gt_dir) / img_gt
                )
            case ColorSpace.LAB.value:
                # Computes Equation 1 using Equation 3
                mean_psnr += _compute_psnr_lab(
                    Path(colored_dir) / img_col, Path(gt_dir) / img_gt
                )
            case _:
                msg = f"The selected color space {color_space} is not implemented."
                raise ValueError(msg)
    # Computes the mean as specified in Section 2.3
    return mean_psnr / len(img_col_list)


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _compute_psnr_rgb(colored_img: str | Path, gt_img: str | Path) -> float:
    """Compute the PSNR on the three RGB channels.

    It computes the Equation 1 of our article with the MSE described in the Equation 2.

    Args:
        colored_img: Path to the colorized image.
        gt_img: Path to the ground truth image.

    Returns:
        The mean PSNR on the RGB channels.
    """
    img_color = imread(colored_img)
    img_gt = imread(gt_img)
    if img_color.shape != img_gt.shape:
        img_color = reshape_same_as(img_color, img_gt)
    # Computes Equation 1
    return peak_signal_noise_ratio(img_gt, img_color)


def _compute_psnr_lab(colored_img: str | Path, gt_img: str | Path) -> float:
    """Compute the PSNR on the chrominance channels of the CIELAB color space.

    It computes the Equation 1 of our article with the MSE described in the Equation 3.

    Args:
        colored_img: Path to the colorized image.
        gt_img: Path to the ground truth image.

    Returns:
        The mean PSNR on the a* and b* channels.
    """
    img_color = imread(colored_img)
    img_gt = imread(gt_img)
    if img_color.shape != img_gt.shape:
        img_color = reshape_same_as(img_color, img_gt)
    img_color = rgb2lab(img_color)
    img_gt = rgb2lab(img_gt)
    # Computes Equation 1
    return peak_signal_noise_ratio(
        img_gt[:, :, 1:3], img_color[:, :, 1:3], data_range=AB_DATA_RANGE
    )


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute the PSNR for an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("reference_path", type=str, help="Path to the reference file.")
    parser.add_argument(
        "-cs",
        "--color_space",
        default=ColorSpace.RGB.value,
        choices=ColorSpace.list(),
        type=str,
        help="Color space in which to compute the PSNR (default : %(default)s).",
    )

    args = parser.parse_args()
    match args.color_space:
        case ColorSpace.RGB.value:
            print(_compute_psnr_rgb(args.image_path, args.reference_path))  # noqa: T201
        case ColorSpace.LAB.value:
            print(_compute_psnr_lab(args.image_path, args.reference_path))  # noqa: T201
