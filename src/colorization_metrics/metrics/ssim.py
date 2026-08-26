"""Compute the Structural Similarity Index (SSIM).

This module use the implementation in scikit-image of SSIM presented in [Wang et al.](https://doi.org/10.1109/TIP.2003.819861).

Usage:
    For an image:
    >>> ssim = compute_ssim("path_to_colored_image", "path_to_gt_image")

    For a video:
    >>> ssim = compute_ssim_dir("path_to_colored_frames_dir", "path_to_gt_frames_dir")

Note:
    The SSIM values range from -1 to 1, where 1 indicates perfect similarity.
        The SSIM metric considers luminance, contrast, and structure.
"""

from pathlib import Path

from skimage.io import imread
from skimage.metrics import structural_similarity
from skimage.util import dtype_limits, img_as_ubyte

from colorization_metrics.utils import (
    assess_correct_pairing,
    get_dir_imgs,
    reshape_same_as,
)

# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def compute_ssim(img_color_path: str | Path, img_gt_path: str | Path) -> float:
    """Compute the SSIM on the three RGB channels.

    It computes the Equation 10 of our article.

    Args:
        img_color_path: Path of the colored image.
        img_gt_path: Path of the ground truth image.

    Returns:
        The SSIM on the RGB channels.
    """
    img_color = img_as_ubyte(imread(img_color_path))
    img_gt = img_as_ubyte(imread(img_gt_path))
    if img_color.shape != img_gt.shape:
        img_color = reshape_same_as(img_color, img_gt)
    # Computes Equation 10
    return structural_similarity(
        img_color,
        img_gt,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        data_range=dtype_limits(img_color)[1] - dtype_limits(img_color)[0],
    )


def compute_ssim_dir(
    colored_dir: str | Path, gt_dir: str | Path, pair_step: int = 0
) -> float:
    """Compute the SSIM on a colorized video.

    It computes the Equation 10 of our article.
    The score of the video is computed as the mean of all frames SSIM,
        as specified in Section 2.3.

    Args:
        colored_dir: Directory containing the colored frames to measure.
        gt_dir: Directory containing the ground truth frames.
        pair_step: Constant index step between two paired files.

    Returns:
        The mean SSIM.
    """
    img_col_list = get_dir_imgs(colored_dir)
    img_gt_list = get_dir_imgs(gt_dir)
    assess_correct_pairing(img_col_list, img_gt_list, pair_step)
    mean_ssim = 0.0
    for img_col, img_gt in zip(img_col_list, img_gt_list, strict=False):
        # Computes Equation 10
        mean_ssim += compute_ssim(Path(colored_dir) / img_col, Path(gt_dir) / img_gt)
    # Computes the mean as specified in Section 2.3
    return mean_ssim / len(img_col_list)


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute the SSIM for an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("reference_path", type=str, help="Path to the reference file")

    args = parser.parse_args()
    print(compute_ssim(args.image_path, args.reference_path))  # noqa: T201
