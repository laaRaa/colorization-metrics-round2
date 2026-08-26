"""Compute the Fréchet Inception Distance (FID).

This module use the official implementation of the FID presented in [Heusel et al.](https://doi.org/10.48550/arXiv.1706.08500).

Usage:
    For a video:
    >>> fid = compute_fid_dir("path_to_colored_frames_dir", "path_to_gt_frames_dir")
"""

import contextlib
import shutil
import tempfile
from io import StringIO
from pathlib import Path

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths

# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def compute_fid(
    colored_img: str | Path,
    gt_img: str | Path,
    batch_size: int = 1,
    fid_dims: int = 64,
    verbose: bool = False,  # noqa: FBT001, FBT002
) -> float:
    """Compute the Fréchet Inception Distance (FID) between two image directories.

    It computes the Equation 12 of our article.

    This function computes the FID metric between two datasets of images,
    representing the colorized images and the ground truth images.

    Args:
        colored_img: Path to the directory containing colorized images.
        gt_img: Path to the directory containing ground truth images.
        batch_size: Batch size for image loading.
        fid_dims: Dimensions for feature extraction.
        verbose: Whether to enable console output.

    Returns:
        The computed FID.
    """
    with (
        tempfile.TemporaryDirectory() as temp_colored,
        tempfile.TemporaryDirectory() as temp_gt,
    ):
        shutil.copy(colored_img, Path(temp_colored) / "image_1.png")
        shutil.copy(colored_img, Path(temp_colored) / "image_2.png")
        shutil.copy(gt_img, Path(temp_gt) / "image_1.png")
        shutil.copy(gt_img, Path(temp_gt) / "image_2.png")
        return compute_fid_dir(temp_colored, temp_gt, batch_size, fid_dims, verbose)


def compute_fid_dir(
    colored_dir: str | Path,
    gt_dir: str | Path,
    batch_size: int = 1,
    fid_dims: int = 2048,
    verbose: bool = False,  # noqa: FBT001, FBT002
) -> float:
    """Compute the Fréchet Inception Distance (FID) between two image directories.

    It computes the Equation 12 of our article.

    This function computes the FID metric between two datasets of images,
    representing the colorized images and the ground truth images.

    Args:
        colored_dir: Path to the directory containing colorized images.
        gt_dir: Path to the directory containing ground truth images.
        batch_size: Batch size for image loading.
        fid_dims: Dimensions for feature extraction.
        verbose: Whether to enable console output.

    Returns:
        The computed FID.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if verbose:
        # Computes Equation 12
        score = calculate_fid_given_paths(
            [str(colored_dir), str(gt_dir)], batch_size, device, fid_dims
        )
    else:
        with (
            contextlib.redirect_stdout(StringIO()),
            contextlib.redirect_stderr(StringIO()),
        ):
            # Computes Equation 12
            score = calculate_fid_given_paths(
                [str(colored_dir), str(gt_dir)], batch_size, device, fid_dims
            )
    return score


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute the FID for an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("reference_path", type=str, help="Path to the reference file")
    parser.add_argument(
        "-d",
        "--fid_dims",
        default=2048,
        choices=[64, 192, 768, 2048],
        type=int,
        help="Number of dimensions of the used feature layer in Inception\
            (default : %(default)s).",
    )

    args = parser.parse_args()
    print(compute_fid(args.image_path, args.reference_path, fid_dims=args.fid_dims))  # noqa: T201
