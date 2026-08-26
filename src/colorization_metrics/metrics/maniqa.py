"""Compute the MANIQA measure.

Compute the Multi-dimension Attention Network for No-Reference Image Quality
    Assessment (MANIQA).
This module use the [python module](https://gitlab.com/nifra/MANIQA) of MANIQA,
forked from the [official implementation](https://github.com/IIGROUP/MANIQA)
of [Yang et al.](https://doi.org/10.1109/CVPRW56347.2022.00126).

Usage:
    For an image:
    >>> maniqa = compute_maniqa("path_to_colored_image")

    For a video:
    >>> maniqa = compute_maniqa_dir("path_to_colored_frames_dir")
"""

from pathlib import Path

import torch

from colorization_metrics.utils import get_dir_imgs


def _infer_score_cpu_safe(img_path: str) -> float:
    """Run MANIQA on CPU when the checkpoint was saved for CUDA.

    This is a demo/runtime compatibility layer for the IPOL CPU-only environment.
    It does not change the metric definition or the network architecture; it only
    makes sure that a CUDA-serialized checkpoint can be loaded on a CPU machine by
    mapping it to CPU before deserialization.
    """
    from maniqa.inference import infer_score

    try:
        return infer_score(img_path)
    except RuntimeError as exc:
        msg = str(exc)
        if "Attempting to deserialize object on a CUDA device" not in msg:
            raise

        original_torch_load = torch.load

        def _cpu_map_location(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device("cpu"))
            return original_torch_load(*args, **kwargs)

        torch.load = _cpu_map_location
        try:
            return infer_score(img_path)
        finally:
            torch.load = original_torch_load

# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def compute_maniqa(img_path: str | Path) -> float:
    """Compute the MANIQA measure on an image.

    Args:
        img_path: Path to the image file to measure.

    Returns:
        The MANIQA score of the image.
    """
    try:
        return _infer_score_cpu_safe(str(img_path))
    except (RuntimeError, ValueError):
        return float("nan")


def compute_maniqa_dir(img_dir: str | Path) -> float:
    """Compute the MANIQA measure on multiple images.

    Args:
        img_dir: Path to the directory containing the images to measure.

    Returns:
        The mean MANIQA score of all images.
    """
    img_list = get_dir_imgs(img_dir)
    mean_maniqa = 0.0
    try:
        for img_name in img_list:
            mean_maniqa += _infer_score_cpu_safe(str(Path(img_dir) / img_name))
    except (RuntimeError, ValueError):
        return float("nan")
    return mean_maniqa / len(img_list)


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the MANIQA score for an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()
    print(compute_maniqa(args.image_path))  # noqa: T201
