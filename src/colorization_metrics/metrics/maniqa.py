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

import sys
import traceback
from pathlib import Path

import torch

from colorization_metrics.utils import get_dir_imgs


def _force_cpu_for_maniqa() -> None:
    """Keep the external MANIQA inference code on CPU when needed.

    The upstream implementation calls `.cuda()` internally during inference. In
    CPU-only or constrained demo environments, those calls may fail unless the
    checkpoint and tensors are mapped to CPU before loading and inference. This
    compatibility shim preserves the original metric definition and architecture
    while keeping the demo executable in such environments.
    """
    if torch.cuda.is_available():
        return

    original_tensor_cuda = torch.Tensor.cuda
    original_module_cuda = torch.nn.Module.cuda

    def _tensor_cuda(self, *args, **kwargs):
        return self.to(torch.device("cpu"))

    def _module_cuda(self, *args, **kwargs):
        return self.to(torch.device("cpu"))

    torch.Tensor.cuda = _tensor_cuda
    torch.nn.Module.cuda = _module_cuda

    return None


def _infer_score_cpu_safe(img_path: str) -> float:
    """Run MANIQA on CPU when the checkpoint was saved for CUDA.

    This is a demo/runtime compatibility layer for constrained execution
    environments. It does not change the metric definition or the network
    architecture; it only ensures that a CUDA-serialized checkpoint can be loaded
    on CPU by mapping it before deserialization, and that the legacy MANIQA
    inference library does not trigger CUDA calls in CPU-only settings.
    """
    from maniqa.inference import infer_score

    _force_cpu_for_maniqa()

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
    except Exception as exc:  # pragma: no cover - demo/runtime guard
        print(
            f"MANIQA failed for {img_path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
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
    except Exception as exc:  # pragma: no cover - demo/runtime guard
        print(
            f"MANIQA batch failed for {img_dir}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
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
