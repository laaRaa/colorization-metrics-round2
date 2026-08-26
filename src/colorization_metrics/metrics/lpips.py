"""Compute the Learned Perceptual Image Patch Similarity (LPIPS) metric.

This module uses the official implementation of the LPIPS presented in [Zhang et al.](https://doi.org/10.1109/CVPR.2018.00068).

Usage:
    For an image:
    >>> lpips = compute_lpips("path_to_colored_image", "path_to_gt_image")

    For a video:
    >>> lpips = compute_lpips_dir("path_to_colored_frames_dir", "path_to_gt_frames_dir")


Note:
    LPIPS is a perceptual metric that measures the similarity between two images.
    The `net_type` parameter specifies the network architecture used
    for computing the metric.

"""

import warnings
from enum import Enum
from pathlib import Path

from lpips import LPIPS, im2tensor
from skimage.io import imread
from skimage.util import img_as_ubyte

from colorization_metrics.utils import (
    assess_correct_pairing,
    get_dir_imgs,
    reshape_same_as,
)

# Network cache to avoid recreating LPIPS networks multiple times
warnings.filterwarnings("ignore", category=UserWarning)
_LPIPS_NETWORK_CACHE: dict[str, LPIPS] = {}

# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------


class LPIPSNetworks(Enum):
    """Enumeration representing available LPIPS networks.

    Attributes:
        ALEX_NET: AlexNet architecture.
        VGG: VGG architecture.
        SQUEEZE_NET: SqueezeNet architecture.
    """

    ALEX_NET = "alex"
    VGG = "vgg"
    SQUEEZE_NET = "squeeze"

    @classmethod
    def list(cls) -> list[str]:
        """Return a list of supported LPIPS network architectures.

        Returns:
            A list of network types as strings.
        """
        return [net.value for net in cls]


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def compute_lpips(
    colored_image: str | Path,
    gt_image: str | Path,
    net_type: str = LPIPSNetworks.ALEX_NET.value,
) -> float:
    """Compute the LPIPS metric introduced in [Zhang et al.](https://doi.org/10.1109/CVPR.2018.00068).

    It computes the Equation 11 of our article.

    Args:
        colored_image: Path of the colorized image to measure.
        gt_image: Path of the ground truth image to measure against.
        net_type: Name of the network architecture to use. Defaults to "alex".

    Returns:
        The LPIPS of the colorization.
    """
    lpips_network = _get_lpips_network(LPIPSNetworks(net_type))
    color_img = img_as_ubyte(imread(colored_image))
    gt_img = img_as_ubyte(imread(gt_image))

    if color_img.shape != gt_img.shape:
        color_img = reshape_same_as(color_img, gt_img)

    color_img = im2tensor(color_img)
    gt_img = im2tensor(gt_img)

    # Computes Equation 11
    return float(lpips_network(color_img, gt_img))


def compute_lpips_dir(
    colored_dir: str | Path,
    gt_dir: str | Path,
    net_type: str = LPIPSNetworks.ALEX_NET.value,
    pair_step: int = 0,
) -> float:
    """Compute the LPIPS metric introduced in [Zhang et al.](https://doi.org/10.1109/CVPR.2018.00068).

    It computes the Equation 11 of our article.
    The score of the video is computed as the mean of all frames LPIPS,
        as specified in Section 2.3.

    Args:
        colored_dir: Directory containing the colorized video frames to measure.
        gt_dir: Directory containing the ground truth video frames to measure against.
        net_type: Name of the network architecture to use. Defaults to "alex".
        pair_step: Constant index step between two paired files. Defaults to 0.

    Returns:
        The LPIPS of the colorization.
    """
    img_col_list = get_dir_imgs(colored_dir)
    img_gt_list = get_dir_imgs(gt_dir)
    assess_correct_pairing(img_col_list, img_gt_list, pair_step)

    lpips_network = _get_lpips_network(LPIPSNetworks(net_type))

    mean_lpips = 0
    for img_col_name, img_gt_name in zip(img_col_list, img_gt_list, strict=False):
        img_color = img_as_ubyte(imread(Path(colored_dir) / img_col_name))
        img_gt = img_as_ubyte(imread(Path(gt_dir) / img_gt_name))

        if img_color.shape != img_gt.shape:
            img_color = reshape_same_as(img_color, img_gt)

        img_color = im2tensor(img_color)
        img_gt = im2tensor(img_gt)

        # Computes Equation 11
        mean_lpips += float(lpips_network(img_color, img_gt))

    # Computes the mean as specified in Section 2.3
    return mean_lpips / len(img_col_list)


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _get_lpips_network(net_type: LPIPSNetworks) -> LPIPS:
    """Retrieve or initialize the appropriate LPIPS network for the given type.

    Args:
        net_type: The network type, should be one of the LPIPSNetworks values.

    Returns:
        The corresponding LPIPS network instance.
    """
    net_name = net_type.value
    if net_name not in _LPIPS_NETWORK_CACHE:
        _LPIPS_NETWORK_CACHE[net_name] = LPIPS(net=net_name, verbose=False)
    return _LPIPS_NETWORK_CACHE[net_name]


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the LPIPS score for an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("reference_path", type=str, help="Path to the reference file")
    parser.add_argument(
        "-n",
        "--network",
        default=LPIPSNetworks.ALEX_NET.value,
        choices=LPIPSNetworks.list(),
        type=str,
        help="CNN with which to compute the LPIPS (default : %(default)s).",
    )

    args = parser.parse_args()
    print(compute_lpips(args.image_path, args.reference_path, args.network))  # noqa: T201
