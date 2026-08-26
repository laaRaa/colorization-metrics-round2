"""Compute colorization metrics on video frames.

This script computes various image quality metrics, including PSNR, SSIM,
    LPIPS, FID, CDC, and COLORFULNESS. The metrics are computed
    on colored video frames, optionally compared with ground truth frames.

Usage:
    $ `python metric.py -c _yourColorizationDir_ -gt _yourGroundTruthDir_`

Notes:
    Directories must contain **numbered** image files, for example `img_00024.png`
    or `007.jpg`. This is necessary because some metrics use the temporal order
    of images and need to know how to classify image files.
    As far as the index is concerned, there may be a difference
    between your colorization and the ground truth, e.g. the former starts at 0
    and the latter starts at one. The `pair_step` argument is for such cases.
"""

import logging
import warnings
from argparse import ArgumentParser, Namespace

from colorization_metrics.evaluate import MetricParameters, compute_metrics
from colorization_metrics.metrics.colorfulness import ColorfulnessMetric
from colorization_metrics.metrics.lpips import LPIPSNetworks
from colorization_metrics.utils import ColorSpace


def _parse() -> Namespace:
    """Function to parse command line arguments.

    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = ArgumentParser(
        prog="metric",
        description="Compute colorization metrics on your ... colorizations !\
            You can compare with ground truth images or just use perceptual metrics.",
        epilog="Thanks for using me :)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
        help="Indicate in output current metrics being processed.",
    )
    parser.add_argument(
        "-w",
        "--write_results",
        default=False,
        action="store_true",
        help="Write results in a text file in the colored directory.",
    )

    inputs = parser.add_argument_group("inputs")
    inputs.add_argument(
        "-c",
        "--colored",
        required=True,
        default=None,
        type=str,
        help="Path to the directory containing colored video frames.\
        Directories must contain **numbered** image files, see the notes in the code.\
        You can also use video files if you want.",
    )
    inputs.add_argument(
        "-gt",
        "--ground_truth",
        dest="gt",
        default=None,
        type=str,
        help="Path to the directory containing ground truth video frames.\
        Directories must contain **numbered** image files, see the notes in the code.\
        You can also use video files if you want.",
    )

    options = parser.add_argument_group("metrics")
    options.add_argument(
        "--color_space",
        default=ColorSpace.LAB.value,
        choices=ColorSpace.list(),
        type=str,
        help="Color space in which to compute the metrics (default : %(default)s).",
    )
    options.add_argument(
        "--pair_step",
        default=0,
        type=int,
        help="Constant index step between two paired files (default : %(default)s).",
    )
    options.add_argument(
        "--fid_dims",
        default=64,
        choices=[64, 192, 768, 2048],
        type=int,
        help="Number of dimensions of the used feature layer in Inception\
            (default : %(default)s).",
    )
    options.add_argument(
        "--colorfulness_type",
        default=ColorfulnessMetric.RG_YB_CHANNELS.value,
        choices=ColorfulnessMetric.list(),
        type=int,
        help="Select the type of colorfulness measure you want to use\
            (default : %(default)s).",
    )
    options.add_argument(
        "--LPIPS_net",
        default=LPIPSNetworks.ALEX_NET.value,
        choices=LPIPSNetworks.list(),
        type=str,
        help="Network used in LPIPS measure (default : %(default)s).",
    )

    return parser.parse_args()


def main() -> None:
    """Main function for the script.

    This function orchestrates the computation of various image quality metrics,
    logs the progress, and optionally writes the results to a file.

    Returns:
        int: Exit code (0 for success).
    """
    args = _parse()
    logging.basicConfig(level=args.loglevel)
    if args.loglevel != logging.INFO:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    params = MetricParameters(
        pair_step=args.pair_step,
        color_space=args.color_space.lower(),
        lpips_net=args.LPIPS_net,
        fid_dims=args.fid_dims,
        colorfulness_type=args.colorfulness_type,
    )
    compute_metrics(
        colored=args.colored, gt=args.gt, params=params, save_results=args.write_results
    )


if __name__ == "__main__":
    main()
