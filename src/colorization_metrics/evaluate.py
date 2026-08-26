"""This module provides functions for calculating various image quality metrics.

Primarily, this module provides the `compute_metrics` method,
which can be used in colorization software to calculate metrics directly.
"""

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from colorization_metrics.metrics import (
    brisque,
    cdc,
    colorfulness,
    fid,
    lpips,
    maniqa,
    niqe,
    psnr,
    ssim,
)
from colorization_metrics.utils import (
    ColorSpace,
    ImageExtension,
    VideoExtension,
    extract_frames,
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

RESULT_FILE = "results.txt"
"""Name of the result file."""


@dataclass
class MetricParameters:
    """Parameters for image quality metric computation.

    Attributes:
        pair_step: Frame interval between distorted and ground-truth pairs.
        color_space: Color space used for metric computation.
        lpips_net: Backbone network used in LPIPS computation.
        fid_batch_size: Batch size used for FID feature extraction.
        fid_dims: Dimensionality of the FID feature embedding.
        colorfulness_type: Variant of the colorfulness metric to apply.
    """

    pair_step: int = 0
    color_space: str = ColorSpace.LAB.value
    lpips_net: str = lpips.LPIPSNetworks.ALEX_NET.value
    fid_batch_size: int = 1
    fid_dims: int = 64
    colorfulness_type: int = colorfulness.ColorfulnessMetric.RG_YB_CHANNELS.value


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def prepare_directory(path: str | Path, temp_dir: str | Path) -> Path:
    """Prepare a directory for frame extraction.

    Prepares a directory for frame extraction by creating a temporary directory
    if the input is a file, or validating the directory if it's already a folder.

    Args:
        path: The path to the video file or directory.
        temp_dir: The base directory where temporary directories can be created.

    Returns:
        The path to the prepared directory containing frames or the input directory.

    Raises:
        FileNotFoundError: If the given path does not exist.
        ValueError: If the path is neither a file nor a directory.
    """
    path = Path(path)
    temp_dir = Path(temp_dir)

    if not path.exists():
        msg = f"Path does not exist: {path}"
        raise FileNotFoundError(msg)

    if path.is_file():
        new_dir = temp_dir / path.stem
        if not new_dir.exists():
            new_dir.mkdir(parents=True)
            extract_frames(path, new_dir)
        return new_dir
    if path.is_dir():
        return path
    msg = f"Invalid path type: {path}"
    raise ValueError(msg)


def write_results(results: dict[str, float]) -> str:
    """Write all metrics results in a nice paragraph.

    Args:
        results: Metric names and results associated.

    Returns:
        Paragraph of results in the form of "Metric_name = number".
    """
    recap = ""
    for metric, result in results.items():
        recap += f"{metric} = {result}\n"
    return recap


def compute_metrics(
    colored: str | Path,
    gt: str | Path | None = None,
    params: MetricParameters | None = None,
    save_results: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Compute various metrics on colorizations.

    Args:
        colored: Path to colorized image, colorized video
            or directory containing colorized frames.
        gt: Path to image, video or directory containing ground truth images (optional).
        params: Parameters of the different metrics.
        save_results: Whether to save results to a file.
    """
    if params is None:
        params = MetricParameters()
    colored = Path(colored)
    if gt is not None:
        gt = Path(gt)

    if colored.is_file() and ImageExtension.is_valid(colored.suffix):
        _compute_metrics_on_img(colored, gt, params, save_results)
    elif (
        colored.is_file() and VideoExtension.is_valid(colored.suffix)
    ) or colored.is_dir():
        with tempfile.TemporaryDirectory() as temp_directory:
            colored_used = prepare_directory(colored, Path(temp_directory) / "colored")
            gt_used = prepare_directory(gt, Path(temp_directory) / "gt") if gt else None

            if colored.is_file() and gt is not None and gt.is_file():
                params.pair_step = 0
            _compute_metrics_on_dirs(colored_used, gt_used, params, save_results)

            if save_results and colored.is_file():
                shutil.move(colored_used / RESULT_FILE, colored.parent / RESULT_FILE)
    else:
        msg = f"File {colored} is not supported."
        raise ValueError(msg)


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _compute_metrics_on_img(
    colored_img: str | Path,
    gt_img: str | Path | None = None,
    params: MetricParameters | None = None,
    save_results: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Compute various image quality metrics for a colorized image.

    Args:
        colored_img: Path to the colorized image.
        gt_img: Path to the ground truth image.
        params: Parameters of the different metrics.
        save_results: Whether to save results to a file.
    """
    if params is None:
        params = MetricParameters()
    colored_img = Path(colored_img)
    if gt_img is not None:
        gt_img = Path(gt_img)
    metrics = {}
    # Run metrics using ground truth
    logger = logging.getLogger("Metrics computation")
    if gt_img:
        logger.info("Started PSNR computation.")
        metrics["PSNR"] = psnr.compute_psnr(
            colored_img, gt_img, ColorSpace(params.color_space).value
        )
        logger.info("Started SSIM computation.")
        metrics["SSIM"] = ssim.compute_ssim(colored_img, gt_img)
        logger.info("Started LPIPS computation.")
        metrics["LPIPS"] = lpips.compute_lpips(
            colored_img, gt_img, lpips.LPIPSNetworks(params.lpips_net)
        )
        logger.info("Started FID computation.")
        metrics["FID"] = fid.compute_fid(
            colored_img, gt_img, params.fid_batch_size, params.fid_dims
        )
    # Run always available metrics
    logger.info("Started BRISQUE computation.")
    metrics["BRISQUE"] = brisque.compute_brisque(colored_img)
    logger.info("Started MANIQA computation.")
    metrics["MANIQA"] = maniqa.compute_maniqa(colored_img)
    logger.info("Started NIQE computation.")
    metrics["NIQE"] = niqe.compute_niqe(colored_img)
    logger.info("Started COLORFULNESS computation.")
    metrics["COLORFULNESS"] = colorfulness.compute_colorfulness(
        colored_img, params.colorfulness_type
    )

    if save_results:
        with colored_img.with_name(f"{colored_img.stem}_{RESULT_FILE}").open(
            "w", encoding="utf-8"
        ) as f:
            logger.info("Writing results in %s.", f.name)
            results = write_results(metrics)
            f.write(results)
            logger.info(results)
    else:
        print("--- Metrics :")  # noqa: T201
        print(write_results(metrics))  # noqa: T201


def _compute_metrics_on_dirs(
    colored_dir: str | Path,
    gt_dir: str | Path | None = None,
    params: MetricParameters | None = None,
    save_results: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Compute various video quality metrics for a set of colorized frames.

    Args:
        colored_dir: Path to the directory containing colorized frames.
        gt_dir: Path to the directory containing ground truth frames.
        params: Parameters of the different metrics.
        save_results: Whether to save results to a file.
    """
    if params is None:
        params = MetricParameters()
    colored_dir = Path(colored_dir)
    if gt_dir is not None:
        gt_dir = Path(gt_dir)
    metrics = {}
    # Run metrics using ground truth
    logger = logging.getLogger("Metrics computation")
    if gt_dir:
        logger.info("Started PSNR computation.")
        metrics["PSNR"] = psnr.compute_psnr_dir(
            colored_dir, gt_dir, ColorSpace(params.color_space).value, params.pair_step
        )
        logger.info("Started SSIM computation.")
        metrics["SSIM"] = ssim.compute_ssim_dir(colored_dir, gt_dir, params.pair_step)
        logger.info("Started LPIPS computation.")
        metrics["LPIPS"] = lpips.compute_lpips_dir(
            colored_dir, gt_dir, lpips.LPIPSNetworks(params.lpips_net), params.pair_step
        )
        logger.info("Started FID computation.")
        metrics["FID"] = fid.compute_fid_dir(
            colored_dir, gt_dir, params.fid_batch_size, params.fid_dims
        )
    # Run videos only metrics
    logger.info("Started CDC computation.")
    metrics["CDC"] = cdc.compute_cdc_dir(colored_dir)
    # Run always available metrics
    logger.info("Started BRISQUE computation.")
    metrics["BRISQUE"] = brisque.compute_brisque_dir(colored_dir)
    logger.info("Started MANIQA computation.")
    metrics["MANIQA"] = maniqa.compute_maniqa_dir(colored_dir)
    logger.info("Started NIQE computation.")
    metrics["NIQE"] = niqe.compute_niqe_dir(colored_dir)
    logger.info("Started COLORFULNESS computation.")
    metrics["COLORFULNESS"] = colorfulness.compute_colorfulness_dir(
        colored_dir, params.colorfulness_type
    )

    if save_results:
        with (colored_dir / RESULT_FILE).open("w", encoding="utf-8") as f:
            logger.info("Writing results in %s.", f.name)
            results = write_results(metrics)
            f.write(results)
            logger.info(results)
    else:
        print("--- Metrics :")  # noqa: T201
        print(write_results(metrics))  # noqa: T201
