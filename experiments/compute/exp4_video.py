"""Script to compute metrics on videos."""

import random
import sys
from collections.abc import Callable
from functools import partial
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
from tqdm import tqdm

from colorization_metrics.evaluate import RESULT_FILE, prepare_directory, write_results
from colorization_metrics.metrics.brisque import FeaturesUsed, compute_brisque_dir
from colorization_metrics.metrics.cdc import compute_cdc_dir
from colorization_metrics.metrics.colorfulness import (
    ColorfulnessMetric,
    compute_colorfulness_dir,
)
from colorization_metrics.metrics.fid import compute_fid_dir
from colorization_metrics.metrics.lpips import LPIPSNetworks, compute_lpips_dir
from colorization_metrics.metrics.niqe import compute_niqe_dir
from colorization_metrics.metrics.psnr import compute_psnr_dir
from colorization_metrics.metrics.ssim import compute_ssim_dir
from colorization_metrics.utils import RANDOM_SEED, ColorSpace, InterruptSignalHandler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset_utils import (  # pylint: disable=import-error
    ALTERATIONS as IMAGE_ALTERATIONS,
)
from utils.dataset_utils import (  # pylint: disable=import-error
    DATA_DIR,
    EXPERIMENTS_DIR,
    PLOT_EXTENSIONS,
    RESULTS_DIR,
    Alteration,
    Metric,
)
from utils.image_deterioration import enhance_contrast  # pylint: disable=import-error
from utils.video2frames import create_video_from_images  # pylint: disable=import-error

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


VIDEOS_DIR = DATA_DIR / "videos"
OUTPUT_DIR = EXPERIMENTS_DIR / "exp4_video"
METRICS_VIDEO = {
    "psnr_rgb": Metric(
        compute_function=partial(compute_psnr_dir, color_space=ColorSpace.RGB.value),
        name="PSNR in RGB color space",
        uses_reference=True,
        higher_is_better=True,
    ),
    "psnr_lab": Metric(
        compute_function=partial(compute_psnr_dir, color_space=ColorSpace.LAB.value),
        name="PSNR on a* and b* chrominances",
        uses_reference=True,
        higher_is_better=True,
    ),
    "ssim": Metric(
        compute_function=compute_ssim_dir,
        name="SSIM",
        uses_reference=True,
        higher_is_better=True,
    ),
    "lpips_alex": Metric(
        compute_function=partial(
            compute_lpips_dir, net_type=LPIPSNetworks.ALEX_NET.value
        ),
        name="LPIPS with a AlexNet network",
        uses_reference=True,
        higher_is_better=False,
    ),
    "lpips_vgg": Metric(
        compute_function=partial(compute_lpips_dir, net_type=LPIPSNetworks.VGG.value),
        name="LPIPS with a VGG network",
        uses_reference=True,
        higher_is_better=False,
    ),
    "fid_64": Metric(
        compute_function=partial(compute_fid_dir, fid_dims=64),
        name="FID with 64 features",
        uses_reference=True,
        higher_is_better=False,
    ),
    "fid_192": Metric(
        compute_function=partial(compute_fid_dir, fid_dims=192),
        name="FID with 192 features",
        uses_reference=True,
        higher_is_better=False,
    ),
    "fid_768": Metric(
        compute_function=partial(compute_fid_dir, fid_dims=768),
        name="FID with 768 features",
        uses_reference=True,
        higher_is_better=False,
    ),
    "fid_2048": Metric(
        compute_function=partial(compute_fid_dir, fid_dims=2048),
        name="FID with 2048 features",
        uses_reference=True,
        higher_is_better=False,
    ),
    "brisque": Metric(
        compute_function=compute_brisque_dir,
        name="BRISQUE",
        uses_reference=False,
        higher_is_better=False,
    ),
    "brisque_cor": Metric(
        compute_function=partial(
            compute_brisque_dir, method=FeaturesUsed.RGB_CORRELATION.value
        ),
        name="BRISQUE - RGB Correlation",
        uses_reference=False,
        higher_is_better=False,
    ),
    "niqe": Metric(
        compute_function=compute_niqe_dir,
        name="NIQE",
        uses_reference=False,
        higher_is_better=False,
    ),
    "niqe_cor": Metric(
        compute_function=partial(
            compute_niqe_dir, method=FeaturesUsed.RGB_CORRELATION.value
        ),
        name="NIQE - RGB Correlation",
        uses_reference=False,
        higher_is_better=False,
    ),
    "colorfulness_1": Metric(
        compute_function=partial(
            compute_colorfulness_dir, metric=ColorfulnessMetric.AB_CHANNELS.value
        ),
        name="Colorfulness version 1",
        uses_reference=False,
        higher_is_better=True,
    ),
    "colorfulness_2": Metric(
        compute_function=partial(
            compute_colorfulness_dir, metric=ColorfulnessMetric.CHROMA_MEAN.value
        ),
        name="Colorfulness version 2",
        uses_reference=False,
        higher_is_better=True,
    ),
    "colorfulness_3": Metric(
        compute_function=partial(
            compute_colorfulness_dir, metric=ColorfulnessMetric.RG_YB_CHANNELS.value
        ),
        name="Colorfulness version 3",
        uses_reference=False,
        higher_is_better=True,
    ),
    "cdc_rgb": Metric(
        compute_function=compute_cdc_dir,
        name="CDC - RGB",
        uses_reference=False,
        higher_is_better=False,
    ),
    "cdc_lab": Metric(
        compute_function=partial(compute_cdc_dir, color_space=ColorSpace.LAB),
        name="CDC - a*b*",
        uses_reference=False,
        higher_is_better=False,
    ),
}

ALTERATIONS = IMAGE_ALTERATIONS
ALTERATIONS["contrast"] = Alteration(
    alteration_function=enhance_contrast, name="Contrasted", type_color="yellow"
)
TEMPORAL_ALTERATIONS = {
    "Lines on all frames": "lines",
    "One frame": "one_frame",
    "Every four frames": "every_four_frame",
}


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def main(
    video_dir: str | Path = VIDEOS_DIR,
    video_to_plot: str = "sheeps",
    compute_only_alterations_for_plot: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """Processes videos in the specified directory by applying a series of alterations.

    Args:
        video_dir: Path to the directory containing video files.
        video_to_plot: Name of the video to plot graphs.
        compute_only_alterations_for_plot: If True, compute only alterations
            and metrics needed for the graph plotted.
    """
    video_dir = Path(video_dir)

    # Select alterations to use
    alterations_to_use = [
        "translated_all",
        "chrominance_translated",
        # "luminance_translated",
        # "siggraph17",
        "desaturated",
        "sepia",
        "altered_hue",
    ]
    if compute_only_alterations_for_plot:
        alterations = {
            key: value
            for key, value in ALTERATIONS.items()
            if key in alterations_to_use
        }
    else:
        alterations = ALTERATIONS

    # Apply alterations and measures videos
    videos_paths = sorted(
        [video_dir / f for f in video_dir.iterdir() if f.suffix == ".mkv"]
    )
    for path in videos_paths:
        print(f"=== {path}")  # noqa: T201
        for suffix, alteration in alterations.items():
            if not (OUTPUT_DIR / path.stem / f"{suffix}_lines" / RESULT_FILE).is_file():
                gt_dir, alt_dirs = _video_alteration(
                    path, suffix, alteration.alteration_function
                )
                for directory in alt_dirs:
                    results = _compute_metrics(directory, gt_dir)
                    with (
                        (Path(directory) / RESULT_FILE).open(
                            "w", encoding="utf-8"
                        ) as f,
                        InterruptSignalHandler(),
                    ):
                        f.write(results)
    metrics_results = _recover_analysis(video_to_plot, alterations_to_use)
    _plot_metrics(video_to_plot, metrics_results)


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _video_alteration(
    video_path: str | Path, alteration_suffix: str, processing_func: Callable
) -> tuple[str, tuple[str]]:
    """Applies a specified alteration to a video and saves the results.

    This function extracts frames from a video, processes them using a given
    alteration function, and reassembles altered frames into new videos.

    Args:
        video_path: Path to the input video file.
        alteration_suffix: Suffix used to identify the alteration type in
            the output directory and filenames.
        processing_func: A function that processes individual frames.
            It must accept a frame as its first argument and can accept additional
            keyword arguments.

    Notes:
        - Three types of output videos are generated:
            1. `*_lines.mkv`: Frames where only a specified region of lines is altered.
            2. `*_one_frame.mkv`: A video where only one frame is altered.
            3. `*_every_four_frames.mkv`: A video where every fourth frame is altered.
        - Frame alterations are localized based on a random seed for reproducibility.
        - Outputs are saved in a structured directory under `results/videos/<base_filename>`.
    """  # noqa: E501
    video_path = Path(video_path)

    # Frames extraction
    frames_dir = prepare_directory(video_path, "data/videos/frames")
    frames_files = sorted([f for f in frames_dir.iterdir() if f.suffix == ".png"])

    # Video parameters
    frame_nb = len(frames_files)
    height, _, _ = img_as_ubyte(imread(frames_dir / frames_files[0])).shape
    frame_digits = max([3, len(str(abs(frame_nb)))])
    altered_lines_nb = max(1, ceil(height / frame_nb))
    margin_percent = max(0.1, altered_lines_nb / height)

    # Alteration localization
    random.seed(RANDOM_SEED)
    random_percent = random.uniform(margin_percent, 1 - margin_percent)  # noqa: S311
    altered_frame_id = round(random_percent * frame_nb)
    altered_line_id = round(random_percent * height)
    print(  # noqa: T201
        f"Modified lines: {altered_line_id} to {altered_line_id + altered_lines_nb - 1}"
    )
    print(f"Modified frame: {altered_frame_id}")  # noqa: T201

    # Output on storage
    base_filename = video_path.stem
    line_alt_dir = RESULTS_DIR / "videos" / base_filename / f"{alteration_suffix}_lines"

    frame_alt_dir = (
        RESULTS_DIR / "videos" / base_filename / f"{alteration_suffix}_one_frame"
    )

    four_frames_alt_dir = (
        RESULTS_DIR / "videos" / base_filename / f"{alteration_suffix}_every_four_frame"
    )
    line_alt_dir.mkdir(parents=True, exist_ok=True)
    frame_alt_dir.mkdir(parents=True, exist_ok=True)
    four_frames_alt_dir.mkdir(parents=True, exist_ok=True)

    # Frames alteration
    for i, image_path in tqdm(
        enumerate(frames_files),
        desc=f"{alteration_suffix} on {base_filename}",
        total=frame_nb,
        ncols=90,
    ):
        image = img_as_float(imread(frames_dir / image_path))
        image_line_alt = image.copy()
        image_frame_alt = processing_func(image.copy())
        image_line_alt[altered_line_id : altered_line_id + altered_lines_nb, :] = (
            image_frame_alt[altered_line_id : altered_line_id + altered_lines_nb, :]
        )
        imsave(line_alt_dir / f"{i:0{frame_digits}d}.png", img_as_ubyte(image_line_alt))
        imsave(
            frame_alt_dir / f"{i:0{frame_digits}d}.png",
            img_as_ubyte(image_frame_alt)
            if i == altered_frame_id
            else img_as_ubyte(image),
        )
        imsave(
            four_frames_alt_dir / f"{i:0{frame_digits}d}.png",
            img_as_ubyte(image_frame_alt) if i % 4 == 0 else img_as_ubyte(image),
        )

    # Video creation
    video_dir = RESULTS_DIR / "videos" / base_filename
    create_video_from_images(line_alt_dir, video_dir / f"{alteration_suffix}_lines.mkv")
    create_video_from_images(
        frame_alt_dir, video_dir / f"{alteration_suffix}_one_frame.mkv"
    )
    create_video_from_images(
        four_frames_alt_dir, video_dir / f"{alteration_suffix}_every_four_frames.mkv"
    )
    return (frames_dir, (line_alt_dir, frame_alt_dir, four_frames_alt_dir))


def _compute_metrics(alterations_dir: str, gt_dir: str) -> str:
    """Computes various metrics to evaluate the quality of altered videos.

    The function calculates metrics such as PSNR, SSIM, LPIPS, BRISQUE, NIQE,
    colorfulness, and FID for the frames in the specified directories. It aggregates
    the results into a dictionary and writes them to a file.

    Args:
        alterations_dir: Path to the directory containing altered images.
        gt_dir: Path to the directory containing ground truth images.

    Returns:
        Path to the file where the metrics results are written.
    """
    metrics = {}
    for suffix, metric in METRICS_VIDEO.items():
        if metric.uses_reference:
            metrics[suffix] = metric.compute_function(alterations_dir, gt_dir)
        else:
            metrics[suffix] = metric.compute_function(alterations_dir)

    return write_results(metrics)


def _plot_metrics(
    video_name: str,
    metric_data: dict[str, dict[str, float]],
    graph_title: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Plot dataset metrics for altered images and save the plots.

    Args:
        video_name: Name of the video
        metric_data: A dictionary where each key is
            a metric name (e.g., "psnr_rgb") and the value is another dictionary. This
            inner dictionary maps altered image filenames to lists of metric values.
        graph_title: Put the title on the graph.
    """
    for metric_name, altered_images in metric_data.items():
        higher_is_better = METRICS_VIDEO[metric_name].higher_is_better

        # Sort the altered images based on the median metric value
        sorted_alterations = sorted(
            altered_images.items(),
            key=lambda alteration: alteration[1],
            reverse=higher_is_better,
        )
        alteration_name, metric_values = zip(*sorted_alterations, strict=False)
        alteration_color = {
            ALTERATIONS[alteration_name].name: ALTERATIONS[alteration_name].type_color
            for alteration_name in sorted_alterations
        }

        # Plot the graph for this metric
        plt.figure(figsize=(10, 6))
        plt.semilogy(metric_values, color="black", marker="o")
        plt.grid(visible=True, which="both", axis="y")

        # Spines removal
        ax = plt.gca()
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # Customize x-tick labels and add color coding
        plt.xticks(
            ticks=range(len(metric_values)),
            labels=alteration_name,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        plt.xlabel("Altered Image Filename")
        for label in plt.gca().get_xticklabels():
            text = label.get_text().split(" - ")[0]
            if text in alteration_color:
                label.set_color(alteration_color[text])

        # Customize y-labels
        plt.ylabel(metric_name)
        if graph_title:
            plt.title(f"{METRICS_VIDEO[metric_name].name} for the {video_name} video")
        plt.tight_layout()

        # Save the plot
        for ext in PLOT_EXTENSIONS:
            plot_path = RESULTS_DIR / "videos" / video_name / f"{metric_name}.{ext}"
            plt.savefig(plot_path)
        plt.close()


def _recover_analysis(
    video_name: str, alterations: dict[str, str]
) -> dict[str, dict[str, float]]:
    results_path = RESULTS_DIR / "videos" / video_name
    compiled_metrics: dict[str, dict[str, list[float]]] = {}
    for metric_suffix in METRICS_VIDEO:
        compiled_metrics[metric_suffix] = {}
    for space_alt, space_prefix in alterations.items():
        for tempo_alt, tempo_suffix in TEMPORAL_ALTERATIONS.items():
            result_file = (
                results_path / (space_prefix + "_" + tempo_suffix) / RESULT_FILE
            )
            alteration_res = _read_result_file(result_file)
            for metric, value in alteration_res.items():
                compiled_metrics[metric][space_alt + " - " + tempo_alt] = value
    return compiled_metrics


def _read_result_file(file: str | Path) -> dict[str, float]:
    """Reads a result file and parses its metrics into a dictionary.

    Args:
        file: Path to the result file containing metrics.

    Returns:
        A dictionary where keys are metric names and values
        are the corresponding float values.
        Metrics with `inf` are stored as `float('inf')`.
    """
    results = {}
    with Path(file).open(encoding="utf-8") as f:
        for line in f:
            key, value = map(str.strip, line.split("=", 1))
            if value.lower() == "inf":
                results[key] = float("inf")
            else:
                results[key] = float(value)
    return results


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    main(VIDEOS_DIR, "dance-twirl")
