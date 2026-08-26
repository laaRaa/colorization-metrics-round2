"""Measure and report execution time of image quality metrics.

This script evaluates the computational cost of several image quality metrics
defined in the `METRICS` dictionary. It measures execution time for each metric
on a test image, with or without a reference image depending on the metric type.
The results are formatted into a LaTeX table for reporting.
"""

import json
import statistics
import sys
import time
import warnings
from pathlib import Path

from skimage.io import imread
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset_utils import (  # pylint: disable=import-error
    EXPERIMENTS_DIR,
    METRICS,
    Metric,
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


IMAGES_DIR = Path("data") / "pictures"
OUTPUT_DIR = EXPERIMENTS_DIR / "exp5_time"
REFERENCE_IMAGE = IMAGES_DIR / "doggy.jpg"
TEST_IMAGE = IMAGES_DIR / "event.jpg"
RESULT_FILE_NAME = "compute_time"


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def measure_image_time(
    image_path: str | Path = TEST_IMAGE,
    ref_path: str | Path = REFERENCE_IMAGE,
    repeats: int = 100,
) -> None:
    """Measure execution time of image quality metrics on a given image.

    This function computes the runtime of all metrics listed in `METRICS`
    on a given test image. If the metric requires a reference, it is also
    provided. The function prints a LaTeX-formatted table with the results,
    including the metric name and execution time in seconds.

    Args:
        image_path: Path to the test image.
        ref_path: Path to the reference image.
        repeats: Number of compute time measures.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    image_path = Path(image_path)
    ref_path = Path(ref_path)
    image_res = imread(image_path).shape[:2]

    stats_file_numbered = OUTPUT_DIR / f"{RESULT_FILE_NAME}_{repeats}.json"
    if not stats_file_numbered.is_file():
        measured_times: dict[str, tuple[str, tuple[float, float]]] = {}
        for name, metric in tqdm(METRICS.items(), desc="Metrics", ncols=90):
            measured_times[name] = (
                metric.name,
                _measure_metric_times(metric, image_path, ref_path, repeats),
            )

        latex_table = _generate_latex_table(
            measured_times, image_path.stem, image_res, repeats
        )

        stats_file = OUTPUT_DIR / f"{RESULT_FILE_NAME}.json"
        for st_file in (stats_file_numbered, stats_file):
            with st_file.open("w", encoding="utf8") as f:
                json.dump(measured_times, f, indent=2)
    else:
        print(f"Loading computed results in {stats_file_numbered}")  # noqa: T201
        latex_table = _generate_latex_table(
            _get_old_results(repeats), image_path.stem, image_res, repeats
        )

    print(latex_table)  # noqa: T201
    result_file_numbered = OUTPUT_DIR / f"{RESULT_FILE_NAME}_{repeats}.tex"
    result_file = OUTPUT_DIR / f"{RESULT_FILE_NAME}.tex"
    for res_file in (result_file_numbered, result_file):
        with res_file.open("w", encoding="utf8") as f:
            f.write(latex_table)


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _get_old_results(repeats: int) -> dict[str, tuple[str, tuple[float, float]]]:
    stats_file_numbered = OUTPUT_DIR / f"{RESULT_FILE_NAME}_{repeats}.json"
    with stats_file_numbered.open("r", encoding="utf8") as f:
        return json.load(f)


def _measure_metric_times(
    metric: Metric, image_path: Path, ref_path: Path, repeats: int = 100
) -> tuple[float, float]:
    """Measure mean and variance of execution time of a metric over multiple runs.

    Args:
        metric: Metric object containing the compute function and metadata.
        image_path: Path to the input image.
        ref_path: Path to the reference image.
        repeats: Number of times to repeat the measurement.

    Returns:
        Mean and variance of execution times in seconds.
    """
    times = []
    for _ in range(repeats):
        start = time.time()
        if metric.uses_reference:
            _ = metric.compute_function(image_path, ref_path)
        else:
            _ = metric.compute_function(image_path)
        end = time.time()
        times.append(end - start)

    mean_time = statistics.mean(times)
    var_time = statistics.variance(times) if repeats > 1 else 0.0
    return mean_time, var_time


def _generate_latex_table(
    results: dict[str, tuple[str, tuple[float, float]]],
    image_name: str,
    resolution: tuple[int, int],
    repeats: int,
) -> str:
    """Generate a LaTeX table from a dictionary, with an optional caption.

    Args:
        results: Dictionary with metric names and measured times.
        image_name: Name of the image evaluated.
        resolution: Resolution of the image evaluated.
        repeats: Number of times the time is measured for each metrics.

    Returns:
        LaTeX code for the table inside a `table` environment.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Computational time for image ``{image_name}'' of resolution {resolution[0]} \\(\\times\\) {resolution[1]}. For each metrics, the time have been measured {repeats} times.}}",  # noqa: E501
        r"\label{tab:comp_times}",
        r"\begin{tabular}{l|l|l}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Mean time (s)} & \textbf{StD (s)} \\",
        r"\hline",
    ]

    sorted_values = sorted(results.values(), key=lambda x: x[1])
    lines.extend(
        f"{value[0]} & {value[1][0]:.2f} & {value[1][1]:.1e} \\\\"
        for value in sorted_values
    )

    lines += [r"\end{tabular}", r"\end{table}"]

    return "\n".join(line for line in lines if line)


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

print("oui")  # noqa: T201

if __name__ == "__main__":
    measure_image_time(repeats=100)
    measure_image_time(repeats=1000)
