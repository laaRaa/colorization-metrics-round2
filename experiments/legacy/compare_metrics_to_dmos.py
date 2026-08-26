"""Module for processing and visualizing DMOS predictions.

This module provides functionality to read DMOS (Differential Mean Opinion Score)
data from JSON files, process the data to generate sorted plots based on various
metrics, and save the plots as PDF files. The module includes functions to sort
lists based on specific criteria and to generate plots using matplotlib.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from orasis_eval import DISTORTIONS_LIVE_IQA
from orasis_eval import OUTPUT_DIR as RESULT_DIR
from tqdm import tqdm

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


OUTPUT_DIR = os.path.join("results", "dmos_prediction")
"""Output directory."""
DMOS_METRICS = {
    "psnr_rgb",
    "psnr_ab",
    "ssim",
    "brisque",
    "brisque_org",
    "brisque_cor",
    "brisque_fea",
    "brisque_all",
    "niqe",
    "niqe_org",
    "niqe_cor",
    "niqe_fea",
    "niqe_all",
}
"""Set of metrics used to predict the DMOS."""


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def main() -> None:
    """Main function to process DMOS predictions and generate plots.

    This function reads DMOS data from JSON files, processes the data to generate
    sorted plots based on different metrics, and saves the plots as PDF files.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for distortion in DISTORTIONS_LIVE_IQA:
        print(f"--- Distortion studied : {distortion}")
        with open(
            os.path.join(RESULT_DIR, "dmos_" + distortion + ".json"), encoding="utf-8"
        ) as file:
            dmos_data = json.load(file)
        with open(
            os.path.join(RESULT_DIR, "spearman_" + distortion + ".json"),
            encoding="utf-8",
        ) as file:
            spearman_data = json.load(file)
        with open(
            os.path.join(RESULT_DIR, "pearson_" + distortion + ".json"),
            encoding="utf-8",
        ) as file:
            pearson_data = json.load(file)

        for metric_name in DMOS_METRICS:
            correl_names = ["spearman", "pearson"]
            correl_metrics = [spearman_data[metric_name], pearson_data[metric_name]]
            for name, metric in zip(correl_names, correl_metrics):
                metric_mean = np.mean(metric)
                metric_median = np.median(metric)
                plt.figure(figsize=(10, 6))
                plt.hist(metric, 20, label="Coefficients")
                plt.axvline(
                    metric_mean,
                    color="b",
                    linestyle="--",
                    label=f"Mean = {metric_mean:.4f}",
                )
                plt.axvline(
                    metric_median,
                    color="r",
                    linestyle="-",
                    label=f"Median = {metric_median:.4f}",
                )
                plt.ylabel(name)
                plt.title(f"Correlation on {metric_name}, computed with {name}.")
                plt.tight_layout()
                plt.legend()
                plt.grid(True)
                pdf_plot_path = os.path.join(
                    OUTPUT_DIR, f"{distortion}_{metric_name}_{name}.pdf"
                )
                plt.savefig(pdf_plot_path)
                plt.close()

        for iteration in tqdm(
            range(len(dmos_data["dmos"])), desc="Splits train/test", ncols=90
        ):
            real_dmos = dmos_data["dmos"][iteration]
            indices = list(range(len(real_dmos)))
            for metric_name in DMOS_METRICS:
                pred_dmos = dmos_data[metric_name][iteration]
                if "brisque" not in metric_name:
                    true_dmos = normalize_min_max(real_dmos)
                    pred_dmos = normalize_min_max(pred_dmos)
                else:
                    true_dmos = real_dmos
                sort_functions = [_sort_by_first_list, _sort_by_error]
                sort_names = ["DMOS increasing", "prediction error increasing"]
                sort_suffix = ["dmos", "error"]
                for func, name, suffix in zip(sort_functions, sort_names, sort_suffix):
                    plt.figure(figsize=(10, 6))
                    real_sorted, pred_sorted = func(true_dmos, pred_dmos)
                    plt.plot(indices, real_sorted, label="Real DMOS")
                    plt.plot(indices, pred_sorted, label="DMOS predicted")
                    plt.ylabel("DMOS")
                    plt.title(f"DMOS predicted with {metric_name}, sorted by {name}.")
                    plt.tight_layout()
                    plt.legend()
                    plt.grid(True)
                    pdf_plot_path = os.path.join(
                        OUTPUT_DIR,
                        f"{distortion}_{metric_name}_{suffix}_{iteration:03d}.pdf",
                    )
                    plt.savefig(pdf_plot_path)
                    plt.close()


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def normalize_min_max(data):
    """Normalize a list of floats using min-max normalization."""
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def _sort_by_first_list(
    first_list: list[float], second_list: list[float]
) -> tuple[list[float], list[float]]:
    """Sorts two lists based on the values of the first list.

    Args:
        first_list: The primary list to sort by.
        second_list: The secondary list to sort based on the first list.

    Returns:
        Two lists sorted based on the values of the first list.

    Raises:
        ValueError: If the input lists do not have the same length.
    """
    if len(first_list) != len(second_list):
        raise ValueError("Both lists should have same length.")

    first_list_sorted = sorted(first_list)
    second_list_sorted = [y for _, y in sorted(zip(first_list, second_list))]

    return first_list_sorted, second_list_sorted


def _sort_by_error(
    first_list: list[float], second_list: list[float]
) -> tuple[list[float], list[float]]:
    """Sorts two lists based on the absolute error between their elements.

    Args:
        first_list: The first list of values.
        second_list: The second list of values.

    Returns:
        Two lists sorted based on the absolute error between their elements.

    Raises:
        ValueError: If the input lists do not have the same length.
    """
    if len(first_list) != len(second_list):
        raise ValueError("Both lists should have same length.")

    errors = [abs(a - b) for a, b in zip(first_list, second_list)]

    first_list_sorted = [x for _, x in sorted(zip(errors, first_list))]
    second_list_sorted = [y for _, y in sorted(zip(errors, second_list))]

    return first_list_sorted, second_list_sorted


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
