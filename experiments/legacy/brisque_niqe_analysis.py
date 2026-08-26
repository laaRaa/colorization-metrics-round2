import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from colorization_metrics.metrics.brisque import FeaturesUsed, compute_brisque
from colorization_metrics.metrics.niqe import compute_niqe
from colorization_metrics.metrics.psnr import compute_psnr
from colorization_metrics.metrics.ssim import compute_ssim
from colorization_metrics.utils import ColorSpace, moving_average

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.image_deterioration import (
    desaturate_image,
    gaussian_noise,
    gaussian_noise_chrominances,
    initialize_image,
    process_and_save_image,
    shift_hue,
    translate_a_chrominance,
    translate_b_chrominance,
    translate_chrominances,
)

METRIC_NAME_CORRESPONDANCE = {
    "psnr_rgb": ("PSNR - RVB", "-b"),
    "psnr_ab": ("PSNR - a*b*", "--b"),
    "ssim": ("SSIM", "-y"),
    "brisque": ("BRISQUE", "o-g"),
    "brisque_org": ("BRISQUE - Original", "-g"),
    "brisque_cor": ("BRISQUE - Correl", "--g"),
    "brisque_fea": ("BRISQUE - Features", "-.g"),
    "brisque_all": ("BRISQUE - All", ":g"),
    "niqe": ("NIQE", "o-r"),
    "niqe_org": ("NIQE - Original", "-r"),
    "niqe_cor": ("NIQE - Correl", "--r"),
    "niqe_fea": ("NIQE - Features", "-.r"),
    "niqe_all": ("NIQE - All", ":r"),
}
METRIC_GREATER_IS_BETTER = {
    "psnr_rgb": False,
    "psnr_ab": False,
    "ssim": False,
    "brisque": True,
    "brisque_org": True,
    "brisque_cor": True,
    "brisque_fea": True,
    "brisque_all": True,
    "niqe": True,
    "niqe_org": True,
    "niqe_cor": True,
    "niqe_fea": True,
    "niqe_all": True,
}

FEATURES_NB = {"org": 36, "cor": 60, "fea": 108, "all": 132}


def norm_list(l: list[float], greater_is_better: bool = True) -> NDArray[np.float64]:
    array = np.array(l)
    min_value = np.min(array)
    max_value = np.max(array)

    if max_value == min_value:
        return np.ones_like(array, dtype=np.float64)

    if greater_is_better:
        normalized_array = (array - min_value) / (max_value - min_value)
    else:
        normalized_array = (max_value - array) / (max_value - min_value)

    return normalized_array


def plot_moving_average(
    x: list[float] | NDArray[np.float64],
    y: list[float] | NDArray[np.float64],
    window_size: int,
    fmt: str = "",
    **kwargs,
) -> None:
    """Plots the moving average of a given dataset.

    This function calculates the moving average of the y-values
        over a specified window size and plots it against the corresponding x-values.

    Args:
        x: A list or NumPy array of x-values.
        y: A list or NumPy array of y-values.
        window_size: The size of the moving average window.
        fmt: The format string for plotting. Defaults to "".
        **kwargs: Additional keyword arguments to pass to the plot function.

    Raises:
        ValueError: If the window size is not a positive integer.
        ValueError: If the length of x and y are not the same.

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 4, 5, 4, 5])
        >>> plot_moving_average(x, y, window_size=3, fmt="r-")
        >>> plt.show()
    """
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    if len(x) != len(y):
        raise ValueError("The length of x and y must be the same.")

    y_moving_avg = moving_average(y, window_size)
    x_moving_avg = x[(window_size - 1) // 2 : -(window_size // 2)]
    plt.plot(x_moving_avg, y_moving_avg, fmt, **kwargs)


def _plot_result(
    result: dict[str, list[float]],
    parameters: NDArray[np.float64],
    metric_names: list[str],
    images: list[str],
    ylabel: str,
    alt_name: str,
    suffix: str,
    output_dir: str,
) -> None:
    # Boxplots
    plt.figure(figsize=(10, 6))
    for metric_name in metric_names:
        plt.boxplot(
            result[metric_name],
            label=METRIC_NAME_CORRESPONDANCE[metric_name][0],
            tick_labels=parameters,
            showfliers=False,
        )

    plt.xlabel("Parameter")
    plt.ylabel(ylabel)
    plt.title(f"Evaluation of {alt_name} on the images")
    plt.legend()
    plt.grid(True)

    png_plot_path = os.path.join(output_dir, f"boxplot_{metric_names[0]}_{suffix}.png")
    pdf_plot_path = os.path.join(output_dir, f"boxplot_{metric_names[0]}_{suffix}.pdf")
    plt.savefig(png_plot_path)
    plt.savefig(pdf_plot_path)
    plt.close()

    # Median plots
    plt.figure(figsize=(10, 6))
    for metric_name in metric_names:
        plt.plot(
            parameters,
            np.median(result[metric_name], axis=0),
            label=METRIC_NAME_CORRESPONDANCE[metric_name][0],
        )

    plt.xlabel("Parameter")
    plt.ylabel(ylabel)
    plt.title(f"Evaluation of {alt_name} on the images")
    plt.legend()
    plt.grid(True)

    png_plot_path = os.path.join(
        output_dir, f"medianplot_{metric_names[0]}_{suffix}.png"
    )
    pdf_plot_path = os.path.join(
        output_dir, f"medianplot_{metric_names[0]}_{suffix}.pdf"
    )
    plt.savefig(png_plot_path)
    plt.savefig(pdf_plot_path)
    plt.close()

    # Images plots
    for i, img in enumerate(images):
        simple_name, _ = os.path.splitext(img)
        plt.figure(figsize=(10, 6))
        for metric_name in metric_names:
            plt.plot(
                parameters,
                result[metric_name][i],
                label=METRIC_NAME_CORRESPONDANCE[metric_name][0],
            )

        plt.xlabel("Parameter")
        plt.ylabel(ylabel)
        plt.title(f"Evaluation of {alt_name} on the {img} image")
        plt.legend()
        plt.grid(True)

        png_plot_path = os.path.join(
            output_dir, simple_name, f"plot_{metric_names[0]}_{suffix}.png"
        )
        pdf_plot_path = os.path.join(
            output_dir, simple_name, f"plot_{metric_names[0]}_{suffix}.pdf"
        )
        plt.savefig(png_plot_path)
        plt.savefig(pdf_plot_path)
        plt.close()


def _load_metrics(
    image_path: str, output_dir: str, alt_suffix: str
) -> dict[str, list[float]]:
    _, base_filename, _ = initialize_image(image_path, output_dir)
    with open(
        os.path.join(output_dir, base_filename + f"_{alt_suffix}.json"),
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)
    return metrics


def _compute_metrics(
    image_path: str,
    output_dir: str,
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    alt_name: str,
    suffix: str,
    param_name: str,
    parameters: list[float],
) -> None:
    image, base_filename, output_dir_used = initialize_image(image_path, output_dir)

    result_metrics = {
        "psnr_rgb": [],
        "psnr_ab": [],
        "ssim": [],
        "brisque": [],
        "brisque_org": [],
        "brisque_cor": [],
        "brisque_fea": [],
        "brisque_all": [],
        "niqe": [],
        "niqe_org": [],
        "niqe_cor": [],
        "niqe_fea": [],
        "niqe_all": [],
    }
    for param in tqdm(parameters, desc=base_filename, ncols=100):
        transformed_image_file = process_and_save_image(
            image,
            output_dir_used,
            base_filename,
            suffix + f"_{param}",
            function,
            **{param_name: param},
        )
        transformed_image_path = os.path.join(output_dir_used, transformed_image_file)
        result_metrics["psnr_rgb"].append(
            compute_psnr(transformed_image_path, image_path, ColorSpace.RGB.value)
        )
        result_metrics["psnr_ab"].append(
            compute_psnr(transformed_image_path, image_path, ColorSpace.LAB.value)
        )
        result_metrics["ssim"].append(compute_ssim(transformed_image_path, image_path))
        result_metrics["brisque"].append(compute_brisque(transformed_image_path))
        result_metrics["brisque_org"].append(
            compute_brisque(transformed_image_path, method=FeaturesUsed.ORIGINAL.value)
        )
        result_metrics["brisque_cor"].append(
            compute_brisque(
                transformed_image_path, method=FeaturesUsed.RGB_CORRELATION.value
            )
        )
        result_metrics["brisque_fea"].append(
            compute_brisque(
                transformed_image_path, method=FeaturesUsed.RGB_ANALYSIS.value
            )
        )
        result_metrics["brisque_all"].append(
            compute_brisque(transformed_image_path, method=FeaturesUsed.RGB_ALL.value)
        )
        result_metrics["niqe"].append(
            compute_niqe(transformed_image_path) / FEATURES_NB["org"]
        )
        result_metrics["niqe_org"].append(
            compute_niqe(transformed_image_path, method=FeaturesUsed.ORIGINAL.value)
            / FEATURES_NB["org"]
        )
        result_metrics["niqe_cor"].append(
            compute_niqe(
                transformed_image_path, method=FeaturesUsed.RGB_CORRELATION.value
            )
            / FEATURES_NB["cor"]
        )
        result_metrics["niqe_fea"].append(
            compute_niqe(transformed_image_path, method=FeaturesUsed.RGB_ANALYSIS.value)
            / FEATURES_NB["fea"]
        )
        result_metrics["niqe_all"].append(
            compute_niqe(transformed_image_path, method=FeaturesUsed.RGB_ALL.value)
            / FEATURES_NB["all"]
        )

    with open(
        os.path.join(output_dir, base_filename + f"_{suffix}.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result_metrics, file)


def main(
    images: tuple[str],
    nb_param: int = 100,
    compute_all_metrics: bool = True,
    output_dir: str = "results/orasis/color_alteration_boxplot",
):
    functions_list = (
        shift_hue,
        desaturate_image,
        translate_chrominances,
        translate_a_chrominance,
        translate_b_chrominance,
        gaussian_noise,
        gaussian_noise_chrominances,
    )
    suffix_list = ("hue_shift", "desaturation", "blue")
    alt_name_list = ("Hue shift", "Desaturation", "Blue filter")
    parameters_list = (
        np.linspace(0, 360, nb_param + 1)[:-1],
        np.linspace(0, 1, nb_param + 1)[:-1],
        np.linspace(0, 200, nb_param).astype(int),
        np.linspace(0, 200, nb_param).astype(int),
        np.linspace(0, 200, nb_param).astype(int),
        np.linspace(0, 2, nb_param + 1)[1:],
        np.linspace(0, 2, nb_param + 1)[1:],
    )
    param_name_list = (
        "hue_shift",
        "factor",
        "displacement",
        "displacement",
        "displacement",
        "sigma",
        "sigma",
    )

    for function, suffix, alt_name, parameters, param_name in zip(
        functions_list,
        suffix_list,
        alt_name_list,
        parameters_list,
        param_name_list,
        strict=False,
    ):
        # if suffix != "wn_chr":
        #     continue
        print(f"-------- Alteration : {alt_name}")
        accumulated_metrics = {
            "psnr_rgb": [],
            "psnr_ab": [],
            "ssim": [],
            "brisque": [],
            "brisque_org": [],
            "brisque_cor": [],
            "brisque_fea": [],
            "brisque_all": [],
            "niqe": [],
            "niqe_org": [],
            "niqe_cor": [],
            "niqe_fea": [],
            "niqe_all": [],
        }
        for image_file in images:
            image_path = os.path.join("data", "pictures", image_file)
            if compute_all_metrics:
                _compute_metrics(
                    image_path,
                    output_dir,
                    function,
                    alt_name,
                    suffix,
                    param_name,
                    parameters,
                )
            metrics = _load_metrics(image_path, output_dir, suffix)
            for key, value in metrics.items():
                accumulated_metrics[key].append(value)

        accumulated_metrics_np = {
            key: np.array(value) for key, value in accumulated_metrics.items()
        }
        _plot_result(
            accumulated_metrics_np,
            parameters,
            ["psnr_rgb", "psnr_ab"],
            images,
            "PSNR",
            alt_name,
            suffix,
            output_dir,
        )
        _plot_result(
            accumulated_metrics_np,
            parameters,
            ["ssim"],
            images,
            "SSIM",
            alt_name,
            suffix,
            output_dir,
        )
        _plot_result(
            accumulated_metrics_np,
            parameters,
            ["brisque", "brisque_org", "brisque_cor", "brisque_fea", "brisque_all"],
            images,
            "BRISQUE",
            alt_name,
            suffix,
            output_dir,
        )
        _plot_result(
            accumulated_metrics_np,
            parameters,
            ["niqe", "niqe_org", "niqe_cor", "niqe_fea", "niqe_all"],
            images,
            "NIQE",
            alt_name,
            suffix,
            output_dir,
        )


if __name__ == "__main__":
    image_filenames = (
        "event.jpg",
        "joust.jpg",
        "pirate.jpg",
        "room.jpg",
        "cheese.jpg",
        "salad.jpg",
        # "img16.jpg",
        # "gray.png",
    )
    # image_filenames = ["gray.png"]
    sys.exit(main(image_filenames, 500, True))
