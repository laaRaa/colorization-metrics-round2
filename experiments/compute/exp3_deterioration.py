"""Evaluate evolution of IQA metrics for diverse level of degradations."""

import json
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from colorization_metrics.metrics.brisque import FeaturesUsed, compute_brisque
from colorization_metrics.metrics.maniqa import compute_maniqa
from colorization_metrics.metrics.niqe import compute_niqe
from colorization_metrics.metrics.psnr import compute_psnr
from colorization_metrics.metrics.ssim import compute_ssim
from colorization_metrics.utils import ColorSpace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset_utils import (  # pylint: disable=import-error
    DATA_DIR,
    EXPERIMENTS_DIR,
    PLOT_EXTENSIONS,
)
from utils.image_deterioration import (  # pylint: disable=import-error
    desaturate_image,
    gaussian_noise_chrominances,
    initialize_image,
    process_and_save_image,
    shift_hue,
)

OUTPUT_DIR = EXPERIMENTS_DIR / "exp3_deterioration"
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
    "maniqa": ("MANIQA", "o-g"),
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
    "maniqa": False,
}

FEATURES_NB = {"org": 36, "cor": 60, "fea": 108, "all": 132}


def _plot_result(  # noqa: PLR0913
    result: dict[str, list[float]],
    parameters: NDArray[np.float64],
    metric_names: list[str],
    images: list[str],
    ylabel: str,
    alt_name: str,
    suffix: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
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
    plt.grid(visible=True)

    for ext in PLOT_EXTENSIONS:
        plot_path = output_dir / f"medianplot_{metric_names[0]}_{suffix}.{ext}"
        plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    # Images plots
    for i, img in enumerate(images):
        simple_name = Path(img).stem
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
        plt.grid(visible=True)

        for ext in PLOT_EXTENSIONS:
            plot_path = (
                output_dir / simple_name / f"plot_{metric_names[0]}_{suffix}.{ext}"
            )
            plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


def _check_if_metrics_saved(
    image_path: str | Path, output_dir: str | Path, alt_suffix: str
) -> bool:
    _, base_filename, _ = initialize_image(image_path, output_dir)
    file = Path(output_dir) / (base_filename + f"_{alt_suffix}.json")
    return file.exists()


def _load_metrics(
    image_path: str, output_dir: str, alt_suffix: str
) -> dict[str, list[float]]:
    _, base_filename, _ = initialize_image(image_path, output_dir)
    with (Path(output_dir) / (base_filename + f"_{alt_suffix}.json")).open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


def _compute_metrics(  # noqa: PLR0913
    image_path: str | Path,
    output_dir: str | Path,
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
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
        "maniqa": [],
        "niqe": [],
        "niqe_org": [],
        "niqe_cor": [],
        "niqe_fea": [],
        "niqe_all": [],
    }
    for param in tqdm(parameters, desc=base_filename, ncols=100):
        param_function = partial(function, **{param_name: param})
        transformed_image_file = process_and_save_image(
            image, output_dir_used, base_filename, suffix + f"_{param}", param_function
        )
        transformed_image_path = output_dir_used / transformed_image_file
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
        result_metrics["maniqa"].append(compute_maniqa(transformed_image_path))
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

    with (Path(output_dir) / (base_filename + f"_{suffix}.json")).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result_metrics, file)


def main(
    images: tuple[str],
    nb_param: int = 100,
    compute_all_metrics: bool = False,  # noqa: FBT001, FBT002
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Main function."""
    functions_list = (shift_hue, desaturate_image, gaussian_noise_chrominances)
    suffix_list = ("hue_shift", "desaturation", "wn_chr")
    alt_name_list = ("Hue shift", "Desaturation", "White noise on chrominances")
    parameters_list = (
        np.linspace(0, 360, nb_param + 1)[:-1],
        np.linspace(0, 1, nb_param + 1)[:-1],
        np.linspace(0, 2, nb_param + 1)[1:],
    )
    param_name_list = ("hue_shift", "factor", "sigma")

    for function, suffix, alt_name, parameters, param_name in zip(
        functions_list,
        suffix_list,
        alt_name_list,
        parameters_list,
        param_name_list,
        strict=False,
    ):
        print(f"-------- Alteration : {alt_name}")  # noqa: T201
        accumulated_metrics = {
            "psnr_rgb": [],
            "psnr_ab": [],
            "ssim": [],
            "brisque": [],
            "brisque_org": [],
            "brisque_cor": [],
            "brisque_fea": [],
            "brisque_all": [],
            "maniqa": [],
            "niqe": [],
            "niqe_org": [],
            "niqe_cor": [],
            "niqe_fea": [],
            "niqe_all": [],
        }
        for image_file in images:
            image_path = DATA_DIR / "pictures" / image_file
            if compute_all_metrics or not _check_if_metrics_saved(
                image_path, output_dir, suffix
            ):
                _compute_metrics(
                    image_path, output_dir, function, suffix, param_name, parameters
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
            ["maniqa"],
            images,
            "MANIQA",
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
        "img16.jpg",
    )
    sys.exit(main(image_filenames, 40, compute_all_metrics=True))
