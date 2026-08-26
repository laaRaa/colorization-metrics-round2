"""Script to compute metrics on datasets."""

import concurrent
import itertools
import multiprocessing
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from colorization_metrics.utils import RANDOM_SEED, get_dir_imgs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset_utils import (  # pylint: disable=import-error
    ALTERATIONS,
    CPU_METRICS,
    DATA_DIR,
    EXPERIMENTS_DIR,
    GPU_METRICS,
    METRICS,
    PLOT_EXTENSIONS,
    REF_ALTERATION,
    REF_ALTERATION_NAME,
    DatasetAvailables,
    ImageDownloadError,
    Metric,
    ResultsManager,
)
from utils.image_deterioration import (  # pylint: disable=import-error
    initialize_image,
    process_and_save_image,
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


OUTPUT_DIR = EXPERIMENTS_DIR / "exp2_dataset"
ONE_HUNDRED = 100.0
"""One hundred in float."""
ALTERATIONS_WITH_REF = ALTERATIONS
"""Complete alterations dict."""
ALTERATIONS_WITH_REF[REF_ALTERATION_NAME] = REF_ALTERATION

# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


# TODO: Add possibility to remove images from results
def main(  # noqa: PLR0913
    dataset: str = DatasetAvailables.UNSPLASH.value,
    sample_percentage: float = 100.0,
    parallelism: bool = False,  # noqa: FBT001, FBT002
    metrics_used: dict[str, Metric] = CPU_METRICS,
    metrics_to_recompute: tuple[str] = (),
    alterations_to_recompute: tuple[str] = (),
    graph_title: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Main function to process a dataset and plot its metrics.

    This function orchestrates the processing of a specified dataset
        and the plotting of its metrics. It handles sampling, parallelism,
        and selective recomputation of metrics and alterations.

    Args:
        dataset: The name of the dataset to process.
        sample_percentage: The percentage of the dataset to sample.
        parallelism: Whether to enable parallel processing.
        metrics_used: Metrics considered.
        metrics_to_recompute: A tuple of metric names to recompute.
        alterations_to_recompute: A tuple of alteration names to recompute.
        graph_title: Whether to display the graph title in plots.
    """
    if not 0 <= sample_percentage <= 100:  # noqa: PLR2004
        msg = f"Percentage must be between 0 and 100, inclusive. Got {sample_percentage} instead."  # noqa: E501
        raise ValueError(msg)

    _process_dataset(
        dataset,
        sample_percentage,
        parallelism,
        metrics_used,
        metrics_to_recompute,
        alterations_to_recompute,
    )
    print("Start plotting")  # noqa: T201
    _plot_dataset_metrics(dataset, graph_title)
    print("Finish plotting")  # noqa: T201


def get_dataset_information(
    results: ResultsManager,
) -> tuple[str, Path, str, Path, Path]:
    """Retrieve information about the dataset being used.

    This function extracts relevant information about the dataset from the
        ResultsManager instance. It returns specific details such as the dataset name,
        the path to the removed samples file, the image file extension,
        and the output directory for results.

    Args:
        results: An instance of ResultsManager that manages the computation results
                 and provides access to dataset information.

    Returns:
        A tuple containing:
            - The name of the dataset.
            - The directory where images are stored.
            - The file extension used for images in the dataset.
            - The path to the file listing removed samples.
            - The output directory where results are stored.

    Raises:
        NotImplementedError: If the dataset is not available.
    """
    dataset_name = results.dataset_name
    match dataset_name:
        case DatasetAvailables.UNSPLASH.value:
            images_dir = DATA_DIR / "unsplash" / "images"
            image_extension = ".jpg"
            removed_samples = DATA_DIR / "unsplash" / "removed.txt"
            output_dir = OUTPUT_DIR / "unsplash"
            return (
                dataset_name,
                images_dir,
                image_extension,
                removed_samples,
                output_dir,
            )
        case DatasetAvailables.PICTURES.value:
            images_dir = DATA_DIR / "pictures"
            image_extension = ".jpg"
            removed_samples = DATA_DIR / "pictures" / "removed.txt"
            output_dir = OUTPUT_DIR / "pictures"
            return (
                dataset_name,
                images_dir,
                image_extension,
                removed_samples,
                output_dir,
            )
        case _:
            msg = f"This dataset is not available, please use one of the following: {DatasetAvailables.list()}"  # noqa: E501
            raise NotImplementedError(msg)


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _metric_worker(
    item: tuple[str, str, str],
    download_dir: str,
    output_dir: str,
    images_dict: dict[str, str],
) -> tuple[str, str, str, float]:
    """Function executed by each worker to compute an item."""
    metric_name, image_filename, alteration_name = item
    try:
        score = _compute_metric_on_item(download_dir, output_dir, item, images_dict)
    except ImageDownloadError:
        return None
    return metric_name, image_filename, alteration_name, score


def _dataset_download(image_url: str, download_dir: str) -> bool:
    """Download an image if needed and return if it was needed."""
    try:
        _download_image(image_url, download_dir)
    except ImageDownloadError:
        return False
    return True


def _store_metric_result(
    manager: ResultsManager, result: tuple[str, str, str, float]
) -> None:
    metric_name, image_filename, alteration_name, value = result
    manager.set_value(metric_name, image_filename, alteration_name, value)


def _process_dataset(  # noqa: C901, PLR0912, PLR0913
    dataset: str = DatasetAvailables.UNSPLASH.value,
    sample_percentage: float = 100.0,
    parallelism: bool = False,  # noqa: FBT001, FBT002
    metrics_used: dict[str, Metric] = CPU_METRICS,
    metrics_to_recompute: tuple[str, ...] = (),
    alterations_to_recompute: tuple[str, ...] = (),
) -> None:
    """Process the dataset and compute metrics.

    Args:
        dataset: The dataset to process.
        sample_percentage: The percentage of the dataset to sample.
        parallelism: Whether to use parallel processing.
        metrics_used: Metrics considered.
        metrics_to_recompute: Metrics to recompute.
        alterations_to_recompute: Alterations to recompute.
    """
    if not 0 <= sample_percentage <= 100:  # noqa: PLR2004
        msg = f"Percentage must be between 0 and 100, inclusive. Got {sample_percentage} instead."  # noqa: E501
        raise ValueError(msg)

    print("Initialize results dataframe")  # noqa: T201
    results = ResultsManager(dataset)
    _, download_dir, _, removed_samples, output_dir = get_dataset_information(results)

    # Determine what needs to be computed
    items_to_compute, images_dict = _determine_items_to_compute(
        results,
        parallelism,
        sample_percentage,
        metrics_used,
        metrics_to_recompute,
        alterations_to_recompute,
    )
    removed_samples_list = []

    if parallelism:
        for _, image_url in tqdm(
            images_dict.items(), desc="Download if needed", ncols=90
        ):
            if not _dataset_download(image_url, download_dir):
                removed_samples_list.append(image_url)

        _compute_alteration_before_parallelism(
            items_to_compute, images_dict, download_dir, output_dir
        )
        workers_nb = int(multiprocessing.cpu_count() * 0.75)
        print(  # noqa: T201
            f"Start computation with parallelism using multiprocessing, with {workers_nb} workers."  # noqa: E501
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers_nb) as executor:
            future_to_item = {
                executor.submit(
                    _metric_worker, item, download_dir, output_dir, images_dict
                ): item
                for item in tqdm(items_to_compute, desc="Submit tasks", ncols=90)
            }

            try:
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_item),
                    total=len(future_to_item),
                    desc="Processing",
                    ncols=90,
                ):
                    try:
                        result = future.result()
                        if result:
                            _store_metric_result(results, result)
                    finally:
                        del future
                        del result

            except KeyboardInterrupt:
                print(  # noqa: T201
                    "Interruption received, waiting for current calculations to finish..."  # noqa: E501
                )
                executor.shutdown(wait=True, cancel_futures=True)

    else:
        print("Start computation without parallelism")  # noqa: T201
        try:
            for item in tqdm(items_to_compute, desc="Metric computation", ncols=90):
                result = _metric_worker(item, download_dir, output_dir, images_dict)
                if result:
                    _store_metric_result(results, result)
        except KeyboardInterrupt:
            print("Interruption received, waiting for current calculation to finish...")  # noqa: T201

    # Write removed samples to file
    unique_removed_samples = set(removed_samples_list)
    with removed_samples.open("a", encoding="utf-8") as f:
        for url in unique_removed_samples:
            f.write(f"{url}\n")

    # Save results
    print("Saving last results")  # noqa: T201
    results.save_dataset()


def _plot_dataset_metrics(
    dataset: str = DatasetAvailables.UNSPLASH.value,
    graph_title: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Plot and save dataset metrics for altered images.

    This function generates box plots for various metrics of altered images
        in a specified dataset.
    The plots are saved as both PNG and PDF files in a designated output directory.

    Args:
        dataset: The name of the dataset to plot metrics for.
        graph_title: Whether to display the graph title.
    """
    results = ResultsManager(dataset)
    _, _, _, _, output_dir = get_dataset_information(results)

    for metric_name, metric in METRICS.items():
        metric_result = results.get_sub_metric_dataframe(metric_name)
        greater_is_better = metric.higher_is_better
        alteration_dict = ALTERATIONS_WITH_REF if metric.uses_reference else ALTERATIONS

        # Sort the altered images based on the median metric value
        median_values = metric_result.groupby("Alteration")["Value"].median()
        sorted_alterations = median_values.sort_values(
            ascending=not greater_is_better
        ).index.tolist()
        alteration_title = [
            alteration_dict[alteration_name].name
            for alteration_name in sorted_alterations
            if alteration_name in alteration_dict
        ]
        alteration_color = {
            alteration_dict[alteration_name].name: alteration_dict[
                alteration_name
            ].type_color
            for alteration_name in sorted_alterations
            if alteration_name in alteration_dict
        }

        metric_values = [
            metric_result[metric_result["Alteration"] == alteration]["Value"]
            .dropna()
            .tolist()
            for alteration in sorted_alterations
            if alteration in alteration_dict
        ]

        plt.figure(figsize=(10, 6))
        plt.boxplot(metric_values, tick_labels=alteration_title, showfliers=False)

        # Spines removal
        ax = plt.gca()
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # Add medians value
        medians = [np.median(values) for values in metric_values]
        for i, median in enumerate(medians):
            position = i + 1
            y_max = (
                plt.ylim()[1] * (1 - 0.01 * i)
                if greater_is_better
                else plt.ylim()[1] * (1 - 0.01 * (len(medians) - i))
            )

            plt.text(
                position,
                y_max,
                f"{median:.2f}",
                ha="center",
                va="top",
                fontsize=9,
                color="black",
            )

        # Customize x-tick labels and add color coding
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.xlabel("Altered Image Filename")
        for label in plt.gca().get_xticklabels():
            text = label.get_text()
            if text in alteration_color:
                label.set_color(alteration_color[text])

        # Customize y-labels
        plt.ylabel(metric_name)
        if graph_title:
            plt.title(f"{metric_name} for all altered images")
        plt.tight_layout()

        # Save the plot
        for ext in PLOT_EXTENSIONS:
            plot_path = output_dir / f"{metric_name}.{ext}"
            plt.savefig(plot_path)
        plt.close()


def _load_dataset(dataset_name: str = "unsplash") -> list[str]:
    """Load a dataset based on the specified name.

    Args:
        dataset_name: Name of the dataset to load. Defaults to "unsplash".

    Returns:
        A list of image URLs from the dataset.

    Raises:
        NotImplementedError: If the dataset name is not supported.
    """
    match dataset_name:
        case DatasetAvailables.UNSPLASH.value:
            dataset_path = DATA_DIR / "unsplash"
            cache_file = dataset_path / "unsplash_image_urls.csv"
            if cache_file.exists():
                image_urls = pd.read_csv(cache_file, header=None)[0].tolist()
            else:
                documents = [
                    "photos",
                    "keywords",
                    "collections",
                    "conversions",
                    "colors",
                ]
                datasets = {}
                for doc in documents:
                    files = dataset_path.glob(f"{doc}.tsv*")
                    subsets = [
                        pd.read_csv(filename, sep="\t", header=0) for filename in files
                    ]
                    datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)
                image_urls = datasets["photos"]["photo_image_url"].tolist()
                pd.DataFrame(image_urls).to_csv(cache_file, index=False, header=False)
                del datasets
            return image_urls
        case DatasetAvailables.PICTURES.value:
            dataset_path = DATA_DIR / "unsplash"
            return get_dir_imgs(dataset_path)
        case _:
            msg = f"This dataset is not available, please use one\
                of the following: {DatasetAvailables.list()}"
            raise NotImplementedError(msg)


def _sample_dataset(
    resources: list[str],
    removed_samples: str | Path,
    sample_percentage: float,
    random_seed: int = RANDOM_SEED,
) -> tuple[str]:
    """Sample a subset of resources after excluding specified samples.

    Args:
        resources: A list of resource URLs to sample from.
        removed_samples: Path to a file containing samples to exclude.
        sample_percentage: Percentage of resources to retain (0-100).
        random_seed: Seed for randomization. Defaults to 1984.

    Returns:
        A list of sampled resource URLs.
    """
    random.seed(random_seed)
    random.shuffle(resources)
    with Path(removed_samples).open(encoding="utf-8") as file:
        for line in file:
            sample = line.strip()
            if sample in resources:
                resources.remove(sample)
    resources_nb = len(resources)
    sample_size = int(resources_nb * sample_percentage / ONE_HUNDRED)
    return resources[:sample_size] if sample_percentage < ONE_HUNDRED else resources


def _get_image_filename(
    resource_path: str | Path, image_extension: str = ".jpg"
) -> str:
    """Extract the filename from an image URL or local file path.

    This function takes a resource path, which can be either a URL or a local file path,
        and extracts the filename. It then ensures that the filename ends
        with the specified image extension.

    Args:
        resource_path: The URL or local file path of the image.
        image_extension: The desired image file extension.

    Returns:
        The filename with the specified image extension.
    """
    image_filename = Path(resource_path).name

    if image_filename.endswith(image_extension):
        return image_filename

    return image_filename + image_extension


def _download_image(image_url: str | Path, download_dir: str | Path) -> str:
    """Download an image if not already downloaded and return the local path.

    Args:
        image_url: URL of the image to download.
        download_dir: Directory where the image should be saved.

    Returns:
        The local file path of the downloaded (or previously existing) image.

    Raises:
        RequestException: If there is a network issue during the request.
    """
    image_url = Path(image_url)
    download_dir = Path(download_dir)

    download_dir.mkdir(parents=True, exist_ok=True)
    image_filename = image_url.name
    image_path = download_dir / (image_filename + ".jpg")

    if not image_path.exists():
        try:
            response = requests.get(image_url.name + "?fm=jpg&w=400", timeout=100)
            response.raise_for_status()
            with image_path.open("wb") as img_file:
                img_file.write(response.content)
        except RequestException as e:
            msg = f"Failed to download image from {image_url}"
            raise RequestException(msg) from e

    return image_path


def _compute_alteration_before_parallelism(
    items: tuple[tuple[str, str, str], ...],
    images_dict: dict[str, str],
    download_dir: str,
    output_dir: str,
) -> None:
    """Alters images based on combinations of image and alteration names.

    This function processes a list of items, extracts image filenames
    and alteration names, then applies alterations to each image
    in parallel using ProcessPoolExecutor.

    Args:
        items: A tuple of tuples, where each inner tuple contains three strings.
            The second and third strings represent image filenames and alteration names.
        images_dict: A dictionary mapping image filenames to their URLs.
        download_dir: The directory where images are downloaded.
        output_dir: The directory where altered images are saved.
    """
    images = {image for _, image, _ in items}
    alterations = {alteration for _, _, alteration in items}

    all_combinations = list(itertools.product(images, alterations))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_combination = {
            executor.submit(
                _compute_alteration,
                images_dict[image_filename],
                alteration_name,
                download_dir,
                output_dir,
            ): (image_filename, alteration_name)
            for image_filename, alteration_name in tqdm(
                all_combinations, desc="Submit tasks", ncols=90
            )
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_combination),
            total=len(all_combinations),
            desc="Alter images",
            ncols=90,
        ):
            combination = future_to_combination[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{combination} generated an exception: {exc}")  # noqa: T201
                raise


def _compute_alteration(
    image_url: str | Path, alteration_name: str, download_dir: str, output_dir: str
) -> tuple[str, str]:
    # Get the path to the local image by downloading if needed
    if Path(image_url).is_file():
        image_path = image_url
    else:
        try:
            image_path = _download_image(image_url, download_dir)
        except RequestException as e:
            raise ImageDownloadError(image_url) from e

    # Apply alteration to the image
    image, base_filename, output_dir_used = initialize_image(image_path, output_dir)
    alteration = ALTERATIONS_WITH_REF[alteration_name]
    altered_filename = process_and_save_image(
        image,
        output_dir_used,
        base_filename,
        alteration_name,
        alteration.alteration_function,
    )
    altered_path = output_dir_used / altered_filename
    return image_path, altered_path


def _determine_items_to_compute(  # noqa: PLR0913
    results: ResultsManager,
    parallelism: bool,  # noqa: FBT001
    sample_percentage: float = 100.0,
    metrics_used: dict[str, Metric] = CPU_METRICS,
    metrics_to_recompute: tuple[str] = (),
    alterations_to_recompute: tuple[str] = (),
) -> tuple[tuple[tuple[str, str, str], ...], dict[str, str]]:
    """Determine the combinations of metrics, images, and alterations to be computed.

    This function identifies which combinations of metrics, images, and alterations
        are missing from the results and need to be computed. It handles loading
        and sampling datasets, and ensures that specified metrics
        and alterations are recomputed.

    Args:
        results: An instance of ResultsManager that manages the computation results.
        parallelism: Indicate you want to parallelize the metrics calculations.
        sample_percentage: The percentage of the dataset to sample.
        metrics_used: Metrics considered.
        metrics_to_recompute: A tuple of metric names to recompute.
        alterations_to_recompute: A tuple of alteration names to recompute.

    Returns:
        A tuple containing:
            - A tuple of missing combinations, each represented as a tuple of strings
              (metric, image, alteration).
            - A dictionary mapping image paths to their corresponding filenames.

    Raises:
        NotImplementedError: If the dataset is not available.
    """
    dataset_name, _, image_extension, removed_samples, output_dir = (
        get_dataset_information(results)
    )

    print("Start dataset loading")  # noqa: T201
    resources = _load_dataset(dataset_name)
    images_path = _sample_dataset(resources, removed_samples, sample_percentage)
    print("Finish dataset loading")  # noqa: T201

    for metric_name in metrics_to_recompute:
        print(f"-- Remove results computed with {metric_name}.")  # noqa: T201
        results.remove_metric(metric_name)
    for alteration in alterations_to_recompute:
        print(f"-- Remove alteration computed with {alteration}.")  # noqa: T201
        results.remove_alteration(alteration)
        directories = [
            entry for entry in output_dir.iterdir() if (output_dir / entry).is_dir()
        ]
        for directory in tqdm(directories, desc="Alteration removal", ncols=90):
            directory_path = output_dir / directory
            for filename in directory_path.iterdir():
                if filename.name == f"{alteration}.jpg":
                    (directory_path / filename).unlink()

    metrics_list = metrics_used.keys()
    images_list = [_get_image_filename(path, image_extension) for path in images_path]
    images_dict = dict(zip(images_list, images_path, strict=False))
    alterations_list = ALTERATIONS.keys()
    if parallelism:
        metrics_list = [
            metric
            for metric in metrics_list
            if "fid" not in metric  # and "lpips" not in metric
        ]

    print("Get list of metrics computations to do")  # noqa: T201
    items_to_compute = results.find_missing_combinations(
        metrics_list, images_list, alterations_list
    )

    return items_to_compute, images_dict


def _compute_metric_on_item(
    download_dir: str,
    output_dir: str,
    item: tuple[str, str, str],
    images_dict: dict[str, str],
) -> float:
    """Compute a specified metric on an image with a given alteration.

    This function takes an item, which is a combination of a metric, an image,
        and an alteration,  and computes the metric score for the altered image.
        It handles downloading the image if necessary, applying the alteration,
        and return the computed metric score.

    Args:
        download_dir: Directory where images are downloaded,
            if not already present locally.
        removed_samples: Path to a file where failed downloads are logged.
        output_dir: Directory where the altered images are saved.
        item: A tuple containing the metric name, image filename, and alteration name
              for which the metric needs to be computed.
        images_dict: A dictionary mapping image filenames to their corresponding URLs.
    """
    # Decompress item informations
    metric_name, image_filename, alteration_name = item
    metric = METRICS[metric_name]
    image_url = images_dict[image_filename]

    # Compute the metric score
    image_path, altered_path = _compute_alteration(
        image_url, alteration_name, download_dir, output_dir
    )
    if metric.uses_reference:
        score = metric.compute_function(altered_path, image_path)
    else:
        score = metric.compute_function(altered_path)
    return score


# ------------------------------------------------------------
# SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    # For LPIPS, FID and MANIQA
    GPU_PERCENT = 10
    print(f"---- GPU metrics on {GPU_PERCENT}% of the dataset:")  # noqa: T201
    main(sample_percentage=GPU_PERCENT, parallelism=False, metrics_used=GPU_METRICS)
    # For the rest
    for per in range(10, 101, 10):
        print(f"---- Sample percentage: {per}")  # noqa: T201
        main(sample_percentage=per, parallelism=True, metrics_used=CPU_METRICS)
