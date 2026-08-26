"""This module provides tools and classes for managing image alterations and metrics.

It includes functionalities for applying various image alterations,
computing image quality metrics,
and managing datasets with these alterations and metrics.

Classes:
- DatasetAvailables: Enumeration for available datasets.
- AlterationColor: Enumeration for different types of color alterations.
- Metric: Data class representing an image quality metric.
- Alteration: Data class representing an image alteration.
- ResultsManager: Manages results for a dataset, including loading, saving,
    and updating values.

Constants:
- METRICS: Dictionary of available metrics.
- ALTERATIONS: Dictionary of available alterations.
"""

import contextlib
import itertools
import shutil
import signal
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from utils import image_deterioration  # pylint: disable=import-error

from colorization_metrics.metrics.brisque import FeaturesUsed, compute_brisque
from colorization_metrics.metrics.colorfulness import (
    ColorfulnessMetric,
    compute_colorfulness,
)
from colorization_metrics.metrics.fid import compute_fid
from colorization_metrics.metrics.lpips import LPIPSNetworks, compute_lpips
from colorization_metrics.metrics.maniqa import compute_maniqa
from colorization_metrics.metrics.niqe import compute_niqe
from colorization_metrics.metrics.psnr import compute_psnr
from colorization_metrics.metrics.ssim import compute_ssim
from colorization_metrics.utils import ColorSpace

# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------


class ImageDownloadError(Exception):
    """Custom exception raised when an error occurs while downloading an image.

    Attributes:
        image_url: The URL of the image that failed to download.
        message: Explanation of the error.
    """

    def __init__(  # noqa: D107
        self, image_url: str, message: str = "Error downloading the image"
    ) -> None:
        self.image_url = image_url
        self.message = message
        super().__init__(self.message)


class DatasetAvailables(Enum):
    """Enumeration for available datasets."""

    UNSPLASH = "unsplash"
    PICTURES = "pictures"

    @classmethod
    def list(cls) -> list[str]:
        """Return a list of supported datasets.

        Returns:
            A list of available datasets as strings.
        """
        return [space.value for space in cls]


class AlterationColor(Enum):
    """Enum representing different types of color alterations.

    Attributes:
        DESATURATION: Represents a desaturation alteration.
        DECORRELATION: Represents a decorrelation alteration.
        SHIFT: Represents a hue alteration.
        MONOTONE: Represents a monocolor alteration.
        NO_DEFECT: Represents a geometric transformation, indicating no defect.
    """

    DESATURATION = "gray"
    DECORRELATION = "red"
    SHIFT = "magenta"
    MONOTONE = "blue"
    NO_DEFECT = "green"
    COLORIZATION = "black"


@dataclass
class Metric:
    """A class to represent an image quality metric.

    Attributes:
        compute_function (Callable): Function used to compute the metric.
            It should accept the necessary arguments to perform the computation,
            such as an image and optionally a reference image.
        name (str): Human-readable name of the metric,
            used for display or documentation purposes.
        uses_reference (bool): Indicates whether the metric requires a reference image
            for computation.
        higher_is_better (bool): Indicates whether a higher score
            means better image quality.
    """

    compute_function: Callable
    name: str
    uses_reference: bool
    higher_is_better: bool


@dataclass
class Alteration:
    """A class to represent an image alteration."""

    alteration_function: Callable
    name: str
    type_color: str


class ResultsManager:
    """Manages results for a dataset, including loading, saving, and updating values."""

    def __init__(self, dataset_name: str = "default", batch_size: int = 1000) -> None:
        """Initializes the ResultsManager with a specified dataset name.

        Args:
            dataset_name: The name of the dataset to manage.
            batch_size: The number of rows to accumulate before writing to disk.
        """
        self._dataset_name = dataset_name
        self._saving_path = Path("results") / "dataframes" / f"{dataset_name}.h5"
        self._batch_size = batch_size
        self._buffer = []

        self._saving_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_indexed()

        with pd.HDFStore(self._saving_path, mode="a") as store:
            if "df" not in store:
                print(f"Creating new HDF5 dataset for {dataset_name}.")  # noqa: T201
                empty_df = pd.DataFrame(
                    columns=["Metric", "Alteration", "Image", "Value"]
                )
                empty_df["Value"] = empty_df["Value"].astype(float)
                store.put(
                    "df",
                    empty_df,
                    format="table",
                    min_itemsize={"Metric": 256, "Alteration": 256, "Image": 256},
                )

    @property
    def dataset_name(self) -> str:
        """The name of the managed dataset."""
        return self._dataset_name

    def save_dataset(self) -> None:
        """Saves the dataset to the file system if it has been updated."""
        if self._buffer:
            self._flush_buffer()

    # TODO: Faire en sorte que les items soient triés sur les images
    #       dans l'ordre de image_list
    # Il faut éviter l'usage de Categorical car c'est complexe de rajouter
    # de nouvelles valeurs, faut creuser.
    def find_missing_combinations(
        self, metric_list: list[str], image_list: list[str], alteration_list: list[str]
    ) -> tuple[tuple[str, str, str], ...]:
        """Identify missing combinations of metrics, images, and alterations.

        Args:
            metric_list: List of metric names.
            image_list: List of image identifiers.
            alteration_list: List of alteration names.

        Returns:
            Missing combinations as (metric, image, alteration) tuples.
        """
        all_combinations = list(
            itertools.product(metric_list, image_list, alteration_list)
        )

        referenceless_metric_list = [
            m for m in metric_list if not METRICS[m].uses_reference
        ]

        all_combinations += list(
            itertools.product(
                referenceless_metric_list, image_list, [REF_ALTERATION_NAME]
            )
        )

        all_combinations_set = set(all_combinations)

        with pd.HDFStore(self._saving_path, mode="r") as store:
            if "df" in store:
                existing_df = store.select(
                    "/df", columns=["Metric", "Image", "Alteration"]
                )
                existing_set = set(map(tuple, existing_df.to_numpy()))
            else:
                existing_set = set()

        missing_set = all_combinations_set - existing_set

        return tuple(sorted(missing_set, key=lambda x: (x[1], x[2], x[0])))

    def get_value(self, metric: str, image: str, alteration: str) -> float | None:
        """Retrieves the value for a given metric, image, and alteration."""
        query = (
            f"Metric == '{metric}' & Image == '{image}' & Alteration == '{alteration}'"
        )
        with pd.HDFStore(self._saving_path, mode="r") as store:
            result = store.select("/df", where=query)
        return result["Value"].to_numpy()[0] if not result.empty else None

    def get_sub_metric_dataframe(self, metric_name: str) -> pd.DataFrame:
        """Retrieve a sub-dataframe filtered by the specified metric name."""
        query = f"Metric == '{metric_name}'"
        with pd.HDFStore(self._saving_path, mode="r") as store:
            return store.select("/df", where=query)

    def set_value(self, metric: str, image: str, alteration: str, value: float) -> None:
        """Sets the value for a given metric, image, and alteration."""
        self._buffer.append(
            {"Metric": metric, "Image": image, "Alteration": alteration, "Value": value}
        )
        if len(self._buffer) >= self._batch_size:
            self._flush_buffer()

    def remove_value(self, metric: str, image: str, alteration: str) -> None:
        """Removes the value for a given metric, image, and alteration."""
        query = (
            f"Metric == '{metric}' & Image == '{image}' & Alteration == '{alteration}'"
        )
        with pd.HDFStore(self._saving_path, mode="a") as store:
            store.remove("df", where=query)

    def _remove_entries(self, column: str, value: str) -> None:
        """Rewrites the dataset without the specified column value."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        with pd.HDFStore(self._saving_path, mode="r") as store:
            total_rows = len(store.select("/df"))
            chunk_size = 10000
            total_chunks = (total_rows // chunk_size) + (
                1 if total_rows % chunk_size != 0 else 0
            )
            print(f"Total rows: {total_rows}, Total chunks to process: {total_chunks}")  # noqa: T201

            with (
                pd.HDFStore(temp_path, mode="w") as temp_store,
                tqdm(
                    total=total_chunks, desc="Rewriting dataset", unit="chunk", ncols=90
                ) as pbar,
            ):
                for chunk in store.select("/df", chunksize=chunk_size):
                    filtered_chunk = chunk[chunk[column] != value]
                    temp_store.append(
                        "df",
                        filtered_chunk,
                        format="table",
                        data_columns=["Metric", "Alteration"],
                        min_itemsize={"Metric": 256, "Alteration": 256, "Image": 256},
                    )
                    pbar.update(1)

        try:
            shutil.copy2(temp_path, self._saving_path)
            Path(temp_path).unlink()
        except OSError as e:
            print(f"Error replacing file: {e}")
            raise e

    def remove_metric(self, metric: str) -> None:
        """Rewrites the dataset without the specified metric in a temporary file."""
        self._remove_entries("Metric", metric)

    def remove_alteration(self, alteration: str) -> None:
        """Rewrites the dataset without the specified alteration in a temporary file."""
        self._remove_entries("Alteration", alteration)

    def _ensure_indexed(self) -> None:
        """Ensures the DataFrame is indexed on 'Metric' and 'Alteration'."""
        with pd.HDFStore(self._saving_path, mode="a") as store:
            if "df" not in store:
                print("No 'df' found in the HDF5 file.")  # noqa: T201
                return
            storer = store.get_storer("df")
            if not {"Metric", "Alteration"}.issubset(set(storer.data_columns or [])):
                print("Indexing the HDF5 file.")  # noqa: T201
                df = store["df"]
                df.to_hdf(
                    self._saving_path,
                    key="df",
                    format="table",
                    data_columns=["Metric", "Alteration"],
                    mode="w",
                )

    def _flush_buffer(self) -> None:
        """Flushes the buffer to the HDF5 store."""
        if not self._buffer:
            return
        with self._ignore_signals():
            buffer_df = pd.DataFrame(self._buffer)

            with pd.HDFStore(self._saving_path, mode="a") as store:
                if "df" not in store:
                    print("No 'df' found in HDF5 file, creating new dataset.")  # noqa: T201
                    store.put(
                        "df",
                        buffer_df,
                        format="table",
                        min_itemsize={"Metric": 256, "Alteration": 256, "Image": 256},
                        data_columns=["Metric", "Alteration"],
                    )
                else:
                    store.append(
                        "df",
                        buffer_df,
                        format="table",
                        data_columns=["Metric", "Alteration"],
                    )

            self._buffer = []

    @contextlib.contextmanager
    def _ignore_signals(self) -> None:
        """Context manager to ignore signals during critical operations."""
        original_handlers = {}

        def handler(signum, _) -> None:
            print(f"Signal {signum} received, delaying until operation complete.")  # noqa: T201

        for sig in (signal.SIGINT, signal.SIGTERM):
            original_handlers[sig] = signal.signal(sig, handler)
        try:
            yield
        finally:
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

DATA_DIR = Path("data")
"""Directory for data."""
RESULTS_DIR = Path("results")
"""Directory for results."""
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
"""Directory for experiments."""
PLOT_EXTENSIONS = ("png", "svg", "pdf")
"""File extensions used for plots."""
CPU_METRICS = {
    "psnr_rgb": Metric(
        compute_function=partial(compute_psnr, color_space=ColorSpace.RGB.value),
        name="PSNR in RGB color space",
        uses_reference=True,
        higher_is_better=True,
    ),
    "psnr_lab": Metric(
        compute_function=partial(compute_psnr, color_space=ColorSpace.LAB.value),
        name="PSNR on a* and b* chrominances",
        uses_reference=True,
        higher_is_better=True,
    ),
    "ssim": Metric(
        compute_function=compute_ssim,
        name="SSIM",
        uses_reference=True,
        higher_is_better=True,
    ),
    "brisque": Metric(
        compute_function=compute_brisque,
        name="BRISQUE",
        uses_reference=False,
        higher_is_better=False,
    ),
    "brisque_org": Metric(
        compute_function=partial(compute_brisque, method=FeaturesUsed.ORIGINAL.value),
        name="BRISQUE - Original Retrained",
        uses_reference=False,
        higher_is_better=False,
    ),
    "brisque_cor": Metric(
        compute_function=partial(
            compute_brisque, method=FeaturesUsed.RGB_CORRELATION.value
        ),
        name="BRISQUE - RGB Correlation",
        uses_reference=False,
        higher_is_better=False,
    ),
    "brisque_ana": Metric(
        compute_function=partial(
            compute_brisque, method=FeaturesUsed.RGB_ANALYSIS.value
        ),
        name="BRISQUE - RGB All",
        uses_reference=False,
        higher_is_better=False,
    ),
    "brisque_all": Metric(
        compute_function=partial(compute_brisque, method=FeaturesUsed.RGB_ALL.value),
        name="BRISQUE - RGB Analysis",
        uses_reference=False,
        higher_is_better=False,
    ),
    "brisque_lab": Metric(
        compute_function=partial(compute_brisque, method=FeaturesUsed.CIELAB.value),
        name="BRISQUE - CIELAB Correlation",
        uses_reference=False,
        higher_is_better=False,
    ),
    "niqe": Metric(
        compute_function=compute_niqe,
        name="NIQE",
        uses_reference=False,
        higher_is_better=False,
    ),
    "niqe_cor": Metric(
        compute_function=partial(
            compute_niqe, method=FeaturesUsed.RGB_CORRELATION.value
        ),
        name="NIQE - RGB Correlation",
        uses_reference=False,
        higher_is_better=False,
    ),
    "niqe_ana": Metric(
        compute_function=partial(compute_niqe, method=FeaturesUsed.RGB_ANALYSIS.value),
        name="NIQE - RGB Analysis",
        uses_reference=False,
        higher_is_better=False,
    ),
    "niqe_all": Metric(
        compute_function=partial(compute_niqe, method=FeaturesUsed.RGB_ALL.value),
        name="NIQE - RGB All",
        uses_reference=False,
        higher_is_better=False,
    ),
    "colorfulness_1": Metric(
        compute_function=partial(
            compute_colorfulness, metric=ColorfulnessMetric.AB_CHANNELS.value
        ),
        name="Colorfulness version 1",
        uses_reference=False,
        higher_is_better=True,
    ),
    "colorfulness_2": Metric(
        compute_function=partial(
            compute_colorfulness, metric=ColorfulnessMetric.CHROMA_MEAN.value
        ),
        name="Colorfulness version 2",
        uses_reference=False,
        higher_is_better=True,
    ),
    "colorfulness_3": Metric(
        compute_function=partial(
            compute_colorfulness, metric=ColorfulnessMetric.RG_YB_CHANNELS.value
        ),
        name="Colorfulness version 3",
        uses_reference=False,
        higher_is_better=True,
    ),
}
GPU_METRICS = {
    "maniqa": Metric(
        compute_function=compute_maniqa,
        name="MANIQA",
        uses_reference=False,
        higher_is_better=True,
    ),
    "lpips_alex": Metric(
        compute_function=partial(compute_lpips, net_type=LPIPSNetworks.ALEX_NET.value),
        name="LPIPS with a AlexNet network",
        uses_reference=True,
        higher_is_better=False,
    ),
    "lpips_vgg": Metric(
        compute_function=partial(compute_lpips, net_type=LPIPSNetworks.VGG.value),
        name="LPIPS with a VGG network",
        uses_reference=True,
        higher_is_better=False,
    ),
    "fid_64": Metric(
        compute_function=partial(compute_fid, fid_dims=64),
        name="FID with 64 features",
        uses_reference=True,
        higher_is_better=False,
    ),
    "fid_192": Metric(
        compute_function=partial(compute_fid, fid_dims=192),
        name="FID with 192 features",
        uses_reference=True,
        higher_is_better=False,
    ),
    "fid_768": Metric(
        compute_function=partial(compute_fid, fid_dims=768),
        name="FID with 768 features",
        uses_reference=True,
        higher_is_better=False,
    ),
    "fid_2048": Metric(
        compute_function=partial(compute_fid, fid_dims=2048),
        name="FID with 2048 features",
        uses_reference=True,
        higher_is_better=False,
    ),
}
METRICS = CPU_METRICS | GPU_METRICS

ALTERATIONS = {
    "oversaturated": Alteration(
        alteration_function=partial(image_deterioration.oversaturate_image, factor=2),
        name="Oversaturated",
        type_color=AlterationColor.DESATURATION.value,
    ),
    "desaturated": Alteration(
        alteration_function=partial(image_deterioration.desaturate_image, factor=0.8),
        name="Desaturated",
        type_color=AlterationColor.DESATURATION.value,
    ),
    "centered_desaturated": Alteration(
        alteration_function=partial(
            image_deterioration.desaturate_central_patch, central_region_size=0.5
        ),
        name="Desaturated in the center",
        type_color=AlterationColor.DESATURATION.value,
    ),
    "gray_from_red": Alteration(
        alteration_function=image_deterioration.get_gray_from_rgb_channels,
        name="Gray from red channel",
        type_color=AlterationColor.DESATURATION.value,
    ),
    "luminance_only": Alteration(
        alteration_function=image_deterioration.remove_chrominance,
        name="Only luminance",
        type_color=AlterationColor.DESATURATION.value,
    ),
    "halo_zoom": Alteration(
        alteration_function=image_deterioration.zoom_chrominance,
        name="Halo by zoom",
        type_color=AlterationColor.DECORRELATION.value,
    ),
    "blured_edges": Alteration(
        alteration_function=partial(
            image_deterioration.blur_chrominances, sigma=200, std_cut=4e-2
        ),
        name="Chrominances edges blurred",
        type_color=AlterationColor.DECORRELATION.value,
    ),
    "chrominance_translated": Alteration(
        alteration_function=image_deterioration.translate_chrominances,
        name="Chrominances translated",
        type_color=AlterationColor.DECORRELATION.value,
    ),
    "luminance_translated": Alteration(
        alteration_function=image_deterioration.translate_luminance,
        name="Luminance translated",
        type_color=AlterationColor.DECORRELATION.value,
    ),
    "partially_translated_1": Alteration(
        alteration_function=partial(
            image_deterioration.translate_chrominances_partially,
            displacement=20,
            direction=image_deterioration.ImageDirections.WEST.value,
        ),
        name="Partial translation 1",
        type_color=AlterationColor.DECORRELATION.value,
    ),
    "partially_translated_2": Alteration(
        alteration_function=partial(
            image_deterioration.translate_chrominances_partially,
            displacement=40,
            direction=image_deterioration.ImageDirections.NORTHEAST.value,
        ),
        name="Partial translation 2",
        type_color=AlterationColor.DECORRELATION.value,
    ),
    "hue_shifted": Alteration(
        alteration_function=partial(image_deterioration.shift_hue, hue_shift=120),
        name="Hue shifted",
        type_color=AlterationColor.SHIFT.value,
    ),
    "altered_hue": Alteration(
        alteration_function=image_deterioration.alter_hue,
        name="Hue altered",
        type_color=AlterationColor.SHIFT.value,
    ),
    "mean_chrominance": Alteration(
        alteration_function=image_deterioration.mean_image_chrominances,
        name="Mean chrominance",
        type_color=AlterationColor.MONOTONE.value,
    ),
    "sepia": Alteration(
        alteration_function=image_deterioration.apply_sepia_filter,
        name="Sepia filter",
        type_color=AlterationColor.MONOTONE.value,
    ),
    "red": Alteration(
        alteration_function=image_deterioration.apply_red_filter,
        name="Red filter",
        type_color=AlterationColor.MONOTONE.value,
    ),
    "green": Alteration(
        alteration_function=image_deterioration.apply_green_filter,
        name="Green filter",
        type_color=AlterationColor.MONOTONE.value,
    ),
    "blue": Alteration(
        alteration_function=image_deterioration.apply_blue_filter,
        name="Blue filter",
        type_color=AlterationColor.MONOTONE.value,
    ),
    "translated_all": Alteration(
        alteration_function=partial(
            image_deterioration.translate_image,
            displacement=40,
            direction=image_deterioration.ImageDirections.NORTHWEST.value,
        ),
        name="Image translated",
        type_color=AlterationColor.NO_DEFECT.value,
    ),
    "horizontal_flip": Alteration(
        alteration_function=image_deterioration.horizontal_flip,
        name="Horizontal flip",
        type_color=AlterationColor.NO_DEFECT.value,
    ),
    "vertical_flip": Alteration(
        alteration_function=image_deterioration.vertical_flip,
        name="Vertical flip",
        type_color=AlterationColor.NO_DEFECT.value,
    ),
    "cic_eccv": Alteration(
        alteration_function=image_deterioration.colorize_cic_eccv,
        name="CIC - ECCV",
        type_color=AlterationColor.COLORIZATION.value,
    ),
    "cic_siggraph": Alteration(
        alteration_function=image_deterioration.colorize_cic_siggraph,
        name="CIC - SIGGRAPH",
        type_color=AlterationColor.COLORIZATION.value,
    ),
}
REF_ALTERATION = Alteration(
    alteration_function=lambda x: x,
    name="Reference",
    type_color=AlterationColor.NO_DEFECT.value,
)
"""Reference image for referenceless metric."""
REF_ALTERATION_NAME = "reference"
"""Reference image name for referenceless metric."""
