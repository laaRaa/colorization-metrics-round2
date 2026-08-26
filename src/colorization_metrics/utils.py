"""Provides utility functions for various tasks.

This module provides utility functions for working with image files,
    specifically retrieving the names of image files in a directory and ensuring correct
    pairing of files based on their numbering.

Constants:
    IMAGE_EXTENSIONS (tuple[str]): List of supported image file extensions.
    COLOR_SPACES (tuple[str]): List of supported color spaces.
"""

import re
import signal
import sys
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Self, TypeVar

import av
import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

COLOR_IMAGE_SHAPE = 3
"""Shape of a NumPy array of a color image."""
COLOR_IMAGE_CHANNELS = 3
"""Channel number of a NumPy array of a color image."""
RANDOM_SEED = 1984
"""Seed used for random."""
METRICS_MODELS_DIR = Path("results") / "metrics_model"
"""Directory where metrics models are saved for later use."""


# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------


class InterruptSignalHandler:
    """A context manager for handling interrupt signals (SIGINT, SIGTERM).

    This class captures and handles interrupt signals during a block of code
    execution. It restores the original signal handlers upon exit and ensures
    that any received signal is processed before program termination.

    Attributes:
        signal_received (bool): Flag indicating whether a signal was received.
        original_handlers (dict): Mapping of signals to their original handlers.
    """

    def __init__(self) -> None:
        """Initialize the signal handler context manager."""
        self.signal_received = False
        self.original_handlers = {}

    def __enter__(self) -> "InterruptSignalHandler":
        """Enter the context and set up signal handlers for SIGINT and SIGTERM.

        Returns:
            InterruptSignalHandler: The context manager instance.
        """
        for sig in [signal.SIGINT, signal.SIGTERM]:
            self.original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the context, restore original handlers and handle received signals.

        Args:
            exc_type: The exception type, if any.
            exc_value: The exception instance, if any.
            traceback: The traceback object, if any.
        """
        for sig, handler in self.original_handlers.items():
            signal.signal(sig, handler)
        if self.signal_received:
            print("Stopping the program since a signal has been received.")  # noqa: T201
            sys.exit(0)

    def _handle_signal(self, signum: int) -> None:
        """Handle the received signal by marking it as captured.

        Args:
            signum: The signal number that was received.
        """
        print(f"\nSignal {signal.Signals(signum).name} captured.")  # noqa: T201
        self.signal_received = True


class ColorSpace(Enum):
    """Enumeration representing color spaces for the metrics.

    Attributes:
        RGB: RGB (Red, Green, Blue) color space, commonly used in
            digital imaging.
        LAB: CIELAB color space, designed to be perceptually uniform
            for more accurate color comparisons.
    """

    RGB = "rgb"
    LAB = "lab"

    @classmethod
    def list(cls) -> list[str]:
        """Return a list of supported color spaces.

        Returns:
            A list of color spaces as strings.
        """
        return [space.value for space in cls]


class FormatEnum(Enum):
    """Base class for file format/extension enums."""

    @classmethod
    def list(cls) -> list[str]:
        """Return a list of all extensions (in order)."""
        return [e.value for e in cls]

    @classmethod
    def set(cls) -> set[str]:
        """Return a set of all extensions (for fast lookup)."""
        return {e.value for e in cls}

    @classmethod
    def is_valid(cls, ext: str) -> bool:
        """Check if the extension is supported."""
        return ext.lower().lstrip(".") in cls.set()

    @classmethod
    def from_ext(cls, ext: str) -> Self:
        """Return the enum member matching a given extension (case-insensitive)."""
        ext = ext.lower()
        for member in cls:
            if member.value == ext:
                return member
        msg = f"Unsupported extension: {ext}"
        raise ValueError(msg)


class ImageExtension(FormatEnum):
    """Enumeration of supported image file extensions."""

    BMP = "bmp"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    WEBP = "webp"


VideoExtension = Enum(
    "VideoFormat", {fmt.upper(): fmt for fmt in av.formats_available}, type=FormatEnum
)
"""Enumeration of supported video file extensions."""


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def moving_average(data: list[float], window_size: int = 5) -> np.ndarray:
    """Calculate the moving average of a list of numerical data.

    The moving average is a common method used to smooth out short-term fluctuations
    and highlight longer-term trends or cycles.

    Args:
        data: A list of numerical data points to calculate the moving average for.
        window_size: The size of the moving window.

    Returns:
        An array containing the moving averages. The length of the array will be
            `len(data) - window_size + 1`.

    Raises:
        ValueError: If the window size is less than 1.

    Example:
        >>> moving_average([1, 2, 3, 4, 5], window_size=3)
        array([2., 3., 4.])
    """
    if window_size < 1:
        msg = f"Window size must be at least 1, got {window_size} instead."
        raise ValueError(msg)

    data = np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def extract_frames(
    video_path: str | Path,
    output_directory: str | Path,
    extension: ImageExtension = ImageExtension.PNG,
) -> None:
    """Extract frames from a video and save them as image files.

    Args:
        video_path: Path to the video file.
        output_directory: Directory where the extracted frames will be saved.
        extension: Image file extension for saved frames.
    """
    video_path = Path(video_path)
    output_directory = Path(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    container = av.open(video_path)
    for i, frame in enumerate(container.decode(video=0)):
        index = frame.index if hasattr(frame, "index") else i
        frame.to_image().save(output_directory / f"{index:05d}.{extension.value}")


def get_dir_imgs(directory: str | Path) -> list[str]:
    """Retrieves the names of all image files in a directory.

    Args:
        directory: Directory from which you want to retrieve list of files.

    Returns:
        List of image files.
    """
    return sorted(
        f.name
        for f in Path(directory).iterdir()
        if f.is_file() and f.suffix[1:].lower() in ImageExtension.list()
    )


def reshape_same_as(img_to_reshape: NDArray, img_correct_shape: NDArray) -> NDArray:
    """Reshapes the input image to match the shape of another image.

    This function uses bilinear interpolation to give one image the shape of another.

    Args:
        img_to_reshape: The image to be reshaped.
        img_correct_shape: The image whose shape `img_to_reshape` will be matched with.

    Returns:
        The reshaped image.

    Raises:
        ValueError: If `img_to_reshape` and `img_correct_shape` have different
            numbers of dimensions.
    """
    height = img_correct_shape.shape[0]
    width = img_correct_shape.shape[1]
    return resize(img_to_reshape, (height, width), order=1)


def assess_correct_pairing(
    list_1: list[str], list_2: list[str], pair_step: int = 0
) -> bool:
    """Assess that two list have the share the same indices.

    Ensure that the files contained in the two lists sorted in ascending
        order are correctly matched by their numbering. A list can contain more
        files as long as they are at the end of the list.

    Args:
        list_1: One list containing numbered file names.
        list_2: Another list containing numbered file names.
        pair_step: Constant index step between two paired files.

    Raises:
        ValueError: Files are not correctly paired, even taking pair_step into account.
    """
    difference = None
    for file_1, file_2 in zip(list_1, list_2, strict=False):
        match_1 = re.findall(r"\d+", file_1)
        match_2 = re.findall(r"\d+", file_2)
        if difference is None and abs(int(match_1[-1]) - int(match_2[-1])) == pair_step:
            # Store the index difference and direction of first tuple
            difference = int(match_1[-1]) - int(match_2[-1])
        elif (
            difference is None
            or match_1 is None
            or match_2 is None
            or int(match_1[-1]) - int(match_2[-1]) != difference
        ):
            msg = f"The {file_1} and {file_2} files are not correctly paired.\
                    Have you checked the pair_step parameter ?"
            raise ValueError(msg)
    return True
