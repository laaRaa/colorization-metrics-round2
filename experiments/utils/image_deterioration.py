"""Script to generate defects and compare metrics."""

import os
import shutil
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import cic
import numpy as np
from imageio.core.util import image_as_uint
from numpy.matlib import imag
from numpy.typing import NDArray
from skimage.color import hsv2rgb, lab2rgb, rgb2hsv, rgb2lab
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_float, img_as_ubyte, random_noise

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------


IMAGES_DIR = "data/pictures"
DEOLDIFY_DIR = "~/Code/02 Colorisation vidéo/DeOldify"
CIC_MODELS = {}


# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------


class ImageDirections(Enum):
    """Directions in which the chrominance channels of an image can be translated.

    This Enum defines the four cardinal and diagonal directions for moving
    the chrominance channels of an image in the Lab color space without interpolation.

    Attributes:
        NORTH: Represents upward translation.
        EAST: Represents rightward translation.
        SOUTH: Represents downward translation.
        WEST: Represents leftward translation.
        NORTHEAST: Represents upward-rightward diagonal translation.
        NORTHWEST: Represents upward-leftward diagonal translation.
        SOUTHEAST: Represents downward-rightward diagonal translation.
        SOUTHWEST: Represents downward-leftward diagonal translation.
    """

    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"
    NORTHEAST = "NE"
    NORTHWEST = "NW"
    SOUTHEAST = "SE"
    SOUTHWEST = "SW"

    @classmethod
    def list(cls):
        """Return a list of supported directions.

        Returns:
            A list of directions as strings.
        """
        return [ext.value for ext in cls]


# ------------------------------------------------------------
# ALTERATIONS
# ------------------------------------------------------------


def gaussian_noise_chrominances(
    image: NDArray[np.float64], sigma: float = 0.5
) -> NDArray[np.float64]:
    """Adds Gaussian noise to the chrominance channels of the image.

    Args:
        image: Original RGB image.
        sigma: The standard deviation of the Gaussian noise to be added
            to the chrominance channels.

    Returns:
        Image with Gaussian noise added to the chrominance channels.
    """
    lab_image = rgb2lab(image)

    lab_image[:, :, 1] = (
        random_noise(lab_image[:, :, 1] / 127, mode="gaussian", var=sigma**2) * 127
    )
    lab_image[:, :, 2] = (
        random_noise(lab_image[:, :, 2] / 127, mode="gaussian", var=sigma**2) * 127
    )

    result_image = lab2rgb(lab_image)

    return result_image


def desaturate_image(
    image: NDArray[np.float64], factor: float = 0.5
) -> NDArray[np.float64]:
    """Desaturates the image by reducing the saturation in the HSV color space.

    Args:
        image: Original RGB image.
        factor: Factor to reduce saturation by, between 0 (nothing) and 1 (gray).

    Returns:
        Desaturated image.
    """
    hsv_image = rgb2hsv(image)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * (1 - factor)
    return hsv2rgb(hsv_image)


def oversaturate_image(
    image: NDArray[np.float64], factor: float = 1.0
) -> NDArray[np.float64]:
    """Oversaturates the image by augmenting the saturation in the HSV color space.

    Args:
        image: Original RGB image.
        factor: Factor to augment saturation by. A factor of 1 means a saturation
            multiplied by 2, a factor of 2 means a saturation multiplied by 3.

    Returns:
        Oversaturated image.
    """
    hsv_image = rgb2hsv(image)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 + factor), 0.0, 1.0)
    return hsv2rgb(hsv_image)


def enhance_contrast(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply adaptive histogram equalization to increase the contrast of an image.

    This function enhances the contrast of the input image by applying adaptive
    histogram equalization (CLAHE).

    Args:
        image: Input image.

    Returns:
        Contrast-enhanced image.
    """
    return img_as_float(equalize_adapthist(img_as_ubyte(image)))


def desaturate_central_patch(
    image: NDArray[np.float64],
    desaturation_level: float = 1,
    central_region_size: float = 0.5,
) -> NDArray[np.float64]:
    """Desaturates a central patch of the image based on the specified size and level.

    This function extracts a centered patch from the input image, with its size defined
    by `central_region_size` relative to the original image size, and desaturates it
    by reducing the color intensity according to the specified `desaturation_level`.

    Args:
        image: Input RGB image as a NumPy array.
        desaturation_level: Level by which to reduce the saturation.
            A level of 0 means no desaturation, while 1 means complete desaturation.
        central_region_size: The size of the central patch to desaturate, expressed
            as a percentage of the original image's dimensions. Must be between 0 and 1.

    Returns:
        The image with the central patch desaturated.

    Raises:
        ValueError: If `central_region_size` is not between 0 and 1.
    """
    if not (0 < central_region_size <= 1):
        raise ValueError("Percentage must be between 0 and 1.")

    h, w = image.shape[:2]
    new_h = int(h * central_region_size)
    new_w = int(w * central_region_size)

    # Extract the centered sub-image
    start_i = (h - new_h) // 2
    start_j = (w - new_w) // 2
    end_i = start_i + new_h
    end_j = start_j + new_w
    center_image = image[start_i:end_i, start_j:end_j]

    # Desaturates the centered sub-image
    altered_image = image.copy()
    altered_image[start_i:end_i, start_j:end_j] = desaturate_image(
        center_image, desaturation_level
    )
    return altered_image


def remove_chrominance(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transforms the image in its gray counterpart by removing the chrominances.

    Args:
        image: Input RGB image as a NumPy array.

    Returns:
        The RGB image with the modified luminance, while chrominance remains unchanged.
    """
    lab_image = rgb2lab(image)
    lab_image[:, :, 1:] = 0
    luminance_image = lab2rgb(lab_image)
    return luminance_image


def remove_luminance(
    image: NDArray[np.float64], final_luminance: int = 50
) -> NDArray[np.float64]:
    """Removes or normalizes the luminance of an image by setting it to a fixed value.

    This function converts an input RGB image to the Lab color space, sets the L
    (luminance) channel to a specified constant value, and then converts the image
    back to the RGB color space. The chrominance channels (a, b) remain unchanged.

    Args:
        image: Input RGB image as a NumPy array.
        final_luminance: The desired final luminance value for the entire image,
            expressed as a percentage (0-100) of the maximum luminance.

    Returns:
        The RGB image with the modified luminance, while chrominance remains unchanged.

    Raises:
        ValueError: If `final_luminance` is not between 0 and 100.
    """
    if not (0 <= final_luminance <= 100):  # noqa: PLR2004
        raise ValueError("final_luminance must be between 0 and 100.")

    lab_image = rgb2lab(image)
    lab_image[:, :, 0] = final_luminance
    chrominance_image = lab2rgb(lab_image)
    return chrominance_image


def mean_image_chrominances(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Desaturates the image by setting all chrominances to their mean values.

    Args:
        image: Original RGB image.

    Returns:
        Desaturated image where chrominances are replaced by their mean values.
    """
    lab_image = rgb2lab(image)

    mean_a = np.mean(lab_image[:, :, 1])
    mean_b = np.mean(lab_image[:, :, 2])
    lab_image[:, :, 1] = mean_a
    lab_image[:, :, 2] = mean_b

    mean_chrominance_image = lab2rgb(lab_image)
    return mean_chrominance_image


def blur_chrominances(
    image: NDArray[np.float64], sigma: float = 100, std_cut: float = 4.0
) -> NDArray[np.float64]:
    """Blur the chrominance channels (a, b) of an image in CIELab color space.

    Args:
        image: Input image in RGB format as a NumPy array.
        kernel: Kernel size for Gaussian blur.
        sigma: Standard deviation for Gaussian blur.
        std_cut: Standard deviation where to cut the kernel.


    Returns:
        Image with blurred chrominance channels in RGB format.
    """
    lab_image = rgb2lab(image)
    l_channel, a_channel, b_channel = np.split(lab_image, 3, axis=-1)  # pylint: disable = unbalanced-tuple-unpacking

    a_blurred = gaussian(a_channel, sigma, truncate=std_cut)
    b_blurred = gaussian(b_channel, sigma, truncate=std_cut)

    lab_blurred = np.concatenate((l_channel, a_blurred, b_blurred), axis=-1)
    rgb_blurred = lab2rgb(lab_blurred)

    return rgb_blurred


def shift_hue(image: NDArray[np.float64], hue_shift: float) -> NDArray[np.float64]:
    """Change the hue of an image in the HSV color space.

    Args:
        image: Original RGB image.
        hue_shift: Hue shift value in degrees (between 0 and 360).

    Returns:
        Image with modified hue.
    """
    hsv_image = rgb2hsv(image)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift / 360) % 1
    hue_changed_image = hsv2rgb(hsv_image)
    return hue_changed_image


def alter_hue(
    image: NDArray[np.float64], origin: int = 50, factor: float = 5
) -> NDArray[np.float64]:
    """ALter the hue of an image in the HSV color space.

    The hue is shifted with a non linear transformation.

    Args:
        image: Original RGB image.
        origin: Origin in degrees to use before the multiplication by the factor.
        factor: Factor by which the translated hue is multiplied.

    Returns:
        Image with modified hue.
    """
    hsv_image = rgb2hsv(image)
    hue = (hsv_image[:, :, 0] - origin / 360) % 1
    hsv_image[:, :, 0] = ((hue * factor) + origin / 360) % 1
    hue_changed_image = hsv2rgb(hsv_image)
    return hue_changed_image


def apply_sepia_filter(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies a sepia filter to the input image.

    Args:
        image: Original RGB image.

    Returns:
        Image with sepia filter applied.
    """
    line = np.array([0.131, 0.534, 0.272])
    sepia_filter = np.array([1.44 * line, 1.28 * line, line])
    sepia_image = image @ sepia_filter.T
    sepia_image = np.clip(sepia_image, 0, 1)
    return sepia_image


def apply_red_filter(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies a red filter to the input image.

    Args:
        image: Original RGB image.

    Returns:
        Image with red filter applied.
    """
    red = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]) / 3
    red_image = image @ red.T
    red_image = np.clip(red_image, 0, 1)
    return red_image


def apply_green_filter(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies a green filter to the input image.

    Args:
        image: Original RGB image.

    Returns:
        Image with green filter applied.
    """
    green = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) / 3
    green_image = image @ green.T
    green_image = np.clip(green_image, 0, 1)
    return green_image


def apply_blue_filter(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies a blue filter to the input image.

    Args:
        image: Original RGB image.

    Returns:
        Image with blue filter applied.
    """
    blue = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]) / 3
    blue_image = image @ blue.T
    blue_image = np.clip(blue_image, 0, 1)
    return blue_image


def get_gray_from_rgb_channels(
    image: NDArray[np.float64], channel: int = 0
) -> NDArray[np.float64]:
    """Applies a red, green or blue filter to the input image.

    Args:
        image: Original RGB image.
        channel: Which RGB channel to keep.

    Returns:
        Image with color filter applied.
    """
    match channel:
        case 0:
            conv_filter = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        case 1:
            conv_filter = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        case 2:
            conv_filter = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    gray_image = image @ conv_filter.T
    gray_image = np.clip(gray_image, 0, 1)
    return gray_image


def translate(
    image: NDArray[np.float64], displacement: int, direction: str
) -> NDArray[np.float64]:
    """General translation function for moving parts of the image in a circular manner.

    Args:
        image: Image or specific channel to be translated.
        displacement: Number of pixels for the translation.
        direction: Direction in which to translate the image or channel.

    Returns:
        Translated image or channel.
    """
    match direction:
        case ImageDirections.NORTH.value:
            return np.roll(image, -displacement, axis=0)
        case ImageDirections.SOUTH.value:
            return np.roll(image, displacement, axis=0)
        case ImageDirections.EAST.value:
            return np.roll(image, displacement, axis=1)
        case ImageDirections.WEST.value:
            return np.roll(image, -displacement, axis=1)
        case ImageDirections.NORTHEAST.value:
            return translate(
                translate(image, displacement, ImageDirections.NORTH.value),
                displacement,
                ImageDirections.EAST.value,
            )
        case ImageDirections.NORTHWEST.value:
            return translate(
                translate(image, displacement, ImageDirections.NORTH.value),
                displacement,
                ImageDirections.WEST.value,
            )
        case ImageDirections.SOUTHEAST.value:
            return translate(
                translate(image, displacement, ImageDirections.SOUTH.value),
                displacement,
                ImageDirections.EAST.value,
            )
        case ImageDirections.SOUTHWEST.value:
            return translate(
                translate(image, displacement, ImageDirections.SOUTH.value),
                displacement,
                ImageDirections.WEST.value,
            )
        case _:
            raise ValueError(
                f"Invalid direction, use instead {ImageDirections.list()}."
            )


def translate_image(
    image: NDArray[np.float64],
    displacement: int = 20,
    direction: str = ImageDirections.WEST.value,
) -> NDArray[np.float64]:
    """Translates the entire image in the specified direction.

    Args:
        image: Original RGB image.
        displacement: The number of pixels to translate the image.
        direction: The direction in which to translate the image.

    Returns:
        Image with the translated content.
    """
    return translate(image, displacement, direction)


def vertical_flip(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flips the image vertically.

    Args:
        image: Original RGB image.

    Returns:
        Image with vertical symmetry.
    """
    return image[::-1, :]


def horizontal_flip(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flips the image horizontally.

    Args:
        image: Original RGB image.

    Returns:
        Image with horizontal symmetry.
    """
    return image[:, ::-1, :]


def translate_chrominances(
    image: NDArray[np.float64],
    displacement: int = 20,
    direction: str = ImageDirections.WEST.value,
) -> NDArray[np.float64]:
    """Translates the chrominance channels of the image in the specified direction.

    Args:
        image: Original RGB image.
        displacement: The number of pixels by which to translate the chrominances.
        direction: The direction in which to move the chrominance channels.

    Returns:
        Image with translated chrominance channels.
    """
    lab_image = rgb2lab(image)

    lab_image[:, :, 1] = translate(lab_image[:, :, 1], displacement, direction)
    lab_image[:, :, 2] = translate(lab_image[:, :, 2], displacement, direction)

    result_image = lab2rgb(lab_image)

    return result_image


def translate_luminance(
    image: NDArray[np.float64],
    displacement: int = 20,
    direction: str = ImageDirections.WEST.value,
) -> NDArray[np.float64]:
    """Translates the luminance (L) channel of the image in the specified direction.

    Args:
        image: Original RGB image.
        displacement: The number of pixels by which to translate the luminance channel.
        direction: The direction in which to move the luminance channel.

    Returns:
        Image with translated luminance channel.
    """
    lab_image = rgb2lab(image)

    lab_image[:, :, 0] = translate(lab_image[:, :, 0], displacement, direction)

    result_image = lab2rgb(lab_image)

    return result_image


def translate_chrominances_partially(
    image: NDArray[np.float64],
    displacement: int = 20,
    direction: str = ImageDirections.WEST.value,
    percentage_displaced: float = 0.5,
) -> NDArray[np.float64]:
    """Translates a portion of the chrominance channels of an image.

    Translates the chrominance channels (a, b) of the image in the Lab color space
    by the specified displacement in pixels and direction, but only for a central
    portion of the image defined by `percentage_displaced`. The luminance (L) remains
    unchanged, and no interpolation is applied.

    Args:
        image: Input RGB image as a NumPy array.
        displacement: The number of pixels to translate the chrominance channels.
        direction: The direction in which to translate the chrominance channels.
        percentage_displaced: The percentage of the image's central region
            to be affected by the translation.
            Must be between 0 and 1 (e.g., 0.5 for 50% of the central area).

    Returns:
        Image with the chrominance channels partially translated.

    Raises:
        ValueError: If `percentage_displaced` is not between 0 and 1.
    """
    if not (0 < percentage_displaced <= 1):
        raise ValueError("Percentage must be between 0 and 1.")

    h, w = image.shape[:2]
    new_h = int(h * percentage_displaced)
    new_w = int(w * percentage_displaced)

    # Extract the centered sub-image
    start_i = (h - new_h) // 2
    start_j = (w - new_w) // 2
    end_i = start_i + new_h
    end_j = start_j + new_w
    center_image = image[start_i:end_i, start_j:end_j]

    # Translate the chrominances of the centered sub-image
    altered_image = image.copy()
    altered_image[start_i:end_i, start_j:end_j] = translate_chrominances(
        center_image, displacement, direction
    )
    return altered_image


def zoom_chrominance(
    image: NDArray[np.float64], zoom_factor: float = 1.2
) -> NDArray[np.float64]:
    """Applies a zoom effect to the chrominance channels.

    Applies a zoom effect to the chrominance channels (a, b) in the Lab color space,
    stretching the zoomed area to cover the entire image.

    Args:
        image: Original RGB image.
        zoom_factor: Factor by which to zoom the chrominance channels.

    Returns:
        Image with zoom applied to the chrominance channels.
    """
    # Convert the image from BGR to Lab color space
    lab_image = rgb2lab(image)

    # Split the L, a, b channels
    l_channel, a_channel, b_channel = np.split(lab_image, 3, axis=-1)  # pylint: disable = unbalanced-tuple-unpacking

    # Get the center of the image
    center_x, center_y = a_channel.shape[1] // 2, a_channel.shape[0] // 2

    # Define the size of the zoomed region
    zoomed_width = int(a_channel.shape[1] / zoom_factor)
    zoomed_height = int(a_channel.shape[0] / zoom_factor)

    # Calculate the cropping box for zoom (centered)
    start_x = center_x - zoomed_width // 2
    start_y = center_y - zoomed_height // 2
    end_x = start_x + zoomed_width
    end_y = start_y + zoomed_height

    # Crop the zoomed region from a and b channels
    a_zoomed_region = a_channel[start_y:end_y, start_x:end_x]
    b_zoomed_region = b_channel[start_y:end_y, start_x:end_x]

    # Resize the zoomed region back to the original image size
    a_zoomed = resize(
        a_zoomed_region, (a_channel.shape[0], a_channel.shape[1]), order=1
    )
    b_zoomed = resize(
        b_zoomed_region, (b_channel.shape[0], b_channel.shape[1]), order=1
    )

    # Merge the L channel with the zoomed a and b channels
    lab_zoomed = np.concatenate([l_channel, a_zoomed, b_zoomed], axis=-1)

    # Convert the Lab image back to BGR
    result_image = lab2rgb(lab_zoomed)

    return result_image


def colorize_cic_eccv(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Colorize an image with CIC in its ECCV version.

    Uses the "Colorful Image Colorization" algorithm of [Zhang et al.](https://doi.org/10.1007/978-3-319-46487-9_40).

    Args:
        image: Original RGB image.

    Returns:
        Image colorized with CIC.
    """
    if "eccv" not in CIC_MODELS:
        CIC_MODELS["eccv"] = cic.eccv16().eval()

    (tens_l_orig, tens_l_rs) = cic.preprocess_img(image_as_uint(image))
    return img_as_float(
        cic.postprocess_tens(tens_l_orig, CIC_MODELS["eccv"](tens_l_rs))
    )


def colorize_cic_siggraph(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Colorize an image with CIC in its SIGGRAPH version.

    Uses the "Colorful Image Colorization" algorithm of [Zhang et al.](https://doi.org/10.1145/3072959.3073703).

    Args:
        image: Original RGB image.

    Returns:
        Image colorized with CIC.
    """
    if "siggraph" not in CIC_MODELS:
        CIC_MODELS["siggraph"] = cic.siggraph17().eval()

    (tens_l_orig, tens_l_rs) = cic.preprocess_img(image_as_uint(image))
    return img_as_float(
        cic.postprocess_tens(tens_l_orig, CIC_MODELS["siggraph"](tens_l_rs))
    )


# ------------------------------------------------------------
# PUBLIC FUNCTIONS
# ------------------------------------------------------------


def initialize_image(
    image_path: str | Path, output_dir: str | Path
) -> tuple[NDArray[np.float64], str, Path]:
    """Load an image from a specified path and generate derived paths for output.

    Args:
        image_path: Path to the input image.
        output_dir: Directory where processed results will be saved.

    Returns:
        A tuple containing:
            - The image as a float64 NumPy array.
            - The base filename extracted from `image_path`.
            - The directory path for saving the processed image.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    image = img_as_float(_safe_imread(image_path))
    base_filename = image_path.name
    output_dir_used = output_dir / image_path.stem

    return image, base_filename, output_dir_used


def process_and_save_image(
    image: NDArray[np.float64],
    output_dir: str,
    base_filename: str,
    suffix: str,
    processing_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> str:
    """Process an image using a given function and save the result.

    Args:
        image: The original image to be processed.
        output_dir: The directory where the processed image will be saved.
        base_filename: The base filename of the input image.
        suffix: Suffix to append to the filename for the processed image.
        processing_func: The function used to process the image.

    Returns:
        The filename of the saved processed image.
    """
    processed_filename = _derive_filename(base_filename, suffix, "png")
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(os.path.join(output_dir, processed_filename)):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".tmp", dir=output_dir
        ) as temp_file:
            temp_filename = temp_file.name
            processed_image = processing_func(image)
            _save_image(output_dir, temp_filename, processed_image)
        shutil.move(
            os.path.join(output_dir, temp_filename),
            os.path.join(output_dir, processed_filename),
        )

    return processed_filename


# ------------------------------------------------------------
# PRIVATE FUNCTIONS
# ------------------------------------------------------------


def _derive_filename(base_filename: str, suffix: str, extension: str = None) -> str:
    """Derives a new filename by adding a suffix before the file extension.

    Args:
        base_filename: The original filename (e.g., "image.jpg").
        suffix: The suffix to add (e.g., "deoldify").
        extension: The extension of the new filename.

    Returns:
        The new filename with the suffix (e.g., "image_deoldify.jpg").
    """
    name, ext = os.path.splitext(base_filename)
    if extension:
        return f"{name}_{suffix}.{extension}"
    return f"{name}_{suffix}{ext}"


def _safe_imread(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Le fichier {image_path} n'existe pas.")
    try:
        image = imread(image_path)
        return image
    except OSError as e:
        raise OSError(f"Erreur lors de la lecture de l'image {image_path}: {e}") from e


def _save_image(output_dir: str, filename: str, image: NDArray[np.float64]):
    """Saves the image to the specified directory with the given filename.

    Args:
        output_dir: Directory where to save the image.
        filename: Name of the image file.
        image: Image to save.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    image = (image * 255).astype(np.uint8)
    imsave(output_path, image)
