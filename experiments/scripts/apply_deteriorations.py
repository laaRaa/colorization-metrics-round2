"""Apply all alterations to an image."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset_utils import (  # pylint: disable=import-error
    ALTERATIONS,
    DATA_DIR,
    RESULTS_DIR,
)
from utils.image_deterioration import (  # pylint: disable=import-error
    initialize_image,
    process_and_save_image,
)

name = sys.argv[1]
image, filename, out_dir = initialize_image(
    DATA_DIR / "pictures" / "{name}.jpg", RESULTS_DIR / "pictures_bis"
)

for alt_name, alt in ALTERATIONS.items():
    altered_filename = process_and_save_image(
        image, out_dir, filename, alt_name, alt.alteration_function
    )
    altered_path = out_dir / altered_filename
