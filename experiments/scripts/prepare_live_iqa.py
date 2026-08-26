"""Prepare the LIVE IQA dataset for BRISQUE training.

This script extracts distorted images from the LIVE Image Quality Assessment Database
(Release 2) and saves their associated DMOS values in .npy format.
The images are categorized based on various distortion types,
and only distorted images (where org==0) are copied to a specified output directory.
This script uses the realigned DMOS values available on the official website.

Database Reference:
    - H.R. Sheikh, M.F. Sabir and A.C. Bovik,
    "A statistical evaluation of recent full reference image quality assessment algorithms",
    IEEE Transactions on Image Processing, vol. 15, no. 11, pp. 3440-3451, Nov. 2006.
    - H.R. Sheikh, Z.Wang, L. Cormack and A.C. Bovik,
    "LIVE Image Quality Assessment Database Release 2",
    http://live.ece.utexas.edu/research/quality.
    - Z. Wang, A.C. Bovik, H.R. Sheikh and E.P. Simoncelli,
    "Image quality assessment: from error visibility to structural similarity",
    IEEE Transactions on Image Processing , vol.13, no.4, pp. 600- 612, April 2004.
"""  # noqa: E501

import json
import os
import shutil
import sys

import scipy.io

# Parameters
BASE_IMG_DIR = os.path.join("data", "live_iqa")
DMOS_MAT_FILE = os.path.join(BASE_IMG_DIR, "dmos_realigned.mat")
REF_MAT_FILE = os.path.join(BASE_IMG_DIR, "refnames_all.mat")
OUTPUT_DIR = os.path.join(BASE_IMG_DIR, "sorted")
DMOS_OUTPUT_JSON = os.path.join(BASE_IMG_DIR, "dmos_sorted.json")


def main() -> None:
    """Main function to prepare LIVE IQA."""
    # Load DMOS data from the .mat files
    mat_data = scipy.io.loadmat(DMOS_MAT_FILE)
    dmos = mat_data["dmos_new"].flatten()
    orgs = mat_data["orgs"].flatten()
    refs = scipy.io.loadmat(REF_MAT_FILE)["refnames_all"].flatten()

    # Define the index ranges for each distortion type
    distortion_types = {
        "jp2k": (0, 227),
        "jpeg": (227, 460),
        "wn": (460, 634),
        "gblur": (634, 808),
        "fastfading": (808, 982),
    }

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # List to store DMOS values for distorted images
    distorted_dmos = {"dmos": [], "images": [], "references": [], "distortions": []}
    distorted_count = 0
    # Iterate through each distortion type
    for distortion, (start_idx, end_idx) in distortion_types.items():
        distortion_dir = os.path.join(BASE_IMG_DIR, distortion)

        for i in range(start_idx, end_idx):
            if orgs[i] == 0:  # org == 0 means distorted image
                image_name = f"img{i - start_idx + 1}.bmp"
                source_path = os.path.join(distortion_dir, image_name)
                new_image_name = f"img{distorted_count:03}.bmp"
                destination_path = os.path.join(OUTPUT_DIR, new_image_name)
                distorted_count += 1

                # Copy the image if it exists
                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_path)
                    distorted_dmos["dmos"].append(dmos[i])
                    distorted_dmos["images"].append(new_image_name)
                    distorted_dmos["references"].append(str(refs[i][0]))
                    distorted_dmos["distortions"].append(distortion)
                else:
                    print(f"Warning: {image_name} not found in {distortion_dir}")

    # Save the DMOS values
    with open(DMOS_OUTPUT_JSON, "w", encoding="utf8") as f:
        json.dump(distorted_dmos, f, indent=4)

    print(f"{distorted_count} distorted images copied to {OUTPUT_DIR}.")
    print(f"DMOS values saved to {DMOS_OUTPUT_JSON}.")


if __name__ == "__main__":
    sys.exit(main())
