"""Colorize using a locally installed CIC."""

import subprocess
from pathlib import Path


def run_cic_script(
    img_path: str | Path,
    output_folder: str | Path,
    cic_dir: str | Path,
    use_gpu: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """Executes the Colorful Image Colorization (CIC) script.

    This function runs the CIC script (`out_call.py`) within the given
    directory (`cic_dir`), leveraging a virtual environment defined in a `.venv`
    subfolder within `cic_dir`. The function also provides an option to enable
    GPU processing if available.

    Args:
        img_path: Path to the input image file to be colorized.
        output_folder: Path to the directory where processed images will be saved.
        cic_dir: Path to the directory containing the CIC script
            and its associated virtual environment (`.venv`).
        use_gpu: Enables GPU processing if set to True

    Raises:
        FileNotFoundError: If the specified image file, output directory, or
            virtual environment does not exist.
        subprocess.CalledProcessError: If script execution fails, due to issues
            in the script, environment, or file paths.
    """
    img_path = Path(img_path)
    output_folder = Path(output_folder)
    cic_dir = Path(cic_dir)

    # Setup paths for the virtual environment and the CIC script
    cic_dir_expanded = cic_dir.expanduser()
    venv_path = cic_dir_expanded / ".venv"
    python_executable = venv_path / "bin" / "python"
    script_file = cic_dir_expanded / "out_call.py"

    # Build command for subprocess
    command = [str(python_executable), script_file, img_path, output_folder]
    if use_gpu:
        command.append("--use_gpu")

    # Execute command with subprocess, raise error if fails
    _ = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
