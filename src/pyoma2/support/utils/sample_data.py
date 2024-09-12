import logging
import os
import typing
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SAMPLE_DATA_DEFAULT_LOCAL_DIR: Path = Path("./.pyoma2_data/test_data/")


def get_sample_data(
    filename: str,
    folder: str,
    local_dir: typing.Union[str, Path] = SAMPLE_DATA_DEFAULT_LOCAL_DIR,
):
    """
    Download a sample data file from the specified GitHub repository if it doesn't exist locally.

    Args:
    filename (str): Name of the file to download.
    folder (str): Folder in the GitHub repository where the file is located.
    local_dir (str): Local directory to save the file.

    Returns:
    str: Path to the local file.

    Raises:
    Exception: If there is an error downloading the file.
    """
    try:
        local_dir = str(local_dir)
        github_raw_url = f"https://raw.githubusercontent.com/dagghe/pyOMA-test-data/main/test_data/{folder}/{filename}"
        local_file_path = Path(local_dir) / folder / filename

        if not local_file_path.exists():
            logger.info("Downloading %s from GitHub...", filename)
            # Create the directory if it doesn't exist
            os.makedirs(local_file_path.parent, exist_ok=True)
            # Download the file
            response = requests.get(url=github_raw_url, timeout=60)
            response.raise_for_status()  # Raise an exception for HTTP errors
            # Save the file locally
            with open(local_file_path, "wb") as f:
                f.write(response.content)
            logger.info("Downloaded %s successfully.", filename)
        else:
            logger.info("%s already exists locally.", filename)
    except Exception as e:
        logger.error("Error downloading %s: %s", filename, e)
        raise e
    return str(local_file_path)
