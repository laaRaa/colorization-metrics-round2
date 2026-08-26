"""HDF5 store inspection utility.

This module provides a utility function to inspect the contents of an HDF5
file (HDFStore) used with pandas. It allows you to check if a specific key
exists in the store, identify the type of the stored object, and print basic
information if it's a DataFrame.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_hdf_store(hdfs_path: str | Path, key: str) -> None:
    """Inspect the content of a given key in an HDF5 store.

    Args:
        hdfs_path: Path to the HDF5 file.
        key: Key to access within the HDF5 store.

    Logs:
        - Whether the key exists.
        - The type of the retrieved object.
        - DataFrame columns and head, if applicable.
    """
    with pd.HDFStore(hdfs_path, mode="r") as store:
        if key in store:
            storer = store[key]
            logger.info("Object type: %s", type(storer))

            if isinstance(storer, pd.DataFrame):
                logger.info("DataFrame columns: %s", storer.columns.tolist())
                logger.info("First rows:\n%s", storer.head())
            else:
                logger.warning("Object is not a pandas DataFrame.")
        else:
            logger.warning("Key not found in store", extra={key: key})
