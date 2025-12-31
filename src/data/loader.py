import pandas as pd
from pathlib import Path
from typing import Union
from src.utils.logger import logger

def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from CSV file with error handling and logging.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        ValueError: If the file format is invalid.
    """
    path = Path(path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    try:
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Empty data file: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise ValueError(f"Invalid file format: {e}")
