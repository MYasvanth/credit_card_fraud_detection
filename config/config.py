"""
Configuration settings for the project.
"""

import os
from pathlib import Path


class Config:
    """
    Configuration class for paths and hyperparameters.
    """

    # Paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
    DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
    MODELS_DIR = ROOT_DIR / "models"
    REPORTS_FIGURES_DIR = ROOT_DIR / "reports" / "figures"

    # Data
    RAW_DATA_FILE = DATA_RAW_DIR / "creditcard.csv"
    PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_creditcard.csv"

    # Model hyperparameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MODELS = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": RANDOM_STATE,
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": RANDOM_STATE,
        },
    }

    # Preprocessing
    SCALING_METHOD = "standard"  # or "minmax"
    HANDLE_IMBALANCE = True

    # Logging
    LOG_LEVEL = "INFO"
