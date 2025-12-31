"""Constants for the credit card fraud detection project."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
EXTERNAL_DATA_PATH = DATA_PATH / "external"

# Model paths
MODELS_PATH = PROJECT_ROOT / "models"

# Reports paths
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"

# Config paths
CONFIG_PATH = PROJECT_ROOT / "configs"

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = "credit_card_fraud_detection"
MLFLOW_MODEL_REGISTRY_NAME = "credit_card_fraud_detector"

# MLflow experiment configurations
MLFLOW_EXPERIMENTS = {
    "training": "credit_card_fraud_detection",
    "ab_testing": "credit_card_fraud_ab_testing", 
    "model_comparison": "credit_card_fraud_model_comparison",
    "monitoring": "credit_card_fraud_monitoring",
    "retraining": "credit_card_fraud_retraining"
}

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering
NUMERICAL_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000