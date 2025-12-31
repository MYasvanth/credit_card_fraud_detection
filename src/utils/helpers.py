"""
Helper functions for the credit card fraud detection project.
"""

import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from .logger import logger
from .constants import MODELS_PATH, REPORTS_PATH

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def save_model(model: Any, model_name: str) -> None:
    """
    Save model to disk.

    Args:
        model: Trained model object.
        model_name: Name of the model file.
    """
    os.makedirs(MODELS_PATH, exist_ok=True)
    model_path = os.path.join(MODELS_PATH, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

def load_model(model_name: str) -> Any:
    """
    Load model from disk.

    Args:
        model_name: Name of the model file.

    Returns:
        Loaded model object.
    """
    model_path = os.path.join(MODELS_PATH, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model

def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    """
    Evaluate model performance.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with evaluation metrics.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Model evaluation completed.")
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

def save_report(report: Dict[str, Any], report_name: str) -> None:
    """
    Save evaluation report to disk.

    Args:
        report: Evaluation report dictionary.
        report_name: Name of the report file.
    """
    os.makedirs(REPORTS_PATH, exist_ok=True)
    report_path = os.path.join(REPORTS_PATH, f"{report_name}.json")
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Report saved to {report_path}")
