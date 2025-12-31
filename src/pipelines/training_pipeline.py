"""ZenML training pipeline for credit card fraud detection."""

from typing import Dict, Any, Tuple
import pandas as pd
from zenml import pipeline, step
from zenml.logger import get_logger
from sklearn.model_selection import train_test_split

from ..steps.validate_data_step import validate_data_step
from ..steps.feature_engineering_step import feature_engineering_step, prepare_features_step
from ..steps.tune_step import tune_step
from ..steps.register_model_step import register_model_step
from ..data.loader import load_data
from ..utils.constants import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE

logger = get_logger(__name__)


@step
def load_data_step(data_path: str) -> pd.DataFrame:
    """Load data from specified path."""
    data = load_data(data_path)
    logger.info(f"Loaded data with shape: {data.shape}")
    return data


@step
def split_data_step(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train, validation, and test sets."""
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Val target distribution: {y_val.value_counts().to_dict()}")
    logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


@step
def apply_test_transforms_step(
    X_test: pd.DataFrame,
    artifacts: Dict[str, Any]
) -> pd.DataFrame:
    """Apply transformations to test data (without SMOTE)."""
    X_test_processed = X_test.copy()
    
    if "scaler" in artifacts:
        X_test_processed = artifacts["scaler"].transform(X_test_processed)
        logger.info("Applied scaling to test data")
    
    if "pca_transformer" in artifacts:
        X_test_processed = artifacts["pca_transformer"].transform(X_test_processed)
        logger.info("Applied PCA to test data")
    
    logger.info(f"Test data processed shape: {X_test_processed.shape}")
    return X_test_processed


@pipeline
def training_pipeline(
    data_path: str,
    validation_config: Dict[str, Any],
    feature_config: Dict[str, Any],
    model_config: Dict[str, Any],
    tuning_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Complete training pipeline for credit card fraud detection.
    
    Args:
        data_path: Path to training data
        validation_config: Data validation configuration
        feature_config: Feature engineering configuration
        model_config: Model configuration
        tuning_config: Hyperparameter tuning configuration
        
    Returns:
        Pipeline results including model and metrics
    """
    # Load and validate data
    raw_data = load_data_step(data_path)
    validated_data, validation_report = validate_data_step(raw_data, validation_config)
    
    # Prepare features and target
    X, y = prepare_features_step(validated_data)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_step(X, y)
    
    # Feature engineering
    X_train_processed, X_val_processed, y_train_processed, y_val_processed, artifacts = feature_engineering_step(
        X_train, X_val, y_train, y_val, feature_config
    )
    
    # Apply same transformations to test set (without SMOTE)
    X_test_processed = apply_test_transforms_step(X_test, artifacts)
    
    # Hyperparameter tuning
    best_model, tuning_results = tune_step(
        X_train_processed, y_train_processed,
        X_val_processed, y_val_processed,
        model_config, tuning_config
    )
    
    # Model registration
    registration_results = register_model_step(
        best_model, X_test_processed, y_test, tuning_results, 
        model_config, feature_config, validation_report
    )
    
    return {
        "model": best_model,
        "registration_results": registration_results,
        "tuning_results": tuning_results,
        "validation_report": validation_report,
        "artifacts": artifacts
    }


@pipeline
def quick_training_pipeline(
    data_path: str,
    model_type: str = "random_forest"
) -> Dict[str, Any]:
    """
    Quick training pipeline with minimal configuration.
    
    Args:
        data_path: Path to training data
        model_type: Type of model to train
        
    Returns:
        Pipeline results
    """
    # Default configurations
    validation_config = {
        "required_columns": ["Class"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
        "target_column": "Class",
        "numerical_columns": [f"V{i}" for i in range(1, 29)] + ["Amount"]
    }
    
    feature_config = {
        "scaling": {"enabled": True, "method": "standard"},
        "pca": {"enabled": False},
        "smote": {"enabled": True, "method": "smote"}
    }
    
    model_config = {"type": model_type}
    
    tuning_config = {
        "cv_folds": 3,
        "scoring": "f1",
        "n_jobs": -1
    }
    
    return training_pipeline(
        data_path, validation_config, feature_config, model_config, tuning_config
    )