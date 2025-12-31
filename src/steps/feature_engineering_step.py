"""ZenML step for feature engineering pipeline."""

from typing import Tuple, Dict, Any
import pandas as pd
from zenml import step
from zenml.logger import get_logger

from ..features.scaler import FeatureScaler
from ..features.pca_transformer import PCATransformer
from ..features.smote import SMOTETransformer
from ..utils.constants import TARGET_COLUMN, NUMERICAL_FEATURES

logger = get_logger(__name__)


@step
def feature_engineering_step(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Apply feature engineering transformations to training and validation data.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        config: Feature engineering configuration

    Returns:
        Tuple of (X_train_processed, X_val_processed, y_train_processed, y_val_processed, artifacts)
    """
    artifacts = {}
    
    # Feature scaling
    scaling_config = config.get("scaling", {})
    if scaling_config.get("enabled", True):
        scaler = FeatureScaler(
            method=scaling_config.get("method", "standard"),
            exclude_columns=scaling_config.get("exclude_columns", [])
        )
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Store scaler
        artifacts["scaler"] = scaler
        
        logger.info(f"Applied {scaling_config.get('method', 'standard')} scaling")
    else:
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
    
    # PCA transformation
    pca_config = config.get("pca", {})
    if pca_config.get("enabled", False):
        pca_transformer = PCATransformer(
            n_components=pca_config.get("n_components"),
            variance_threshold=pca_config.get("variance_threshold", 0.95)
        )
        
        X_train_pca = pca_transformer.fit_transform(X_train_scaled)
        X_val_pca = pca_transformer.transform(X_val_scaled)
        
        # Store PCA transformer
        artifacts["pca_transformer"] = pca_transformer
        
        # Skip plotting for now
        
        X_train_final = X_train_pca
        X_val_final = X_val_pca
        
        logger.info(f"Applied PCA with {pca_transformer.n_components} components")
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
    
    # SMOTE oversampling (only on training data)
    smote_config = config.get("smote", {})
    if smote_config.get("enabled", True):
        smote_transformer = SMOTETransformer(
            method=smote_config.get("method", "smote"),
            sampling_strategy=smote_config.get("sampling_strategy", "auto"),
            random_state=smote_config.get("random_state", 42)
        )
        
        X_train_resampled, y_train_resampled = smote_transformer.fit_transform(
            X_train_final, y_train
        )
        
        # Store SMOTE transformer
        artifacts["smote_transformer"] = smote_transformer
        
        # Log sampling information
        logger.info(f"Applied SMOTE with method: {smote_config.get('method', 'smote')}")
        
        X_train_final = X_train_resampled
        y_train_final = y_train_resampled
    else:
        y_train_final = y_train.copy()
    
    # Validation data remains unchanged (no SMOTE)
    y_val_final = y_val.copy()
    
    # Log final shapes
    logger.info(f"Final training data shape: {X_train_final.shape}")
    logger.info(f"Final validation data shape: {X_val_final.shape}")
    logger.info(f"Training target distribution: {y_train_final.value_counts().to_dict()}")

    return X_train_final, X_val_final, y_train_final, y_val_final, artifacts


@step
def prepare_features_step(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from raw data.
    
    Args:
        data: Input DataFrame
        target_column: Name of target column
        
    Returns:
        Features and target
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Log basic statistics
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"Missing values in features: {X.isnull().sum().sum()}")
    
    return X, y