"""ZenML step for hyperparameter tuning."""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from zenml import step
from zenml.logger import get_logger

from ..models.tune import HyperparameterTuner
from ..utils.constants import RANDOM_STATE

logger = get_logger(__name__)


@step
def tune_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_config: Dict[str, Any],
    tuning_config: Dict[str, Any]
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Perform hyperparameter tuning for the specified model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_config: Model configuration
        tuning_config: Tuning configuration
        
    Returns:
        Best model and tuning results
    """
    logger.info(f"Starting hyperparameter tuning for {model_config.get('type', 'random_forest')}")
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    
    try:
        from ..models.tune import HyperparameterTuner
        
        tuner = HyperparameterTuner(
            model_type=model_config.get("type", "random_forest"),
            param_grid=tuning_config.get("param_grid", {}),
            cv_folds=tuning_config.get("cv_folds", 3),  # Reduced for faster execution
            scoring=tuning_config.get("scoring", "f1"),
            n_jobs=tuning_config.get("n_jobs", -1),
            random_state=tuning_config.get("random_state", RANDOM_STATE)
        )
        
        logger.info("Hyperparameter tuner initialized, starting tuning...")
        
        # Convert data to avoid MaskedArray issues
        import numpy as np
        X_train_clean = pd.DataFrame(np.asarray(X_train), columns=X_train.columns)
        y_train_clean = pd.Series(np.asarray(y_train), name=y_train.name)
        X_val_clean = pd.DataFrame(np.asarray(X_val), columns=X_val.columns)
        y_val_clean = pd.Series(np.asarray(y_val), name=y_val.name)
        
        # Perform tuning
        best_model, tuning_results = tuner.tune(X_train_clean, y_train_clean)
        
        logger.info("Tuning completed, evaluating on validation set...")
        
        # Evaluate on validation set
        val_score = tuner.evaluate_model(best_model, X_val_clean, y_val_clean)
        
        # Log results
        logger.info(f"Best parameters: {tuning_results['best_params']}")
        logger.info(f"Best CV score: {tuning_results['best_score']:.4f}")
        logger.info(f"Validation score: {val_score:.4f}")
        
        # Add validation score to results
        tuning_results["validation_score"] = val_score
        tuning_results["model_type"] = model_config.get("type")
        
        return best_model, tuning_results
        
    except ImportError as e:
        logger.warning(f"HyperparameterTuner not available: {e}")
        logger.info("Falling back to simple model training...")
        
        # Fallback to simple training
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
        
        logger.info("Training fallback RandomForest model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = model.predict(X_val)
        val_score = f1_score(y_val, y_val_pred, average='weighted')
        
        logger.info(f"Fallback model validation F1 score: {val_score:.4f}")
        
        tuning_results = {
            "best_params": model.get_params(),
            "best_score": val_score,
            "validation_score": val_score,
            "model_type": "random_forest",
            "method": "fallback"
        }
        
        return model, tuning_results


@step
def quick_tune_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    n_trials: int = 50
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Quick hyperparameter tuning with Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to tune
        n_trials: Number of optimization trials
        
    Returns:
        Best model and tuning results
    """
    tuner = HyperparameterTuner(
        model_type=model_type,
        random_state=RANDOM_STATE
    )
    
    # Perform Optuna tuning
    best_model, tuning_results = tuner.tune_with_optuna(
        X_train, y_train, n_trials=n_trials
    )
    
    logger.info(f"Optuna tuning completed with {n_trials} trials")
    logger.info(f"Best parameters: {tuning_results['best_params']}")
    logger.info(f"Best score: {tuning_results['best_score']:.4f}")
    
    return best_model, tuning_results