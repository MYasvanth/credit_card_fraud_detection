"""ZenML step for model registration with MLflow."""

from typing import Dict, Any, Optional
import pandas as pd
from sklearn.base import BaseEstimator
import mlflow
import mlflow.sklearn
from zenml import step
from zenml.logger import get_logger
from pathlib import Path

from ..utils.constants import MLFLOW_EXPERIMENT_NAME, MODELS_PATH
from ..utils.helpers import evaluate_model

logger = get_logger(__name__)


@step
def register_model_step(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tuning_results: Dict[str, Any],
    model_config: Dict[str, Any],
    feature_config: Dict[str, Any],
    validation_report: Dict[str, Any],
    model_name: str = "credit_card_fraud_detector"
) -> Dict[str, Any]:
    """
    Register model with MLflow and evaluate performance.
    
    Args:
        model: Trained model to register
        X_test: Test features
        y_test: Test target
        tuning_results: Hyperparameter tuning results
        model_config: Model configuration
        feature_config: Feature engineering configuration
        validation_report: Data validation report
        model_name: Name for model registration
        
    Returns:
        Registration results and model URI
    """
    # Set MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        # Log model parameters
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
        
        # Log training metadata
        mlflow.log_params(tuning_results.get("best_params", {}))
        mlflow.log_metric("cv_score", tuning_results.get("best_score", 0))
        mlflow.log_metric("validation_score", tuning_results.get("validation_score", 0))
        
        # Evaluate model on test set
        y_pred = model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred)
        
        # Log test metrics
        classification_report = test_metrics["classification_report"]
        mlflow.log_metric("test_accuracy", classification_report["accuracy"])
        mlflow.log_metric("test_precision", classification_report["1"]["precision"])
        mlflow.log_metric("test_recall", classification_report["1"]["recall"])
        mlflow.log_metric("test_f1", classification_report["1"]["f1-score"])
        
        # Log additional metadata
        mlflow.log_param("model_type", model_config.get("type", "unknown"))
        
        # Log model
        model_uri = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        ).model_uri
        
        run_id = run.info.run_id
        
    logger.info(f"Model registered with MLflow. Run ID: {run_id}")
    logger.info(f"Model URI: {model_uri}")
    logger.info(f"Test F1 Score: {classification_report['1']['f1-score']:.4f}")
    
    registration_results = {
        "run_id": run_id,
        "model_uri": model_uri,
        "model_name": model_name,
        "test_metrics": test_metrics,
        "experiment_name": MLFLOW_EXPERIMENT_NAME
    }
    
    return registration_results


@step
def promote_model_step(
    model_name: str,
    model_version: Optional[str] = None,
    stage: str = "Staging"
) -> Dict[str, Any]:
    """
    Promote model to a specific stage in MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        model_version: Version to promote (latest if None)
        stage: Target stage ("Staging", "Production", "Archived")
        
    Returns:
        Promotion results
    """
    client = mlflow.tracking.MlflowClient()
    
    # Get latest version if not specified
    if model_version is None:
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if not latest_versions:
            raise ValueError(f"No versions found for model {model_name}")
        model_version = latest_versions[0].version
    
    # Transition model to new stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage
    )
    
    logger.info(f"Model {model_name} version {model_version} promoted to {stage}")
    
    return {
        "model_name": model_name,
        "model_version": model_version,
        "stage": stage,
        "promotion_timestamp": pd.Timestamp.now().isoformat()
    }