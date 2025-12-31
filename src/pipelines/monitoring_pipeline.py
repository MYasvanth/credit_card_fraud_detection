"""ZenML monitoring pipeline for model drift detection and performance monitoring."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from zenml import pipeline, step
from zenml.logger import get_logger
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import mlflow

from ..utils.constants import MLFLOW_EXPERIMENT_NAME
from ..data.loader import load_data

logger = get_logger(__name__)


@step
def load_reference_data_step(reference_data_path: str) -> pd.DataFrame:
    """Load reference data for drift detection."""
    reference_data = load_data(reference_data_path)
    logger.info(f"Loaded reference data with shape: {reference_data.shape}")
    return reference_data


@step
def load_current_data_step(current_data_path: str) -> pd.DataFrame:
    """Load current production data."""
    current_data = load_data(current_data_path)
    logger.info(f"Loaded current data with shape: {current_data.shape}")
    return current_data


@step
def detect_data_drift_step(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    drift_threshold: float = 0.05,
    numerical_columns: Optional[list] = None
) -> Dict[str, Any]:
    """
    Detect data drift using statistical tests.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset
        drift_threshold: P-value threshold for drift detection
        numerical_columns: List of numerical columns to check
        
    Returns:
        Drift detection results
    """
    if numerical_columns is None:
        numerical_columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()
    
    drift_results = {
        "drift_detected": False,
        "drifted_features": [],
        "drift_scores": {},
        "summary": {}
    }
    
    drifted_features = []
    
    for column in numerical_columns:
        if column in reference_data.columns and column in current_data.columns:
            # Kolmogorov-Smirnov test for distribution drift
            ks_statistic, p_value = stats.ks_2samp(
                reference_data[column].dropna(),
                current_data[column].dropna()
            )
            
            drift_results["drift_scores"][column] = {
                "ks_statistic": ks_statistic,
                "p_value": p_value,
                "drift_detected": p_value < drift_threshold
            }
            
            if p_value < drift_threshold:
                drifted_features.append(column)
                logger.warning(f"Drift detected in feature {column}: p-value = {p_value:.6f}")
    
    drift_results["drifted_features"] = drifted_features
    drift_results["drift_detected"] = len(drifted_features) > 0
    
    # Summary statistics
    drift_results["summary"] = {
        "total_features_checked": len(numerical_columns),
        "features_with_drift": len(drifted_features),
        "drift_percentage": len(drifted_features) / len(numerical_columns) * 100
    }
    
    logger.info(f"Drift detection completed. {len(drifted_features)} out of {len(numerical_columns)} features show drift")
    
    return drift_results


@step
def evaluate_model_performance_step(
    model_predictions: pd.Series,
    true_labels: pd.Series,
    model_name: str = "current_model"
) -> Dict[str, Any]:
    """
    Evaluate current model performance.
    
    Args:
        model_predictions: Model predictions
        true_labels: True labels
        model_name: Name of the model
        
    Returns:
        Performance metrics
    """
    # Calculate metrics
    report = classification_report(true_labels, model_predictions, output_dict=True)
    cm = confusion_matrix(true_labels, model_predictions)
    
    performance_metrics = {
        "model_name": model_name,
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
    logger.info(f"Model performance - F1: {report['1']['f1-score']:.4f}, "
               f"Precision: {report['1']['precision']:.4f}, "
               f"Recall: {report['1']['recall']:.4f}")
    
    return performance_metrics


@step
def check_performance_degradation_step(
    current_metrics: Dict[str, Any],
    baseline_f1: float,
    degradation_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Check if model performance has degraded significantly.
    
    Args:
        current_metrics: Current model performance metrics
        baseline_f1: Baseline F1 score
        degradation_threshold: Threshold for significant degradation
        
    Returns:
        Performance degradation analysis
    """
    current_f1 = current_metrics["f1_score"]
    f1_drop = baseline_f1 - current_f1
    degradation_percentage = (f1_drop / baseline_f1) * 100
    
    performance_degraded = f1_drop > degradation_threshold
    
    degradation_results = {
        "performance_degraded": performance_degraded,
        "baseline_f1": baseline_f1,
        "current_f1": current_f1,
        "f1_drop": f1_drop,
        "degradation_percentage": degradation_percentage,
        "threshold": degradation_threshold
    }
    
    if performance_degraded:
        logger.warning(f"Performance degradation detected! F1 dropped by {degradation_percentage:.2f}%")
    else:
        logger.info(f"Model performance is stable. F1 change: {degradation_percentage:.2f}%")
    
    return degradation_results


@step
def log_monitoring_results_step(
    drift_results: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    degradation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Log monitoring results to MLflow.
    
    Args:
        drift_results: Data drift detection results
        performance_metrics: Model performance metrics
        degradation_results: Performance degradation analysis
        
    Returns:
        Monitoring summary
    """
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_NAME}_monitoring")
    
    with mlflow.start_run():
        # Log drift metrics
        mlflow.log_metric("drift_detected", int(drift_results["drift_detected"]))
        mlflow.log_metric("features_with_drift", drift_results["summary"]["features_with_drift"])
        mlflow.log_metric("drift_percentage", drift_results["summary"]["drift_percentage"])
        
        # Log performance metrics
        mlflow.log_metric("current_accuracy", performance_metrics["accuracy"])
        mlflow.log_metric("current_precision", performance_metrics["precision"])
        mlflow.log_metric("current_recall", performance_metrics["recall"])
        mlflow.log_metric("current_f1", performance_metrics["f1_score"])
        
        # Log degradation metrics
        mlflow.log_metric("performance_degraded", int(degradation_results["performance_degraded"]))
        mlflow.log_metric("f1_drop", degradation_results["f1_drop"])
        mlflow.log_metric("degradation_percentage", degradation_results["degradation_percentage"])
        
        # Log parameters
        mlflow.log_param("model_name", performance_metrics["model_name"])
        mlflow.log_param("baseline_f1", degradation_results["baseline_f1"])
    
    monitoring_summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "drift_detected": drift_results["drift_detected"],
        "performance_degraded": degradation_results["performance_degraded"],
        "current_f1": performance_metrics["f1_score"],
        "recommendations": []
    }
    
    # Generate recommendations
    if drift_results["drift_detected"]:
        monitoring_summary["recommendations"].append("Data drift detected - consider retraining model")
    
    if degradation_results["performance_degraded"]:
        monitoring_summary["recommendations"].append("Performance degradation detected - investigate and retrain")
    
    if not drift_results["drift_detected"] and not degradation_results["performance_degraded"]:
        monitoring_summary["recommendations"].append("Model is performing well - continue monitoring")
    
    logger.info(f"Monitoring results logged. Summary: {monitoring_summary}")
    
    return monitoring_summary


@pipeline
def monitoring_pipeline(
    reference_data_path: str,
    current_data_path: str,
    model_predictions: pd.Series,
    true_labels: pd.Series,
    baseline_f1: float,
    drift_threshold: float = 0.05,
    degradation_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Complete monitoring pipeline for model drift and performance monitoring.
    
    Args:
        reference_data_path: Path to reference dataset
        current_data_path: Path to current dataset
        model_predictions: Current model predictions
        true_labels: True labels for current data
        baseline_f1: Baseline F1 score for comparison
        drift_threshold: Threshold for drift detection
        degradation_threshold: Threshold for performance degradation
        
    Returns:
        Monitoring results and recommendations
    """
    # Load data
    reference_data = load_reference_data_step(reference_data_path)
    current_data = load_current_data_step(current_data_path)
    
    # Detect data drift
    drift_results = detect_data_drift_step(
        reference_data, current_data, drift_threshold
    )
    
    # Evaluate current performance
    performance_metrics = evaluate_model_performance_step(
        model_predictions, true_labels
    )
    
    # Check for performance degradation
    degradation_results = check_performance_degradation_step(
        performance_metrics, baseline_f1, degradation_threshold
    )
    
    # Log results
    monitoring_summary = log_monitoring_results_step(
        drift_results, performance_metrics, degradation_results
    )
    
    return {
        "drift_results": drift_results,
        "performance_metrics": performance_metrics,
        "degradation_results": degradation_results,
        "monitoring_summary": monitoring_summary
    }