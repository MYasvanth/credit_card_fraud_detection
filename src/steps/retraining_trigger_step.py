"""ZenML step for automated retraining triggers based on monitoring results."""

from typing import Dict, Any, Optional
import pandas as pd
import mlflow
from zenml import step
from zenml.logger import get_logger

from ..utils.constants import MLFLOW_EXPERIMENT_NAME

logger = get_logger(__name__)


@step
def evaluate_retraining_criteria_step(
    monitoring_results: Dict[str, Any],
    retraining_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate whether retraining should be triggered based on monitoring results.

    Args:
        monitoring_results: Results from monitoring pipeline
        retraining_config: Configuration for retraining criteria

    Returns:
        Retraining decision and reasoning
    """
    drift_results = monitoring_results.get("drift_results", {})
    performance_results = monitoring_results.get("degradation_results", {})

    # Extract thresholds
    drift_threshold = retraining_config.get("drift_threshold", 0.1)  # 10% features with drift
    performance_drop_threshold = retraining_config.get("performance_drop_threshold", 0.05)  # 5% F1 drop
    time_since_last_training = retraining_config.get("max_days_since_training", 30)

    # Check drift criteria
    drift_detected = drift_results.get("drift_detected", False)
    drift_percentage = drift_results.get("summary", {}).get("drift_percentage", 0)

    # Check performance criteria
    performance_degraded = performance_results.get("performance_degraded", False)
    degradation_percentage = performance_results.get("degradation_percentage", 0)

    # Check time-based criteria (simplified - in practice, check against last training timestamp)
    time_based_trigger = False  # Placeholder - would check actual timestamps

    # Decision logic
    retraining_reasons = []

    if drift_detected and drift_percentage >= drift_threshold:
        retraining_reasons.append(f"Data drift detected in {drift_percentage:.1f}% of features")

    if performance_degraded and abs(degradation_percentage) >= performance_drop_threshold:
        retraining_reasons.append(f"Performance degradation of {abs(degradation_percentage):.1f}% detected")

    if time_based_trigger:
        retraining_reasons.append(f"Time-based retraining trigger (>{time_since_last_training} days)")

    # Overall decision
    should_retrain = len(retraining_reasons) > 0

    decision = {
        "should_retrain": should_retrain,
        "retraining_reasons": retraining_reasons,
        "criteria_evaluated": {
            "drift_threshold": drift_threshold,
            "drift_percentage": drift_percentage,
            "performance_drop_threshold": performance_drop_threshold,
            "degradation_percentage": degradation_percentage,
            "time_based_trigger": time_based_trigger
        },
        "monitoring_summary": monitoring_results.get("monitoring_summary", {})
    }

    if should_retrain:
        logger.info(f"Retraining triggered: {', '.join(retraining_reasons)}")
    else:
        logger.info("No retraining needed based on current criteria")

    return decision


@step
def trigger_retraining_pipeline_step(
    retraining_decision: Dict[str, Any],
    pipeline_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Trigger automated retraining pipeline if criteria are met.

    Args:
        retraining_decision: Decision from retraining criteria evaluation
        pipeline_config: Configuration for triggering retraining pipeline

    Returns:
        Trigger results
    """
    if not retraining_decision.get("should_retrain", False):
        logger.info("Retraining not triggered - criteria not met")
        return {
            "triggered": False,
            "reason": "Retraining criteria not met",
            "decision": retraining_decision
        }

    try:
        # In a real implementation, this would trigger the training pipeline
        # For now, we'll log the intent and prepare the trigger

        trigger_info = {
            "triggered": True,
            "timestamp": pd.Timestamp.now().isoformat(),
            "reasons": retraining_decision.get("retraining_reasons", []),
            "pipeline_config": pipeline_config,
            "monitoring_snapshot": retraining_decision.get("monitoring_summary", {})
        }

        # Log to MLflow
        mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_NAME}_retraining")

        with mlflow.start_run():
            mlflow.log_param("retraining_triggered", True)
            mlflow.log_param("trigger_timestamp", trigger_info["timestamp"])
            mlflow.log_param("retraining_reasons", ", ".join(trigger_info["reasons"]))

            # Log monitoring metrics at time of trigger
            monitoring = trigger_info["monitoring_snapshot"]
            if monitoring:
                mlflow.log_metric("drift_detected", int(monitoring.get("drift_detected", False)))
                mlflow.log_metric("performance_degraded", int(monitoring.get("performance_degraded", False)))
                mlflow.log_metric("current_f1", monitoring.get("current_f1", 0))

        logger.info(f"Retraining pipeline triggered: {trigger_info}")

        # Here you would typically:
        # 1. Call the training pipeline API
        # 2. Update model registry status
        # 3. Send notifications
        # 4. Update monitoring dashboards

        return trigger_info

    except Exception as e:
        logger.error(f"Failed to trigger retraining pipeline: {e}")
        return {
            "triggered": False,
            "error": str(e),
            "decision": retraining_decision
        }


@step
def update_model_registry_status_step(
    retraining_decision: Dict[str, Any],
    model_name: str = "credit_card_fraud_detector"
) -> Dict[str, Any]:
    """
    Update model registry status based on retraining decision.

    Args:
        retraining_decision: Retraining decision
        model_name: Name of the model in registry

    Returns:
        Registry update results
    """
    client = mlflow.tracking.MlflowClient()

    try:
        # Get current production model
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            logger.warning(f"No production version found for {model_name}")
            return {"updated": False, "reason": "No production model found"}

        prod_version = prod_versions[0]

        # Add tags based on retraining decision
        if retraining_decision.get("should_retrain"):
            client.set_model_version_tag(
                model_name,
                prod_version.version,
                "retraining_needed",
                "true"
            )
            client.set_model_version_tag(
                model_name,
                prod_version.version,
                "retraining_reasons",
                ", ".join(retraining_decision.get("retraining_reasons", []))
            )
            logger.info(f"Marked production model {model_name} v{prod_version.version} for retraining")
        else:
            # Clear any existing retraining tags
            try:
                client.delete_model_version_tag(model_name, prod_version.version, "retraining_needed")
                client.delete_model_version_tag(model_name, prod_version.version, "retraining_reasons")
            except:
                pass  # Tags might not exist
            logger.info(f"Cleared retraining tags for stable model {model_name} v{prod_version.version}")

        return {
            "updated": True,
            "model_name": model_name,
            "version": prod_version.version,
            "retraining_needed": retraining_decision.get("should_retrain", False)
        }

    except Exception as e:
        logger.error(f"Failed to update model registry: {e}")
        return {"updated": False, "error": str(e)}


@step
def send_retraining_notifications_step(
    retraining_decision: Dict[str, Any],
    notification_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send notifications about retraining decisions.

    Args:
        retraining_decision: Retraining decision
        notification_config: Notification configuration

    Returns:
        Notification results
    """
    # This is a placeholder for actual notification logic
    # In practice, you might integrate with Slack, email, etc.

    if retraining_decision.get("should_retrain"):
        message = {
            "type": "retraining_triggered",
            "message": "Automated retraining triggered",
            "reasons": retraining_decision.get("retraining_reasons", []),
            "timestamp": pd.Timestamp.now().isoformat(),
            "monitoring_data": retraining_decision.get("monitoring_summary", {})
        }
        logger.info(f"Retraining notification: {message}")

        # Here you would send to notification service
        # e.g., slack_webhook.send(message)

    else:
        message = {
            "type": "retraining_not_needed",
            "message": "Model performance stable - no retraining needed",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        logger.info(f"Stability notification: {message}")

    return {
        "notification_sent": True,
        "message": message,
        "channels": notification_config.get("channels", ["log"])
    }
