"""ZenML pipeline for automated retraining workflow."""

from typing import Dict, Any
import pandas as pd
from zenml import pipeline
from zenml.logger import get_logger

from ..steps.validate_data_step import validate_data_step
from ..steps.feature_engineering_step import feature_engineering_step
from ..steps.tune_step import tune_step
from ..steps.register_model_step import register_model_step
from ..steps.retraining_trigger_step import (
    evaluate_retraining_criteria_step,
    trigger_retraining_pipeline_step,
    update_model_registry_status_step,
    send_retraining_notifications_step
)

logger = get_logger(__name__)


@pipeline
def retraining_pipeline(
    monitoring_results: Dict[str, Any],
    data_path: str,
    validation_config: Dict[str, Any],
    feature_config: Dict[str, Any],
    model_config: Dict[str, Any],
    retraining_config: Dict[str, Any],
    notification_config: Dict[str, Any],
    mlflow_config: Dict[str, Any]
):
    """
    Complete automated retraining pipeline triggered by monitoring results.

    Args:
        monitoring_results: Results from monitoring pipeline
        data_path: Path to new training data
        validation_config: Data validation configuration
        feature_config: Feature engineering configuration
        model_config: Model training configuration
        retraining_config: Retraining criteria configuration
        notification_config: Notification settings
        mlflow_config: MLflow tracking configuration
    """
    logger.info("Starting automated retraining pipeline...")

    # Step 1: Evaluate retraining criteria
    retraining_decision = evaluate_retraining_criteria_step(
        monitoring_results, retraining_config
    )

    # Step 2: Update model registry status
    registry_update = update_model_registry_status_step(
        retraining_decision, model_config.get("name", "credit_card_fraud_detector")
    )

    # Step 3: Send notifications
    notifications = send_retraining_notifications_step(
        retraining_decision, notification_config
    )

    # Step 4: Trigger retraining if needed
    retraining_trigger = trigger_retraining_pipeline_step(
        retraining_decision, mlflow_config
    )

    # Only proceed with training if retraining is triggered
    if retraining_decision.get("should_retrain", False):
        logger.info("Retraining criteria met - proceeding with model training...")

        # Step 5: Load and validate new data
        validated_data, validation_report = validate_data_step(
            data_path=data_path,
            validation_config=validation_config
        )

        # Step 6: Feature engineering
        processed_data = feature_engineering_step(
            data=validated_data,
            config=feature_config
        )

        # Step 7: Model training and tuning
        trained_model, model_metrics = tune_step(
            data=processed_data,
            config=model_config
        )

        # Step 8: Register new model
        model_uri = register_model_step(
            model=trained_model,
            X_test=model_metrics.get("X_test"),
            y_test=model_metrics.get("y_test"),
            tuning_results=model_metrics.get("tuning_results", {}),
            model_config=model_config,
            feature_config=feature_config,
            validation_report=validation_report,
            model_name=model_config.get("name", "credit_card_fraud_detector")
        )

        logger.info("Retraining pipeline completed successfully")
        return {
            "retraining_decision": retraining_decision,
            "registry_update": registry_update,
            "notifications": notifications,
            "retraining_trigger": retraining_trigger,
            "new_model_uri": model_uri,
            "training_completed": True
        }
    else:
        logger.info("Retraining criteria not met - skipping training")
        return {
            "retraining_decision": retraining_decision,
            "registry_update": registry_update,
            "notifications": notifications,
            "retraining_trigger": retraining_trigger,
            "training_completed": False
        }


@pipeline
def ab_testing_pipeline(
    model_a_name: str,
    model_b_name: str,
    test_data_path: str,
    ab_config: Dict[str, Any],
    mlflow_config: Dict[str, Any]
):
    """
    A/B testing pipeline for model comparison.

    Args:
        model_a_name: Name of model A in registry
        model_b_name: Name of model B in registry
        test_data_path: Path to test data
        ab_config: A/B testing configuration
        mlflow_config: MLflow configuration
    """
    logger.info(f"Starting A/B testing pipeline: {model_a_name} vs {model_b_name}")

    # Load test data
    from ..data.loader import load_data
    test_data = load_data(test_data_path)

    # Run A/B test simulation
    from ..utils.ab_testing import run_ab_test_simulation
    ab_results = run_ab_test_simulation(
        model_a_name, model_b_name, test_data, ab_config
    )

    # Log results to MLflow
    import mlflow
    mlflow.set_experiment(mlflow_config.get("experiment_name", "ab_testing"))

    with mlflow.start_run():
        mlflow.log_param("model_a", model_a_name)
        mlflow.log_param("model_b", model_b_name)
        mlflow.log_param("traffic_split", ab_config.get("traffic_split", 0.5))
        mlflow.log_param("test_samples", len(test_data))

        # Log metrics
        for group in ["A", "B"]:
            if group in ab_results.get("metrics", {}):
                metrics = ab_results["metrics"][group]
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{group}_{metric}", value)

        if "comparison" in ab_results:
            comp = ab_results["comparison"]
            mlflow.log_metric("f1_improvement", comp.get("f1_improvement", 0))
            mlflow.log_param("winner", comp.get("winner"))
            mlflow.log_param("significant", comp.get("significant", False))

    logger.info("A/B testing pipeline completed")
    return ab_results


@pipeline
def model_comparison_pipeline(
    model_names: list,
    test_data_path: str,
    comparison_config: Dict[str, Any],
    mlflow_config: Dict[str, Any]
):
    """
    Model comparison pipeline for evaluating multiple models.

    Args:
        model_names: List of model names to compare
        test_data_path: Path to test data
        comparison_config: Comparison configuration
        mlflow_config: MLflow configuration
    """
    logger.info(f"Starting model comparison pipeline for {len(model_names)} models")

    # Load test data
    from ..data.loader import load_data
    test_data = load_data(test_data_path)

    # Run model comparison
    from ..utils.model_comparison import compare_models_from_registry
    comparator = compare_models_from_registry(
        model_names, test_data, target_column="Class"
    )

    # Generate comparison report
    report = comparator.generate_comparison_report()

    # Generate plots
    plots = comparator.plot_comparison_charts(
        save_path=comparison_config.get("output_dir", "reports/comparison")
    )

    # Log to MLflow
    run_id = comparator.log_comparison_to_mlflow()

    logger.info("Model comparison pipeline completed")
    return {
        "comparison_report": report,
        "mlflow_run_id": run_id,
        "plots_generated": len(plots)
    }
