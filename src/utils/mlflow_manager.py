"""Enhanced MLflow experiment management and tracking utilities."""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

from .constants import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENTS, MLFLOW_MODEL_REGISTRY_NAME
from .logger import logger

class MLflowManager:
    """Enhanced MLflow experiment manager with better organization and tracking."""
    
    def __init__(self, tracking_uri: str = None):
        """Initialize MLflow manager."""
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(self.tracking_uri)
        
        # Initialize experiments
        self._setup_experiments()
    
    def _setup_experiments(self):
        """Setup all required experiments."""
        for exp_type, exp_name in MLFLOW_EXPERIMENTS.items():
            try:
                # Check if experiment already exists before creating
                existing_exp = self.client.get_experiment_by_name(exp_name)
                if existing_exp is None:
                    self.client.create_experiment(exp_name)
                    logger.info(f"Created experiment: {exp_name}")
                else:
                    logger.debug(f"Experiment {exp_name} already exists. Skipping creation.")
            except mlflow.exceptions.MlflowException as e:
                logger.error(f"Error creating experiment {exp_name}: {e}")
    
    def start_training_run(self, run_name: str = None, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Start a training experiment run."""
        mlflow.set_experiment(MLFLOW_EXPERIMENTS["training"])
        
        run_tags = {"experiment_type": "training"}
        if tags:
            run_tags.update(tags)
        
        return mlflow.start_run(run_name=run_name, tags=run_tags)
    
    def start_ab_test_run(self, model_a: str, model_b: str, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Start an A/B testing experiment run."""
        mlflow.set_experiment(MLFLOW_EXPERIMENTS["ab_testing"])
        
        run_tags = {
            "experiment_type": "ab_testing",
            "model_a": model_a,
            "model_b": model_b
        }
        if tags:
            run_tags.update(tags)
        
        return mlflow.start_run(run_name=f"ab_test_{model_a}_vs_{model_b}", tags=run_tags)
    
    def start_comparison_run(self, models: List[str], tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Start a model comparison experiment run."""
        mlflow.set_experiment(MLFLOW_EXPERIMENTS["model_comparison"])
        
        run_tags = {
            "experiment_type": "model_comparison",
            "models_compared": ",".join(models),
            "n_models": len(models)
        }
        if tags:
            run_tags.update(tags)
        
        return mlflow.start_run(run_name=f"comparison_{len(models)}_models", tags=run_tags)
    
    def start_monitoring_run(self, model_name: str, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Start a monitoring experiment run."""
        mlflow.set_experiment(MLFLOW_EXPERIMENTS["monitoring"])
        
        run_tags = {
            "experiment_type": "monitoring",
            "monitored_model": model_name
        }
        if tags:
            run_tags.update(tags)
        
        return mlflow.start_run(run_name=f"monitoring_{model_name}", tags=run_tags)
    
    def start_retraining_run(self, trigger_reason: str, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Start a retraining experiment run."""
        mlflow.set_experiment(MLFLOW_EXPERIMENTS["retraining"])
        
        run_tags = {
            "experiment_type": "retraining",
            "trigger_reason": trigger_reason
        }
        if tags:
            run_tags.update(tags)
        
        return mlflow.start_run(run_name=f"retraining_{trigger_reason}", tags=run_tags)
    
    def log_model_with_metadata(
        self, 
        model: Any, 
        model_name: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        artifacts: Dict[str, str] = None,
        stage: str = "Staging"
    ) -> str:
        """Log model with comprehensive metadata."""
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log additional artifacts
        if artifacts:
            for artifact_name, artifact_path in artifacts.items():
                if Path(artifact_path).exists():
                    mlflow.log_artifact(artifact_path, artifact_name)
        
        # Transition to specified stage
        try:
            latest_version = self.client.get_latest_versions(model_name, stages=["None"])[0]
            self.client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage=stage
            )
            logger.info(f"Model {model_name} v{latest_version.version} transitioned to {stage}")
        except Exception as e:
            logger.warning(f"Failed to transition model stage: {e}")
        
        return model_info.model_uri
    
    def log_experiment_summary(self, summary: Dict[str, Any]):
        """Log experiment summary with structured data."""
        
        # Log summary as parameters and metrics
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            elif isinstance(value, str):
                mlflow.log_param(key, value)
            elif isinstance(value, (list, dict)):
                mlflow.log_param(key, json.dumps(value))
        
        # Save detailed summary as artifact
        summary_path = "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        mlflow.log_artifact(summary_path)
        Path(summary_path).unlink()  # Clean up
    
    def compare_model_versions(self, model_name: str, stages: List[str] = None) -> Dict[str, Any]:
        """Compare different versions/stages of a model."""
        if stages is None:
            stages = ["Production", "Staging"]
        
        comparison = {}
        
        for stage in stages:
            try:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if versions:
                    version = versions[0]
                    run = self.client.get_run(version.run_id)
                    
                    comparison[stage] = {
                        "version": version.version,
                        "run_id": version.run_id,
                        "metrics": run.data.metrics,
                        "params": run.data.params,
                        "creation_timestamp": version.creation_timestamp
                    }
            except Exception as e:
                logger.warning(f"Failed to get {stage} model: {e}")
                comparison[stage] = {"error": str(e)}
        
        return comparison
    
    def get_experiment_runs(self, experiment_type_or_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get runs from a specific experiment type or name."""
        # Check if it's an experiment type (key in MLFLOW_EXPERIMENTS)
        experiment_name = MLFLOW_EXPERIMENTS.get(experiment_type_or_name)
        if not experiment_name:
            # Assume it's already an experiment name
            experiment_name = experiment_type_or_name

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=max_results,
                    order_by=["start_time DESC"]
                )
                return runs.to_dict('records') if not runs.empty else []
            else:
                logger.warning(f"Experiment '{experiment_name}' not found")
                return []
        except Exception as e:
            logger.error(f"Failed to get experiment runs: {e}")
            return []
    
    def cleanup_old_runs(self, experiment_type: str, keep_last_n: int = 50):
        """Clean up old experiment runs to save space."""
        runs = self.get_experiment_runs(experiment_type, max_results=1000)
        
        if len(runs) > keep_last_n:
            runs_to_delete = runs[keep_last_n:]
            
            for run in runs_to_delete:
                try:
                    self.client.delete_run(run['run_id'])
                    logger.info(f"Deleted old run: {run['run_id']}")
                except Exception as e:
                    logger.warning(f"Failed to delete run {run['run_id']}: {e}")
    
    def export_experiment_data(self, experiment_type: str, output_path: str):
        """Export experiment data to CSV for analysis."""
        runs = self.get_experiment_runs(experiment_type, max_results=1000)
        
        if runs:
            df = pd.DataFrame(runs)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(runs)} runs to {output_path}")
        else:
            logger.warning(f"No runs found for experiment type: {experiment_type}")
    
    def get_model_performance_history(self, model_name: str) -> pd.DataFrame:
        """Get performance history of a model across versions."""
        try:
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            history = []
            for version in model_versions:
                run = self.client.get_run(version.run_id)
                
                history.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "creation_time": version.creation_timestamp,
                    **run.data.metrics,
                    **run.data.params
                })
            
            return pd.DataFrame(history).sort_values("creation_time")
        
        except Exception as e:
            logger.error(f"Failed to get model history: {e}")
            return pd.DataFrame()

# Global instance
mlflow_manager = MLflowManager()
