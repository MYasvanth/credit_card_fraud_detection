"""Deployment runner for model training and serving pipelines."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
from datetime import datetime

from ..utils.logger import logger
from ..utils.constants import CONFIG_PATH, MODELS_PATH, DATA_PATH
from ..pipelines.training_pipeline import training_pipeline, quick_training_pipeline
from ..pipelines.monitoring_pipeline import monitoring_pipeline


class PipelineRunner:
    """Runner for executing ML pipelines."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize pipeline runner.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else CONFIG_PATH
        self.configs = self._load_configs()
        
    def _load_configs(self) -> Dict[str, Any]:
        """Load all configuration files."""
        configs = {}
        
        config_files = {
            'data': 'data.yaml',
            'training': 'training.yaml',
            'model': 'model.yaml',
            'evaluation': 'evaluation.yaml',
            'monitoring': 'monitoring.yaml',
            'deployment': 'deployment.yaml'
        }
        
        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                with open(config_path, 'r') as f:
                    configs[config_name] = yaml.safe_load(f)
                logger.info(f"Loaded {config_name} configuration from {config_path}")
            else:
                logger.warning(f"Configuration file not found: {config_path}")
                configs[config_name] = {}
        
        return configs
    
    def run_training_pipeline(self, 
                            data_path: Optional[str] = None,
                            quick_mode: bool = False) -> Dict[str, Any]:
        """
        Run the training pipeline.
        
        Args:
            data_path: Path to training data
            quick_mode: Whether to use quick training mode
            
        Returns:
            Training results
        """
        logger.info("Starting training pipeline...")
        
        # Use default data path if not provided
        if data_path is None:
            data_path = str(DATA_PATH / "raw" / "creditcard.csv")
        
        try:
            if quick_mode:
                logger.info("Running quick training pipeline")
                results = quick_training_pipeline(
                    data_path=data_path,
                    model_type=self.configs['model'].get('type', 'random_forest')
                )
            else:
                logger.info("Running full training pipeline")
                
                # Prepare configurations
                validation_config = self._prepare_validation_config()
                feature_config = self._prepare_feature_config()
                model_config = self._prepare_model_config()
                tuning_config = self._prepare_tuning_config()
                
                results = training_pipeline(
                    data_path=data_path,
                    validation_config=validation_config,
                    feature_config=feature_config,
                    model_config=model_config,
                    tuning_config=tuning_config
                )
            
            # Handle ZenML pipeline response
            if hasattr(results, 'steps'):
                # Extract results from ZenML pipeline response
                pipeline_results = {}
                for step_name, step_output in results.steps.items():
                    if hasattr(step_output, 'outputs'):
                        pipeline_results[step_name] = step_output.outputs
                results = pipeline_results
            
            # Save training results
            self._save_training_results(results)
            
            logger.info("Training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def run_monitoring_pipeline(self,
                              reference_data_path: str,
                              current_data_path: str,
                              predictions_path: str,
                              baseline_f1: float) -> Dict[str, Any]:
        """
        Run the monitoring pipeline.
        
        Args:
            reference_data_path: Path to reference dataset
            current_data_path: Path to current dataset
            predictions_path: Path to predictions file
            baseline_f1: Baseline F1 score
            
        Returns:
            Monitoring results
        """
        logger.info("Starting monitoring pipeline...")
        
        try:
            # Load predictions and true labels
            import pandas as pd
            predictions_df = pd.read_csv(predictions_path)

            results = monitoring_pipeline(
                reference_data_path=reference_data_path,
                current_data_path=current_data_path,
                model_predictions=predictions_df['prediction'],
                true_labels=predictions_df['true_label'],
                baseline_f1=baseline_f1,
                drift_threshold=self.configs['monitoring'].get('drift_threshold', 0.05),
                degradation_threshold=self.configs['monitoring'].get('degradation_threshold', 0.05)
            )

            logger.info("Monitoring pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Monitoring pipeline failed: {e}")
            raise

    def _prepare_validation_config(self) -> Dict[str, Any]:
        """Prepare data validation configuration."""
        data_config = self.configs.get('data', {})

        return {
            'required_columns': data_config.get('required_columns', []),
            'target_column': data_config.get('target_column', 'Class'),
            'numerical_columns': data_config.get('numerical_columns', []),
            'categorical_columns': data_config.get('categorical_columns', [])
        }

    def _prepare_feature_config(self) -> Dict[str, Any]:
        """Prepare feature engineering configuration."""
        return {
            'scaling': {
                'enabled': True,
                'method': 'standard',
                'exclude_columns': ['Class']
            },
            'pca': {
                'enabled': self.configs.get('training', {}).get('use_pca', False),
                'n_components': self.configs.get('training', {}).get('pca_components'),
                'variance_threshold': 0.95
            },
            'smote': {
                'enabled': self.configs.get('training', {}).get('use_smote', True),
                'method': 'smote',
                'sampling_strategy': 'auto'
            }
        }

    def _prepare_model_config(self) -> Dict[str, Any]:
        """Prepare model configuration."""
        model_config = self.configs.get('model', {})

        return {
            'type': model_config.get('type', 'random_forest'),
            'parameters': model_config.get('parameters', {})
        }

    def _prepare_tuning_config(self) -> Dict[str, Any]:
        """Prepare hyperparameter tuning configuration."""
        training_config = self.configs.get('training', {})

        return {
            'cv_folds': training_config.get('cv_folds', 5),
            'scoring': training_config.get('scoring', 'f1'),
            'n_jobs': training_config.get('n_jobs', -1),
            'param_grid': training_config.get('param_grid', {})
        }

    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save training results to disk."""
        # Create results directory
        results_dir = Path(MODELS_PATH) / "training_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"training_results_{timestamp}.json"

        # Prepare serializable results
        serializable_results = {
            'timestamp': timestamp,
            'registration_results': results.get('registration_results', {}),
            'tuning_results': results.get('tuning_results', {}),
            'validation_report': results.get('validation_report', {})
        }

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Training results saved to {results_file}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="ML Pipeline Runner")
    parser.add_argument('command', choices=['train', 'monitor'],
                       help='Pipeline to run')
    parser.add_argument('--data-path', type=str,
                       help='Path to training data')
    parser.add_argument('--quick', action='store_true',
                       help='Use quick training mode')
    parser.add_argument('--config-dir', type=str,
                       help='Configuration directory')

    # Monitoring specific arguments
    parser.add_argument('--reference-data', type=str,
                       help='Path to reference data for monitoring')
    parser.add_argument('--current-data', type=str,
                       help='Path to current data for monitoring')
    parser.add_argument('--predictions', type=str,
                       help='Path to predictions file')
    parser.add_argument('--baseline-f1', type=float,
                       help='Baseline F1 score for comparison')

    args = parser.parse_args()

    # Initialize runner
    runner = PipelineRunner(config_dir=args.config_dir)

    try:
        if args.command == 'train':
            results = runner.run_training_pipeline(
                data_path=args.data_path,
                quick_mode=args.quick
            )
            print(f"Training completed. Model URI: {results['registration_results']['model_uri']}")

        elif args.command == 'monitor':
            if not all([args.reference_data, args.current_data,
                       args.predictions, args.baseline_f1]):
                print("Error: Monitoring requires --reference-data, --current-data, "
                     "--predictions, and --baseline-f1 arguments")
                sys.exit(1)

            results = runner.run_monitoring_pipeline(
                reference_data_path=args.reference_data,
                current_data_path=args.current_data,
                predictions_path=args.predictions,
                baseline_f1=args.baseline_f1
            )

            summary = results['monitoring_summary']
            print(f"Monitoring completed. Drift detected: {summary['drift_detected']}, "
                 f"Performance degraded: {summary['performance_degraded']}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


def run_full_pipeline() -> bool:
    """
    Run the complete MLOps pipeline: training -> deployment -> monitoring.

    Returns:
        True if all steps successful, False otherwise.
    """
    logger.info("Starting full MLOps pipeline execution")

    runner = PipelineRunner()

    # Step 1: Training
    try:
        results = runner.run_training_pipeline()
        logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return False

    # Step 2: Deployment (placeholder)
    logger.info("Deployment step (placeholder)")

    # Step 3: Monitoring (placeholder)
    logger.info("Monitoring step (placeholder)")

    logger.info("Full MLOps pipeline completed successfully")
    return True


if __name__ == "__main__":
    main()
