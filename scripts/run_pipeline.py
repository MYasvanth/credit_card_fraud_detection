#!/usr/bin/env python3
"""Script to run ML pipelines for credit card fraud detection."""

import argparse
import sys
import os
from pathlib import Path

# Change to project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Add src to path
sys.path.insert(0, project_root)

from src.deployment.runner import PipelineRunner
from src.utils.logger import logger


def main():
    """Main function for running pipelines."""
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection ML Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training
  python run_pipeline.py train --quick
  
  # Full training with custom data
  python run_pipeline.py train --data-path /path/to/data.csv
  
  # Run monitoring
  python run_pipeline.py monitor --reference-data ref.csv --current-data curr.csv --predictions pred.csv --baseline-f1 0.85
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run training pipeline')
    train_parser.add_argument(
        '--data-path', 
        type=str,
        help='Path to training data CSV file'
    )
    train_parser.add_argument(
        '--quick', 
        action='store_true',
        help='Use quick training mode with default settings'
    )
    train_parser.add_argument(
        '--config-dir',
        type=str,
        help='Directory containing configuration files'
    )
    
    # Monitoring command
    monitor_parser = subparsers.add_parser('monitor', help='Run monitoring pipeline')
    monitor_parser.add_argument(
        '--reference-data',
        type=str,
        required=True,
        help='Path to reference dataset CSV file'
    )
    monitor_parser.add_argument(
        '--current-data',
        type=str,
        required=True,
        help='Path to current dataset CSV file'
    )
    monitor_parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV file (must have "prediction" and "true_label" columns)'
    )
    monitor_parser.add_argument(
        '--baseline-f1',
        type=float,
        required=True,
        help='Baseline F1 score for performance comparison'
    )
    monitor_parser.add_argument(
        '--config-dir',
        type=str,
        help='Directory containing configuration files'
    )
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the API server'
    )
    api_parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind the API server'
    )
    api_parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'train':
            run_training(args)
        elif args.command == 'monitor':
            run_monitoring(args)
        elif args.command == 'api':
            run_api_server(args)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


def run_training(args):
    """Run training pipeline."""
    logger.info("Starting training pipeline...")
    
    # Initialize runner
    runner = PipelineRunner(config_dir=args.config_dir)
    
    # Run training
    results = runner.run_training_pipeline(
        data_path=args.data_path,
        quick_mode=args.quick
    )
    
    # Print results
    registration_results = results.get('registration_results', {})
    tuning_results = results.get('tuning_results', {})
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*50)
    
    if 'model_uri' in registration_results:
        print(f"Model URI: {registration_results['model_uri']}")
    
    if 'run_id' in registration_results:
        print(f"MLflow Run ID: {registration_results['run_id']}")
    
    if 'best_score' in tuning_results:
        print(f"Best CV Score: {tuning_results['best_score']:.4f}")
    
    if 'test_metrics' in registration_results:
        test_metrics = registration_results['test_metrics']
        if 'classification_report' in test_metrics:
            f1_score = test_metrics['classification_report']['1']['f1-score']
            precision = test_metrics['classification_report']['1']['precision']
            recall = test_metrics['classification_report']['1']['recall']
            
            print(f"Test F1 Score: {f1_score:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
    
    print("="*50)


def run_monitoring(args):
    """Run monitoring pipeline."""
    logger.info("Starting monitoring pipeline...")
    
    # Initialize runner
    runner = PipelineRunner(config_dir=args.config_dir)
    
    # Run monitoring
    results = runner.run_monitoring_pipeline(
        reference_data_path=args.reference_data,
        current_data_path=args.current_data,
        predictions_path=args.predictions,
        baseline_f1=args.baseline_f1
    )
    
    # Print results
    monitoring_summary = results.get('monitoring_summary', {})
    drift_results = results.get('drift_results', {})
    performance_metrics = results.get('performance_metrics', {})
    
    print("\n" + "="*50)
    print("MONITORING COMPLETED")
    print("="*50)
    
    print(f"Timestamp: {monitoring_summary.get('timestamp', 'N/A')}")
    print(f"Data Drift Detected: {drift_results.get('drift_detected', False)}")
    print(f"Performance Degraded: {monitoring_summary.get('performance_degraded', False)}")
    print(f"Current F1 Score: {performance_metrics.get('f1_score', 0):.4f}")
    
    if drift_results.get('drift_detected'):
        drifted_features = drift_results.get('drifted_features', [])
        print(f"Features with Drift: {len(drifted_features)}")
        if drifted_features:
            print(f"Drifted Features: {', '.join(drifted_features[:5])}{'...' if len(drifted_features) > 5 else ''}")
    
    recommendations = monitoring_summary.get('recommendations', [])
    if recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("="*50)


def run_api_server(args):
    """Run API server."""
    logger.info("Starting API server...")
    
    try:
        import uvicorn
        from src.api.main import app
        
        print(f"\nStarting API server on {args.host}:{args.port}")
        print(f"API Documentation: http://{args.host}:{args.port}/docs")
        print("Press Ctrl+C to stop the server\n")
        
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
        
    except ImportError:
        logger.error("uvicorn is required to run the API server. Install it with: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("API server stopped by user")


if __name__ == "__main__":
    main()