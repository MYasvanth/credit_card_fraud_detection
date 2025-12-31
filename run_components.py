#!/usr/bin/env python3
"""
Unified runner for Credit Card Fraud Detection ML Components
Supports: Model Serving, A/B Testing, Model Comparison, and Automated Retraining
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger

def run_model_serving(args):
    """Start model serving API."""
    logger.info("Starting model serving...")
    
    cmd = [
        sys.executable, "scripts/serve_model.py",
        "--model_name", args.model_name,
        "--stage", args.stage,
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    print(f"Starting FastAPI server: {args.model_name} ({args.stage})")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print("Endpoints:")
    print(f"  - Health: http://{args.host}:{args.port}/health")
    print(f"  - Predict: http://{args.host}:{args.port}/predict")
    print(f"  - Batch Predict: http://{args.host}:{args.port}/predict/batch")
    print(f"  - API Docs: http://{args.host}:{args.port}/docs")
    
    subprocess.run(cmd)

def run_ab_testing(args):
    """Run A/B testing between two models."""
    logger.info(f"Running A/B test: {args.model_a} vs {args.model_b}")
    
    cmd = [
        sys.executable, "scripts/run_ab_test.py",
        "--model-a", args.model_a,
        "--model-b", args.model_b,
        "--test-data", args.test_data,
        "--traffic-split", str(args.traffic_split),
        "--n-simulations", str(args.n_simulations)
    ]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    subprocess.run(cmd)

def run_model_comparison(args):
    """Run model comparison analysis."""
    logger.info(f"Comparing models: {', '.join(args.models)}")
    
    # Create comparison script
    comparison_script = f"""
import sys
sys.path.insert(0, '{project_root}')

from src.utils.model_comparison import compare_models_from_registry
from src.data.loader import load_data
import pandas as pd

# Load test data
test_data = pd.read_csv('{args.test_data}')

# Compare models
comparator = compare_models_from_registry(
    {args.models}, 
    test_data, 
    target_column='Class'
)

# Evaluate models
results = comparator.evaluate_all_models()

# Generate report
report = comparator.generate_comparison_report()

# Generate plots
plots = comparator.plot_comparison_charts(save_path='{args.output_dir}')

# Log to MLflow
run_id = comparator.log_comparison_to_mlflow()

print("\\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
print(f"Models compared: {len(report['metrics_comparison'])} models")
print(f"Test samples: {{report['test_data_info']['n_samples']}}")
print(f"Best overall model: {{report['best_models']['overall']}}")
print(f"Best F1 model: {{report['best_models']['by_f1']}}")
print(f"Best AUC model: {{report['best_models']['by_auc']}}")
print(f"MLflow run ID: {{run_id}}")
print(f"Plots saved to: {args.output_dir}")
print("="*60)

# Print detailed metrics
import pandas as pd
metrics_df = pd.DataFrame(report['metrics_comparison'])
print("\\nDetailed Metrics:")
print(metrics_df.round(4))
"""
    
    # Write and execute script
    script_path = project_root / "temp_comparison.py"
    with open(script_path, 'w') as f:
        f.write(comparison_script)
    
    try:
        subprocess.run([sys.executable, str(script_path)])
    finally:
        if script_path.exists():
            script_path.unlink()

def run_automated_retraining(args):
    """Run automated retraining pipeline."""
    logger.info("Starting automated retraining pipeline...")
    
    # Create retraining script
    retraining_script = f"""
import sys
sys.path.insert(0, '{project_root}')

from src.deployment.runner import PipelineRunner
from src.utils.logger import logger
import json

# Initialize runner
runner = PipelineRunner()

# Simulate monitoring results that trigger retraining
monitoring_results = {{
    'performance_metrics': {{
        'f1_score': {args.current_f1},
        'precision': 0.85,
        'recall': 0.80
    }},
    'drift_results': {{
        'drift_detected': {str(args.drift_detected).lower()},
        'drifted_features': ['V1', 'V2'] if {str(args.drift_detected).lower()} else []
    }},
    'baseline_f1': {args.baseline_f1},
    'performance_degraded': {str(args.current_f1 < args.baseline_f1).lower()}
}}

print("\\n" + "="*60)
print("AUTOMATED RETRAINING PIPELINE")
print("="*60)
print(f"Current F1 Score: {args.current_f1}")
print(f"Baseline F1 Score: {args.baseline_f1}")
print(f"Performance Degraded: {args.current_f1 < args.baseline_f1}")
print(f"Data Drift Detected: {args.drift_detected}")

# Determine if retraining should be triggered
should_retrain = (
    {args.current_f1} < {args.baseline_f1} or 
    {str(args.drift_detected).lower()}
)

print(f"Retraining Decision: {{'RETRAIN' if should_retrain else 'NO ACTION'}}")

if should_retrain:
    print("\\nTriggering retraining...")
    
    # Run training pipeline
    results = runner.run_training_pipeline(
        data_path='{args.data_path}',
        quick_mode={str(args.quick).lower()}
    )
    
    print("\\nRetraining Results:")
    if 'registration_results' in results:
        reg_results = results['registration_results']
        print(f"  New Model URI: {{reg_results.get('model_uri', 'N/A')}}")
        print(f"  MLflow Run ID: {{reg_results.get('run_id', 'N/A')}}")
        
        if 'test_metrics' in reg_results:
            test_metrics = reg_results['test_metrics']
            if 'classification_report' in test_metrics:
                new_f1 = test_metrics['classification_report']['1']['f1-score']
                print(f"  New F1 Score: {{new_f1:.4f}}")
                improvement = (new_f1 - {args.current_f1}) / {args.current_f1} * 100
                print(f"  Improvement: {{improvement:+.2f}}%")
    
    print("\\nRetraining completed successfully!")
else:
    print("\\nNo retraining needed. Model performance is acceptable.")

print("="*60)
"""
    
    # Write and execute script
    script_path = project_root / "temp_retraining.py"
    with open(script_path, 'w') as f:
        f.write(retraining_script)
    
    try:
        subprocess.run([sys.executable, str(script_path)])
    finally:
        if script_path.exists():
            script_path.unlink()

def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection ML Components Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='component', help='Component to run')
    
    # Model Serving
    serve_parser = subparsers.add_parser('serve', help='Start model serving API')
    serve_parser.add_argument('--model_name', default='credit_card_fraud_detector', help='Model name')
    serve_parser.add_argument('--stage', default='Production', help='Model stage')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port')
    
    # A/B Testing
    ab_parser = subparsers.add_parser('ab-test', help='Run A/B testing')
    ab_parser.add_argument('--model-a', required=True, help='Model A name')
    ab_parser.add_argument('--model-b', required=True, help='Model B name')
    ab_parser.add_argument('--test-data', default='data/raw/creditcard.csv', help='Test data path')
    ab_parser.add_argument('--traffic-split', type=float, default=0.5, help='Traffic split to model B')
    ab_parser.add_argument('--n-simulations', type=int, default=1000, help='Number of simulations')
    ab_parser.add_argument('--output', help='Output JSON file')
    
    # Model Comparison
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', nargs='+', required=True, help='Model names to compare')
    compare_parser.add_argument('--test-data', default='data/raw/creditcard.csv', help='Test data path')
    compare_parser.add_argument('--output-dir', default='reports/comparison', help='Output directory')
    
    # Automated Retraining
    retrain_parser = subparsers.add_parser('retrain', help='Run automated retraining')
    retrain_parser.add_argument('--current-f1', type=float, required=True, help='Current model F1 score')
    retrain_parser.add_argument('--baseline-f1', type=float, required=True, help='Baseline F1 score')
    retrain_parser.add_argument('--drift-detected', action='store_true', help='Data drift detected')
    retrain_parser.add_argument('--data-path', default='data/raw/creditcard.csv', help='Training data path')
    retrain_parser.add_argument('--quick', action='store_true', help='Quick training mode')
    
    args = parser.parse_args()
    
    if not args.component:
        parser.print_help()
        return
    
    # Ensure output directories exist
    os.makedirs('reports/comparison', exist_ok=True)
    
    try:
        if args.component == 'serve':
            run_model_serving(args)
        elif args.component == 'ab-test':
            run_ab_testing(args)
        elif args.component == 'compare':
            run_model_comparison(args)
        elif args.component == 'retrain':
            run_automated_retraining(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error running {args.component}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()