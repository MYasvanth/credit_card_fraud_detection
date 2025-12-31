#!/usr/bin/env python3
"""Enhanced MLflow dashboard and experiment management."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import mlflow
from src.utils.mlflow_manager import mlflow_manager
from src.utils.logger import logger

def show_experiments_summary():
    """Show summary of all experiments."""
    print("\n" + "="*80)
    print("MLFLOW EXPERIMENTS SUMMARY")
    print("="*80)
    
    # Use search_experiments() for MLflow 3.x compatibility
    experiments = mlflow.search_experiments()
    for exp in experiments:
        try:
            exp_name = exp.name
            # Pass experiment name directly to get_experiment_runs
            runs = mlflow_manager.get_experiment_runs(exp_name, max_results=10)
            print(f"\n📊 {exp_name.upper()}")
            print(f"   Total Runs: {len(runs)}")
            
            if runs:
                latest_run = runs[0]
                print(f"   Latest Run: {latest_run.get('run_id', 'N/A')[:8]}...")
                print(f"   Status: {latest_run.get('status', 'N/A')}")
                
                # Show key metrics if available
                # Fix: keys should not include 'metrics.' prefix in keys, adjust accordingly
                metrics = {k: v for k, v in latest_run.items() 
                          if k.startswith('metrics') and v is not None}
                if metrics:
                    print("   Key Metrics:")
                    for metric, value in list(metrics.items())[:3]:
                        metric_name = metric.replace('metrics.', '')
                        print(f"     {metric_name}: {value:.4f}")
        
        except Exception as e:
            print(f"\n❌ {exp_name}: Error - {e}")
    
    print("="*80)

def show_model_registry():
    """Show registered models and their versions."""
    print("\n" + "="*80)
    print("MODEL REGISTRY")
    print("="*80)
    
    try:
        models = mlflow_manager.client.search_registered_models()
        
        if not models:
            print("No registered models found.")
            return
        
        for model in models:
            print(f"\n🤖 {model.name}")
            print(f"   Description: {model.description or 'No description'}")
            
            # Get latest versions
            versions = mlflow_manager.client.get_latest_versions(model.name)
            
            for version in versions:
                print(f"   📦 Version {version.version} ({version.current_stage})")
                print(f"      Run ID: {version.run_id}")
                print(f"      Created: {version.creation_timestamp}")
                
                # Get run metrics
                try:
                    run = mlflow_manager.client.get_run(version.run_id)
                    key_metrics = ['accuracy', 'fraud_f1', 'fraud_precision', 'fraud_recall']
                    
                    for metric in key_metrics:
                        if metric in run.data.metrics:
                            print(f"      {metric}: {run.data.metrics[metric]:.4f}")
                
                except Exception as e:
                    print(f"      Error getting metrics: {e}")
    
    except Exception as e:
        print(f"Error accessing model registry: {e}")
    
    print("="*80)

def compare_model_stages():
    """Compare Production vs Staging models."""
    print("\n" + "="*80)
    print("PRODUCTION vs STAGING COMPARISON")
    print("="*80)
    
    try:
        models = mlflow_manager.client.list_registered_models()
        
        for model in models:
            print(f"\n🔄 {model.name}")
            
            comparison = mlflow_manager.compare_model_versions(
                model.name, 
                stages=["Production", "Staging"]
            )
            
            for stage, info in comparison.items():
                if "error" in info:
                    print(f"   {stage}: {info['error']}")
                else:
                    print(f"   {stage} (v{info['version']}):")
                    
                    # Show key metrics
                    metrics = info.get('metrics', {})
                    key_metrics = ['accuracy', 'fraud_f1', 'fraud_precision', 'fraud_recall']
                    
                    for metric in key_metrics:
                        if metric in metrics:
                            print(f"     {metric}: {metrics[metric]:.4f}")
    
    except Exception as e:
        print(f"Error comparing models: {e}")
    
    print("="*80)

def export_experiment_data(experiment_type: str, output_file: str):
    """Export experiment data to CSV."""
    try:
        mlflow_manager.export_experiment_data(experiment_type, output_file)
        print(f"✅ Exported {experiment_type} data to {output_file}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

def cleanup_experiments(experiment_type: str, keep_last_n: int):
    """Clean up old experiment runs."""
    try:
        mlflow_manager.cleanup_old_runs(experiment_type, keep_last_n)
        print(f"✅ Cleaned up {experiment_type} experiments, kept last {keep_last_n}")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

def show_model_performance_history(model_name: str):
    """Show performance history of a model."""
    print(f"\n" + "="*80)
    print(f"PERFORMANCE HISTORY: {model_name}")
    print("="*80)
    
    try:
        history = mlflow_manager.get_model_performance_history(model_name)
        
        if history.empty:
            print("No performance history found.")
            return
        
        # Show summary statistics
        print("\n📈 Performance Trends:")
        key_metrics = ['accuracy', 'fraud_f1', 'fraud_precision', 'fraud_recall']
        
        for metric in key_metrics:
            if metric in history.columns:
                values = history[metric].dropna()
                if not values.empty:
                    print(f"   {metric}:")
                    print(f"     Latest: {values.iloc[-1]:.4f}")
                    print(f"     Best: {values.max():.4f}")
                    print(f"     Trend: {'+' if values.iloc[-1] > values.iloc[0] else '-'}")
        
        # Show recent versions
        print(f"\n📦 Recent Versions:")
        recent = history.tail(5)[['version', 'stage', 'accuracy', 'fraud_f1']].round(4)
        print(recent.to_string(index=False))
    
    except Exception as e:
        print(f"Error getting performance history: {e}")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="MLflow Dashboard and Management")
    parser.add_argument('command', choices=[
        'summary', 'registry', 'compare', 'export', 'cleanup', 'history'
    ], help='Command to execute')
    
    parser.add_argument('--experiment-type', help='Experiment type for export/cleanup')
    parser.add_argument('--output-file', help='Output file for export')
    parser.add_argument('--keep-last-n', type=int, default=50, help='Number of runs to keep')
    parser.add_argument('--model-name', help='Model name for history')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'summary':
            show_experiments_summary()
        
        elif args.command == 'registry':
            show_model_registry()
        
        elif args.command == 'compare':
            compare_model_stages()
        
        elif args.command == 'export':
            if not args.experiment_type or not args.output_file:
                print("❌ Export requires --experiment-type and --output-file")
                return
            export_experiment_data(args.experiment_type, args.output_file)
        
        elif args.command == 'cleanup':
            if not args.experiment_type:
                print("❌ Cleanup requires --experiment-type")
                return
            cleanup_experiments(args.experiment_type, args.keep_last_n)
        
        elif args.command == 'history':
            if not args.model_name:
                print("❌ History requires --model-name")
                return
            show_model_performance_history(args.model_name)
    
    except Exception as e:
        logger.error(f"Dashboard command failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
