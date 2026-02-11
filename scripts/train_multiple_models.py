#!/usr/bin/env python3
"""Train multiple model types for fraud detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mlflow

from src.features.scaler import FeatureScaler
from src.data.loader import load_data
from src.utils.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

def train_model_type(model_type, data):
    """Train a specific model type with SMOTE and proper evaluation."""
    from src.models.train import train_and_evaluate_model
    
    print(f"\nTraining IMPROVED {model_type.upper()} with SMOTE...")
    
    try:
        model, scaler, metrics = train_and_evaluate_model(
            data=data,
            model_type=model_type,
            test_size=0.2,
            use_smote=True,
            log_mlflow=True
        )
        
        print(f"SUCCESS {model_type.upper()} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['fraud_f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        print(f"  Fraud Detection: {metrics['fraud_detected']} ({metrics['fraud_recall']*100:.1f}%)")
        
        return model
        
    except Exception as e:
        print(f"ERROR {model_type.upper()} failed: {e}")
        return None

def main():
    """Train multiple model types."""
    
    print("Training Multiple Fraud Detection Models")
    print("=" * 60)
    
    # Load real data
    data_path = "data/raw/creditcard.csv"
    data = load_data(data_path)
    
    # Drop Time column if exists
    if 'Time' in data.columns:
        data = data.drop('Time', axis=1)
    
    fraud_count = int(data['Class'].sum())
    fraud_rate = fraud_count / len(data) * 100
    
    print(f"Dataset: {len(data)} samples (REAL DATA)")
    print(f"Fraud cases: {fraud_count} ({fraud_rate:.2f}%)")
    print(f"Normal cases: {len(data) - fraud_count} ({100-fraud_rate:.2f}%)")
    
    # Model types to train with SMOTE
    model_types = ["random_forest", "logistic_regression", "xgboost", "svm"]
    
    results = {}
    
    for model_type in model_types:
        try:
            model = train_model_type(model_type, data)
            if model:
                results[model_type] = "SUCCESS"
            else:
                results[model_type] = "SKIPPED"
        except Exception as e:
            print(f"ERROR {model_type.upper()} failed: {e}")
            results[model_type] = f"FAILED: {e}"
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY (REAL DATA)")
    print("=" * 60)
    for model_type, status in results.items():
        print(f"{model_type.upper():<20}: {status}")
    print("=" * 60)
    print(f"Dataset: {len(data)} real transactions")
    print("Models trained with:")
    print("+ SMOTE oversampling")
    print("+ Class weight balancing")
    print("+ Optimized hyperparameters")
    print("+ Feature scaling")
    print("+ Comprehensive evaluation")

if __name__ == "__main__":
    main()