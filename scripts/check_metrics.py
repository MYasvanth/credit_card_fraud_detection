"""Quick model evaluation metrics checker."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json

def check_model_metrics():
    """Check evaluation metrics for trained models."""
    
    print("="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    
    # Load model metadata
    metadata_path = "models/model_metadata.json"
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("SAVED MODEL METADATA:")
        print(f"  Model Type: {metadata.get('model_type', 'Unknown')}")
        print(f"  Features: {metadata.get('n_features', 'Unknown')}")
        print(f"  Training Samples: {metadata.get('n_samples', 'Unknown')}")
        print(f"  Test Accuracy: {metadata.get('test_accuracy', 0):.4f}")
        print(f"  Test F1: {metadata.get('test_f1', 0):.4f}")
        print(f"  Test Precision: {metadata.get('test_precision', 0):.4f}")
        print(f"  Test Recall: {metadata.get('test_recall', 0):.4f}")
        print()
    
    # Test models on new data
    models_to_test = {
        "Basic Model": "models/fraud_detection_model.pkl",
        "MLflow Model": "models/mlflow_fraud_model.pkl"
    }
    
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    X_test = np.random.randn(n_samples, 29)
    y_test = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    print("LIVE MODEL EVALUATION:")
    print(f"Test Dataset: {n_samples} samples, {sum(y_test)} fraud cases ({sum(y_test)/len(y_test)*100:.1f}%)")
    print("-" * 60)
    
    for name, model_path in models_to_test.items():
        if Path(model_path).exists():
            try:
                model = joblib.load(model_path)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                
                print(f"{name}:")
                print(f"  Accuracy: {report['accuracy']:.4f}")
                
                if '1' in report:
                    print(f"  Fraud Precision: {report['1']['precision']:.4f}")
                    print(f"  Fraud Recall: {report['1']['recall']:.4f}")
                    print(f"  Fraud F1-Score: {report['1']['f1-score']:.4f}")
                else:
                    print(f"  Fraud Precision: 0.0000 (No fraud predicted)")
                    print(f"  Fraud Recall: 0.0000 (No fraud predicted)")
                    print(f"  Fraud F1-Score: 0.0000 (No fraud predicted)")
                
                print(f"  ROC AUC: {auc:.4f}")
                
                # Confusion Matrix
                tn, fp, fn, tp = cm.ravel()
                print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
                
                # Fraud Detection Stats
                fraud_detected = tp
                fraud_missed = fn
                false_alarms = fp
                
                print(f"  Fraud Cases Detected: {fraud_detected}/{sum(y_test)} ({fraud_detected/max(sum(y_test),1)*100:.1f}%)")
                print(f"  False Alarms: {false_alarms} ({false_alarms/len(y_test)*100:.2f}%)")
                print()
                
            except Exception as e:
                print(f"{name}: Error loading model - {e}")
                print()
        else:
            print(f"{name}: Model file not found - {model_path}")
            print()
    
    print("="*60)
    print("EVALUATION SUMMARY:")
    print("- Accuracy: Overall correctness")
    print("- Precision: Of predicted fraud, how many were actually fraud")
    print("- Recall: Of actual fraud, how many were detected")
    print("- F1-Score: Harmonic mean of precision and recall")
    print("- ROC AUC: Area under receiver operating characteristic curve")
    print("="*60)

if __name__ == "__main__":
    check_model_metrics()