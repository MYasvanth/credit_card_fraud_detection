"""Simple A/B test without command line arguments."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score

def simple_ab_test():
    """Run simple A/B test between two models."""
    
    print("Starting simple A/B test...")
    
    # Check if models exist
    model_a_path = "models/fraud_detection_model.pkl"
    model_b_path = "models/mlflow_fraud_model.pkl"
    
    if not Path(model_a_path).exists():
        print(f"Model A not found: {model_a_path}")
        return
    
    if not Path(model_b_path).exists():
        print(f"Model B not found: {model_b_path}")
        return
    
    # Load models
    model_a = joblib.load(model_a_path)
    model_b = joblib.load(model_b_path)
    
    # Load actual credit card data
    data = pd.read_csv('data/raw/creditcard.csv')
    
    # Remove Time column as it's not used in training
    X = data.drop(['Class', 'Time'], axis=1)
    y = data['Class']
    
    # Use last 1000 samples as test set
    X_test = X.tail(1000)
    y_test = y.tail(1000)
    
    print(f"Test set: {len(X_test)} samples, {y_test.sum()} fraud cases")
    
    # Run predictions
    pred_a = model_a.predict(X_test)
    pred_b = model_b.predict(X_test)
    
    # Calculate metrics
    acc_a = accuracy_score(y_test, pred_a)
    acc_b = accuracy_score(y_test, pred_b)
    f1_a = f1_score(y_test, pred_a, average='weighted')
    f1_b = f1_score(y_test, pred_b, average='weighted')
    
    print("\n" + "="*50)
    print("A/B TEST RESULTS")
    print("="*50)
    print(f"Model A (Basic): Accuracy {acc_a:.4f}, F1 {f1_a:.4f}")
    print(f"Model B (MLflow): Accuracy {acc_b:.4f}, F1 {f1_b:.4f}")
    
    winner = "A" if f1_a > f1_b else "B"
    improvement = abs(f1_a - f1_b) / min(f1_a, f1_b) * 100
    
    print(f"Winner: Model {winner}")
    print(f"Improvement: {improvement:.2f}%")
    print("="*50)

if __name__ == "__main__":
    simple_ab_test()