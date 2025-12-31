#!/usr/bin/env python3
"""Minimal working example without pandas/numpy issues."""

import sys
import os
import json
from pathlib import Path

# Create minimal model files
def create_minimal_setup():
    """Create minimal model files for testing components."""
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/inference_samples").mkdir(parents=True, exist_ok=True)
    
    # Create dummy model metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_samples": 2000,
        "n_features": 29,
        "test_accuracy": 0.9850,
        "fraud_f1": 0.8234,
        "fraud_precision": 0.8567,
        "fraud_recall": 0.7923
    }
    
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create sample predictions for monitoring
    predictions_data = """prediction,true_label,confidence
0,0,0.95
1,1,0.87
0,0,0.92
1,0,0.73
0,0,0.98
1,1,0.89
0,0,0.94
0,1,0.65
1,1,0.91
0,0,0.97"""
    
    with open("data/inference_samples/sample_predictions.csv", "w") as f:
        f.write(predictions_data)
    
    print("✅ Minimal setup created successfully!")
    print("\nNow you can run:")
    print("1. Model Serving: python run_components.py serve")
    print("2. A/B Testing: python run_components.py ab-test --model-a model1 --model-b model2")
    print("3. Model Comparison: python run_components.py compare --models model1 model2")
    print("4. Automated Retraining: python run_components.py retrain --current-f1 0.82 --baseline-f1 0.85")

if __name__ == "__main__":
    create_minimal_setup()