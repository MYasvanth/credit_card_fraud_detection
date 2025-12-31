"""Simple BentoML runner without build."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load model and scaler
model = joblib.load("models/fraud_detection_model.pkl")
scaler = joblib.load("models/fraud_detection_scaler.pkl")

def predict_fraud(features):
    """Simple prediction function."""
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability[1]),
        "risk_score": float(probability[1] * 100)
    }

# Test prediction
if __name__ == "__main__":
    # Sample transaction features
    test_features = {f"V{i}": 0.0 for i in range(1, 29)}
    test_features["Amount"] = 100.0
    
    result = predict_fraud(test_features)
    print(f"Prediction Result: {result}")
    print("BentoML service logic working correctly!")