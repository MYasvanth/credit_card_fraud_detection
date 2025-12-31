"""Simplified BentoML service for fraud detection."""

import bentoml
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/fraud_detection_model.pkl")
scaler = joblib.load("models/fraud_detection_scaler.pkl")

# Create BentoML service
svc = bentoml.Service("fraud_detection")

@bentoml.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def predict(input_data):
    """Predict fraud probability."""
    features = input_data["features"]
    
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

@bentoml.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def health():
    """Health check."""
    return {"status": "healthy", "model": "fraud_detection_model"}