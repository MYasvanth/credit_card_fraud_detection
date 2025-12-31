"""Simple model serving without MLflow registry."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Load model and scaler
model = joblib.load("models/mlflow_fraud_model.pkl")
scaler = joblib.load("models/mlflow_fraud_scaler.pkl")

app = FastAPI(title="Fraud Detection API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Validate input
        if len(request.features) != 29:
            raise HTTPException(status_code=400, detail="Expected 29 features")
        
        # Prepare data
        X = np.array(request.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "low"
        elif probability < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Fraud Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)