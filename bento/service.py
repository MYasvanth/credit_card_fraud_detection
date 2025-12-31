"""BentoML service for credit card fraud detection."""

import bentoml
import numpy as np
import pandas as pd
from typing import Dict, List
import joblib
from pathlib import Path

from src.features.scaler import FeatureScaler
from src.features.pca_transformer import PCATransformer
from src.utils.constants import NUMERICAL_FEATURES, MODELS_PATH
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.drift_detector import DriftDetector

# Load model and artifacts
model = bentoml.sklearn.get("fraud_detection_model:latest")
scaler_path = Path(MODELS_PATH) / "feature_scaler.pkl"
scaler = joblib.load(scaler_path) if scaler_path.exists() else None

svc = bentoml.Service("fraud_detection_service", runners=[model])

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def predict(input_data: Dict) -> Dict:
    """Predict fraud for single transaction."""
    # Preprocess input
    df = pd.DataFrame([input_data["features"]])
    
    # Ensure all features present
    for feature in NUMERICAL_FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0
    
    df = df[NUMERICAL_FEATURES]
    
    # Apply scaling if available
    if scaler:
        df = scaler.transform(df)
    
    # Make prediction
    prediction = model.run(df)
    probability = model.run_batch(df, method="predict_proba")[0]
    
    fraud_prob = float(probability[1])
    
    return {
        "prediction": int(prediction[0]),
        "probability": fraud_prob,
        "risk_score": fraud_prob * 100,
        "confidence": "high" if fraud_prob > 0.7 else "medium" if fraud_prob > 0.3 else "low"
    }

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def predict_batch(input_data: Dict) -> List[Dict]:
    """Predict fraud for multiple transactions."""
    transactions = input_data["transactions"]
    results = []
    
    for transaction in transactions:
        result = predict({"features": transaction})
        results.append(result)
    
    return results

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def health() -> Dict:
    """Health check endpoint."""
    try:
        # Test prediction with dummy data
        dummy_features = {feature: 0.0 for feature in NUMERICAL_FEATURES}
        test_result = predict({"features": dummy_features})
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "test_prediction": test_result["prediction"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }