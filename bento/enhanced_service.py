"""Enhanced BentoML service with advanced features."""

import bentoml
import pandas as pd
import numpy as np
from typing import Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.utils.constants import NUMERICAL_FEATURES
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.drift_detector import DriftDetector

# Load models
fraud_model = bentoml.sklearn.get("fraud_detection_model:latest")
baseline_model = bentoml.sklearn.get("fraud_detection_baseline:latest")

# Initialize monitoring
performance_tracker = PerformanceTracker({"f1_score": 0.85, "precision": 0.80})
drift_detector = None  # Initialize with reference data

svc = bentoml.Service("enhanced_fraud_detection")

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
async def predict_with_monitoring(input_data: Dict) -> Dict:
    """Enhanced prediction with monitoring."""
    features = input_data["features"]
    
    # Preprocess
    df = pd.DataFrame([features])
    for feature in NUMERICAL_FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0
    df = df[NUMERICAL_FEATURES]
    
    # Predict with main model
    prediction = await asyncio.get_event_loop().run_in_executor(
        None, fraud_model.run, df
    )
    probability = await asyncio.get_event_loop().run_in_executor(
        None, lambda: fraud_model.run_batch(df, method="predict_proba")[0]
    )
    
    # A/B test with baseline model
    baseline_pred = await asyncio.get_event_loop().run_in_executor(
        None, baseline_model.run, df
    )
    
    result = {
        "prediction": int(prediction[0]),
        "probability": float(probability[1]),
        "risk_score": float(probability[1] * 100),
        "model_version": "latest",
        "baseline_prediction": int(baseline_pred[0]),
        "confidence": "high" if probability[1] > 0.7 else "medium" if probability[1] > 0.3 else "low"
    }
    
    # Log for monitoring
    if drift_detector:
        drift_score = drift_detector.detect_feature_drift(df)
        result["drift_detected"] = any(d["drift_detected"] for d in drift_score.values())
    
    return result

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def batch_predict_optimized(input_data: Dict) -> List[Dict]:
    """Optimized batch prediction."""
    transactions = input_data["transactions"]
    
    # Batch preprocessing
    df_list = []
    for transaction in transactions:
        df = pd.DataFrame([transaction])
        for feature in NUMERICAL_FEATURES:
            if feature not in df.columns:
                df[feature] = 0.0
        df_list.append(df[NUMERICAL_FEATURES])
    
    batch_df = pd.concat(df_list, ignore_index=True)
    
    # Batch prediction
    predictions = fraud_model.run_batch(batch_df)
    probabilities = fraud_model.run_batch(batch_df, method="predict_proba")
    
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            "transaction_id": i,
            "prediction": int(pred),
            "probability": float(prob[1]),
            "risk_score": float(prob[1] * 100)
        })
    
    return results

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
def model_comparison(input_data: Dict) -> Dict:
    """Compare multiple model versions."""
    features = input_data["features"]
    df = pd.DataFrame([features])
    
    # Get predictions from different models
    main_pred = fraud_model.run(df)[0]
    main_prob = fraud_model.run_batch(df, method="predict_proba")[0]
    
    baseline_pred = baseline_model.run(df)[0]
    baseline_prob = baseline_model.run_batch(df, method="predict_proba")[0]
    
    return {
        "main_model": {
            "prediction": int(main_pred),
            "probability": float(main_prob[1]),
            "version": "latest"
        },
        "baseline_model": {
            "prediction": int(baseline_pred),
            "probability": float(baseline_prob[1]),
            "version": "baseline"
        },
        "agreement": main_pred == baseline_pred,
        "probability_diff": abs(float(main_prob[1]) - float(baseline_prob[1]))
    }