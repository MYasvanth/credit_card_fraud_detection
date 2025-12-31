"""Enhanced BentoML-style service with production features."""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
import time
from functools import wraps
from collections import defaultdict
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics tracking
metrics = defaultdict(int)
response_times = []
metrics_lock = threading.Lock()

# Load model and scaler
try:
    model = joblib.load("models/fraud_detection_model.pkl")
    scaler = joblib.load("models/fraud_detection_scaler.pkl")
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    scaler = None

app = Flask(__name__)

def track_metrics(func):
    """Decorator to track API metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            with metrics_lock:
                metrics['requests_total'] += 1
                metrics['requests_success'] += 1
                response_times.append(time.time() - start_time)
            return result
        except Exception as e:
            with metrics_lock:
                metrics['requests_total'] += 1
                metrics['requests_error'] += 1
            raise
    return wrapper

def validate_features(features):
    """Validate input features."""
    required_features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    missing = [f for f in required_features if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing[:5]}...")
    return True

@app.route('/predict', methods=['POST'])
@track_metrics
def predict():
    """Predict fraud probability."""
    try:
        data = request.get_json()
        features = data["features"]
        
        # Validate features
        validate_features(features)
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0]
        
        result = {
            "prediction": int(prediction),
            "fraud_probability": float(probability[1]),
            "risk_score": float(probability[1] * 100),
            "confidence": "high" if probability[1] > 0.7 else "medium" if probability[1] > 0.3 else "low"
        }
        
        logger.info(f"Prediction made: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    status = "healthy" if model is not None else "unhealthy"
    with metrics_lock:
        avg_response_time = sum(response_times[-100:]) / len(response_times[-100:]) if response_times else 0
    
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "service": "fraud_detection_bento",
        "metrics": {
            "total_requests": metrics['requests_total'],
            "success_rate": metrics['requests_success'] / max(metrics['requests_total'], 1),
            "avg_response_time_ms": avg_response_time * 1000
        }
    })

@app.route('/batch_predict', methods=['POST'])
@track_metrics
def batch_predict():
    """Batch prediction."""
    try:
        data = request.get_json()
        transactions = data["transactions"]
        results = []
        
        for transaction in transactions:
            df = pd.DataFrame([transaction])
            df_scaled = scaler.transform(df)
            prediction = model.predict(df_scaled)[0]
            probability = model.predict_proba(df_scaled)[0]
            
            results.append({
                "prediction": int(prediction),
                "fraud_probability": float(probability[1]),
                "risk_score": float(probability[1] * 100)
            })
        
        logger.info(f"Batch prediction completed for {len(transactions)} transactions")
        return jsonify({"predictions": results, "count": len(results)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get service metrics."""
    with metrics_lock:
        return jsonify(dict(metrics))

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    try:
        import json
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    except:
        return jsonify({"model": "fraud_detection", "version": "1.0"})

if __name__ == '__main__':
    print("Starting Enhanced BentoML-style Fraud Detection Service...")
    print("Endpoints:")
    print("  POST /predict - Single prediction")
    print("  POST /batch_predict - Batch predictions") 
    print("  GET /health - Health check with metrics")
    print("  GET /metrics - Service metrics")
    print("  GET /model/info - Model information")
    print("Service running on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)