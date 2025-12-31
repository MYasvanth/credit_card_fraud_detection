"""Simple working BentoML service."""

from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
try:
    model = joblib.load("models/fraud_detection_model.pkl")
    scaler = joblib.load("models/fraud_detection_scaler.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return jsonify({
        "service": "BentoML Fraud Detection",
        "status": "running",
        "endpoints": ["/health", "/predict"]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    features = data["features"]
    
    df = pd.DataFrame([features])
    df_scaled = scaler.transform(df)
    
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    return jsonify({
        "prediction": int(prediction),
        "fraud_probability": float(probability[1]),
        "risk_score": float(probability[1] * 100)
    })

if __name__ == '__main__':
    print("Starting Simple BentoML Service...")
    print("URL: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)