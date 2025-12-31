"""Utility functions for the API module."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
import joblib
from pathlib import Path
import json

from ..utils.logger import logger
from ..utils.constants import MODELS_PATH, NUMERICAL_FEATURES
from ..features.scaler import FeatureScaler
from ..features.pca_transformer import PCATransformer


class PredictionRequest(BaseModel):
    """Request model for fraud prediction."""
    
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Validate that all required features are present."""
        required_features = NUMERICAL_FEATURES
        missing_features = set(required_features) - set(v.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        return v


class PredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    
    prediction: int = Field(..., description="Predicted class (0: normal, 1: fraud)")
    probability: float = Field(..., description="Probability of fraud")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    risk_score: float = Field(..., description="Risk score (0-100)")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    transactions: List[Dict[str, float]] = Field(..., description="List of transactions")
    
    @field_validator('transactions')
    @classmethod
    def validate_transactions(cls, v):
        """Validate that all transactions have required features."""
        if not v:
            raise ValueError("At least one transaction is required")
        
        required_features = NUMERICAL_FEATURES
        for i, transaction in enumerate(v):
            missing_features = set(required_features) - set(transaction.keys())
            if missing_features:
                raise ValueError(f"Transaction {i}: Missing required features: {missing_features}")
        return v


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str
    model_version: str
    model_type: str
    features_count: int
    training_date: str
    performance_metrics: Dict[str, float]


def preprocess_input(
    features: Dict[str, float],
    scaler: Optional[FeatureScaler] = None,
    pca_transformer: Optional[PCATransformer] = None
) -> pd.DataFrame:
    """
    Preprocess input features for prediction.
    
    Args:
        features: Input features dictionary
        scaler: Feature scaler (optional)
        pca_transformer: PCA transformer (optional)
        
    Returns:
        Preprocessed features DataFrame
    """
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Ensure all required features are present with correct order
    for feature in NUMERICAL_FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Reorder columns to match training data
    df = df[NUMERICAL_FEATURES]
    
    # Apply scaling if available
    if scaler is not None:
        df = scaler.transform(df)
        logger.debug("Applied feature scaling")
    
    # Apply PCA if available
    if pca_transformer is not None:
        df = pca_transformer.transform(df)
        logger.debug("Applied PCA transformation")
    
    return df


def postprocess_prediction(
    prediction: np.ndarray,
    probability: np.ndarray
) -> PredictionResponse:
    """
    Postprocess model prediction into response format.
    
    Args:
        prediction: Model prediction (0 or 1)
        probability: Prediction probabilities
        
    Returns:
        Formatted prediction response
    """
    pred_class = int(prediction[0])
    fraud_prob = float(probability[0][1]) if len(probability[0]) > 1 else float(probability[0])
    
    # Determine confidence level
    if fraud_prob < 0.3:
        confidence = "low"
    elif fraud_prob < 0.7:
        confidence = "medium"
    else:
        confidence = "high"
    
    # Calculate risk score (0-100)
    risk_score = fraud_prob * 100
    
    return PredictionResponse(
        prediction=pred_class,
        probability=fraud_prob,
        confidence=confidence,
        risk_score=risk_score
    )


def load_preprocessing_artifacts() -> Tuple[Optional[FeatureScaler], Optional[PCATransformer]]:
    """
    Load preprocessing artifacts (scaler, PCA transformer).
    
    Returns:
        Tuple of scaler and PCA transformer (None if not found)
    """
    scaler = None
    pca_transformer = None
    
    # Try to load scaler
    try:
        scaler = FeatureScaler.load("feature_scaler")
        logger.info("Feature scaler loaded successfully")
    except FileNotFoundError:
        logger.warning("Feature scaler not found, skipping scaling")
    
    # Try to load PCA transformer
    try:
        pca_transformer = PCATransformer.load("pca_transformer")
        logger.info("PCA transformer loaded successfully")
    except FileNotFoundError:
        logger.info("PCA transformer not found, skipping PCA")
    
    return scaler, pca_transformer


def validate_model_health(model: Any) -> Dict[str, Any]:
    """
    Validate model health and readiness.
    
    Args:
        model: Loaded model object
        
    Returns:
        Health check results
    """
    health_status = {
        "status": "healthy",
        "checks": {},
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Check if model has required methods
    required_methods = ["predict", "predict_proba"]
    for method in required_methods:
        health_status["checks"][f"has_{method}"] = hasattr(model, method)
        if not hasattr(model, method):
            health_status["status"] = "unhealthy"
    
    # Test prediction with dummy data
    try:
        dummy_features = {feature: 0.0 for feature in NUMERICAL_FEATURES}
        dummy_df = pd.DataFrame([dummy_features])
        
        prediction = model.predict(dummy_df)
        probability = model.predict_proba(dummy_df)
        
        health_status["checks"]["can_predict"] = True
        health_status["checks"]["prediction_shape"] = prediction.shape
        health_status["checks"]["probability_shape"] = probability.shape
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["can_predict"] = False
        health_status["checks"]["prediction_error"] = str(e)
        logger.error(f"Model health check failed: {e}")
    
    return health_status


def create_model_info(model: Any, model_metadata: Dict[str, Any]) -> ModelInfo:
    """
    Create model information response.
    
    Args:
        model: Loaded model object
        model_metadata: Model metadata
        
    Returns:
        Model information
    """
    return ModelInfo(
        model_name=model_metadata.get("model_name", "credit_card_fraud_detector"),
        model_version=model_metadata.get("model_version", "1.0.0"),
        model_type=model_metadata.get("model_type", "unknown"),
        features_count=len(NUMERICAL_FEATURES),
        training_date=model_metadata.get("training_date", "unknown"),
        performance_metrics=model_metadata.get("performance_metrics", {})
    )


def log_prediction_request(
    request_id: str,
    features: Dict[str, float],
    prediction: PredictionResponse,
    processing_time: float
) -> None:
    """
    Log prediction request for monitoring and auditing.
    
    Args:
        request_id: Unique request identifier
        features: Input features
        prediction: Prediction response
        processing_time: Time taken for prediction
    """
    log_entry = {
        "request_id": request_id,
        "timestamp": pd.Timestamp.now().isoformat(),
        "prediction": prediction.prediction,
        "probability": prediction.probability,
        "risk_score": prediction.risk_score,
        "processing_time_ms": processing_time * 1000,
        "feature_count": len(features)
    }
    
    logger.info(f"Prediction logged: {json.dumps(log_entry)}")


class APIError(Exception):
    """Custom API error class."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)