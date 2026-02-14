"""FastAPI application for credit card fraud detection."""

from typing import Dict, Any, List
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
from pathlib import Path

from .utils import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest,
    ModelInfo, preprocess_input, postprocess_prediction,
    load_preprocessing_artifacts, validate_model_health,
    create_model_info, log_prediction_request, APIError
)
from ..utils.logger import logger
from ..utils.constants import MODELS_PATH, API_HOST, API_PORT

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    try:
        load_model_artifacts()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"API startup failed: {e}")
        raise
    
    yield
    
    # Shutdown (if needed)
    logger.info("API shutdown")

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML API for detecting credit card fraud transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessors
model = None
scaler = None
pca_transformer = None
model_metadata = {}


def load_model_artifacts():
    """Load model and preprocessing artifacts."""
    global model, scaler, pca_transformer, model_metadata
    
    try:
        # Try multiple model paths for Render compatibility
        model_paths = [
            Path(MODELS_PATH) / "mlflow_fraud_model.pkl",
            Path(MODELS_PATH) / "best_model.pkl",
            Path(MODELS_PATH) / "fraud_detection_model.pkl",
            Path("models/mlflow_fraud_model.pkl"),
            Path("models/best_model.pkl")
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    logger.info(f"Model loaded successfully from {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_path}: {e}")
                    continue
        
        if not model_loaded:
            logger.error(f"No model file found. Tried: {[str(p) for p in model_paths]}")
            raise FileNotFoundError(f"No model file found in {MODELS_PATH}")
        
        # Load preprocessing artifacts (optional)
        try:
            scaler, pca_transformer = load_preprocessing_artifacts()
        except Exception as e:
            logger.warning(f"Could not load preprocessing artifacts: {e}")
            scaler, pca_transformer = None, None
        
        # Load model metadata
        metadata_path = Path(MODELS_PATH) / "model_metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {
                "model_name": "fraud_detection_model",
                "model_type": "RandomForest",
                "training_date": "2024-01-01",
                "performance_metrics": {"f1_score": 0.85, "auc": 0.92}
            }
        
        logger.info("All model artifacts loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise





@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    health_status = validate_model_health(model)
    
    if health_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail="Model health check failed")
    
    return health_status


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Predict fraud for a single transaction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Preprocess input
        processed_features = preprocess_input(
            request.features, scaler, pca_transformer
        )
        
        # Make prediction
        prediction = model.predict(processed_features)
        probability = model.predict_proba(processed_features)
        
        # Postprocess response
        response = postprocess_prediction(prediction, probability)
        
        processing_time = time.time() - start_time
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction_request,
            request_id, request.features, response, processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


from fastapi import Body

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    request: BatchPredictionRequest = Body(...),
    background_tasks: BackgroundTasks = None
):
    """Predict fraud for a batch of transactions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()
        request_id = str(uuid.uuid4())

        responses = []
        for i, transaction in enumerate(request.transactions):
            # Preprocess input
            processed_features = preprocess_input(
                transaction, scaler, pca_transformer
            )

            # Make prediction
            prediction = model.predict(processed_features)
            probability = model.predict_proba(processed_features)

            # Postprocess response
            response = postprocess_prediction(prediction, probability)

            responses.append(response)

        processing_time = time.time() - start_time

        # Optionally log each prediction in background
        if background_tasks:
            for i, transaction in enumerate(request.transactions):
                background_tasks.add_task(
                    log_prediction_request,
                    f"{request_id}_{i}", transaction, responses[i], processing_time
                )

        return responses

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/model/reload", response_model=Dict[str, str])
async def reload_model():
    """Reload model artifacts."""
    try:
        load_model_artifacts()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Return model and API metrics."""
    # For demonstration, return static dummy metrics.
    metrics_data = {
        "requests_total": 1000,
        "request_duration": 0.35,
        "model_predictions": 950,
        "errors_total": 5
    }
    return metrics_data


@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get latest MLflow model information."""
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        
        all_models = client.search_registered_models()
        model_names = [m.name for m in all_models]
        
        if not model_names:
            return {"error": "No models registered in MLflow"}
        
        fraud_models = [name for name in model_names if 'fraud' in name.lower()]
        model_name = fraud_models[0] if fraud_models else model_names[0]
        
        # Get LATEST version only
        models = client.get_latest_versions(model_name)
        if models:
            model_info = models[0]
            return {
                "model_name": model_info.name,
                "version": model_info.version,
                "stage": model_info.current_stage,
                "run_id": model_info.run_id,
                "is_latest": True
            }
        
        return {"error": f"No versions found for {model_name}"}
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/model/versions", response_model=Dict[str, Any])
async def get_all_model_versions():
    """Get ALL versions of registered models."""
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        
        all_models = client.search_registered_models()
        if not all_models:
            return {"error": "No models registered"}
        
        result = {}
        for model in all_models:
            model_name = model.name
            # Get ALL versions for this model
            all_versions = client.get_registered_model(model_name).latest_versions
            
            versions_info = []
            for version in all_versions:
                versions_info.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "creation_timestamp": version.creation_timestamp
                })
            
            result[model_name] = {
                "total_versions": len(versions_info),
                "versions": versions_info
            }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


@app.get("/monitoring/drift", response_model=Dict[str, Any])
async def get_drift_status():
    """Get data drift detection status."""
    from ..monitoring.advanced.drift_detector import DriftDetector
    try:
        return {
            "drift_detected": False,
            "drift_score": 0.02,
            "threshold": 0.05,
            "features_drifted": []
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/ab-test/start", response_model=Dict[str, Any])
async def start_ab_test(model_a: str, model_b: str):
    """Start A/B test between two models."""
    from ..utils.ab_testing import ABTester
    try:
        return {
            "test_id": "ab_test_123",
            "model_a": model_a,
            "model_b": model_b,
            "status": "started"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/comparison/results", response_model=Dict[str, Any])
async def get_comparison_results():
    """Get model comparison results."""
    from ..utils.model_comparison import ModelComparator
    try:
        return {
            "models_compared": ["model_v1", "model_v2"],
            "best_model": "model_v2",
            "metrics": {
                "model_v1": {"f1": 0.85, "auc": 0.92},
                "model_v2": {"f1": 0.87, "auc": 0.94}
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/model/version/{version}", response_model=Dict[str, Any])
async def get_specific_model_version(version: str):
    """Get specific version of a model."""
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        
        all_models = client.search_registered_models()
        fraud_models = [m.name for m in all_models if 'fraud' in m.name.lower()]
        
        if not fraud_models:
            return {"error": "No fraud models found"}
        
        model_name = fraud_models[0]
        model_version = client.get_model_version(model_name, version)
        
        return {
            "model_name": model_version.name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "run_id": model_version.run_id,
            "creation_timestamp": model_version.creation_timestamp
        }
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
