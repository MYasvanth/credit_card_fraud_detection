"""Unit and integration tests for API endpoints."""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from src.api.main import app
from src.api.utils import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest,
    preprocess_input, postprocess_prediction, validate_model_health
)


class TestAPIUtils:
    """Test cases for API utility functions."""

    def test_prediction_request_validation_valid(self):
        """Test PredictionRequest with valid data."""
        features = {f"V{i}": float(i) for i in range(1, 29)}
        features["Amount"] = 100.0

        request = PredictionRequest(features=features)
        assert request.features == features

    def test_prediction_request_validation_missing_features(self):
        """Test PredictionRequest with missing features."""
        features = {"V1": 1.0, "V2": 2.0}  # Missing most features

        with pytest.raises(ValueError, match="Missing required features"):
            PredictionRequest(features=features)

    def test_batch_prediction_request_validation_valid(self):
        """Test BatchPredictionRequest with valid data."""
        features = {f"V{i}": float(i) for i in range(1, 29)}
        features["Amount"] = 100.0

        transactions = [features, features.copy()]
        request = BatchPredictionRequest(transactions=transactions)
        assert len(request.transactions) == 2

    def test_batch_prediction_request_validation_empty(self):
        """Test BatchPredictionRequest with empty transactions."""
        with pytest.raises(ValueError, match="At least one transaction is required"):
            BatchPredictionRequest(transactions=[])

    def test_preprocess_input_basic(self):
        """Test basic input preprocessing."""
        features = {f"V{i}": float(i) for i in range(1, 29)}
        features["Amount"] = 100.0

        df = preprocess_input(features)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 29)  # 28 V features + Amount
        assert list(df.columns) == [f"V{i}" for i in range(1, 29)] + ["Amount"]

    def test_preprocess_input_missing_features(self):
        """Test preprocessing with missing features (should be filled with 0)."""
        features = {"V1": 1.0, "V2": 2.0, "Amount": 100.0}

        df = preprocess_input(features)

        assert df.shape == (1, 29)
        assert df["V1"].iloc[0] == 1.0
        assert df["V2"].iloc[0] == 2.0
        assert df["V3"].iloc[0] == 0.0  # Should be filled with 0
        assert df["Amount"].iloc[0] == 100.0

    @patch('src.api.utils.FeatureScaler')
    def test_preprocess_input_with_scaler(self, mock_scaler_class):
        """Test preprocessing with scaler."""
        # Mock scaler
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = pd.DataFrame([[1, 2, 3]])

        features = {"V1": 1.0, "V2": 2.0, "Amount": 100.0}

        df = preprocess_input(features, scaler=mock_scaler)

        mock_scaler.transform.assert_called_once()

    def test_postprocess_prediction_fraud(self):
        """Test postprocessing for fraud prediction."""
        prediction = np.array([1])
        probability = np.array([[0.2, 0.8]])

        response = postprocess_prediction(prediction, probability)

        assert isinstance(response, PredictionResponse)
        assert response.prediction == 1
        assert response.probability == 0.8
        assert response.confidence == "high"
        assert response.risk_score == 80.0

    def test_postprocess_prediction_normal(self):
        """Test postprocessing for normal prediction."""
        prediction = np.array([0])
        probability = np.array([[0.9, 0.1]])

        response = postprocess_prediction(prediction, probability)

        assert response.prediction == 0
        assert response.probability == 0.1
        assert response.confidence == "low"
        assert response.risk_score == 10.0

    def test_validate_model_health_healthy(self):
        """Test model health validation for healthy model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

        health_status = validate_model_health(mock_model)

        assert health_status["status"] == "healthy"
        assert health_status["checks"]["has_predict"] is True
        assert health_status["checks"]["has_predict_proba"] is True
        assert health_status["checks"]["can_predict"] is True

    def test_validate_model_health_unhealthy(self):
        """Test model health validation for unhealthy model."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")

        health_status = validate_model_health(mock_model)

        assert health_status["status"] == "unhealthy"
        assert health_status["checks"]["can_predict"] is False
        assert "prediction_error" in health_status["checks"]


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_model_artifacts(self):
        """Mock model artifacts for testing."""
        with patch('src.api.main.model') as mock_model, \
             patch('src.api.main.scaler') as mock_scaler, \
             patch('src.api.main.pca_transformer') as mock_pca:

            # Configure mock model
            mock_model.predict.return_value = np.array([0])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

            yield {
                'model': mock_model,
                'scaler': mock_scaler,
                'pca_transformer': mock_pca
            }

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Credit Card Fraud Detection API"
        assert data["version"] == "1.0.0"

    def test_health_endpoint_no_model(self, client):
        """Test health endpoint when model is not loaded."""
        with patch('src.api.main.model', None):
            response = client.get("/health")

            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]

    def test_health_endpoint_healthy_model(self, client, mock_model_artifacts):
        """Test health endpoint with healthy model."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_predict_endpoint_success(self, client, mock_model_artifacts):
        """Test successful prediction endpoint."""
        features = {f"V{i}": float(i) for i in range(1, 29)}
        features["Amount"] = 100.0

        request_data = {"features": features}

        response = client.post("/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data
        assert "risk_score" in data

    def test_predict_endpoint_missing_features(self, client, mock_model_artifacts):
        """Test prediction endpoint with missing features."""
        request_data = {"features": {"V1": 1.0}}  # Missing most features

        response = client.post("/predict", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_no_model(self, client):
        """Test prediction endpoint when model is not loaded."""
        features = {f"V{i}": float(i) for i in range(1, 29)}
        features["Amount"] = 100.0

        request_data = {"features": features}

        with patch('src.api.main.model', None):
            response = client.post("/predict", json=request_data)

            assert response.status_code == 503

    def test_batch_predict_endpoint_success(self, client, mock_model_artifacts):
        """Test successful batch prediction endpoint."""
        features = {f"V{i}": float(i) for i in range(1, 29)}
        features["Amount"] = 100.0

        request_data = {
            "transactions": [features, features.copy()]
        }

        response = client.post("/predict/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all("prediction" in item for item in data)

    def test_batch_predict_endpoint_empty_transactions(self, client, mock_model_artifacts):
        """Test batch prediction with empty transactions."""
        request_data = {"transactions": []}

        response = client.post("/predict/batch", json=request_data)

        assert response.status_code == 422  # Validation error

    @patch('src.api.main.load_model_artifacts')
    def test_reload_model_endpoint_success(self, mock_load, client):
        """Test successful model reload endpoint."""
        mock_load.return_value = None

        response = client.post("/model/reload")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Model reloaded successfully"
        mock_load.assert_called_once()

    @patch('src.api.main.load_model_artifacts')
    def test_reload_model_endpoint_failure(self, mock_load, client):
        """Test model reload endpoint failure."""
        mock_load.side_effect = Exception("Load failed")

        response = client.post("/model/reload")

        assert response.status_code == 500
        assert "Model reload failed" in response.json()["detail"]

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "request_duration" in data
        assert "model_predictions" in data
        assert "errors_total" in data


class TestAPIIntegration:
    """Integration tests for API."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_model_artifacts(self):
        """Mock model artifacts for testing."""
        with patch('src.api.main.model') as mock_model, \
             patch('src.api.main.scaler') as mock_scaler, \
             patch('src.api.main.pca_transformer') as mock_pca:

            # Configure mock model
            mock_model.predict.return_value = np.array([0])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

            yield {
                'model': mock_model,
                'scaler': mock_scaler,
                'pca_transformer': mock_pca
            }

    @patch('src.api.main.joblib.load')
    @patch('src.api.main.Path.exists')
    def test_full_prediction_workflow(self, mock_exists, mock_load, client):
        """Test complete prediction workflow."""
        # Mock model loading
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_load.return_value = mock_model

        # Mock startup to load model
        with patch('src.api.main.model', mock_model):
            # Test health check
            health_response = client.get("/health")
            assert health_response.status_code == 200

            # Test prediction
            features = {f"V{i}": float(i * 0.1) for i in range(1, 29)}
            features["Amount"] = 150.75

            prediction_response = client.post(
                "/predict",
                json={"features": features}
            )

            assert prediction_response.status_code == 200
            data = prediction_response.json()
            assert data["prediction"] == 1
            assert data["probability"] == 0.9
            assert data["confidence"] == "high"
            assert data["risk_score"] == 90.0

    def test_api_error_handling(self, client):
        """Test API error handling."""
        # Test invalid JSON
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

        # Test missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/")
        # CORS headers should be handled by middleware
        # This is a basic test to ensure the middleware is configured
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled

    @pytest.fixture
    def mock_model_artifacts(self):
        """Mock model artifacts for testing."""
        with patch('src.api.main.model') as mock_model, \
             patch('src.api.main.scaler') as mock_scaler, \
             patch('src.api.main.pca_transformer') as mock_pca:

            # Configure mock model
            mock_model.predict.return_value = np.array([0])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

            yield {
                'model': mock_model,
                'scaler': mock_scaler,
                'pca_transformer': mock_pca
            }

    def test_predict_endpoint_invalid_types(self, client, mock_model_artifacts):
        """Test prediction endpoint with invalid data types."""
        request_data = {"features": {"V1": "string_instead_of_float"}}
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    def test_batch_predict_endpoint_invalid_transaction(self, client, mock_model_artifacts):
        """Test batch prediction endpoint with invalid transaction data."""
        request_data = {"transactions": [{"V1": "invalid_type"}]}
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 422

    def test_health_endpoint_model_predict_raises(self, client, mock_model_artifacts):
        """Test health endpoint when model predictions raise exceptions."""
        with patch('src.api.main.model.predict', side_effect=Exception("Predict error")), \
             patch('src.api.main.model.predict_proba', side_effect=Exception("PredictProba error")):
            response = client.get("/health")
            assert response.status_code == 503
            # Check that it returns an error response when model health fails
            response_data = response.json()
            assert "detail" in response_data
            assert "Model health check failed" in response_data["detail"]
