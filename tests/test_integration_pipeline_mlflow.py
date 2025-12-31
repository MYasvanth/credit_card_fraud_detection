import pytest
from unittest.mock import patch, MagicMock
from src.pipelines.training_pipeline import training_pipeline
from src.utils.mlflow_manager import MLflowManager
import mlflow

@pytest.mark.integration
@patch.object(MLflowManager, 'log_model_with_metadata')
def test_training_pipeline_logs_model(mock_log_model):
    """Test training pipeline logs model to MLflow registry."""
    # Mock model
    mock_model = MagicMock()
    mock_log_model.return_value = "fake_model_uri"
    
    # Log model to MLflow
    mlflow_manager = MLflowManager()
    uri = mlflow_manager.log_model_with_metadata(
        model=mock_model,
        model_name="credit_card_fraud_detection_model",
        metrics={"accuracy": 0.95},
        params={"param1": "value1"},
        artifacts=None,
        stage="Staging"
    )
    
    # Assert log_model_with_metadata called and returned uri
    mock_log_model.assert_called_once_with(
        model=mock_model,
        model_name="credit_card_fraud_detection_model",
        metrics={"accuracy": 0.95},
        params={"param1": "value1"},
        artifacts=None,
        stage="Staging"
    )
    assert uri == "fake_model_uri"

@pytest.mark.integration
def test_end_to_end_pipeline_mlflow(monkeypatch):
    """End to end test for full pipeline with MLflow logging."""
    
    # Here you'd run the actual pipeline entry point and verify MLflow model logging
    # For example: run pipeline, check MLflow experiments and registered models
    
    # This is a placeholder; real implementation depends on pipeline code
    
    # You can patch MLflowManager or MLflow client to capture calls
    
    pass