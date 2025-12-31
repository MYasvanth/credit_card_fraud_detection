import pytest
from unittest.mock import patch, MagicMock
from src.utils import mlflow_manager
import mlflow

@pytest.fixture
def mlflow_manager_instance():
    with patch('mlflow.set_tracking_uri'), \
         patch.object(mlflow_manager.MLflowManager, '_setup_experiments'):
        return mlflow_manager.MLflowManager(tracking_uri="http://fake_tracking_uri")

def test_setup_experiments_creates_new_experiment(mlflow_manager_instance):
    with patch('mlflow.create_experiment') as mock_create:
        # Simulate some calls succeed, some raise exists error
        def create_side_effect(*args, **kwargs):
            if mock_create.call_count == 1:
                raise mlflow.exceptions.MlflowException("exists")
            return None
        mock_create.side_effect = create_side_effect
        mlflow_manager_instance._setup_experiments()
        assert mock_create.call_count == 5  # Number of experiments in MLFLOW_EXPERIMENTS

# Removed duplicate test_export_experiment_data function

# Existing test_log_model_with_metadata_success and others remain unchanged here

def test_start_training_run_tags(mlflow_manager_instance):
    with patch('mlflow.set_experiment') as mock_set_exp, \
         patch('mlflow.start_run') as mock_start_run:
        mock_start_run.return_value = MagicMock()
        run = mlflow_manager_instance.start_training_run(run_name="train_run", tags={"tag1":"value1"})
        mock_set_exp.assert_called_with(mlflow_manager.MLFLOW_EXPERIMENTS["training"])
        mock_start_run.assert_called()
        assert run is not None

def test_log_model_with_metadata_success(mlflow_manager_instance):
    mock_model = MagicMock()
    mock_model_info = MagicMock()
    mock_model_info.model_uri = "fake_uri"

    with patch('mlflow.log_params') as mock_log_params, \
         patch('mlflow.log_metrics') as mock_log_metrics, \
         patch('mlflow.sklearn.log_model') as mock_log_model, \
         patch.object(mlflow_manager_instance.client, 'get_latest_versions') as mock_get_versions, \
         patch.object(mlflow_manager_instance.client, 'transition_model_version_stage') as mock_transition:

        mock_log_model.return_value = mock_model_info
        mock_get_versions.return_value = [MagicMock(version=1)]
        uri = mlflow_manager_instance.log_model_with_metadata(
            model=mock_model,
            model_name="test_model",
            metrics={"acc": 0.95},
            params={"param1": 1},
            artifacts=None,
            stage="Staging"
        )
        mock_log_params.assert_called()
        mock_log_metrics.assert_called()
        mock_log_model.assert_called()
        mock_get_versions.assert_called()
        mock_transition.assert_called()
        assert uri == "fake_uri"

def test_log_model_with_metadata_transition_failure(mlflow_manager_instance):
    mock_model = MagicMock()

    with patch('mlflow.log_params'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.sklearn.log_model', return_value=MagicMock(model_uri="uri")), \
         patch.object(mlflow_manager_instance.client, 'get_latest_versions', side_effect=Exception("fail")), \
         patch.object(mlflow_manager_instance.client, 'transition_model_version_stage') as mock_transition:

        # Transition should not be called due to exception
        uri = mlflow_manager_instance.log_model_with_metadata(
            model=mock_model,
            model_name="test_model",
            metrics={},
            params={},
            artifacts=None,
            stage="Production"
        )
        mock_transition.assert_not_called()
        assert uri == "uri"

def test_compare_model_versions(mlflow_manager_instance):
    with patch.object(mlflow_manager_instance.client, 'get_latest_versions') as mock_get_versions, \
         patch.object(mlflow_manager_instance.client, 'get_run') as mock_get_run:
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.run_id = "run_1"
        mock_version.creation_timestamp = 123456789
        mock_get_versions.return_value = [mock_version]

        mock_run = MagicMock()
        mock_run.data.metrics = {"accuracy": 0.9}
        mock_run.data.params = {"param1": "val"}
        mock_get_run.return_value = mock_run

        result = mlflow_manager_instance.compare_model_versions("model", stages=["Production"])
        assert "Production" in result
        assert result["Production"]["version"] == 1

def test_get_experiment_runs_and_cleanup(mlflow_manager_instance):
    with patch('mlflow.get_experiment_by_name') as mock_get_exp, \
         patch('mlflow.search_runs') as mock_search_runs, \
         patch.object(mlflow_manager_instance.client, 'delete_run') as mock_delete:

        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp1"
        mock_get_exp.return_value = mock_exp

        mock_runs_df = MagicMock()
        mock_runs_df.empty = False
        mock_runs_df.to_dict.return_value = [{"run_id": "1"}, {"run_id": "2"}]
        mock_search_runs.return_value = mock_runs_df

        runs = mlflow_manager_instance.get_experiment_runs("training")
        assert isinstance(runs, list)
        assert len(runs) == 2

        mlflow_manager_instance.cleanup_old_runs("training", keep_last_n=1)
        mock_delete.assert_called_once_with("2")

def test_export_experiment_data(mlflow_manager_instance, tmp_path):
    with patch('mlflow.get_experiment_by_name') as mock_get_exp, \
         patch('mlflow.search_runs') as mock_search_runs, \
         patch('pandas.DataFrame') as mock_df_class:

        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp1"
        mock_get_exp.return_value = mock_exp

        mock_runs_df = MagicMock()
        mock_runs_df.empty = False
        mock_runs_df.to_dict.return_value = [{"run_id": "1"}]
        mock_search_runs.return_value = mock_runs_df

        mock_df = MagicMock()
        mock_df_class.return_value = mock_df

        mlflow_manager_instance.export_experiment_data("training", str(tmp_path / "export.csv"))
        mock_df.to_csv.assert_called_once()

def test_get_model_performance_history_empty_and_nonempty(mlflow_manager_instance):
    with patch.object(mlflow_manager_instance.client, 'search_model_versions') as mock_search_versions, \
         patch.object(mlflow_manager_instance.client, 'get_run') as mock_get_run:
        
        # Empty history
        mock_search_versions.return_value = []
        df = mlflow_manager_instance.get_model_performance_history("model")
        assert df.empty

        # Non-empty history
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.run_id = "run_1"
        mock_version.current_stage = "Production"
        mock_version.creation_timestamp = 123456789
        mock_search_versions.return_value = [mock_version]

        mock_run = MagicMock()
        mock_run.data.metrics = {"accuracy": 0.9}
        mock_run.data.params = {"param1": "val"}
        mock_get_run.return_value = mock_run

        df = mlflow_manager_instance.get_model_performance_history("model")
        assert not df.empty
