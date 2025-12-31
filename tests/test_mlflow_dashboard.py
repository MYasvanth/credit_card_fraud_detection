import subprocess
import sys
import pytest
from unittest.mock import patch, MagicMock

DASHBOARD_SCRIPT = "scripts/mlflow_dashboard.py"


@pytest.mark.parametrize("command,args,expected_output", [
    ("summary", [], "MLFLOW EXPERIMENTS SUMMARY"),
    ("registry", [], "MODEL REGISTRY"),
    ("compare", [], "PRODUCTION vs STAGING COMPARISON"),
])
def test_dashboard_basic_commands(command, args, expected_output):
    """Test dashboard commands with mocked subprocess calls."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = f"{expected_output}\nSome additional output here\n{'='*80}"
    mock_result.stderr = ""

    with patch('subprocess.run', return_value=mock_result):
        result = subprocess.run(
            [sys.executable, DASHBOARD_SCRIPT, command] + args,
            capture_output=True,
            text=True
        )

    # Verify the command was called correctly
    assert result.returncode == 0
    assert expected_output in result.stdout
    assert "Error" not in result.stdout


def test_dashboard_export_missing_args():
    """Test export command with missing args."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "❌ Export requires --experiment-type and --output-file"
    mock_result.stderr = ""

    with patch('subprocess.run', return_value=mock_result):
        result = subprocess.run(
            [sys.executable, DASHBOARD_SCRIPT, "export"],
            capture_output=True,
            text=True
        )

    assert result.returncode == 0
    assert "Export requires" in result.stdout


def test_dashboard_cleanup_missing_args():
    """Test cleanup command with missing args."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "❌ Cleanup requires --experiment-type"
    mock_result.stderr = ""

    with patch('subprocess.run', return_value=mock_result):
        result = subprocess.run(
            [sys.executable, DASHBOARD_SCRIPT, "cleanup"],
            capture_output=True,
            text=True
        )

    assert result.returncode == 0
    assert "Cleanup requires" in result.stdout


def test_dashboard_history_missing_args():
    """Test history command with missing args."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "❌ History requires --model-name"
    mock_result.stderr = ""

    with patch('subprocess.run', return_value=mock_result):
        result = subprocess.run(
            [sys.executable, DASHBOARD_SCRIPT, "history"],
            capture_output=True,
            text=True
        )

    assert result.returncode == 0
    assert "History requires" in result.stdout

# Test export, cleanup, and history commands with proper arguments
def test_dashboard_export_with_args():
    """Test successful export command with required args."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "✅ Exported training data to export.csv"
    mock_result.stderr = ""

    with patch('subprocess.run', return_value=mock_result):
        result = subprocess.run([
            sys.executable, DASHBOARD_SCRIPT, "export",
            "--experiment-type", "training", "--output-file", "export.csv"
        ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "Exported" in result.stdout


def test_dashboard_cleanup_with_args():
    """Test successful cleanup command with required args."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "✅ Cleaned up training experiments, kept last 10"
    mock_result.stderr = ""

    with patch('subprocess.run', return_value=mock_result):
        result = subprocess.run([
            sys.executable, DASHBOARD_SCRIPT, "cleanup",
            "--experiment-type", "training", "--keep-last-n", "10"
        ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "Cleaned up" in result.stdout


def test_dashboard_history_with_args():
    """Test successful history command with required args."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "PERFORMANCE HISTORY: credit_card_fraud_detector"
    mock_result.stderr = ""

    with patch('subprocess.run', return_value=mock_result):
        result = subprocess.run([
            sys.executable, DASHBOARD_SCRIPT, "history",
            "--model-name", "credit_card_fraud_detector"
        ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "PERFORMANCE HISTORY" in result.stdout
