"""Utilities for A/B testing of ML models."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
import mlflow
import mlflow.sklearn

from .constants import MLFLOW_EXPERIMENT_NAME
from .logger import logger
from .mlflow_manager import mlflow_manager


class ABTester:
    """Class for running A/B tests between two models."""

    def __init__(
        self,
        model_a: Any,
        model_b: Any,
        test_data: pd.DataFrame,
        target_column: str = "Class",
        traffic_split: float = 0.5
    ):
        """
        Initialize A/B tester.

        Args:
            model_a: First model to test
            model_b: Second model to test
            test_data: Test dataset
            target_column: Name of target column
            traffic_split: Fraction of traffic to route to model B
        """
        self.model_a = model_a
        self.model_b = model_b
        self.X_test = test_data.drop(columns=[target_column])
        self.y_test = test_data[target_column]
        self.traffic_split = traffic_split
        self.results = {}

    def run_test(self, n_samples: int = None) -> Dict[str, Any]:
        """
        Run A/B test simulation.

        Args:
            n_samples: Number of samples to test (None for all)

        Returns:
            A/B test results
        """
        if n_samples is None:
            n_samples = len(self.X_test)
        else:
            n_samples = min(n_samples, len(self.X_test))

        logger.info(f"Running A/B test with {n_samples} samples, {self.traffic_split:.1%} traffic to B")

        # Generate predictions for both models
        pred_a = self.model_a.predict(self.X_test[:n_samples])
        pred_b = self.model_b.predict(self.X_test[:n_samples])
        proba_a = self.model_a.predict_proba(self.X_test[:n_samples])[:, 1]
        proba_b = self.model_b.predict_proba(self.X_test[:n_samples])[:, 1]

        # Simulate traffic routing
        routes = np.random.random(n_samples) < self.traffic_split
        routed_predictions = np.where(routes, pred_b, pred_a)
        routed_probabilities = np.where(routes, proba_b, proba_a)

        # Calculate metrics for each group
        group_a_mask = ~routes
        group_b_mask = routes

        results = {
            "test_info": {
                "total_samples": n_samples,
                "traffic_split": self.traffic_split,
                "group_a_samples": group_a_mask.sum(),
                "group_b_samples": group_b_mask.sum()
            },
            "group_a": self._calculate_metrics(
                self.y_test[:n_samples][group_a_mask],
                routed_predictions[group_a_mask],
                routed_probabilities[group_a_mask]
            ),
            "group_b": self._calculate_metrics(
                self.y_test[:n_samples][group_b_mask],
                routed_predictions[group_b_mask],
                routed_probabilities[group_b_mask]
            )
        }

        # Statistical comparison
        results["comparison"] = self._compare_groups(
            results["group_a"], results["group_b"]
        )

        self.results = results
        logger.info(f"A/B test completed. Winner: {results['comparison']['winner']}")

        return results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics for a group."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "sample_size": len(y_true)
        }

    def _compare_groups(self, group_a: Dict[str, Any], group_b: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance between two groups."""
        metrics_to_compare = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

        comparison = {}
        winner = None
        max_improvement = 0

        for metric in metrics_to_compare:
            val_a = group_a[metric]
            val_b = group_b[metric]
            improvement = (val_b - val_a) / val_a * 100 if val_a != 0 else 0

            comparison[f"{metric}_improvement"] = improvement

            if abs(improvement) > abs(max_improvement):
                max_improvement = improvement
                winner = "B" if improvement > 0 else "A"

        # Statistical significance (simplified t-test on F1 scores)
        # In practice, you'd use proper statistical testing for A/B tests
        f1_a = group_a["f1_score"]
        f1_b = group_b["f1_score"]
        n_a = group_a["sample_size"]
        n_b = group_b["sample_size"]

        # Pooled standard error approximation
        se = np.sqrt(f1_a * (1 - f1_a) / n_a + f1_b * (1 - f1_b) / n_b)
        t_stat = (f1_b - f1_a) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        comparison.update({
            "winner": winner,
            "max_improvement": max_improvement,
            "statistical_significance": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        })

        return comparison

    def log_to_mlflow(self) -> str:
        """
        Log A/B test results to MLflow.

        Returns:
            MLflow run ID
        """
        with mlflow_manager.start_ab_test_run("model_a", "model_b"):
            # Log test parameters
            mlflow.log_param("traffic_split", self.traffic_split)
            mlflow.log_param("total_samples", self.results["test_info"]["total_samples"])

            # Log group metrics
            for group in ["group_a", "group_b"]:
                if group in self.results:
                    metrics = self.results[group]
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{group}_{metric}", value)

            # Log comparison results
            if "comparison" in self.results:
                comp = self.results["comparison"]
                mlflow.log_param("winner", comp["winner"])
                mlflow.log_metric("max_improvement", comp["max_improvement"])

                sig = comp["statistical_significance"]
                mlflow.log_metric("t_statistic", sig["t_statistic"])
                mlflow.log_metric("p_value", sig["p_value"])
                mlflow.log_param("significant", sig["significant"])

        run_id = mlflow.active_run().info.run_id
        logger.info(f"A/B test results logged to MLflow run {run_id}")

        return run_id


def run_ab_test_simulation(
    model_a_name: str,
    model_b_name: str,
    test_data: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run A/B test simulation between two models from registry.

    Args:
        model_a_name: Name of first model
        model_b_name: Name of second model
        test_data: Test dataset
        config: A/B test configuration

    Returns:
        A/B test results
    """
    # Load models from registry
    model_a = mlflow.sklearn.load_model(f"models:/{model_a_name}/Production")
    model_b = mlflow.sklearn.load_model(f"models:/{model_b_name}/Production")

    # Initialize tester
    tester = ABTester(
        model_a, model_b, test_data,
        traffic_split=config.get("traffic_split", 0.5)
    )

    # Run test
    results = tester.run_test(n_samples=config.get("n_samples"))

    # Log to MLflow
    run_id = tester.log_to_mlflow()

    results["mlflow_run_id"] = run_id

    return results


def sequential_ab_test(
    model_a_name: str,
    model_b_name: str,
    test_data: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run sequential A/B test with early stopping criteria.

    Args:
        model_a_name: Name of control model
        model_b_name: Name of treatment model
        test_data: Test dataset
        config: Sequential test configuration

    Returns:
        Sequential test results
    """
    logger.info("Running sequential A/B test")

    # Load models
    model_a = mlflow.sklearn.load_model(f"models:/{model_a_name}/Production")
    model_b = mlflow.sklearn.load_model(f"models:/{model_b_name}/Production")

    batch_size = config.get("batch_size", 100)
    min_samples = config.get("min_samples", 1000)
    confidence_level = config.get("confidence_level", 0.95)

    results = {
        "batches": [],
        "early_stopped": False,
        "final_decision": None
    }

    cumulative_a_correct = 0
    cumulative_b_correct = 0
    total_a = 0
    total_b = 0

    for i in range(0, len(test_data), batch_size):
        batch_data = test_data.iloc[i:i+batch_size]
        if len(batch_data) == 0:
            break

        # Route traffic
        routes = np.random.random(len(batch_data)) < config.get("traffic_split", 0.5)

        # Get predictions
        pred_a = model_a.predict(batch_data.drop(columns=["Class"]))
        pred_b = model_b.predict(batch_data.drop(columns=["Class"]))
        true_labels = batch_data["Class"].values

        # Calculate batch metrics
        a_correct = (pred_a == true_labels).sum()
        b_correct = (pred_b == true_labels).sum()

        cumulative_a_correct += a_correct
        cumulative_b_correct += b_correct
        total_a += len(pred_a)
        total_b += len(pred_b)

        # Calculate current conversion rates
        rate_a = cumulative_a_correct / total_a if total_a > 0 else 0
        rate_b = cumulative_b_correct / total_b if total_b > 0 else 0

        batch_result = {
            "batch": len(results["batches"]) + 1,
            "samples": len(batch_data),
            "rate_a": rate_a,
            "rate_b": rate_b,
            "improvement": (rate_b - rate_a) / rate_a * 100 if rate_a > 0 else 0,
            "total_a": total_a,
            "total_b": total_b
        }

        results["batches"].append(batch_result)

        # Check early stopping criteria
        if total_a >= min_samples and total_b >= min_samples:
            # Simplified statistical test
            se = np.sqrt(rate_a * (1 - rate_a) / total_a + rate_b * (1 - rate_b) / total_b)
            z_score = (rate_b - rate_a) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            if p_value < (1 - confidence_level):
                results["early_stopped"] = True
                results["final_decision"] = "B" if rate_b > rate_a else "A"
                results["stopping_reason"] = f"Statistical significance reached (p={p_value:.4f})"
                break

    if not results["early_stopped"]:
        results["final_decision"] = "B" if results["batches"][-1]["rate_b"] > results["batches"][-1]["rate_a"] else "A"

    logger.info(f"Sequential A/B test completed. Decision: {results['final_decision']}")

    return results
