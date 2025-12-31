"""Utilities for multi-model comparison and evaluation."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import mlflow
import mlflow.sklearn

from .constants import MLFLOW_EXPERIMENT_NAME
from .logger import logger
from .mlflow_manager import mlflow_manager


class ModelComparator:
    """Class for comparing multiple ML models."""

    def __init__(self, models: Dict[str, Any], test_data: pd.DataFrame, target_column: str = "Class"):
        """
        Initialize model comparator.

        Args:
            models: Dictionary of model names to model objects
            test_data: Test dataset
            target_column: Name of target column
        """
        self.models = models
        self.X_test = test_data.drop(columns=[target_column])
        self.y_test = test_data[target_column]
        self.target_column = target_column
        self.results = {}

    def evaluate_all_models(self) -> Dict[str, Any]:
        """
        Evaluate all models and store results.

        Returns:
            Dictionary of evaluation results for each model
        """
        logger.info(f"Evaluating {len(self.models)} models on {len(self.X_test)} test samples")

        for model_name, model in self.models.items():
            try:
                logger.info(f"Evaluating {model_name}...")

                # Get predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Probability of positive class

                # Calculate metrics
                report = classification_report(self.y_test, y_pred, output_dict=True)
                cm = confusion_matrix(self.y_test, y_pred)

                # ROC curve
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
                avg_precision = average_precision_score(self.y_test, y_pred_proba)

                # Store results
                self.results[model_name] = {
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                    "classification_report": report,
                    "confusion_matrix": cm,
                    "roc_curve": {"fpr": fpr, "tpr": tpr, "auc": roc_auc},
                    "pr_curve": {"precision": precision, "recall": recall, "avg_precision": avg_precision},
                    "metrics": {
                        "accuracy": report["accuracy"],
                        "precision": report["1"]["precision"],
                        "recall": report["1"]["recall"],
                        "f1_score": report["1"]["f1-score"],
                        "roc_auc": roc_auc,
                        "avg_precision": avg_precision
                    }
                }

                logger.info(f"{model_name} - F1: {report['1']['f1-score']:.4f}, AUC: {roc_auc:.4f}")

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                self.results[model_name] = {"error": str(e)}

        return self.results

    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.

        Returns:
            Comparison report with rankings and insights
        """
        if not self.results:
            self.evaluate_all_models()

        # Extract metrics for comparison
        metrics_df = pd.DataFrame({
            model_name: results.get("metrics", {})
            for model_name, results in self.results.items()
            if "metrics" in results
        }).T

        # Rank models by different metrics
        rankings = {}
        for metric in metrics_df.columns:
            if metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "avg_precision"]:
                rankings[metric] = metrics_df[metric].sort_values(ascending=False).index.tolist()

        # Overall ranking (weighted average of key metrics)
        weights = {
            "f1_score": 0.4,
            "roc_auc": 0.3,
            "precision": 0.15,
            "recall": 0.15
        }

        overall_scores = {}
        for model in metrics_df.index:
            score = sum(metrics_df.loc[model, metric] * weight
                       for metric, weight in weights.items()
                       if metric in metrics_df.columns)
            overall_scores[model] = score

        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

        # Statistical significance tests
        significance_tests = self._perform_significance_tests()

        report = {
            "metrics_comparison": metrics_df.to_dict(),
            "rankings": rankings,
            "overall_ranking": overall_ranking,
            "significance_tests": significance_tests,
            "best_models": {
                "by_f1": rankings.get("f1_score", [None])[0],
                "by_auc": rankings.get("roc_auc", [None])[0],
                "overall": overall_ranking[0][0] if overall_ranking else None
            },
            "test_data_info": {
                "n_samples": len(self.X_test),
                "n_features": self.X_test.shape[1],
                "fraud_rate": self.y_test.mean(),
                "class_distribution": self.y_test.value_counts().to_dict()
            }
        }

        logger.info(f"Comparison report generated. Best overall model: {report['best_models']['overall']}")

        return report

    def _perform_significance_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        from scipy import stats

        significance_results = {}
        model_names = [name for name in self.results.keys() if "metrics" in self.results[name]]

        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if i >= j:
                    continue

                try:
                    pred_a = self.results[model_a]["predictions"]
                    pred_b = self.results[model_b]["predictions"]

                    # McNemar test for paired predictions
                    contingency = pd.crosstab(pred_a, pred_b)
                    if contingency.shape == (2, 2):
                        b = contingency.iloc[0, 1]
                        c = contingency.iloc[1, 0]
                        mcnemar_stat = (abs(b - c) ** 2) / (b + c) if (b + c) > 0 else 0

                        # Chi-square approximation
                        p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)

                        significance_results[f"{model_a}_vs_{model_b}"] = {
                            "test": "mcnemar",
                            "statistic": mcnemar_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05
                        }
                    else:
                        significance_results[f"{model_a}_vs_{model_b}"] = {
                            "test": "mcnemar",
                            "note": "Cannot perform test - incomplete contingency table"
                        }

                except Exception as e:
                    logger.error(f"Significance test failed for {model_a} vs {model_b}: {e}")
                    significance_results[f"{model_a}_vs_{model_b}"] = {"error": str(e)}

        return significance_results

    def plot_comparison_charts(self, save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Generate comparison visualization charts.

        Args:
            save_path: Path to save plots (optional)

        Returns:
            Dictionary of matplotlib figures
        """
        if not self.results:
            self.evaluate_all_models()

        figures = {}

        # Metrics comparison bar chart
        metrics_df = pd.DataFrame({
            model_name: results.get("metrics", {})
            for model_name, results in self.results.items()
            if "metrics" in results
        }).T

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Model Performance Comparison", fontsize=16)

        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "avg_precision"]
        axes = axes.ravel()

        for i, metric in enumerate(metrics_to_plot):
            if metric in metrics_df.columns:
                metrics_df[metric].plot(kind='bar', ax=axes[i], color='skyblue')
                axes[i].set_title(f"{metric.replace('_', ' ').title()}")
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        figures["metrics_comparison"] = fig

        # ROC curves
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for model_name, results in self.results.items():
            if "roc_curve" in results:
                roc_data = results["roc_curve"]
                ax_roc.plot(roc_data["fpr"], roc_data["tpr"],
                           label=f'{model_name} (AUC = {roc_data["auc"]:.3f})')

        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curves Comparison')
        ax_roc.legend()
        ax_roc.grid(True)
        figures["roc_curves"] = fig_roc

        # Precision-Recall curves
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        for model_name, results in self.results.items():
            if "pr_curve" in results:
                pr_data = results["pr_curve"]
                ax_pr.plot(pr_data["recall"], pr_data["precision"],
                          label=f'{model_name} (AP = {pr_data["avg_precision"]:.3f})')

        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curves Comparison')
        ax_pr.legend()
        ax_pr.grid(True)
        figures["pr_curves"] = fig_pr

        if save_path:
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/{name}.png", dpi=300, bbox_inches='tight')
                logger.info(f"Saved {name} plot to {save_path}")

        return figures

    def log_comparison_to_mlflow(self) -> str:
        """
        Log comparison results to MLflow.

        Returns:
            MLflow run ID
        """
        model_names = list(self.models.keys())
        
        with mlflow_manager.start_comparison_run(model_names) as run:
            # Log comparison metadata
            mlflow.log_param("n_models", len(self.models))
            mlflow.log_param("test_samples", len(self.X_test))
            mlflow.log_param("fraud_rate", self.y_test.mean())

            # Log individual model metrics
            for model_name, results in self.results.items():
                if "metrics" in results:
                    metrics = results["metrics"]
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{model_name}_{metric_name}", value)

            # Log comparison report
            report = self.generate_comparison_report()
            mlflow.log_param("best_overall_model", report["best_models"]["overall"])
            mlflow.log_param("best_f1_model", report["best_models"]["by_f1"])
            mlflow.log_param("best_auc_model", report["best_models"]["by_auc"])

            # Log significance test results
            sig_tests = report["significance_tests"]
            significant_comparisons = sum(1 for test in sig_tests.values()
                                        if isinstance(test, dict) and test.get("significant", False))
            mlflow.log_metric("significant_comparisons", significant_comparisons)

        logger.info(f"Model comparison logged to MLflow run {run.info.run_id}")

        return run.info.run_id


def compare_models_from_registry(
    model_names: List[str],
    test_data: pd.DataFrame,
    target_column: str = "Class",
    stages: Optional[List[str]] = None
) -> ModelComparator:
    """
    Compare models loaded from MLflow Model Registry.

    Args:
        model_names: List of model names in registry
        test_data: Test dataset
        target_column: Target column name
        stages: Model stages to compare (default: Production)

    Returns:
        Configured ModelComparator instance
    """
    if stages is None:
        stages = ["Production"]

    loaded_models = {}

    for model_name in model_names:
        for stage in stages:
            try:
                model_uri = f"models:/{model_name}/{stage}"
                model = mlflow.sklearn.load_model(model_uri)
                loaded_models[f"{model_name}_{stage}"] = model
                logger.info(f"Loaded {model_name} from {stage}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name} from {stage}: {e}")

    return ModelComparator(loaded_models, test_data, target_column)
