"""Performance tracking and alerting system."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import classification_report, roc_auc_score
import json
from pathlib import Path

from ...utils.logger import logger
from ...utils.constants import REPORTS_PATH


class PerformanceTracker:
    """Track model performance over time and detect degradation."""
    
    def __init__(self, baseline_metrics: Dict[str, float]):
        self.baseline_metrics = baseline_metrics
        self.performance_history = []
        
    def evaluate_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_proba: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate current model performance."""
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"]
        }
        
        if y_proba is not None:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
        
        self.performance_history.append(metrics)
        return metrics
    
    def check_degradation(self, current_metrics: Dict[str, float], 
                         threshold: float = 0.05) -> Dict[str, Any]:
        """Check for performance degradation."""
        degradation_results = {}
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                drop = baseline_value - current_value
                degradation_results[metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "drop": drop,
                    "degraded": drop > threshold
                }
        
        return degradation_results
    
    def save_performance_history(self):
        """Save performance history to file."""
        Path(REPORTS_PATH).mkdir(parents=True, exist_ok=True)
        history_path = Path(REPORTS_PATH) / "performance_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        logger.info(f"Performance history saved to {history_path}")
