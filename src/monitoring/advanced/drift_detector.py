"""Real-time drift detection for production monitoring."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
# Evidently has numpy compatibility issues - using basic drift detection only
EVIDENTLY_AVAILABLE = False

from ...utils.logger import logger


class DriftDetector:
    """Real-time drift detection using statistical tests and Evidently."""
    
    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.05):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        
    def detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in individual features."""
        drift_results = {}
        
        for column in self.reference_data.select_dtypes(include=[np.number]).columns:
            if column in current_data.columns:
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                
                drift_results[column] = {
                    "ks_statistic": ks_stat,
                    "p_value": p_value,
                    "drift_detected": p_value < self.drift_threshold
                }
        
        return drift_results
    
    def generate_drift_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate drift report using available methods."""
        if EVIDENTLY_AVAILABLE:
            try:
                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=self.reference_data, current_data=current_data)
                return report.as_dict()
            except Exception as e:
                logger.warning(f"Evidently report generation failed: {e}")
        
        # Fallback to basic drift detection
        drift_results = self.detect_feature_drift(current_data)
        total_features = len(drift_results)
        drifted_features = sum(1 for r in drift_results.values() if r['drift_detected'])
        
        return {
            "drift_scores": drift_results,
            "summary": {
                "total_features": total_features,
                "drifted_features": drifted_features,
                "drift_percentage": (drifted_features / total_features) * 100 if total_features > 0 else 0
            }
        }
