"""Run monitoring pipeline for production model."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.dashboard_generator import DashboardGenerator
from src.utils.logger import logger
from src.utils.constants import DATA_PATH, REPORTS_PATH


def run_monitoring_pipeline(reference_data_path: str, current_data_path: str,
                          predictions_path: str = None):
    """Run complete monitoring pipeline."""
    
    # Load data
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)
    
    logger.info(f"Reference data shape: {reference_data.shape}")
    logger.info(f"Current data shape: {current_data.shape}")
    
    # Initialize drift detector
    drift_detector = DriftDetector(reference_data)
    
    # Detect drift
    drift_results = drift_detector.detect_feature_drift(current_data)
    drift_report = drift_detector.generate_drift_report(current_data)
    
    # Log drift results
    drifted_features = [f for f, r in drift_results.items() if r["drift_detected"]]
    logger.info(f"Drift detected in {len(drifted_features)} features: {drifted_features}")
    
    # Performance monitoring (if predictions available)
    if predictions_path and Path(predictions_path).exists():
        predictions_df = pd.read_csv(predictions_path)
        
        baseline_metrics = {"f1_score": 0.85, "precision": 0.80, "recall": 0.90}
        tracker = PerformanceTracker(baseline_metrics)
        
        current_metrics = tracker.evaluate_performance(
            predictions_df["true_label"].values,
            predictions_df["prediction"].values,
            predictions_df[["prob_normal", "prob_fraud"]].values
        )
        
        degradation_results = tracker.check_degradation(current_metrics)
        tracker.save_performance_history()
        
        logger.info(f"Current F1 Score: {current_metrics['f1_score']:.4f}")
    
    # Generate dashboards
    dashboard_gen = DashboardGenerator()
    
    # Create drift dashboard
    feature_distributions = {}
    for feature in reference_data.select_dtypes(include=[np.number]).columns[:5]:  # Top 5 features
        feature_distributions[feature] = {
            "reference": reference_data[feature].values,
            "current": current_data[feature].values
        }
    
    drift_dashboard = dashboard_gen.generate_drift_dashboard(
        {"drift_scores": drift_results}, feature_distributions
    )
    
    logger.info("Monitoring pipeline completed successfully")
    
    return {
        "drift_results": drift_results,
        "drift_dashboard": drift_dashboard
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run monitoring pipeline")
    parser.add_argument("--reference-data", required=True, help="Path to reference data")
    parser.add_argument("--current-data", required=True, help="Path to current data")
    parser.add_argument("--predictions", help="Path to predictions file")
    
    args = parser.parse_args()
    
    run_monitoring_pipeline(
        args.reference_data,
        args.current_data,
        args.predictions
    )