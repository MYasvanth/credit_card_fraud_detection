"""Example of advanced monitoring modules usage."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from src.monitoring.advanced.performance_tracker import PerformanceTracker

def demo_advanced_monitoring():
    """Demo advanced monitoring capabilities."""
    print("=== Advanced Monitoring Demo ===")
    
    # Performance Tracker
    baseline = {"f1_score": 0.85, "precision": 0.80, "recall": 0.90}
    tracker = PerformanceTracker(baseline)
    
    # Simulate predictions
    y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    
    metrics = tracker.evaluate_performance(y_true, y_pred)
    print(f"Current Metrics: {metrics}")
    
    degradation = tracker.check_degradation(metrics)
    print(f"Degradation Check: {degradation}")
    
    # Simple drift detection
    reference_data = np.random.normal(100, 30, 1000)
    current_data = np.random.normal(150, 50, 500)
    
    from scipy import stats
    ks_stat, p_value = stats.ks_2samp(reference_data, current_data)
    drift_detected = p_value < 0.05
    
    print(f"Drift Detection: {'DRIFT DETECTED' if drift_detected else 'NO DRIFT'}")
    print(f"KS Statistic: {ks_stat:.3f}, P-value: {p_value:.6f}")

if __name__ == "__main__":
    demo_advanced_monitoring()