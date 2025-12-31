"""Working example of advanced monitoring without external dependencies."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from src.monitoring.advanced.performance_tracker import PerformanceTracker
from src.monitoring.advanced.drift_detector import DriftDetector

def demo_working_advanced():
    """Demo advanced monitoring that actually works."""
    print("=== Working Advanced Monitoring Demo ===")
    
    # 1. Performance Tracker Demo
    print("\n1. Performance Tracker:")
    baseline = {"f1_score": 0.85, "precision": 0.80, "recall": 0.90}
    tracker = PerformanceTracker(baseline)
    
    y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    
    metrics = tracker.evaluate_performance(y_true, y_pred)
    print(f"   Current: F1={metrics['f1_score']:.3f}, Precision={metrics['precision']:.3f}")
    
    degradation = tracker.check_degradation(metrics)
    for metric, result in degradation.items():
        status = "DEGRADED" if result['degraded'] else "OK"
        print(f"   {metric}: {status} (drop: {result['drop']:.3f})")
    
    # 2. Drift Detector Demo
    print("\n2. Drift Detector:")
    reference_data = pd.DataFrame({
        'amount': np.random.normal(100, 30, 1000),
        'time': np.random.uniform(0, 24, 1000)
    })
    
    current_data = pd.DataFrame({
        'amount': np.random.normal(150, 50, 500),  # Drifted
        'time': np.random.uniform(0, 24, 500)
    })
    
    detector = DriftDetector(reference_data)
    drift_results = detector.detect_feature_drift(current_data)
    
    for feature, result in drift_results.items():
        status = "DRIFT" if result['drift_detected'] else "OK"
        print(f"   {feature}: {status} (p-value: {result['p_value']:.4f})")
    
    # 3. Generate drift report
    print("\n3. Drift Report:")
    report = detector.generate_drift_report(current_data)
    summary = report['summary']
    print(f"   Features: {summary['total_features']}")
    print(f"   Drifted: {summary['drifted_features']} ({summary['drift_percentage']:.1f}%)")
    
    print("\n✅ Advanced monitoring working without external dependencies!")

if __name__ == "__main__":
    demo_working_advanced()