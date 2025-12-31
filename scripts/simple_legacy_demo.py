"""Simple demo of legacy monitoring logic without complex dependencies."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.monitoring.performance_tracker import PerformanceTracker

def demo_performance_tracker():
    """Demo: How Performance Tracker Works"""
    print("=== Performance Tracker Logic ===")
    
    # Step 1: Set baseline (what we expect from a good model)
    baseline_metrics = {
        "f1_score": 0.85,
        "precision": 0.80, 
        "recall": 0.90
    }
    tracker = PerformanceTracker(baseline_metrics)
    print(f"Baseline set: {baseline_metrics}")
    
    # Step 2: Simulate current model performance
    y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])  # Some errors
    
    # Step 3: Evaluate current performance
    current_metrics = tracker.evaluate_performance(y_true, y_pred)
    print(f"Current performance: {current_metrics}")
    
    # Step 4: Check for degradation
    degradation_check = tracker.check_degradation(current_metrics, threshold=0.05)
    print(f"Degradation analysis:")
    for metric, analysis in degradation_check.items():
        status = "DEGRADED" if analysis['degraded'] else "OK"
        print(f"  {metric}: {status} (drop: {analysis['drop']:.3f})")
    
    return tracker

def demo_simple_drift_detection():
    """Demo: Simple Drift Detection Logic"""
    print("\n=== Simple Drift Detection Logic ===")
    
    # Step 1: Reference data (training data distribution)
    reference_amounts = np.random.normal(100, 30, 1000)  # Mean=100, Std=30
    print(f"Reference data: Mean={reference_amounts.mean():.1f}, Std={reference_amounts.std():.1f}")
    
    # Step 2: Current data (production data - different distribution)
    current_amounts = np.random.normal(150, 50, 500)  # Mean=150, Std=50 (DRIFTED!)
    print(f"Current data: Mean={current_amounts.mean():.1f}, Std={current_amounts.std():.1f}")
    
    # Step 3: Simple drift detection using statistical test
    from scipy import stats
    ks_statistic, p_value = stats.ks_2samp(reference_amounts, current_amounts)
    
    drift_detected = p_value < 0.05  # 5% significance level
    print(f"KS Test: statistic={ks_statistic:.3f}, p-value={p_value:.6f}")
    print(f"Drift Status: {'DRIFT DETECTED' if drift_detected else 'NO DRIFT'}")
    
    return drift_detected

def demo_simple_dashboard_logic():
    """Demo: Dashboard Generation Logic"""
    print("\n=== Dashboard Generation Logic ===")
    
    # Step 1: Collect metrics over time
    metrics_over_time = [
        {"time": "10:00", "f1_score": 0.85, "precision": 0.80},
        {"time": "11:00", "f1_score": 0.82, "precision": 0.78},
        {"time": "12:00", "f1_score": 0.79, "precision": 0.75},  # Degrading
        {"time": "13:00", "f1_score": 0.76, "precision": 0.72}   # Getting worse
    ]
    
    print("Performance over time:")
    for metric in metrics_over_time:
        print(f"  {metric['time']}: F1={metric['f1_score']}, Precision={metric['precision']}")
    
    # Step 2: Identify trends
    f1_scores = [m['f1_score'] for m in metrics_over_time]
    trend = "DECLINING" if f1_scores[-1] < f1_scores[0] else "STABLE/IMPROVING"
    print(f"Trend Analysis: {trend}")
    
    # Step 3: Generate alerts
    latest_f1 = f1_scores[-1]
    if latest_f1 < 0.8:
        print(f"ALERT: F1 score dropped to {latest_f1} (below 0.8 threshold)")
    else:
        print("Performance within acceptable range")

def main():
    """Run simplified legacy monitoring demos."""
    print("Legacy Monitoring Logic Demo (Simplified)\n")
    
    # Demo 1: Performance Tracking
    tracker = demo_performance_tracker()
    
    # Demo 2: Drift Detection  
    drift_detected = demo_simple_drift_detection()
    
    # Demo 3: Dashboard Logic
    demo_simple_dashboard_logic()
    
    print(f"\n=== Summary ===")
    print(f"Performance Tracker: Tracks {len(tracker.performance_history)} metrics")
    print(f"Drift Detection: {'Drift detected' if drift_detected else 'No drift'}")
    print(f"Dashboard: Would generate HTML visualizations")
    
    print(f"\n=== How Legacy Files Work ===")
    print("1. PerformanceTracker: Compares current vs baseline, stores history")
    print("2. DriftDetector: Uses KS-test to compare data distributions")  
    print("3. DashboardGenerator: Creates Plotly charts from metrics data")
    print("4. All modules work together for comprehensive monitoring")

if __name__ == "__main__":
    main()