"""Example of how to use legacy monitoring modules."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.dashboard_generator import DashboardGenerator

def demo_performance_tracker():
    """Demo: Performance Tracker Logic"""
    print("=== Performance Tracker Demo ===")
    
    # 1. Set baseline metrics (what we expect)
    baseline = {"f1_score": 0.85, "precision": 0.80, "recall": 0.90}
    tracker = PerformanceTracker(baseline)
    
    # 2. Simulate model predictions
    y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    y_proba = np.random.rand(10, 2)
    y_proba[:, 1] = 1 - y_proba[:, 0]  # Make probabilities sum to 1
    
    # 3. Evaluate performance
    metrics = tracker.evaluate_performance(y_true, y_pred, y_proba)
    print(f"Current Metrics: {metrics}")
    
    # 4. Check for degradation
    degradation = tracker.check_degradation(metrics)
    print(f"Degradation Check: {degradation}")
    
    # 5. Save history
    tracker.save_performance_history()
    print("Performance history saved\n")

def demo_drift_detector():
    """Demo: Drift Detector Logic"""
    print("=== Drift Detector Demo ===")
    
    # 1. Create reference data (training data)
    reference_data = pd.DataFrame({
        'amount': np.random.normal(100, 50, 1000),
        'time': np.random.uniform(0, 24, 1000),
        'feature1': np.random.normal(0, 1, 1000)
    })
    
    # 2. Create current data (production data - slightly different)
    current_data = pd.DataFrame({
        'amount': np.random.normal(120, 60, 500),  # Different mean/std
        'time': np.random.uniform(0, 24, 500),
        'feature1': np.random.normal(0.5, 1.2, 500)  # Drifted
    })
    
    # 3. Initialize drift detector
    detector = DriftDetector(reference_data, drift_threshold=0.05)
    
    # 4. Detect drift
    drift_results = detector.detect_feature_drift(current_data)
    print("Drift Detection Results:")
    for feature, result in drift_results.items():
        status = "DRIFT DETECTED" if result['drift_detected'] else "NO DRIFT"
        print(f"  {feature}: {status} (p-value: {result['p_value']:.4f})")
    
    print()

def demo_dashboard_generator():
    """Demo: Dashboard Generator Logic"""
    print("=== Dashboard Generator Demo ===")
    
    # 1. Create sample metrics history
    metrics_history = [
        {"timestamp": "2024-01-01T10:00:00", "f1_score": 0.85, "precision": 0.80, "recall": 0.90},
        {"timestamp": "2024-01-01T11:00:00", "f1_score": 0.82, "precision": 0.78, "recall": 0.87},
        {"timestamp": "2024-01-01T12:00:00", "f1_score": 0.79, "precision": 0.75, "recall": 0.84}
    ]
    
    # 2. Create dashboard generator
    dashboard = DashboardGenerator()
    
    # 3. Generate performance dashboard
    dashboard_path = dashboard.generate_performance_dashboard(metrics_history)
    print(f"Performance dashboard saved to: {dashboard_path}")
    
    # 4. Create sample prediction data
    predictions_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'prediction': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        'confidence': np.random.uniform(0.6, 0.95, 100),
        'processing_time_ms': np.random.uniform(50, 200, 100),
        'risk_score': np.random.uniform(0, 1, 100)
    })
    
    # 5. Generate prediction analysis dashboard
    pred_dashboard_path = dashboard.generate_prediction_analysis_dashboard(predictions_data)
    print(f"Prediction dashboard saved to: {pred_dashboard_path}")
    print()

def main():
    """Run all legacy monitoring demos."""
    print("Legacy Monitoring Modules Demo\n")
    
    try:
        demo_performance_tracker()
        demo_drift_detector() 
        demo_dashboard_generator()
        
        print("✅ All legacy modules working correctly!")
        print("\nLogic Summary:")
        print("1. PerformanceTracker: Compares current vs baseline metrics")
        print("2. DriftDetector: Uses statistical tests (KS test) to detect data changes")
        print("3. DashboardGenerator: Creates interactive HTML dashboards with Plotly")
        
    except Exception as e:
        print(f"❌ Error running legacy modules: {e}")

if __name__ == "__main__":
    main()