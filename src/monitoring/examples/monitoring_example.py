"""Example of minimal fraud detection monitoring usage."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
from src.monitoring.core.monitor import fraud_monitor, monitor_prediction


@monitor_prediction
def predict_fraud(transaction_data):
    """Example prediction function with monitoring."""
    # Simulate prediction logic
    prediction = np.random.choice([0, 1], p=[0.95, 0.05])
    confidence = np.random.uniform(0.6, 0.95)
    return prediction, confidence


def main():
    """Demonstrate monitoring functionality."""
    
    # Simulate some predictions
    for i in range(10):
        result = predict_fraud(f"transaction_{i}")
        print(f"Transaction {i}: Prediction={result[0]}, Confidence={result[1]:.3f}")
    
    # Simulate performance tracking
    y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    
    metrics = fraud_monitor.track_performance(y_true, y_pred)
    print(f"\nPerformance Metrics: {metrics}")
    
    # Check monitoring status
    status = fraud_monitor.get_status()
    print(f"\nMonitoring Status: {status}")
    
    # Show any alerts
    if fraud_monitor.alerts:
        print(f"\nAlerts: {len(fraud_monitor.alerts)}")
        for alert in fraud_monitor.alerts:
            print(f"  - {alert['type']}: {alert['message']}")


if __name__ == "__main__":
    main()