"""Critical fraud detection monitoring - minimal implementation."""

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from ...utils.logger import logger
from ...utils.constants import REPORTS_PATH


class FraudMonitor:
    """Minimal monitoring for critical fraud detection requirements."""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        
    def log_prediction(self, prediction: int, confidence: float, processing_time: float):
        """Log prediction with response time tracking."""
        timestamp = datetime.now().isoformat()
        
        # Critical: Response time monitoring
        if processing_time > 1000:  # > 1 second
            self.alerts.append({
                "timestamp": timestamp,
                "type": "RESPONSE_TIME_ALERT",
                "message": f"Slow response: {processing_time}ms"
            })
            logger.warning(f"Response time alert: {processing_time}ms")
        
        # Log prediction for audit trail
        log_entry = {
            "timestamp": timestamp,
            "prediction": int(prediction),
            "confidence": float(confidence),
            "processing_time_ms": float(processing_time)
        }
        
        # Save to file for compliance
        self._save_prediction_log(log_entry)
        
    def track_performance(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Track model performance and detect degradation."""
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }
        
        self.metrics_history.append(metrics)
        
        # Critical: Performance degradation alert
        if f1 < 0.8:  # Threshold for immediate action
            alert = {
                "timestamp": metrics["timestamp"],
                "type": "PERFORMANCE_ALERT",
                "message": f"F1 score dropped to {f1:.3f}"
            }
            self.alerts.append(alert)
            logger.error(f"Performance alert: F1 score {f1:.3f}")
        
        return metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        if not self.metrics_history:
            return {"status": "NO_DATA", "alerts": len(self.alerts)}
        
        latest = self.metrics_history[-1]
        return {
            "status": "ACTIVE",
            "latest_f1": latest["f1_score"],
            "alerts": len(self.alerts),
            "last_update": latest["timestamp"]
        }
    
    def _save_prediction_log(self, log_entry: Dict[str, Any]):
        """Save prediction log for audit compliance."""
        log_file = Path(REPORTS_PATH) / "prediction_logs.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


# Global monitor instance
fraud_monitor = FraudMonitor()


def monitor_prediction(func):
    """Decorator to monitor prediction functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        processing_time = (time.time() - start_time) * 1000
        
        # Extract prediction and confidence from result
        if isinstance(result, tuple):
            prediction, confidence = result[0], result[1] if len(result) > 1 else 0.5
        else:
            prediction, confidence = result, 0.5
            
        fraud_monitor.log_prediction(prediction, confidence, processing_time)
        return result
    return wrapper
