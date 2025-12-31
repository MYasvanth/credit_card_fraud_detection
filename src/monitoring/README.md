# Fraud Detection Monitoring

## 📁 Logical File Organization

```
monitoring/
├── core/                    # 🔴 CRITICAL - Always use
│   ├── monitor.py          # Main monitoring logic
│   └── __init__.py         # Core exports
├── advanced/               # 🟡 OPTIONAL - Use for detailed analysis
│   ├── performance_tracker.py  # Baseline comparison
│   ├── drift_detector.py      # Statistical drift detection
│   ├── dashboard_generator.py  # Interactive dashboards
│   └── __init__.py            # Advanced exports
├── examples/               # 📚 USAGE EXAMPLES
│   ├── monitoring_example.py     # Core monitoring demo
│   └── advanced_monitoring_example.py # Advanced features demo
├── __init__.py            # Main module interface
└── README.md              # This file
```

## 🚀 Quick Start

### Core Monitoring (Critical)
```python
from src.monitoring import fraud_monitor, monitor_prediction

@monitor_prediction
def predict_fraud(data):
    return prediction, confidence

# Track performance
fraud_monitor.track_performance(y_true, y_pred)
status = fraud_monitor.get_status()
```

### Advanced Monitoring (Optional)
```python
from src.monitoring import PerformanceTracker, DriftDetector

# Performance tracking with baselines
tracker = PerformanceTracker(baseline_metrics)
metrics = tracker.evaluate_performance(y_true, y_pred)

# Drift detection
detector = DriftDetector(reference_data)
drift_results = detector.detect_feature_drift(current_data)
```

## 📊 Monitoring Priorities

| Component | Priority | Use Case |
|-----------|----------|----------|
| Core Monitor | 🔴 Critical | Real-time alerts, audit logs |
| Performance Tracker | 🟡 Important | Baseline comparison |
| Drift Detector | 🟡 Important | Data quality monitoring |
| Dashboard Generator | 🟢 Nice-to-have | Visual reporting |

## 🏃‍♂️ Run Examples

```bash
# Core monitoring (always works)
python src/monitoring/examples/monitoring_example.py

# Advanced monitoring (works without external dependencies)
python src/monitoring/examples/working_advanced_example.py
```

## ⚠️ Dependencies

**Core Monitoring:** No external dependencies (uses only sklearn, numpy, pandas)

**Advanced Monitoring:** 
- Works without external dependencies (basic functionality)
- Optional: Install `plotly` for dashboard generation
- Optional: Install `evidently` for advanced drift detection

```bash
# Optional dependencies
pip install plotly evidently
```