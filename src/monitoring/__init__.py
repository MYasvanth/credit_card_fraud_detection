"""Fraud detection monitoring system.

Core Components (Critical - Always Use):
- fraud_monitor: Global monitoring instance
- monitor_prediction: Decorator for prediction functions

Advanced Components (Optional - Use for detailed analysis):
- PerformanceTracker: Baseline comparison and degradation detection
- DriftDetector: Statistical drift detection
- DashboardGenerator: Interactive HTML dashboards
"""

# Core monitoring (critical)
from .core import fraud_monitor, monitor_prediction

# Advanced monitoring (optional)
try:
    from .advanced import PerformanceTracker, DriftDetector
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

__all__ = ['fraud_monitor', 'monitor_prediction']

if ADVANCED_AVAILABLE:
    __all__.extend(['PerformanceTracker', 'DriftDetector'])