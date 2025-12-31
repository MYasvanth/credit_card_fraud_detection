"""Dashboard generator for model performance and monitoring metrics."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import json

from ...utils.logger import logger
from ..utils.constants import FIGURES_PATH, REPORTS_PATH


class DashboardGenerator:
    """Generate monitoring dashboards for model performance."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize dashboard generator.
        
        Args:
            output_dir: Directory to save dashboard files
        """
        self.output_dir = Path(output_dir) if output_dir else FIGURES_PATH
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_performance_dashboard(
        self,
        metrics_history: List[Dict[str, Any]],
        save_html: bool = True
    ) -> str:
        """
        Generate performance monitoring dashboard.
        
        Args:
            metrics_history: List of performance metrics over time
            save_html: Whether to save as HTML file
            
        Returns:
            Path to generated dashboard
        """
        # Convert to DataFrame
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('F1 Score Over Time', 'Precision vs Recall', 
                          'Prediction Distribution', 'Processing Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # F1 Score over time
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['f1_score'],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Precision vs Recall scatter
        fig.add_trace(
            go.Scatter(
                x=df['recall'],
                y=df['precision'],
                mode='markers',
                name='Precision vs Recall',
                marker=dict(
                    size=10,
                    color=df['f1_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="F1 Score")
                )
            ),
            row=1, col=2
        )
        
        # Prediction distribution (if available)
        if 'fraud_predictions' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=['Normal', 'Fraud'],
                    y=[df['normal_predictions'].iloc[-1], df['fraud_predictions'].iloc[-1]],
                    name='Latest Predictions',
                    marker_color=['green', 'red']
                ),
                row=2, col=1
            )
        
        # Processing time over time (if available)
        if 'processing_time' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['processing_time'],
                    mode='lines+markers',
                    name='Processing Time (ms)',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Model Performance Dashboard",
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="F1 Score", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.update_xaxes(title_text="Prediction Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Processing Time (ms)", row=2, col=2)
        
        # Save dashboard
        if save_html:
            dashboard_path = self.output_dir / "performance_dashboard.html"
            pyo.plot(fig, filename=str(dashboard_path), auto_open=False)
            logger.info(f"Performance dashboard saved to {dashboard_path}")
            return str(dashboard_path)
        
        return fig
    
    def generate_drift_dashboard(
        self,
        drift_results: Dict[str, Any],
        feature_distributions: Dict[str, Dict[str, np.ndarray]],
        save_html: bool = True
    ) -> str:
        """
        Generate data drift monitoring dashboard.
        
        Args:
            drift_results: Drift detection results
            feature_distributions: Feature distributions (reference vs current)
            save_html: Whether to save as HTML file
            
        Returns:
            Path to generated dashboard
        """
        # Create subplots for drift visualization
        n_features = len(feature_distributions)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows + 1, cols=n_cols,
            subplot_titles=list(feature_distributions.keys()) + ['Drift Summary'],
            specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)] +\n                  [[{"colspan": n_cols, "secondary_y": False}, None, None]]\n        )\n        \n        # Plot feature distributions\n        row, col = 1, 1\n        for feature_name, distributions in feature_distributions.items():\n            # Reference distribution\n            fig.add_trace(\n                go.Histogram(\n                    x=distributions['reference'],\n                    name=f'{feature_name} (Reference)',\n                    opacity=0.7,\n                    nbinsx=30,\n                    marker_color='blue'\n                ),\n                row=row, col=col\n            )\n            \n            # Current distribution\n            fig.add_trace(\n                go.Histogram(\n                    x=distributions['current'],\n                    name=f'{feature_name} (Current)',\n                    opacity=0.7,\n                    nbinsx=30,\n                    marker_color='red'\n                ),\n                row=row, col=col\n            )\n            \n            # Move to next subplot\n            col += 1\n            if col > n_cols:\n                col = 1\n                row += 1\n        \n        # Drift summary\n        drift_scores = []\n        feature_names = []\n        colors = []\n        \n        for feature, score_info in drift_results['drift_scores'].items():\n            drift_scores.append(score_info['ks_statistic'])\n            feature_names.append(feature)\n            colors.append('red' if score_info['drift_detected'] else 'green')\n        \n        fig.add_trace(\n            go.Bar(\n                x=feature_names,\n                y=drift_scores,\n                name='KS Statistic',\n                marker_color=colors\n            ),\n            row=n_rows + 1, col=1\n        )\n        \n        # Update layout\n        fig.update_layout(\n            title_text=\"Data Drift Monitoring Dashboard\",\n            showlegend=True,\n            height=300 * (n_rows + 1),\n            width=1200\n        )\n        \n        # Save dashboard\n        if save_html:\n            dashboard_path = self.output_dir / \"drift_dashboard.html\"\n            pyo.plot(fig, filename=str(dashboard_path), auto_open=False)\n            logger.info(f\"Drift dashboard saved to {dashboard_path}\")\n            return str(dashboard_path)\n        \n        return fig\n    \n    def generate_prediction_analysis_dashboard(\n        self,\n        predictions_data: pd.DataFrame,\n        save_html: bool = True\n    ) -> str:\n        \"\"\"\n        Generate prediction analysis dashboard.\n        \n        Args:\n            predictions_data: DataFrame with prediction logs\n            save_html: Whether to save as HTML file\n            \n        Returns:\n            Path to generated dashboard\n        \"\"\"\n        fig = make_subplots(\n            rows=2, cols=2,\n            subplot_titles=('Predictions Over Time', 'Risk Score Distribution',\n                          'Confidence Levels', 'Processing Time Distribution')\n        )\n        \n        # Predictions over time\n        predictions_data['timestamp'] = pd.to_datetime(predictions_data['timestamp'])\n        daily_predictions = predictions_data.groupby(\n            predictions_data['timestamp'].dt.date\n        )['prediction'].agg(['count', 'sum']).reset_index()\n        \n        fig.add_trace(\n            go.Scatter(\n                x=daily_predictions['timestamp'],\n                y=daily_predictions['count'],\n                mode='lines+markers',\n                name='Total Predictions',\n                line=dict(color='blue')\n            ),\n            row=1, col=1\n        )\n        \n        fig.add_trace(\n            go.Scatter(\n                x=daily_predictions['timestamp'],\n                y=daily_predictions['sum'],\n                mode='lines+markers',\n                name='Fraud Predictions',\n                line=dict(color='red')\n            ),\n            row=1, col=1\n        )\n        \n        # Risk score distribution\n        fig.add_trace(\n            go.Histogram(\n                x=predictions_data['risk_score'],\n                name='Risk Score Distribution',\n                nbinsx=50,\n                marker_color='orange'\n            ),\n            row=1, col=2\n        )\n        \n        # Confidence levels\n        confidence_counts = predictions_data['confidence'].value_counts()\n        fig.add_trace(\n            go.Bar(\n                x=confidence_counts.index,\n                y=confidence_counts.values,\n                name='Confidence Levels',\n                marker_color=['green', 'yellow', 'red']\n            ),\n            row=2, col=1\n        )\n        \n        # Processing time distribution\n        fig.add_trace(\n            go.Histogram(\n                x=predictions_data['processing_time_ms'],\n                name='Processing Time (ms)',\n                nbinsx=30,\n                marker_color='purple'\n            ),\n            row=2, col=2\n        )\n        \n        # Update layout\n        fig.update_layout(\n            title_text=\"Prediction Analysis Dashboard\",\n            showlegend=True,\n            height=800,\n            width=1200\n        )\n        \n        # Save dashboard\n        if save_html:\n            dashboard_path = self.output_dir / \"prediction_analysis_dashboard.html\"\n            pyo.plot(fig, filename=str(dashboard_path), auto_open=False)\n            logger.info(f\"Prediction analysis dashboard saved to {dashboard_path}\")\n            return str(dashboard_path)\n        \n        return fig\n    \n    def generate_model_comparison_dashboard(\n        self,\n        model_results: List[Dict[str, Any]],\n        save_html: bool = True\n    ) -> str:\n        \"\"\"\n        Generate model comparison dashboard.\n        \n        Args:\n            model_results: List of model performance results\n            save_html: Whether to save as HTML file\n            \n        Returns:\n            Path to generated dashboard\n        \"\"\"\n        df = pd.DataFrame(model_results)\n        \n        fig = make_subplots(\n            rows=2, cols=2,\n            subplot_titles=('Model Performance Comparison', 'Training Time vs Performance',\n                          'Feature Importance', 'ROC Curves')\n        )\n        \n        # Model performance comparison\n        metrics = ['accuracy', 'precision', 'recall', 'f1_score']\n        for metric in metrics:\n            if metric in df.columns:\n                fig.add_trace(\n                    go.Bar(\n                        x=df['model_name'],\n                        y=df[metric],\n                        name=metric.title(),\n                        opacity=0.8\n                    ),\n                    row=1, col=1\n                )\n        \n        # Training time vs performance\n        if 'training_time' in df.columns and 'f1_score' in df.columns:\n            fig.add_trace(\n                go.Scatter(\n                    x=df['training_time'],\n                    y=df['f1_score'],\n                    mode='markers+text',\n                    text=df['model_name'],\n                    textposition=\"top center\",\n                    name='Training Time vs F1',\n                    marker=dict(size=10)\n                ),\n                row=1, col=2\n            )\n        \n        # Update layout\n        fig.update_layout(\n            title_text=\"Model Comparison Dashboard\",\n            showlegend=True,\n            height=800,\n            width=1200\n        )\n        \n        # Save dashboard\n        if save_html:\n            dashboard_path = self.output_dir / \"model_comparison_dashboard.html\"\n            pyo.plot(fig, filename=str(dashboard_path), auto_open=False)\n            logger.info(f\"Model comparison dashboard saved to {dashboard_path}\")\n            return str(dashboard_path)\n        \n        return fig\n    \n    def save_dashboard_config(self, config: Dict[str, Any]) -> None:\n        \"\"\"\n        Save dashboard configuration.\n        \n        Args:\n            config: Dashboard configuration\n        \"\"\"\n        config_path = self.output_dir / \"dashboard_config.json\"\n        with open(config_path, 'w') as f:\n            json.dump(config, f, indent=2)\n        logger.info(f\"Dashboard configuration saved to {config_path}\")\n\n\ndef create_dashboard_generator(output_dir: Optional[str] = None) -> DashboardGenerator:\n    \"\"\"\n    Factory function to create dashboard generator.\n    \n    Args:\n        output_dir: Output directory for dashboards\n        \n    Returns:\n        DashboardGenerator instance\n    \"\"\"\n    return DashboardGenerator(output_dir)