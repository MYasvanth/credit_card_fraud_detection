"""Feature scaling module for credit card fraud detection."""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path

from ..utils.logger import logger
from ..utils.constants import MODELS_PATH


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Custom feature scaler with support for different scaling methods."""
    
    def __init__(self, method: str = "standard", exclude_columns: Optional[list] = None):
        """
        Initialize the feature scaler.
        
        Args:
            method: Scaling method ('standard' or 'robust')
            exclude_columns: Columns to exclude from scaling
        """
        self.method = method
        self.exclude_columns = exclude_columns or []
        self.scaler = None
        self.feature_columns = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureScaler":
        """
        Fit the scaler to the data.
        
        Args:
            X: Input features
            y: Target variable (unused)
            
        Returns:
            Fitted scaler instance
        """
        # Check for NaN values
        if X.isnull().any().any():
            raise ValueError("Input data contains NaN values. Please handle missing values before scaling.")
            
        self.feature_columns = [col for col in X.columns if col not in self.exclude_columns]
        
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
            
        self.scaler.fit(X[self.feature_columns])
        logger.info(f"Fitted {self.method} scaler on {len(self.feature_columns)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler.
        
        Args:
            X: Input features
            
        Returns:
            Scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler must be fitted before transform")
            
        X_scaled = X.copy()
        X_scaled[self.feature_columns] = self.scaler.transform(X[self.feature_columns])
        
        logger.info(f"Transformed {len(self.feature_columns)} features using {self.method} scaling")
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the scaler and transform the data.
        
        Args:
            X: Input features
            y: Target variable (unused)
            
        Returns:
            Scaled features
        """
        return self.fit(X, y).transform(X)
    
    def save(self, filename: str) -> None:
        """
        Save the fitted scaler to disk.
        
        Args:
            filename: Name of the file to save
        """
        Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
        filepath = Path(MODELS_PATH) / f"{filename}.pkl"
        joblib.dump(self, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    @classmethod
    def load(cls, filename: str) -> "FeatureScaler":
        """
        Load a fitted scaler from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Loaded scaler instance
        """
        filepath = Path(MODELS_PATH) / f"{filename}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
            
        scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")
        return scaler


def create_scaler(method: str = "standard", exclude_columns: Optional[list] = None) -> FeatureScaler:
    """
    Factory function to create a feature scaler.
    
    Args:
        method: Scaling method ('standard' or 'robust')
        exclude_columns: Columns to exclude from scaling
        
    Returns:
        FeatureScaler instance
    """
    return FeatureScaler(method=method, exclude_columns=exclude_columns)