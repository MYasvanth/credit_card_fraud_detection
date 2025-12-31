"""SMOTE oversampling module for handling imbalanced datasets."""

from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path
from collections import Counter

from ..utils.logger import logger
from ..utils.constants import MODELS_PATH


class SMOTETransformer(BaseEstimator, TransformerMixin):
    """SMOTE transformer with multiple oversampling strategies."""
    
    def __init__(self, 
                 strategy: str = "auto",
                 method: str = "smote",
                 random_state: int = 42,
                 k_neighbors: int = 5,
                 sampling_strategy: str = "auto"):
        """
        Initialize SMOTE transformer.
        
        Args:
            strategy: Sampling strategy ('auto', 'minority', 'not majority', 'all')
            method: Oversampling method ('smote', 'borderline', 'adasyn')
            random_state: Random state for reproducibility
            k_neighbors: Number of nearest neighbors
            sampling_strategy: Sampling strategy for the resampler
        """
        self.strategy = strategy
        self.method = method
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.sampler = None
        self.original_distribution = None
        self.resampled_distribution = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SMOTETransformer":
        """
        Fit the SMOTE transformer.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Fitted SMOTE transformer
        """
        self.original_distribution = Counter(y)
        
        if self.method == "smote":
            self.sampler = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.k_neighbors
            )
        elif self.method == "borderline":
            self.sampler = BorderlineSMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.k_neighbors
            )
        elif self.method == "adasyn":
            self.sampler = ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_neighbors=self.k_neighbors
            )
        else:
            raise ValueError(f"Unknown SMOTE method: {self.method}")
            
        self.sampler.fit(X, y)
        
        logger.info(f"SMOTE transformer fitted with method: {self.method}")
        logger.info(f"Original class distribution: {self.original_distribution}")
        
        return self
    
    def transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform the data using SMOTE.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Resampled features and target
        """
        if self.sampler is None:
            raise ValueError("SMOTE transformer must be fitted before transform")
            
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        self.resampled_distribution = Counter(y_resampled)
        
        logger.info(f"Data resampled using {self.method}")
        logger.info(f"Resampled class distribution: {self.resampled_distribution}")
        
        return X_resampled, y_resampled
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        return self.fit(X, y).transform(X, y)
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """
        Get information about the sampling process.
        
        Returns:
            Dictionary with sampling information
        """
        if self.original_distribution is None:
            raise ValueError("SMOTE transformer must be fitted first")
            
        samples_added = {}
        if self.resampled_distribution:
            for class_label in self.original_distribution:
                original_count = self.original_distribution[class_label]
                resampled_count = self.resampled_distribution.get(class_label, 0)
                samples_added[class_label] = resampled_count - original_count
        
        return {
            "method": self.method,
            "original_distribution": dict(self.original_distribution),
            "resampled_distribution": dict(self.resampled_distribution) if self.resampled_distribution else {},
            "samples_added": samples_added
        }


def apply_smote(X: pd.DataFrame, y: pd.Series, 
                method: str = "smote", 
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to the dataset."""
    transformer = SMOTETransformer(method=method, random_state=random_state)
    return transformer.fit_transform(X, y)
