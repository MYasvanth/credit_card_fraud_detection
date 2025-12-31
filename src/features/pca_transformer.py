"""PCA transformer module for dimensionality reduction."""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from ..utils.logger import logger
from ..utils.constants import MODELS_PATH, FIGURES_PATH


class PCATransformer(BaseEstimator, TransformerMixin):
    """PCA transformer with explained variance analysis."""
    
    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = 0.95):
        """
        Initialize PCA transformer.
        
        Args:
            n_components: Number of components to keep
            variance_threshold: Minimum variance to retain if n_components is None
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PCATransformer":
        """
        Fit PCA to the data.
        
        Args:
            X: Input features
            y: Target variable (unused)
            
        Returns:
            Fitted PCA transformer
        """
        self.feature_names = X.columns.tolist()
        
        if self.n_components is None:
            # Determine optimal number of components based on variance threshold
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum_variance >= self.variance_threshold) + 1
            
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA fitted with {self.n_components} components, "
                   f"explaining {explained_variance:.3f} of variance")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted PCA.
        
        Args:
            X: Input features
            
        Returns:
            PCA-transformed features
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted before transform")
            
        X_pca = self.pca.transform(X)
        
        # Create DataFrame with proper column names
        columns = [f"PC{i+1}" for i in range(self.n_components)]
        X_transformed = pd.DataFrame(X_pca, columns=columns, index=X.index)
        
        logger.info(f"Transformed data to {self.n_components} principal components")
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit PCA and transform data.
        
        Args:
            X: Input features
            y: Target variable (unused)
            
        Returns:
            PCA-transformed features
        """
        return self.fit(X, y).transform(X)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio for each component.
        
        Returns:
            Explained variance ratios
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted first")
        return self.pca.explained_variance_ratio_
    
    def get_feature_importance(self, component_idx: int = 0) -> pd.Series:
        """
        Get feature importance for a specific principal component.
        
        Args:
            component_idx: Index of the principal component
            
        Returns:
            Feature importance scores
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted first")
            
        if component_idx >= self.n_components:
            raise ValueError(f"Component index {component_idx} out of range")
            
        importance = np.abs(self.pca.components_[component_idx])
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
    
    def plot_explained_variance(self, save_path: Optional[str] = None) -> None:
        """
        Plot explained variance ratio.
        
        Args:
            save_path: Path to save the plot
        """
        if self.pca is None:
            raise ValueError("PCA must be fitted first")
            
        plt.figure(figsize=(10, 6))
        
        # Individual variance
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                self.pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Individual Explained Variance')
        
        # Cumulative variance
        plt.subplot(1, 2, 2)
        cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-')
        plt.axhline(y=self.variance_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({self.variance_threshold})')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            Path(FIGURES_PATH).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(FIGURES_PATH) / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explained variance plot saved to {save_path}")
        
        plt.show()
    
    def save(self, filename: str) -> None:
        """
        Save fitted PCA transformer.
        
        Args:
            filename: Name of the file to save
        """
        Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
        filepath = Path(MODELS_PATH) / f"{filename}.pkl"
        joblib.dump(self, filepath)
        logger.info(f"PCA transformer saved to {filepath}")
    
    @classmethod
    def load(cls, filename: str) -> "PCATransformer":
        """
        Load fitted PCA transformer.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Loaded PCA transformer
        """
        filepath = Path(MODELS_PATH) / f"{filename}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"PCA transformer file not found: {filepath}")
            
        transformer = joblib.load(filepath)
        logger.info(f"PCA transformer loaded from {filepath}")
        return transformer


def create_pca_transformer(n_components: Optional[int] = None, 
                          variance_threshold: float = 0.95) -> PCATransformer:
    """
    Factory function to create PCA transformer.
    
    Args:
        n_components: Number of components to keep
        variance_threshold: Minimum variance to retain
        
    Returns:
        PCATransformer instance
    """
    return PCATransformer(n_components=n_components, variance_threshold=variance_threshold)