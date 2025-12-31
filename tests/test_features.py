"""Unit tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from unittest.mock import patch, MagicMock

from src.features.scaler import FeatureScaler, create_scaler
from src.features.pca_transformer import PCATransformer, create_pca_transformer
from src.features.smote import SMOTETransformer, apply_smote


class TestFeatureScaler:
    """Test cases for FeatureScaler."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.normal(-2, 0.5, 100),
            'exclude_me': np.random.randint(0, 2, 100)
        })
        return X
    
    def test_standard_scaler_initialization(self):
        """Test FeatureScaler initialization with standard method."""
        scaler = FeatureScaler(method="standard")
        assert scaler.method == "standard"
        assert scaler.exclude_columns == []
        assert scaler.scaler is None
    
    def test_robust_scaler_initialization(self):
        """Test FeatureScaler initialization with robust method."""
        scaler = FeatureScaler(method="robust", exclude_columns=["exclude_me"])
        assert scaler.method == "robust"
        assert scaler.exclude_columns == ["exclude_me"]
    
    def test_invalid_method_raises_error(self):
        """Test that invalid scaling method raises ValueError."""
        scaler = FeatureScaler(method="invalid")
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Unknown scaling method"):
            scaler.fit(X)
    
    def test_fit_transform_standard(self, sample_data):
        """Test fit_transform with standard scaling."""
        scaler = FeatureScaler(method="standard", exclude_columns=["exclude_me"])
        X_scaled = scaler.fit_transform(sample_data)
        
        # Check that excluded column is unchanged
        pd.testing.assert_series_equal(X_scaled["exclude_me"], sample_data["exclude_me"])
        
        # Check that scaled features have mean ~0 and std ~1
        scaled_features = ["feature1", "feature2", "feature3"]
        for feature in scaled_features:
            assert abs(X_scaled[feature].mean()) < 0.1
            assert abs(X_scaled[feature].std() - 1.0) < 0.1
    
    def test_fit_transform_robust(self, sample_data):
        """Test fit_transform with robust scaling."""
        scaler = FeatureScaler(method="robust")
        X_scaled = scaler.fit_transform(sample_data)
        
        # Check that all features are scaled
        assert X_scaled.shape == sample_data.shape
        assert list(X_scaled.columns) == list(sample_data.columns)
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises ValueError."""
        scaler = FeatureScaler()
        
        with pytest.raises(ValueError, match="Scaler must be fitted before transform"):
            scaler.transform(sample_data)
    
    @patch('src.features.scaler.joblib.dump')
    @patch('src.features.scaler.Path')
    def test_save_scaler(self, mock_path, mock_dump, sample_data):
        """Test saving fitted scaler."""
        scaler = FeatureScaler()
        scaler.fit(sample_data)
        
        scaler.save("test_scaler")
        
        mock_dump.assert_called_once()
        mock_path.assert_called()
    
    def test_create_scaler_factory(self):
        """Test create_scaler factory function."""
        scaler = create_scaler(method="robust", exclude_columns=["test"])
        
        assert isinstance(scaler, FeatureScaler)
        assert scaler.method == "robust"
        assert scaler.exclude_columns == ["test"]

    def test_fit_transform_with_nan(self, sample_data):
        """Test fit_transform with data containing NaNs."""
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.loc[0, "feature1"] = np.nan
        scaler = FeatureScaler(method="standard")
        with pytest.raises(ValueError):
            scaler.fit_transform(sample_data_with_nan)


class TestPCATransformer:
    """Test cases for PCATransformer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X, _ = make_classification(n_samples=100, n_features=10, n_informative=8, 
                                 n_redundant=2, random_state=42)
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    def test_pca_initialization_with_components(self):
        """Test PCA initialization with specified components."""
        pca = PCATransformer(n_components=5)
        assert pca.n_components == 5
        assert pca.variance_threshold == 0.95
    
    def test_pca_initialization_with_variance_threshold(self):
        """Test PCA initialization with variance threshold."""
        pca = PCATransformer(variance_threshold=0.9)
        assert pca.n_components is None
        assert pca.variance_threshold == 0.9
    
    def test_fit_transform_with_components(self, sample_data):
        """Test fit_transform with specified components."""
        pca = PCATransformer(n_components=5)
        X_pca = pca.fit_transform(sample_data)
        
        assert X_pca.shape == (100, 5)
        assert list(X_pca.columns) == ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    
    def test_fit_transform_with_variance_threshold(self, sample_data):
        """Test fit_transform with variance threshold."""
        pca = PCATransformer(variance_threshold=0.8)
        X_pca = pca.fit_transform(sample_data)
        
        # Should determine components automatically
        assert X_pca.shape[0] == 100
        assert X_pca.shape[1] <= 10
        assert pca.n_components is not None
    
    def test_get_explained_variance_ratio(self, sample_data):
        """Test getting explained variance ratio."""
        pca = PCATransformer(n_components=3)
        pca.fit(sample_data)
        
        variance_ratio = pca.get_explained_variance_ratio()
        assert len(variance_ratio) == 3
        assert all(0 <= ratio <= 1 for ratio in variance_ratio)
    
    def test_get_feature_importance(self, sample_data):
        """Test getting feature importance for components."""
        pca = PCATransformer(n_components=3)
        pca.fit(sample_data)
        
        importance = pca.get_feature_importance(component_idx=0)
        assert len(importance) == 10
        # Check if feature names are the same ignoring order
        assert set(importance.index.tolist()) == set(sample_data.columns.tolist())
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises ValueError."""
        pca = PCATransformer()
        
        with pytest.raises(ValueError, match="PCA must be fitted before transform"):
            pca.transform(sample_data)
    
    def test_create_pca_transformer_factory(self):
        """Test create_pca_transformer factory function."""
        pca = create_pca_transformer(n_components=5, variance_threshold=0.9)
        
        assert isinstance(pca, PCATransformer)
        assert pca.n_components == 5
        assert pca.variance_threshold == 0.9


class TestSMOTETransformer:
    """Test cases for SMOTETransformer."""
    
    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced sample data for testing."""
        np.random.seed(42)
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=8,
                                 n_redundant=2, n_clusters_per_class=1,
                                 weights=[0.9, 0.1], random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y, name='target')
        return X_df, y_series
    
    def test_smote_initialization(self):
        """Test SMOTE initialization."""
        smote = SMOTETransformer(method="smote", k_neighbors=3)
        assert smote.method == "smote"
        assert smote.k_neighbors == 3
        assert smote.random_state == 42
    
    def test_invalid_method_raises_error(self, imbalanced_data):
        """Test that invalid SMOTE method raises ValueError."""
        X, y = imbalanced_data
        smote = SMOTETransformer(method="invalid")
        
        with pytest.raises(ValueError, match="Unknown SMOTE method"):
            smote.fit(X, y)
    
    def test_fit_transform_smote(self, imbalanced_data):
        """Test fit_transform with SMOTE."""
        X, y = imbalanced_data
        smote = SMOTETransformer(method="smote")
        
        X_resampled, y_resampled = smote.fit_transform(X, y)
        
        # Check that minority class is oversampled
        original_counts = y.value_counts()
        resampled_counts = y_resampled.value_counts()
        
        assert resampled_counts[1] > original_counts[1]  # Minority class increased
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)
    
    def test_fit_transform_borderline(self, imbalanced_data):
        """Test fit_transform with BorderlineSMOTE."""
        X, y = imbalanced_data
        smote = SMOTETransformer(method="borderline")
        
        X_resampled, y_resampled = smote.fit_transform(X, y)
        
        # Check basic properties
        assert X_resampled.shape[1] == X.shape[1]  # Same number of features
        assert len(X_resampled) == len(y_resampled)  # Same number of samples
    
    def test_get_sampling_info(self, imbalanced_data):
        """Test getting sampling information."""
        X, y = imbalanced_data
        smote = SMOTETransformer()
        
        X_resampled, y_resampled = smote.fit_transform(X, y)
        sampling_info = smote.get_sampling_info()
        
        assert "method" in sampling_info
        assert "original_distribution" in sampling_info
        assert "resampled_distribution" in sampling_info
        assert "samples_added" in sampling_info
    
    def test_transform_without_fit_raises_error(self, imbalanced_data):
        """Test that transform without fit raises ValueError."""
        X, y = imbalanced_data
        smote = SMOTETransformer()
        
        with pytest.raises(ValueError, match="SMOTE transformer must be fitted before transform"):
            smote.transform(X, y)
    
    def test_apply_smote_function(self, imbalanced_data):
        """Test apply_smote convenience function."""
        X, y = imbalanced_data
        
        X_resampled, y_resampled = apply_smote(X, y, method="smote")
        
        # Check that resampling occurred
        assert len(X_resampled) >= len(X)
        assert len(y_resampled) >= len(y)
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)


# Integration tests
class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering pipeline."""
    
    @pytest.fixture
    def pipeline_data(self):
        """Create data for pipeline testing."""
        np.random.seed(42)
        X, y = make_classification(n_samples=500, n_features=15, n_informative=10,
                                 n_redundant=5, n_clusters_per_class=1,
                                 weights=[0.8, 0.2], random_state=42)
        X_df = pd.DataFrame(X, columns=[f'V{i+1}' for i in range(15)])
        y_series = pd.Series(y, name='Class')
        return X_df, y_series
    
    def test_full_feature_engineering_pipeline(self, pipeline_data):
        """Test complete feature engineering pipeline."""
        X, y = pipeline_data
        
        # Step 1: Scaling
        scaler = FeatureScaler(method="standard")
        X_scaled = scaler.fit_transform(X)
        
        # Step 2: PCA (optional)
        pca = PCATransformer(n_components=10)
        X_pca = pca.fit_transform(X_scaled)
        
        # Step 3: SMOTE
        smote = SMOTETransformer(method="smote")
        X_final, y_final = smote.fit_transform(X_pca, y)
        
        # Verify pipeline results
        assert X_final.shape[1] == 10  # PCA components
        assert len(X_final) >= len(X)  # SMOTE increased samples
        assert len(y_final) == len(X_final)  # Consistent lengths
        
        # Verify class balance improved
        original_ratio = y.value_counts()[1] / y.value_counts()[0]
        final_ratio = y_final.value_counts()[1] / y_final.value_counts()[0]
        assert final_ratio > original_ratio
