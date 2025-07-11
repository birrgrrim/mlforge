import pytest
import pandas as pd
from mltune.wrappers import RandomForestModelWrapper, XGBoostModelWrapper, LightGBMModelWrapper


class TestWrapperFitPredict:
    """Integration tests for wrapper fit and predict functionality."""
    
    def test_sklearn_wrapper_fit_predict(self):
        """Test RandomForestModelWrapper fit and predict methods."""
        wrapper = RandomForestModelWrapper(features=["feature1", "feature2"])
        X = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [5, 6, 7, 8, 9, 10, 11, 12]
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        
        # Test fit
        result = wrapper.fit(X, y)
        assert result is not None
        
        # Test predict
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)
        # Check that predictions are valid (0 or 1 for classification)
        assert all(pred in [0, 1] for pred in predictions)
    
    @pytest.mark.skipif(XGBoostModelWrapper is None, reason="xgboost not installed")
    def test_xgboost_wrapper_fit_predict(self):
        """Test XGBoostModelWrapper fit and predict methods."""
        wrapper = XGBoostModelWrapper(features=["feature1", "feature2"])
        X = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [5, 6, 7, 8, 9, 10, 11, 12]
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        
        # Test fit
        result = wrapper.fit(X, y)
        assert result is not None
        
        # Test predict
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)
        # Check that predictions are valid (0 or 1 for classification)
        assert all(pred in [0, 1] for pred in predictions)
    
    @pytest.mark.skipif(LightGBMModelWrapper is None, reason="lightgbm not installed")
    def test_lightgbm_wrapper_fit_predict(self):
        """Test LightGBMModelWrapper fit and predict methods."""
        wrapper = LightGBMModelWrapper(features=["feature1", "feature2"])
        X = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [5, 6, 7, 8, 9, 10, 11, 12]
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        
        # Test fit
        result = wrapper.fit(X, y)
        assert result is not None
        
        # Test predict
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)
        # Check that predictions are valid (0 or 1 for classification)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_sklearn_wrapper_fit_predict_with_missing_features(self):
        """Test that wrapper correctly handles missing features in input data."""
        wrapper = RandomForestModelWrapper(features=["feature1"])
        X = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],  # Extra feature not used
            "feature3": [9, 10, 11, 12]  # Another extra feature
        })
        y = pd.Series([0, 1, 0, 1])
        
        # Should work fine - wrapper will only use feature1
        result = wrapper.fit(X, y)
        assert result is not None
        
        predictions = wrapper.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_sklearn_wrapper_predict_without_fit(self):
        """Test that predict raises error when model hasn't been fitted."""
        wrapper = RandomForestModelWrapper(features=["feature1"])
        X = pd.DataFrame({"feature1": [1, 2, 3, 4]})
        
        # Model exists but hasn't been fitted
        with pytest.raises(Exception):
            wrapper.predict(X) 