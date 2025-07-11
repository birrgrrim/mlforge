import pytest
import pandas as pd
import json
from unittest.mock import MagicMock, patch

from mltune.base import BaseModelWrapper, MLModel


class TestBaseModelWrapper:
    """Test the BaseModelWrapper class."""
    
    def test_init_with_none_parameters(self):
        """Test initialization with None parameters."""
        wrapper = BaseModelWrapper()
        assert wrapper.hyperparameters == {}
        assert wrapper.features == []
        assert wrapper.model is None
    
    def test_init_with_parameters(self):
        """Test initialization with provided parameters."""
        hyperparameters = {"n_estimators": 100}
        features = ["feature1", "feature2"]
        
        wrapper = BaseModelWrapper(hyperparameters, features)
        assert wrapper.hyperparameters == hyperparameters
        assert wrapper.features == features
        assert wrapper.model is None
    
    def test_get_model_factory_not_implemented(self):
        """Test that get_model_factory raises NotImplementedError."""
        wrapper = BaseModelWrapper()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement get_model_factory\\(\\)"):
            wrapper.get_model_factory()
    
    def test_to_json(self):
        """Test JSON serialization."""
        hyperparameters = {"n_estimators": 100}
        features = ["feature1", "feature2"]
        wrapper = BaseModelWrapper(hyperparameters, features)
        
        json_str = wrapper.to_json()
        data = json.loads(json_str)
        
        assert data["model_class"] == "BaseModelWrapper"
        assert data["hyperparameters"] == hyperparameters
        assert data["features"] == features
    
    def test_from_json(self):
        """Test JSON deserialization."""
        hyperparameters = {"n_estimators": 100}
        features = ["feature1", "feature2"]
        wrapper = BaseModelWrapper(hyperparameters, features)
        
        json_str = wrapper.to_json()
        new_wrapper = BaseModelWrapper.from_json(json_str)
        
        assert new_wrapper.hyperparameters == hyperparameters
        assert new_wrapper.features == features
        assert new_wrapper.model is None
    
    def test_fit_with_none_model(self):
        """Test fit method raises error when model is None."""
        wrapper = BaseModelWrapper()
        X = pd.DataFrame({"feature1": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        
        with pytest.raises(ValueError, match="Model has not been initialized"):
            wrapper.fit(X, y)
    
    def test_predict_with_none_model(self):
        """Test predict method raises error when model is None."""
        wrapper = BaseModelWrapper()
        X = pd.DataFrame({"feature1": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Model has not been initialized"):
            wrapper.predict(X)
    
    def test_fit_with_model(self):
        """Test fit method with valid model."""
        wrapper = BaseModelWrapper(features=["feature1"])
        
        # Create a mock model that implements the MLModel protocol
        mock_model = MagicMock()
        wrapper.model = mock_model
        
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        
        wrapper.fit(X, y)
        
        # Check that model.fit was called with correct subset of features
        mock_model.fit.assert_called_once()
        called_X = mock_model.fit.call_args[0][0]
        called_y = mock_model.fit.call_args[0][1]
        
        assert list(called_X.columns) == ["feature1"]
        assert called_y.equals(y)
    
    def test_predict_with_model(self):
        """Test predict method with valid model."""
        wrapper = BaseModelWrapper(features=["feature1"])
        
        # Create a mock model that implements the MLModel protocol
        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 0]
        wrapper.model = mock_model
        
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        
        result = wrapper.predict(X)
        
        # Check that model.predict was called with correct subset of features
        mock_model.predict.assert_called_once()
        called_X = mock_model.predict.call_args[0][0]
        
        assert list(called_X.columns) == ["feature1"]
        assert result == [0, 1, 0]
    
    def test_autotune_with_none_best_params(self):
        """Test autotune handles None best_params gracefully."""
        from mltune.sklearn import RandomForestModelWrapper
        
        wrapper = RandomForestModelWrapper(features=["feature1", "feature2"])
        original_model = wrapper.model  # Store the original model
        
        # Mock the tuning function to return None for best_params
        with patch('mltune.base.tune_model_parameters_and_features') as mock_tune:
            mock_tune.return_value = (None, ["feature1"])
            
            X = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
            y = pd.Series([0, 1])
            
            # Should not raise an error
            wrapper.autotune(X, y, hyperparam_initial_info={})
            
            assert wrapper.hyperparameters is None
            assert wrapper.features == ["feature1"]
            # Model should remain the same (not recreated) when best_params is None
            assert wrapper.model is original_model


class TestMLModelProtocol:
    """Test the MLModel protocol."""
    
    def test_protocol_definition(self):
        """Test that the protocol is properly defined."""
        # This test ensures the protocol exists and has the expected methods
        assert hasattr(MLModel, 'fit')
        assert hasattr(MLModel, 'predict')
        
        # Check that the protocol methods are callable
        assert callable(MLModel.fit)
        assert callable(MLModel.predict) 