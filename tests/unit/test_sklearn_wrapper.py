import pytest
from mltune.wrappers import RandomForestModelWrapper
from sklearn.ensemble import RandomForestClassifier


def test_sklearn_wrapper_json():
    """Test JSON serialization/deserialization for sklearn wrapper."""
    params = {"n_estimators": 10}
    features = ["age", "fare"]
    wrapper = RandomForestModelWrapper(params, features)

    # check model type
    assert isinstance(wrapper.model, RandomForestClassifier)

    data = wrapper.to_json()
    wrapper2 = RandomForestModelWrapper.from_json(data)

    assert wrapper2.hyperparameters == params
    assert wrapper2.features == features
    # check model created on reload
    assert isinstance(wrapper2.model, RandomForestClassifier)


class TestSklearnWrapperFactory:
    """Test RandomForestModelWrapper factory function."""
    
    def test_factory_with_empty_params(self):
        """Test factory function with empty parameters."""
        wrapper = RandomForestModelWrapper()
        factory = wrapper.get_model_factory()
        
        model = factory({})
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == 1
    
    def test_factory_with_params(self):
        """Test factory function with provided parameters."""
        wrapper = RandomForestModelWrapper()
        factory = wrapper.get_model_factory()
        
        params = {"n_estimators": 50, "max_depth": 5}
        model = factory(params)
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 1
    
    def test_factory_overrides_random_state(self):
        """Test that factory overrides random_state parameter."""
        wrapper = RandomForestModelWrapper()
        factory = wrapper.get_model_factory()
        
        params = {"random_state": 42}
        model = factory(params)
        
        assert model.random_state == 1  # Should be overridden


class TestSklearnWrapperEdgeCases:
    """Test edge cases for sklearn wrapper."""
    
    def test_sklearn_wrapper_empty_features(self):
        """Test RandomForestModelWrapper with empty features list."""
        wrapper = RandomForestModelWrapper(features=[])
        assert wrapper.features == []
        assert isinstance(wrapper.model, RandomForestClassifier)
    
    def test_sklearn_wrapper_empty_hyperparameters(self):
        """Test RandomForestModelWrapper with empty hyperparameters."""
        wrapper = RandomForestModelWrapper(hyperparameters={})
        assert wrapper.hyperparameters == {}
        assert isinstance(wrapper.model, RandomForestClassifier)
    
    def test_sklearn_wrapper_none_parameters(self):
        """Test RandomForestModelWrapper with None parameters."""
        wrapper = RandomForestModelWrapper(hyperparameters=None, features=None)
        assert wrapper.hyperparameters == {}
        assert wrapper.features == []
        assert isinstance(wrapper.model, RandomForestClassifier) 