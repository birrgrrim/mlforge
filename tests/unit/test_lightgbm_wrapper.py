import pytest
from mltune.wrappers import LightGBMModelWrapper

if LightGBMModelWrapper:
    from lightgbm import LGBMClassifier


@pytest.mark.skipif(LightGBMModelWrapper is None, reason="lightgbm not installed")
def test_lightgbm_wrapper_json():
    """Test JSON serialization/deserialization for LightGBM wrapper."""
    params = {"n_estimators": 10}
    features = ["sibsp", "parch"]
    wrapper = LightGBMModelWrapper(params, features)

    assert isinstance(wrapper.model, LGBMClassifier)

    data = wrapper.to_json()
    wrapper2 = LightGBMModelWrapper.from_json(data)

    assert wrapper2.hyperparameters == params
    assert wrapper2.features == features
    assert isinstance(wrapper2.model, LGBMClassifier)


@pytest.mark.skipif(LightGBMModelWrapper is None, reason="lightgbm not installed")
class TestLightGBMWrapperFactory:
    """Test LightGBMModelWrapper factory function."""
    
    def test_factory_with_empty_params(self):
        """Test factory function with empty parameters."""
        wrapper = LightGBMModelWrapper()
        factory = wrapper.get_model_factory()
        
        model = factory({})
        assert isinstance(model, LGBMClassifier)
        assert model.random_state == 1
        assert model.verbosity == -1
    
    def test_factory_with_params(self):
        """Test factory function with provided parameters."""
        wrapper = LightGBMModelWrapper()
        factory = wrapper.get_model_factory()
        
        params = {"n_estimators": 50, "max_depth": 5}
        model = factory(params)
        
        assert isinstance(model, LGBMClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 1
        assert model.verbosity == -1
    
    def test_factory_overrides_verbosity(self):
        """Test that factory overrides verbosity parameter."""
        wrapper = LightGBMModelWrapper()
        factory = wrapper.get_model_factory()
        
        params = {"verbosity": 1}
        model = factory(params)
        
        assert model.verbosity == -1  # Should be overridden


@pytest.mark.skipif(LightGBMModelWrapper is None, reason="lightgbm not installed")
class TestLightGBMWrapperEdgeCases:
    """Test edge cases for LightGBM wrapper."""
    
    def test_lightgbm_wrapper_empty_features(self):
        """Test LightGBMModelWrapper with empty features list."""
        wrapper = LightGBMModelWrapper(features=[])
        assert wrapper.features == []
        assert isinstance(wrapper.model, LGBMClassifier)
    
    def test_lightgbm_wrapper_empty_hyperparameters(self):
        """Test LightGBMModelWrapper with empty hyperparameters."""
        wrapper = LightGBMModelWrapper(hyperparameters={})
        assert wrapper.hyperparameters == {}
        assert isinstance(wrapper.model, LGBMClassifier)
    
    def test_lightgbm_wrapper_none_parameters(self):
        """Test LightGBMModelWrapper with None parameters."""
        wrapper = LightGBMModelWrapper(hyperparameters=None, features=None)
        assert wrapper.hyperparameters == {}
        assert wrapper.features == []
        assert isinstance(wrapper.model, LGBMClassifier) 