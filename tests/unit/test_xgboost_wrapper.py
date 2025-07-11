import pytest
from mltune.wrappers import XGBoostModelWrapper

if XGBoostModelWrapper:
    from xgboost import XGBClassifier


@pytest.mark.skipif(XGBoostModelWrapper is None, reason="xgboost not installed")
def test_xgboost_wrapper_json():
    """Test JSON serialization/deserialization for XGBoost wrapper."""
    params = {"n_estimators": 10}
    features = ["pclass", "sex"]
    wrapper = XGBoostModelWrapper(params, features)

    assert isinstance(wrapper.model, XGBClassifier)

    data = wrapper.to_json()
    wrapper2 = XGBoostModelWrapper.from_json(data)

    assert wrapper2.hyperparameters == params
    assert wrapper2.features == features
    assert isinstance(wrapper2.model, XGBClassifier)


@pytest.mark.skipif(XGBoostModelWrapper is None, reason="xgboost not installed")
class TestXGBoostWrapperFactory:
    """Test XGBoostModelWrapper factory function."""
    
    def test_factory_with_empty_params(self):
        """Test factory function with empty parameters."""
        wrapper = XGBoostModelWrapper()
        factory = wrapper.get_model_factory()
        
        model = factory({})
        assert isinstance(model, XGBClassifier)
        assert model.random_state == 1
        assert model.eval_metric == "logloss"
    
    def test_factory_with_params(self):
        """Test factory function with provided parameters."""
        wrapper = XGBoostModelWrapper()
        factory = wrapper.get_model_factory()
        
        params = {"n_estimators": 50, "max_depth": 5}
        model = factory(params)
        
        assert isinstance(model, XGBClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 1
        assert model.eval_metric == "logloss"
    
    def test_factory_overrides_eval_metric(self):
        """Test that factory overrides eval_metric parameter."""
        wrapper = XGBoostModelWrapper()
        factory = wrapper.get_model_factory()
        
        params = {"eval_metric": "auc"}
        model = factory(params)
        
        assert model.eval_metric == "logloss"  # Should be overridden


@pytest.mark.skipif(XGBoostModelWrapper is None, reason="xgboost not installed")
class TestXGBoostWrapperEdgeCases:
    """Test edge cases for XGBoost wrapper."""
    
    def test_xgboost_wrapper_empty_features(self):
        """Test XGBoostModelWrapper with empty features list."""
        wrapper = XGBoostModelWrapper(features=[])
        assert wrapper.features == []
        assert isinstance(wrapper.model, XGBClassifier)
    
    def test_xgboost_wrapper_empty_hyperparameters(self):
        """Test XGBoostModelWrapper with empty hyperparameters."""
        wrapper = XGBoostModelWrapper(hyperparameters={})
        assert wrapper.hyperparameters == {}
        assert isinstance(wrapper.model, XGBClassifier)
    
    def test_xgboost_wrapper_none_parameters(self):
        """Test XGBoostModelWrapper with None parameters."""
        wrapper = XGBoostModelWrapper(hyperparameters=None, features=None)
        assert wrapper.hyperparameters == {}
        assert wrapper.features == []
        assert isinstance(wrapper.model, XGBClassifier) 