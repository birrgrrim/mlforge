import pytest

from mlforge.sklearn import SklearnModelWrapper

# Import optional wrappers safely
try:
    from mlforge.xgboost import XGBoostModelWrapper
except ImportError:
    XGBoostModelWrapper = None

try:
    from mlforge.lightgbm import LightGBMModelWrapper
except ImportError:
    LightGBMModelWrapper = None


def test_sklearn_wrapper_json():
    params = {"n_estimators": 10}
    features = ["age", "fare"]
    wrapper = SklearnModelWrapper(params, features)

    data = wrapper.to_json()
    wrapper2 = SklearnModelWrapper.from_json(data)

    assert wrapper2.hyperparameters == params
    assert wrapper2.features == features


@pytest.mark.skipif(XGBoostModelWrapper is None, reason="xgboost not installed")
def test_xgboost_wrapper_json():
    params = {"n_estimators": 10}
    features = ["pclass", "sex"]
    wrapper = XGBoostModelWrapper(params, features)

    data = wrapper.to_json()
    wrapper2 = XGBoostModelWrapper.from_json(data)

    assert wrapper2.hyperparameters == params
    assert wrapper2.features == features


@pytest.mark.skipif(LightGBMModelWrapper is None, reason="lightgbm not installed")
def test_lightgbm_wrapper_json():
    params = {"n_estimators": 10}
    features = ["sibsp", "parch"]
    wrapper = LightGBMModelWrapper(params, features)

    data = wrapper.to_json()
    wrapper2 = LightGBMModelWrapper.from_json(data)

    assert wrapper2.hyperparameters == params
    assert wrapper2.features == features