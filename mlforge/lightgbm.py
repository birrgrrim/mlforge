from .base import BaseModelWrapper

try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError("LightGBMModelWrapper requires 'lightgbm'. Install with: pip install mlforge[lgbm]") from e


class LightGBMModelWrapper(BaseModelWrapper):
    def __init__(self, hyperparameters, features):
        super().__init__(hyperparameters, features)
        self.model = LGBMClassifier(**hyperparameters)