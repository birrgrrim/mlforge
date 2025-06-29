from .base import BaseModelWrapper

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("XGBoostModelWrapper requires 'xgboost'. Install with: pip install mlforge[xgboost]") from e


class XGBoostModelWrapper(BaseModelWrapper):
    def __init__(self, hyperparameters, features):
        super().__init__(hyperparameters, features)
        self.model = XGBClassifier(**hyperparameters)