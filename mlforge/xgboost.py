from typing import Any

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("XGBoostModelWrapper requires 'xgboost'. Install with: pip install mlforge[xgboost]") from e

from .base import BaseModelWrapper


class XGBoostModelWrapper(BaseModelWrapper):
    """
    Wraps xgboost.XGBClassifier.

    Args:
        hyperparameters (dict[str, Any]): model parameters
        features (list[str]): list of feature names
    """

    def __init__(self, hyperparameters: dict[str, Any], features: list[str]):
        super().__init__(hyperparameters, features)
        self.model = XGBClassifier(**hyperparameters)
