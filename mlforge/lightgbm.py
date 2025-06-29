from typing import Any

try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError("LightGBMModelWrapper requires 'lightgbm'. Install with: pip install mlforge[lgbm]") from e

from .base import BaseModelWrapper


class LightGBMModelWrapper(BaseModelWrapper):
    """
    Wraps lightgbm.LGBMClassifier.

    Args:
        hyperparameters (dict[str, Any]): model parameters
        features (list[str]): list of feature names
    """

    def __init__(self, hyperparameters: dict[str, Any], features: list[str]):
        super().__init__(hyperparameters, features)
        self.model = LGBMClassifier(**hyperparameters)
