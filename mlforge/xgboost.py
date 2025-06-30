from typing import Any

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("XGBoostModelWrapper requires 'xgboost'. Install with: pip install mlforge[xgboost]") from e

from .base import BaseModelWrapper


class XGBoostModelWrapper(BaseModelWrapper):
    """
    Wrapper for xgboost.XGBClassifier.

    Initializes the underlying XGBClassifier with given hyperparameters.

    Parameters
    ----------
    hyperparameters : dict of str to Any
        Model hyperparameters to configure RandomForestClassifier.
    features : list of str
        List of feature names to use during training and prediction.
    """

    def __init__(self, hyperparameters: dict[str, Any], features: list[str]):
        super().__init__(hyperparameters, features)
        self.model = XGBClassifier(**hyperparameters)
