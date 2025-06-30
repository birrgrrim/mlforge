from typing import Any

try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError("LightGBMModelWrapper requires 'lightgbm'. Install with: pip install mlforge[lgbm]") from e

from .base import BaseModelWrapper


class LightGBMModelWrapper(BaseModelWrapper):
    """
    Wrapper for lightgbm.LGBMClassifier.

    Initializes the underlying LGBMClassifier with given hyperparameters.

    Parameters
    ----------
    hyperparameters : dict of str to Any
        Model hyperparameters to configure RandomForestClassifier.
    features : list of str
        List of feature names to use during training and prediction.
    """

    def __init__(self, hyperparameters: dict[str, Any], features: list[str]):
        super().__init__(hyperparameters, features)
        self.model = LGBMClassifier(**hyperparameters)
