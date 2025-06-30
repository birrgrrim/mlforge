from typing import Any
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModelWrapper


class RandomForestModelWrapper(BaseModelWrapper):
    """
    Wrapper for sklearn.ensemble.RandomForestClassifier.

    Initializes the underlying RandomForestClassifier with given hyperparameters.

    Parameters
    ----------
    hyperparameters : dict of str to Any
        Model hyperparameters to configure RandomForestClassifier.
    features : list of str
        List of feature names to use during training and prediction.
    """

    def __init__(self, hyperparameters: dict[str, Any], features: list[str]):
        super().__init__(hyperparameters, features)
        self.model = RandomForestClassifier(**hyperparameters)
