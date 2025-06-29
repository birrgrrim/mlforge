from typing import Any
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModelWrapper


class RandomForestModelWrapper(BaseModelWrapper):
    """
    Wraps sklearn.ensemble.RandomForestClassifier.

    Args:
        hyperparameters (dict[str, Any]): model parameters
        features (list[str]): list of feature names
    """

    def __init__(self, hyperparameters: dict[str, Any], features: list[str]):
        super().__init__(hyperparameters, features)
        self.model = RandomForestClassifier(**hyperparameters)
