from sklearn.ensemble import RandomForestClassifier
from .base import BaseModelWrapper

class RandomForestModelWrapper(BaseModelWrapper):
    def __init__(self, hyperparameters, features):
        super().__init__(hyperparameters, features)
        self.model = RandomForestClassifier(**hyperparameters)