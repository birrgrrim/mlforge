import json

class BaseModelWrapper:
    def __init__(self, hyperparameters: dict, features: list):
        self.hyperparameters = hyperparameters
        self.features = features
        self.model = None  # to be set in subclass

    def to_json(self):
        return json.dumps({
            "model_class": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "features": self.features
        })

    @classmethod
    def from_json(cls, json_string):
        data = json.loads(json_string)
        return cls(data["hyperparameters"], data["features"])

    def fit(self, X, y):
        return self.model.fit(X[self.features], y)

    def predict(self, X):
        return self.model.predict(X[self.features])