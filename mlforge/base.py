import json
from typing import Any


class BaseModelWrapper:
    """
    Base wrapper for ML models.

    Stores hyperparameters and feature list, and provides
    JSON serialization and basic fit/predict interface.

    Subclasses must set `self.model` to the actual model instance.

    Attributes
    ----------
    hyperparameters : dict
        Hyperparameters for the model.
    features : list of str
        List of feature names to use.
    model : Any
        Underlying ML model instance (set by subclass).
    """

    def __init__(self, hyperparameters: dict[str, Any], features: list[str]):
        """
        Initialize the wrapper.

        Parameters
        ----------
        hyperparameters : dict of str to Any
            Dictionary of hyperparameters to configure the model.
        features : list of str
            List of feature names to use during training and prediction.
        """
        self.hyperparameters = hyperparameters
        self.features = features
        self.model = None  # to be set in subclass

    def to_json(self) -> str:
        """
        Serialize the wrapper's configuration to a JSON string.

        Returns
        -------
        str
            JSON string representing model class, hyperparameters, and features.
        """
        return json.dumps({
            "model_class": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "features": self.features
        })

    @classmethod
    def from_json(cls, json_string: str) -> 'BaseModelWrapper':
        """
        Deserialize from JSON string to create a new wrapper instance.

        Parameters
        ----------
        json_string : str
            JSON string created by `to_json`.

        Returns
        -------
        BaseModelWrapper
            A new instance of the wrapper with loaded hyperparameters and features.
        """
        data = json.loads(json_string)
        return cls(data["hyperparameters"], data["features"])

    def fit(self, X: Any, y: Any) -> Any:
        """
        Fit the underlying model to training data.

        Parameters
        ----------
        X : Any
            Training feature data (e.g., pandas DataFrame).
        y : Any
            Training target labels.

        Returns
        -------
        Any
            Result of the model's fit method.
        """
        return self.model.fit(X[self.features], y)

    def predict(self, X: Any) -> Any:
        """
        Predict target values using the trained model.

        Parameters
        ----------
        X : Any
            Input feature data.

        Returns
        -------
        Any
            Predicted target values.
        """
        return self.model.predict(X[self.features])
