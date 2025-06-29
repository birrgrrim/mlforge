import json
from typing import Any


class BaseModelWrapper:
    """
    Base wrapper for ML models.

    Stores hyperparameters and feature list, and provides
    JSON serialization and basic fit/predict interface.

    Subclasses must set `self.model` to the actual model instance.

    Attributes:
        hyperparameters (dict): Hyperparameters for the model.
        features (list): List of feature names to use.
        model: Underlying ML model instance (set by subclass).
    """

    def __init__(self, hyperparameters: dict[str, Any], features: list[str]):
        """
        Initialize the wrapper.

        Args:
            hyperparameters: Dictionary of hyperparameters to configure the model.
            features: List of feature names to use during training and prediction.
        """
        self.hyperparameters = hyperparameters
        self.features = features
        self.model = None  # to be set in subclass

    def to_json(self) -> str:
        """
        Serialize the wrapper's configuration to a JSON string.

        Returns:
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

        Args:
            json_string: JSON string created by `to_json`.

        Returns:
            A new instance of the wrapper with loaded hyperparameters and features.
        """
        data = json.loads(json_string)
        return cls(data["hyperparameters"], data["features"])

    def fit(self, X: Any, y: Any) -> Any:
        """
        Fit the underlying model to training data.

        Args:
            X: Training feature data (e.g., pandas DataFrame).
            y: Training target labels.

        Returns:
            Result of the model's fit method.
        """
        return self.model.fit(X[self.features], y)

    def predict(self, X: Any) -> Any:
        """
        Predict target values using the trained model.

        Args:
            X: Input feature data.

        Returns:
            Predicted target values.
        """
        return self.model.predict(X[self.features])
