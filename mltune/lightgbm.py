from typing import Any, Callable

try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError("LightGBMModelWrapper requires 'lightgbm'. Install with: pip install mltune[lgbm]") from e

from .base import BaseModelWrapper


class LightGBMModelWrapper(BaseModelWrapper):
    """
    Wrapper for lightgbm.LGBMClassifier.

    Initializes the underlying LGBMClassifier with given hyperparameters.

    Parameters
    ----------
    hyperparameters : dict of str to Any
        Model hyperparameters to configure LGBMClassifier.
    features : list of str
        List of feature names to use during training and prediction.
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None, features: list[str] | None = None):
        super().__init__(hyperparameters, features)
        self.model = LGBMClassifier(**(self.hyperparameters or {}))

    def get_model_factory(self) -> Callable[[dict[str, Any]], Any]:
        """
        Returns a factory function that creates new LGBMClassifier instances.

        Returns
        -------
        Callable[[dict[str, Any]], Any]
            A factory function: dynamic_params → model instance.
        """

        def factory(dynamic_params: dict[str, Any] = {}) -> Any:
            params = {
                **dynamic_params,
                "random_state": 1,
                "verbosity": -1
            }
            return LGBMClassifier(**params)

        return factory
