from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def get_feature_importance_ranking(model, features: list[str], ascending=True, plot=False) -> list[str]:
    """
    Returns a list of features sorted by importance.

    Parameters
    ----------
    model : fitted model
        Must have `feature_importances_` attribute.
    features : list of str
        Feature names to check.
    ascending : bool, default=True
        If True, sort in ascending order (the least important first).
    plot : bool, default=False
        If True, show matplotlib bar chart of feature importances.

    Returns
    -------
    list of str
        Feature names sorted by importance.
    """
    importances = pd.Series(model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=ascending)

    if plot:
        importances.plot(kind='barh')
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    return importances.index.tolist()


def tune_model_parameters(
        estimator,
        param_grid: dict[str, list[Any]],
        X: Any,
        y: Any,
        features: list[str],
        splits: int = 5,
        verbose: bool = False,
        search_strategy: str = "grid"  # only "grid" supported for now
) -> dict[str, dict[str, Any] | float | Any]:
    """
    Perform hyperparameter tuning using GridSearchCV (with option to extend to other strategies).

    Parameters
    ----------
    estimator : estimator instance
        The ML model (e.g., RandomForestClassifier) to tune.
    param_grid : dict of str to list
        Parameter names and list of values to try.
    X : Any
        Feature dataset. (e.g., pandas DataFrame)
    y : Any
        Target labels. (e.g., pandas Series)
    features : list of str
        Feature names to use during tuning.
    splits : int, default=5
        Number of cross-validation folds.
    verbose : bool, default=False
        If True, print best score and params.
    search_strategy : str, default="grid"
        Search strategy to use (currently only "grid" is supported).

    Returns
    -------
    dict
        Dictionary with:
        - 'best_params': best hyperparameters found.
        - 'best_score': best CV accuracy score (rounded).
        - 'cv_results': full cv_results_ from GridSearchCV.
    """
    if search_strategy != "grid":
        raise NotImplementedError(f"Search strategy '{search_strategy}' not implemented yet.")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=1)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    grid_search.fit(X[features], y)

    if verbose:
        print(f"✅ Best CV accuracy: {round(grid_search.best_score_, 4)}")
        print(f"🏆 Best hyperparameters: {grid_search.best_params_}")

    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_
    }
