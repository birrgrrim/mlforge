from typing import Any, Callable
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from mlforge.plotting import plot_feature_elimination_progression, plot_feature_importances


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
        plot_feature_importances(importances, )

    return importances.index.tolist()


def tune_model_parameters(
        X: Any,
        y: Any,
        estimator,
        hyperparam_initial_info: Any,
        features: list[str],
        splits: int = 5,
        verbose: bool = False,
        search_strategy: str = "grid_search"  # only "grid" supported for now
) -> dict[str, dict[str, Any] | float | Any]:
    """
    Perform hyperparameter tuning using GridSearchCV (with option to extend to other strategies).

    Parameters
    ----------
    estimator : estimator instance
        The ML model (e.g., RandomForestClassifier) to tune.
    hyperparam_initial_info : Any
        e.g. Parameter names and list of values to try for "grid_search".
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
    search_strategy : str, default="grid_search"
        Search strategy to use (currently only "grid_search" is supported).

    Returns
    -------
    dict
        Dictionary with:
        - 'best_params': best hyperparameters found.
        - 'best_score': best CV accuracy score (rounded).
        - 'cv_results': full cv_results_ from GridSearchCV.
    """
    if search_strategy != "grid_search":
        raise NotImplementedError(f"Search strategy '{search_strategy}' not implemented yet.")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=1)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=hyperparam_initial_info,
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


def tune_model_parameters_and_features(
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable[..., Any],
        features: list[str],
        hyperparam_initial_info: Any,
        splits: int = 5,
        feature_selection_strategy: str = "greedy_backward",
        hyperparam_tuning_strategy: str = "grid_search",
        verbose: bool = False,
        plot: bool = False
) -> tuple[dict[str, Any], list[str]]:
    """
    Tune model hyperparameters and perform feature selection.

    Currently supports:
    - Feature selection strategy: "greedy_backward" (backward elimination)
    - Hyperparameter tuning strategy: "grid_search"

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataset.
    y : pd.Series
        Target variable.
    model_factory : Callable[..., Any]
        Callable that returns a new instance of the model to tune.
    features : list of str
        List of feature names to consider.
    hyperparam_initial_info : Any
        Initial hyperparameter information (e.g., grid or ranges).
    splits : int, default=5
        Number of cross-validation folds.
    feature_selection_strategy : str, default="greedy_backward"
        Strategy used for feature selection.
    hyperparam_tuning_strategy : str, default="grid_search"
        Strategy used for hyperparameter tuning.
    verbose : bool, default=False
        If True, print detailed progress information.
    plot : bool, default=False
        If true, show plot with cv/train accuracy

    Returns
    -------
    best_params : dict of str to Any
        Best hyperparameters found.
    best_features : list of str
        Selected feature names.

    Raises
    ------
    NotImplementedError
        If the specified feature selection strategy is not supported.
    """
    if feature_selection_strategy != "greedy_backward":
        raise NotImplementedError(f"Strategy {feature_selection_strategy} not supported")

    def fit_and_score(f, model_params=None):
        if not model_params:
            model_params = tune_model_parameters(
                X, y,
                estimator=model_factory(),
                hyperparam_initial_info=hyperparam_initial_info,
                features=f,
                splits=splits,
                search_strategy=hyperparam_tuning_strategy,
                verbose=False
            )['best_params']
        _model = model_factory(model_params)
        _best_score = cross_val_score(_model, X[f], y, cv=cv, scoring="accuracy").mean()
        _model.fit(X[f], y)
        _train_score = _model.score(X[f], y)
        return _model, model_params, _best_score, _train_score

    # 1. Initial parameter tuning on full feature set
    score_log = []
    current_features = features
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=1)

    # 2. Initial baseline score
    model, base_model_params, best_score, train_score = fit_and_score(current_features)

    if verbose:
        print(f"🔰 Initial CV score: {best_score:.4f}, Train score: {train_score:.4f}")

    # 3. Backward elimination loop
    improved, skip_first = True, True

    while improved:
        improved = False

        if skip_first:  # or self.grid_once: <- TODO: add some hack for lgbm
            skip_first = False
        else:
            # Recalibrate model
            model, base_model_params, best_score, train_score = fit_and_score(current_features)


        score_log.append((len(current_features), best_score, train_score))
        # Sort features by importance (least important first)
        current_features = get_feature_importance_ranking(model, current_features)

        for feature in current_features:
            trial_features = [f for f in current_features if f != feature]
            if not trial_features:
                continue

            _, _, cv_score, train_score = fit_and_score(trial_features, model_params=base_model_params)

            if verbose:
                print(f"Try removing '{feature}'. CV score: {cv_score:.4f}, Train score: {train_score:.4f}")

            if cv_score >= best_score:
                if verbose:
                    print(f"✅ Removed '{feature}'. Improved score: {best_score:.4f} → {cv_score:.4f}")
                current_features = trial_features
                improved = True
                break

    # 4. Final model training with best feature subset
    if verbose:
        print("🎯 Final feature set:")
        print(current_features)
        print(f"✅ Best CV Accuracy: {best_score:.4f}")

    # 📈 Plot accuracy progression
    if plot:
        plot_feature_elimination_progression(score_log)

    return base_model_params, current_features
