import pandas as pd
import matplotlib.pyplot as plt


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
