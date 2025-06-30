import pandas as pd
from unittest.mock import patch, MagicMock

from mlforge.tuning import get_feature_importance_ranking, tune_model_parameters


class DummyModel:
    def __init__(self, importances):
        self.feature_importances_ = importances


def test_get_feature_importance_ranking_ascending():
    model = DummyModel([0.3, 0.1, 0.6])
    features = ['a', 'b', 'c']

    ranking = get_feature_importance_ranking(model, features, ascending=True, plot=False)

    assert ranking == ['b', 'a', 'c']


def test_get_feature_importance_ranking_descending():
    model = DummyModel([0.3, 0.1, 0.6])
    features = ['a', 'b', 'c']

    ranking = get_feature_importance_ranking(model, features, ascending=False, plot=False)

    assert ranking == ['c', 'a', 'b']


def test_tune_model_parameters():
    # Dummy data
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4],
        'f2': [4, 3, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1])
    features = ['f1', 'f2']

    # Dummy param grid and estimator
    param_grid = {'param1': [1, 2]}
    estimator = MagicMock(name='Estimator')

    # Patch GridSearchCV to mock fitting and results
    with patch('mlforge.tuning.GridSearchCV') as MockGridSearchCV:
        mock_grid_search = MagicMock()
        MockGridSearchCV.return_value = mock_grid_search

        # Setup mock attributes
        mock_grid_search.best_params_ = {'param1': 2}
        mock_grid_search.best_score_ = 0.95

        # Make fit do nothing
        mock_grid_search.fit.return_value = None

        # Call the function
        results = tune_model_parameters(
             estimator, param_grid, X, y, features, splits=3, verbose=False
        )

        # Check GridSearchCV was called with correct params
        MockGridSearchCV.assert_called_once()
        args, kwargs = MockGridSearchCV.call_args
        assert kwargs['estimator'] == estimator
        assert kwargs['param_grid'] == param_grid
        assert kwargs['cv'].n_splits == 3
        assert kwargs['scoring'] == 'accuracy'
        assert kwargs['verbose'] == 0
        assert kwargs['n_jobs'] == -1

        # Assert .fit was called once (we won't check exact DataFrame)
        assert mock_grid_search.fit.call_count == 1

        # Optionally check the shape/columns of the DataFrame passed to .fit
        called_X = mock_grid_search.fit.call_args[0][0]
        assert isinstance(called_X, pd.DataFrame)
        assert list(called_X.columns) == features
        assert called_X.to_dict() == X.to_dict()

        # Assert y was passed
        called_y = mock_grid_search.fit.call_args[0][1]
        assert isinstance(called_y, pd.Series)
        assert called_y.tolist() == y.tolist()

        # Check results returned correctly
        assert isinstance(results, dict)
        assert "best_params" in results
        assert "best_score" in results
        assert results['best_params'] == {'param1': 2}
        assert results['best_score'] == 0.95
