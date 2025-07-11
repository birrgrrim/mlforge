import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import pytest

from mltune.tuning import get_feature_importance_ranking, tune_model_parameters, tune_model_parameters_and_features


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
    with patch('mltune.tuning.GridSearchCV') as MockGridSearchCV:
        mock_grid_search = MagicMock()
        MockGridSearchCV.return_value = mock_grid_search

        # Setup mock attributes
        mock_grid_search.best_params_ = {'param1': 2}
        mock_grid_search.best_score_ = 0.95

        # Make fit do nothing
        mock_grid_search.fit.return_value = None

        # Call the function
        results = tune_model_parameters(
            X, y, estimator, param_grid, features, splits=3, verbose=False
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


def test_tune_model_parameters_and_features(monkeypatch):
    # Dummy data
    X = pd.DataFrame({
        'f1': [0, 1, 0, 1],
        'f2': [1, 0, 1, 0],
        'f3': [1, 1, 0, 0]
    })
    y = pd.Series([0, 1, 0, 1])

    features = ['f1', 'f2', 'f3']

    # Mock hyperparameter tuning to return static params
    monkeypatch.setattr(
        "mltune.tuning.tune_model_parameters",
        lambda X, y, estimator, hyperparam_initial_info, features, splits, search_strategy, verbose: {
            "best_params": {"n_estimators": 10},
            "best_score": 0.9
        }
    )

    # Dummy model with feature_importances_ and fit/score
    class DummyModel:
        def __init__(self, **kwargs):
            self.feature_importances_ = np.array([0.2, 0.5, 0.3])
            self.params = kwargs

        def fit(self, X, y):
            return self

        def score(self, X, y):
            return 0.9

        def get_params(self, deep=True):
            return self.params

        def set_params(self, **params):
            self.params.update(params)
            return self

    # Model factory returns dummy model
    model_factory = lambda _ = None: DummyModel()

    # Call function
    params, best_features = tune_model_parameters_and_features(
        X, y,
        model_factory=model_factory,
        features=features,
        hyperparam_initial_info={"n_estimators": [10]},
        splits=2,
        verbose=False,
        plot=False
    )

    # Assertions
    assert isinstance(params, dict)
    assert params["n_estimators"] == 10
    assert isinstance(best_features, list)
    assert set(best_features).issubset(set(features))
    assert len(best_features) >= 1


# New edge case tests
def test_get_feature_importance_ranking_empty_features():
    """Test feature importance ranking with empty features list."""
    model = DummyModel([])  # Empty importances to match empty features
    features = []
    
    ranking = get_feature_importance_ranking(model, features, ascending=True, plot=False)
    
    assert ranking == []


def test_get_feature_importance_ranking_single_feature():
    """Test feature importance ranking with single feature."""
    model = DummyModel([0.5])
    features = ['single_feature']
    
    ranking = get_feature_importance_ranking(model, features, ascending=True, plot=False)
    
    assert ranking == ['single_feature']


def test_get_feature_importance_ranking_equal_importances():
    """Test feature importance ranking with equal importances."""
    model = DummyModel([0.3, 0.3, 0.3])
    features = ['a', 'b', 'c']
    
    ranking = get_feature_importance_ranking(model, features, ascending=True, plot=False)
    
    # Should maintain original order when importances are equal
    assert set(ranking) == set(features)
    assert len(ranking) == 3


def test_tune_model_parameters_invalid_search_strategy():
    """Test tune_model_parameters with invalid search strategy."""
    X = pd.DataFrame({'f1': [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    features = ['f1']
    param_grid = {'param1': [1, 2]}
    estimator = MagicMock()
    
    with pytest.raises(NotImplementedError, match="Search strategy 'invalid_strategy' not implemented yet"):
        tune_model_parameters(
            X, y, estimator, param_grid, features, 
            search_strategy="invalid_strategy"
        )


def test_tune_model_parameters_empty_param_grid():
    """Test tune_model_parameters with empty parameter grid."""
    X = pd.DataFrame({'f1': [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    features = ['f1']
    param_grid = {}
    estimator = MagicMock()
    
    with patch('mltune.tuning.GridSearchCV') as MockGridSearchCV:
        mock_grid_search = MagicMock()
        MockGridSearchCV.return_value = mock_grid_search
        mock_grid_search.best_params_ = {}
        mock_grid_search.best_score_ = 0.8
        mock_grid_search.fit.return_value = None
        
        results = tune_model_parameters(
            X, y, estimator, param_grid, features, verbose=False
        )
        
        assert results['best_params'] == {}
        assert results['best_score'] == 0.8


def test_tune_model_parameters_and_features_invalid_strategy():
    """Test tune_model_parameters_and_features with invalid feature selection strategy."""
    X = pd.DataFrame({'f1': [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    features = ['f1']
    
    def model_factory(_=None):
        return MagicMock()
    
    with pytest.raises(ValueError, match="Unsupported feature_selection_strategy: invalid_strategy"):
        tune_model_parameters_and_features(
            X, y,
            model_factory=model_factory,
            features=features,
            hyperparam_initial_info={},
            feature_selection_strategy="invalid_strategy"
        )


def test_tune_model_parameters_and_features_empty_features():
    """Test tune_model_parameters_and_features with empty features list."""
    X = pd.DataFrame({'f1': [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    features = []
    
    def model_factory(_=None):
        return MagicMock()
    
    # Mock the tuning function
    with patch('mltune.tuning.tune_model_parameters') as mock_tune:
        mock_tune.return_value = {
            "best_params": {"param1": 1},
            "best_score": 0.8
        }
        
        params, best_features = tune_model_parameters_and_features(
            X, y,
            model_factory=model_factory,
            features=features,
            hyperparam_initial_info={},
            feature_selection_strategy="none"
        )
        
        assert params == {"param1": 1}
        assert best_features == []


def test_tune_model_parameters_verbose_output(capsys):
    """Test tune_model_parameters with verbose output."""
    X = pd.DataFrame({'f1': [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    features = ['f1']
    param_grid = {'param1': [1, 2]}
    estimator = MagicMock()
    
    with patch('mltune.tuning.GridSearchCV') as MockGridSearchCV:
        mock_grid_search = MagicMock()
        MockGridSearchCV.return_value = mock_grid_search
        mock_grid_search.best_params_ = {'param1': 2}
        mock_grid_search.best_score_ = 0.95
        mock_grid_search.fit.return_value = None
        
        tune_model_parameters(
            X, y, estimator, param_grid, features, verbose=True
        )
        
        captured = capsys.readouterr()
        assert "✅ Best CV accuracy: 0.95" in captured.out
        assert "🏆 Best hyperparameters: {'param1': 2}" in captured.out
