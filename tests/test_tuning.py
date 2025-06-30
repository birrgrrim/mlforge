from mlforge.tuning import get_feature_importance_ranking


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
