import pytest
from sklearn.base import clone, is_classifier

from dosdetect.models import MODELS, knn, random_forest_tuned
from dosdetect.preprocess import Preprocessor


@pytest.fixture
def split(train_df, test_df):
    prep = Preprocessor(n_features=41)
    X_train, y_train = prep.fit_transform(train_df)
    X_test, y_test = prep.transform(test_df)
    return X_train, y_train, X_test, y_test


@pytest.mark.parametrize("name", list(MODELS))
def test_model_learns_separable_data(name, split):
    X_train, y_train, X_test, y_test = split
    model = MODELS[name]()
    assert is_classifier(model)
    model.fit(X_train, y_train)
    assert (model.predict(X_test) == y_test).mean() > 0.9


@pytest.mark.parametrize("name", list(MODELS))
def test_model_factories_return_fresh_unfitted_models(name):
    a, b = MODELS[name](), MODELS[name]()
    assert a is not b
    clone(a)  # raises if the estimator breaks scikit-learn's conventions


def test_knn_scales_features():
    assert knn()[0].__class__.__name__ == "StandardScaler"


def test_tuned_forest_keeps_five_features(split):
    X_train, y_train, *_ = split
    model = random_forest_tuned().fit(X_train, y_train)
    assert model["select"].get_support().sum() == 5
