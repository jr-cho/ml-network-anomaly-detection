import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone, is_classifier

from dosdetect.data import Preprocessor
from dosdetect.evaluate import TABLE_COLUMNS, evaluate, metrics_table
from dosdetect.models import MODELS, knn, random_forest, random_forest_tuned


@pytest.fixture
def split(train_df, test_df):
    prep = Preprocessor(n_features=41)
    X_train, y_train = prep.fit_transform(train_df)
    X_test, y_test = prep.transform(test_df)
    unseen = (test_df["attack"] == "apache2").to_numpy()
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "unseen": unseen,
    }


@pytest.mark.parametrize("name", list(MODELS))
def test_model_learns_separable_data(name, split):
    model = MODELS[name]()
    assert is_classifier(model)
    clone(model)  # raises if the estimator breaks scikit-learn's conventions
    assert evaluate(model, split)["accuracy"] > 0.9


def test_knn_scales_features():
    assert knn()[0].__class__.__name__ == "StandardScaler"


def test_tuned_forest_keeps_five_features(split):
    model = random_forest_tuned().fit(split["X_train"], split["y_train"])
    assert model["select"].get_support().sum() == 5


class FixedPredictions:
    def __init__(self, y_pred):
        self.y_pred = np.asarray(y_pred)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.y_pred


def fixed_data(unseen):
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X = np.zeros((8, 1))
    return {"X_train": X, "y_train": y, "X_test": X, "y_test": y, "unseen": np.array(unseen, bool)}


Y_PRED = [0, 0, 0, 1, 1, 1, 1, 0]  # 1 false alarm, 1 missed attack


def test_metrics_and_confusion_matrix():
    r = evaluate(FixedPredictions(Y_PRED), fixed_data([0] * 8))
    assert r["accuracy"] == pytest.approx(6 / 8)
    assert r["precision"] == pytest.approx(3 / 4)
    assert r["recall"] == pytest.approx(3 / 4)
    assert r["confusion_matrix"].tolist() == [[3, 1], [1, 3]]
    assert "recall_unseen" not in r and "oob_score" not in r


def test_recall_split_by_seen_and_unseen_attacks():
    r = evaluate(FixedPredictions(Y_PRED), fixed_data([0, 0, 0, 0, 0, 0, 1, 1]))
    assert r["recall_seen"] == pytest.approx(1.0)
    assert r["recall_unseen"] == pytest.approx(0.5)


def test_random_forest_reports_oob_score():
    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    y = (X[:, 0] > 0.5).astype(int)
    d = {"X_train": X, "y_train": y, "X_test": X, "y_test": y, "unseen": np.zeros(200, bool)}
    assert 0.8 < evaluate(random_forest(n_estimators=30), d)["oob_score"] <= 1.0


def test_tuned_forest_can_be_pickled_and_handles_few_features(split):
    model = random_forest_tuned().fit(split["X_train"].iloc[:, :3], split["y_train"])
    assert model["select"].get_support().sum() == 3
    pickle.loads(pickle.dumps(model))


def test_recall_seen_is_nan_without_warning_when_every_attack_is_unseen():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = evaluate(FixedPredictions(Y_PRED), fixed_data([0, 0, 0, 0, 1, 1, 1, 1]))
    assert np.isnan(r["recall_seen"])


def test_metrics_table_keeps_column_order_when_first_result_lacks_unseen():
    holdout = evaluate(FixedPredictions(Y_PRED), fixed_data([0] * 8))
    official = evaluate(FixedPredictions(Y_PRED), fixed_data([0, 0, 0, 0, 0, 0, 1, 1]))
    columns = list(metrics_table({"holdout": holdout, "official": official}).columns)
    assert columns == TABLE_COLUMNS
