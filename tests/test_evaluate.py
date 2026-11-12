import numpy as np
import pytest

from dosdetect.evaluate import evaluate
from dosdetect.models import random_forest


class FixedPredictions:
    """Stub model with preset predictions."""

    def __init__(self, y_pred):
        self.y_pred = np.asarray(y_pred)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.y_pred


Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1])
Y_PRED = np.array([0, 0, 0, 1, 1, 1, 1, 0])  # 1 false alarm, 1 missed attack
X = np.zeros((8, 1))


def test_metrics_and_confusion_matrix():
    r = evaluate(FixedPredictions(Y_PRED), X, Y_TRUE, X, Y_TRUE)
    assert r["accuracy"] == pytest.approx(6 / 8)
    assert r["precision"] == pytest.approx(3 / 4)
    assert r["recall"] == pytest.approx(3 / 4)
    assert r["confusion_matrix"].tolist() == [[3, 1], [1, 3]]
    assert r["train_time_s"] >= 0 and r["predict_time_s"] >= 0


def test_recall_split_by_seen_and_unseen_attacks():
    unseen = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)  # last two attacks are new
    r = evaluate(FixedPredictions(Y_PRED), X, Y_TRUE, X, Y_TRUE, unseen=unseen)
    assert r["recall_seen"] == pytest.approx(1.0)
    assert r["recall_unseen"] == pytest.approx(0.5)


def test_unseen_recall_is_nan_without_unseen_rows():
    r = evaluate(FixedPredictions(Y_PRED), X, Y_TRUE, X, Y_TRUE, unseen=np.zeros(8, dtype=bool))
    assert np.isnan(r["recall_unseen"])


def test_no_oob_score_for_models_without_one():
    assert "oob_score" not in evaluate(FixedPredictions(Y_PRED), X, Y_TRUE, X, Y_TRUE)


def test_random_forest_reports_oob_score():
    rng = np.random.default_rng(0)
    X_big = rng.random((200, 3))
    y_big = (X_big[:, 0] > 0.5).astype(int)
    r = evaluate(random_forest(n_estimators=30), X_big, y_big, X_big, y_big)
    assert 0.8 < r["oob_score"] <= 1.0
