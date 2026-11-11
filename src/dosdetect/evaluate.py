"""Train a model, time it, and score it on the test set."""

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Paper results (Section IV-A).
PAPER_RESULTS = {
    "knn": {"accuracy": 0.9851, "precision": 0.989, "recall": 0.978},
    "naive_bayes": {"accuracy": 0.9395, "precision": 0.9774, "recall": 0.9554},
}


def evaluate(model, X_train, y_train, X_test, y_test, unseen=None):
    """Fit, time and score a model; `unseen` mask adds seen/unseen recall."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start

    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }

    if unseen is not None:
        y_true = np.asarray(y_test)
        y_pred = np.asarray(y_pred)
        unseen = np.asarray(unseen, dtype=bool)
        result["recall_seen"] = _recall_on(y_pred, (y_true == 1) & ~unseen)
        result["recall_unseen"] = _recall_on(y_pred, (y_true == 1) & unseen)

    # OOB score lives on the pipeline's final step.
    final = model[-1] if hasattr(model, "steps") else model
    if hasattr(final, "oob_score_"):
        result["oob_score"] = final.oob_score_

    return result


def _recall_on(y_pred, attack_rows):
    """Share of attack rows predicted DoS (NaN if none)."""
    return y_pred[attack_rows].mean() if attack_rows.any() else float("nan")
