"""Train a model, time it, and score it on the test set."""

import time

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Reported results from the paper (Section IV-A), as fractions.
PAPER_RESULTS = {
    "knn": {"accuracy": 0.9851, "precision": 0.989, "recall": 0.978},
    "naive_bayes": {"accuracy": 0.9395, "precision": 0.9774, "recall": 0.9554},
}


def evaluate(model, X_train, y_train, X_test, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start

    # DoS (label 1) is the positive class.
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }
