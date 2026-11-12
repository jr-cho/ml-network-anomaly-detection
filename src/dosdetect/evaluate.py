"""Fit, time, and score models; tables and plots shared by the notebooks."""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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
TABLE_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "recall_seen",
    "recall_unseen",
    "train_time_s",
    "predict_time_s",
]


def evaluate(model, data):
    """Fit `model` on a `prepare()` dict and score it on the test rows."""
    start = time.perf_counter()
    model.fit(data["X_train"], data["y_train"])
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = np.asarray(model.predict(data["X_test"]))
    predict_time = time.perf_counter() - start

    y_true = np.asarray(data["y_test"])
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }

    # Recall on DoS types seen vs never seen in training (official split only).
    unseen = np.asarray(data["unseen"], dtype=bool)
    if unseen.any():
        result["recall_seen"] = y_pred[(y_true == 1) & ~unseen].mean()
        result["recall_unseen"] = y_pred[unseen].mean()

    final = model[-1] if hasattr(model, "steps") else model  # OOB lives on the last step
    if hasattr(final, "oob_score_"):
        result["oob_score"] = final.oob_score_
    return result


def evaluate_all(make_model, data):
    """Evaluate a fresh `make_model()` on every setup from `prepare_all()`."""
    return {key: evaluate(make_model(), d) for key, d in data.items()}


def metrics_table(results):
    """One row per result; keys like ("official", 8) become "official, 8 features"."""
    rows = {
        (f"{k[0]}, {k[1]} features" if isinstance(k, tuple) else k): {
            c: r[c] for c in TABLE_COLUMNS if c in r
        }
        for k, r in results.items()
    }
    return pd.DataFrame(rows).T.round(4)


def paper_table(name, results):
    """Our 8-feature metrics on both splits vs the paper's (KNN, Naive Bayes)."""
    paper = PAPER_RESULTS[name]
    return pd.concat(
        {
            f"{split}, 8 features": pd.DataFrame(
                {
                    "ours": {m: results[(split, 8)][m] for m in paper},
                    "paper": paper,
                    "difference": {m: results[(split, 8)][m] - paper[m] for m in paper},
                }
            )
            for split in ("official", "holdout")
        },
        axis=1,
    ).round(4)


def plot_confusions(results):
    """Confusion matrices side by side, one per result."""
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.5), squeeze=False)
    for ax, (key, r) in zip(axes[0], results.items(), strict=True):
        ConfusionMatrixDisplay(r["confusion_matrix"], display_labels=["normal", "DoS"]).plot(
            ax=ax, cmap="Blues", values_format=",", colorbar=False
        )
        ax.set_title(f"{key[0]}, {key[1]} features" if isinstance(key, tuple) else key)
    fig.tight_layout()
    return fig
