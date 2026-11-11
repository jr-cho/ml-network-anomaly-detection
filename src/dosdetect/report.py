"""Notebook display helpers."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from dosdetect.evaluate import PAPER_RESULTS, evaluate

METRICS = ["accuracy", "precision", "recall", "f1"]


def run(model, data):
    """Evaluate `model` on a dict from run.prepare()."""
    return evaluate(
        model,
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"],
        unseen=data["unseen"] if data["unseen"].any() else None,
    )


def metrics_table(results):
    """Table of evaluate() results, one row per label."""
    cols = METRICS + ["recall_seen", "recall_unseen", "train_time_s", "predict_time_s"]
    rows = {name: {c: r.get(c) for c in cols if c in r} for name, r in results.items()}
    return pd.DataFrame(rows).T.round(4)


def paper_table(name, result):
    """Our metrics vs the paper's (KNN, Naive Bayes only)."""
    paper = PAPER_RESULTS[name]
    return pd.DataFrame(
        {
            "ours": {m: result[m] for m in paper},
            "paper": paper,
            "difference": {m: result[m] - paper[m] for m in paper},
        }
    ).round(4)


def plot_confusion(cm, title, ax=None):
    ax = ax or plt.subplots(figsize=(4, 3.5))[1]
    ConfusionMatrixDisplay(cm, display_labels=["normal", "DoS"]).plot(
        ax=ax, cmap="Blues", values_format=",", colorbar=False
    )
    ax.set_title(title)
    return ax
