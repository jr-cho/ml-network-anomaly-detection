"""Robustness checks for the Random Forest feature importance results (step 4).

Run after notebooks/rf_feature_importance.py. Checks that the importance
conclusions in BRAINSTORM.md section 10 are not artefacts of one seed, one
metric, or one way of measuring importance:

  1. Seed stability: retrain + re-permute with 3 other seeds.
  2. Metric: permutation importance scored by recall / accuracy / precision.
  3. Retraining: add or remove features and retrain (official split).
  4. Unseen attacks: which features drive detection of attack types
     never seen in training.

Usage (from the repo root, ~4 min):
    python notebooks/rf_importance_checks.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `src` imports work

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, recall_score

from src.data import TEST_FILE, TRAIN_FILE, load
from src.models import random_forest
from src.preprocess import Preprocessor

SEED = 42


def top5(model, X, y, scoring, seed):
    pi = permutation_importance(model, X, y, scoring=scoring, n_repeats=5, random_state=seed)
    s = pd.Series(pi.importances_mean, X.columns).sort_values(ascending=False)
    return s, ", ".join(f"{k} {v:.4f}" for k, v in s.head(5).items())


def main():
    train, test = load(TRAIN_FILE), load(TEST_FILE)
    prep = Preprocessor(n_features=41)
    X_train, y_train = prep.fit_transform(train)
    X_test, y_test = prep.transform(test)
    paper_8 = Preprocessor(n_features=8).fit(train).selected_features
    model = random_forest(seed=SEED).fit(X_train, y_train)

    print("=== 1. Seed stability (F1 permutation, official split) ===")
    base, _ = top5(model, X_test, y_test, "f1", SEED)
    for seed in (0, 1, 2):
        m = random_forest(seed=seed).fit(X_train, y_train)
        s, text = top5(m, X_test, y_test, "f1", seed)
        rho = s.rank().corr(base.rank(), method="spearman")
        print(f"  seed {seed}: rank agreement with seed {SEED} = {rho:.2f} | top 5: {text}")

    print("\n=== 2. Scoring metric (seed 42 model) ===")
    for scoring in ("recall", "accuracy", "precision"):
        print(f"  {scoring:9}: {top5(model, X_test, y_test, scoring, SEED)[1]}")

    print("\n=== 3. Retraining with different feature sets (official split) ===")
    dos = (test["label"] == 1).values
    unseen = (dos & ~test["attack"].isin(train["attack"])).values
    all_41 = list(X_train.columns)
    feature_sets = {
        "41 features": all_41,
        "41 minus src_bytes": [c for c in all_41 if c != "src_bytes"],
        "41 minus src_bytes, dst_bytes": [c for c in all_41 if c not in ("src_bytes", "dst_bytes")],
        "paper 8": paper_8,
        "paper 8 + src_bytes": paper_8 + ["src_bytes"],
        "paper 8 + src_bytes, dst_bytes": paper_8 + ["src_bytes", "dst_bytes"],
    }
    for name, cols in feature_sets.items():
        pred = random_forest(seed=SEED).fit(X_train[cols], y_train).predict(X_test[cols])
        print(f"  {name:32} F1 {f1_score(y_test, pred):.4f}   "
              f"recall seen {pred[dos & ~unseen].mean():.4f}   unseen {pred[unseen].mean():.4f}")

    print("\n=== 4. What drives detection of unseen attack types? ===")
    X_u, y_u = X_test[unseen], y_test[unseen]
    print(f"  recall on unseen rows: {recall_score(y_u, model.predict(X_u)):.4f}")
    print(f"  top 5 by recall drop: {top5(model, X_u, y_u, 'recall', SEED)[1]}")


if __name__ == "__main__":
    main()
