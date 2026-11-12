"""Tune the Random Forest on KDDTrain+ (standard vs attack-out CV), compare on KDDTest+.

Writes results/random_forest/tuning.csv. ~10 min.
Run: python experiments/random_forest/tuning.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from dosdetect.data import TEST_FILE, TRAIN_FILE, load
from dosdetect.paths import RF_RESULTS_DIR
from dosdetect.preprocess import Preprocessor

SEED = 42
EVAL_SEEDS = (0, 1, 2, 3, 4)
N_ITER = 30

# DoS types hidden per fold; small types paired to have enough rows
ATTACK_OUT_FOLDS = [["neptune"], ["smurf"], ["back", "land"], ["teardrop", "pod"]]

SEARCH_SPACE = {
    "select__max_features": [5, 8, 10, 15, 20, 41],
    "rf__max_features": ["sqrt", "log2", 0.3, 0.5],
    "rf__min_samples_leaf": [1, 2, 5, 10, 20],
    "rf__max_depth": [None, 10, 20, 30],
    "rf__max_samples": [None, 0.5, 0.8],
    "rf__class_weight": [None, "balanced", "balanced_subsample"],
    "rf__criterion": ["gini", "entropy"],
}


def pipeline(seed=SEED, n_estimators=100, **params):
    """Top-k features by RF importance, then a Random Forest."""
    pipe = Pipeline(
        [
            (
                "select",
                SelectFromModel(
                    RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=seed),
                    threshold=-np.inf,  # rank only; max_features picks k
                    max_features=41,
                ),
            ),
            ("rf", RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)),
        ]
    )
    return pipe.set_params(**params)


def attack_out_splits(train_df):
    """Folds that each hide one DoS type group plus 1/k of normal rows."""
    rng = np.random.default_rng(SEED)
    normal_idx = np.flatnonzero(train_df["label"].values == 0)
    normal_chunks = np.array_split(rng.permutation(normal_idx), len(ATTACK_OUT_FOLDS))
    attack = train_df["attack"].values
    all_idx = np.arange(len(train_df))
    for held_out, normal_chunk in zip(ATTACK_OUT_FOLDS, normal_chunks, strict=True):
        held_attacks = np.flatnonzero(np.isin(attack, held_out))
        val = np.concatenate([held_attacks, normal_chunk])
        yield np.setdiff1d(all_idx, val), val


def search(X, y, cv, label):
    s = RandomizedSearchCV(
        pipeline(),
        SEARCH_SPACE,
        n_iter=N_ITER,
        scoring="f1",
        cv=cv,
        random_state=SEED,
        n_jobs=1,
        refit=False,
    ).fit(X, y)
    scores = s.cv_results_["mean_test_score"]
    print(
        f"  {label:10} best CV F1 {scores.max():.4f} (spread across candidates "
        f"{scores.min():.4f}–{scores.max():.4f})"
    )
    print(f"  {'':10} {s.best_params_}")
    return s.best_params_


def evaluate(params, X_train, y_train, X_test, y_test, dos, unseen):
    rows = []
    for seed in EVAL_SEEDS:
        pred = pipeline(seed=seed, n_estimators=300, **params).fit(X_train, y_train).predict(X_test)
        rows.append(
            {
                "f1": f1_score(y_test, pred),
                "recall_seen": pred[dos & ~unseen].mean(),
                "recall_unseen": pred[unseen].mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    train, test = load(TRAIN_FILE), load(TEST_FILE)
    prep = Preprocessor(n_features=41)
    X_train, y_train = prep.fit_transform(train)
    X_test, y_test = prep.transform(test)
    dos = (test["label"] == 1).values
    unseen = (dos & ~test["attack"].isin(train["attack"])).values

    print(f"=== Search ({N_ITER} candidates each, training data only) ===")
    standard = search(
        X_train, y_train, StratifiedKFold(3, shuffle=True, random_state=SEED), "standard"
    )
    attack_out = search(X_train, y_train, list(attack_out_splits(train)), "attack-out")

    paper_8 = Preprocessor(n_features=8).fit(train).selected_features
    configs = {
        "default, paper 8": ("paper8", {}),
        "default, 41": ("all", {"select__max_features": 41}),
        "tuned (standard CV)": ("all", standard),
        "tuned (attack-out CV)": ("all", attack_out),
    }

    print(f"\n=== Official test set, mean ± std over seeds {EVAL_SEEDS} ===")
    rows = []
    for name, (cols, params) in configs.items():
        if cols == "paper8":
            Xtr, Xte, params = X_train[paper_8], X_test[paper_8], {"select__max_features": 8}
        else:
            Xtr, Xte = X_train, X_test
        res = evaluate(params, Xtr, y_train, Xte, y_test, dos, unseen)
        mean, std = res.mean(), res.std()
        print(
            f"  {name:22} F1 {mean.f1:.4f} ± {std.f1:.4f}   "
            f"recall seen {mean.recall_seen:.4f}   unseen {mean.recall_unseen:.4f} ± {std.recall_unseen:.4f}"
        )
        rows.append(
            {
                "config": name,
                **{f"{k}_mean": v for k, v in mean.items()},
                **{f"{k}_std": v for k, v in std.items()},
                "params": {k: v for k, v in params.items()},
            }
        )

    pd.DataFrame(rows).to_csv(RF_RESULTS_DIR / "tuning.csv", index=False, float_format="%.4f")
    print("\nSaved results/random_forest/tuning.csv")


if __name__ == "__main__":
    main()
