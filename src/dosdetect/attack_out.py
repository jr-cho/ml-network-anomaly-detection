"""Leave-one-attack-out: how well does each model catch DoS types it never saw?

For each dataset and held-out group (data.ATTACK_GROUPS), the rows outside the group are
split 80/20 (stratified by attack, seed 42). Train is the 80%; test is the 20% plus every
held-out row (capped at 20,000). Feature selection is fit on that training set only.
One seed, so differences of a few points are within noise.

python -m dosdetect.attack_out                       # everything -> results/attack_out.csv
python -m dosdetect.attack_out --datasets nsl-kdd --models knn --n-features 8
A partial run is printed only, unless --out is given.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dosdetect.data import (
    DATASETS,
    FEATURE_COLUMNS,
    FEATURE_COUNTS,
    RESULTS_DIR,
    Preprocessor,
    attack_out_splits,
    load_dataset,
)
from dosdetect.models import MODELS

FULL_RUN = (list(DATASETS), list(FEATURE_COUNTS), list(MODELS))


def score(y_true, y_pred, heldout):
    """Held-out and seen-DoS recall, false-alarm rate on normal rows."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    seen_dos = (y_true == 1) & ~heldout
    normal = y_true == 0
    return {
        "n_heldout": int(heldout.sum()),
        "recall_heldout": y_pred[heldout].mean(),
        "recall_seen": y_pred[seen_dos].mean() if seen_dos.any() else float("nan"),
        "false_alarm_rate": y_pred[normal].mean() if normal.any() else float("nan"),
    }


def run(datasets=DATASETS, feature_counts=FEATURE_COUNTS, models=tuple(MODELS), seed=42):
    """One row per (dataset, n_features, model, group)."""
    rows = []
    for dataset in datasets:
        df = load_dataset(dataset)
        for group, train, test, heldout in attack_out_splits(df, seed):
            for n in feature_counts:
                prep = Preprocessor(n)
                X_train, y_train = prep.fit_transform(train)
                X_test, y_test = prep.transform(test)
                for name in models:
                    y_pred = MODELS[name]().fit(X_train, y_train).predict(X_test)
                    rows.append(
                        {
                            "dataset": dataset,
                            "n_features": n,
                            "model": name,
                            "group": group,
                            **score(y_test, y_pred, heldout),
                        }
                    )
                    print(f"{dataset} {group} {n} {name} done", flush=True)
    return pd.DataFrame(rows)


def summary(df):
    """Mean held-out recall next to mean seen recall, per dataset, features and model."""
    keys = ["dataset", "n_features", "model"]
    return df.groupby(keys)[["recall_heldout", "recall_seen", "false_alarm_rate"]].mean()


def unique(values):
    return list(dict.fromkeys(values))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument(
        "--n-features",
        nargs="+",
        type=int,
        choices=range(1, len(FEATURE_COLUMNS) + 1),
        default=list(FEATURE_COUNTS),
        metavar="N",
        help=f"features to keep, 1-{len(FEATURE_COLUMNS)} (default: 8 41)",
    )
    p.add_argument("--out", type=Path, help="CSV to write (default: attack_out.csv on a full run)")
    args = p.parse_args(argv)
    args.datasets, args.models, args.n_features = map(
        unique, (args.datasets, args.models, args.n_features)
    )
    return args


def main(argv=None):
    args = parse_args(argv)
    df = run(args.datasets, args.n_features, args.models)
    print()
    print(summary(df).to_string(float_format="%.4f"))

    out = args.out
    chosen = (args.datasets, args.n_features, args.models)
    if out is None and all(set(a) == set(b) for a, b in zip(chosen, FULL_RUN, strict=True)):
        out = RESULTS_DIR / "attack_out.csv"
    if out is None:
        print("\nPartial run: not saved (use --out to save).")
        return df
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved {out}")
    return df


if __name__ == "__main__":
    main()
