"""Repeated-run timing benchmark; writes results/timing.csv.

python -m dosdetect.timing                           # everything -> results/timing.csv
python -m dosdetect.timing --repeats 3 --models knn naive_bayes   # a subset: printed only
python -m dosdetect.timing --split official --n-features 8 --out my_timing.csv

Each model is fit and predicted `--repeats` times (default 5), with a fresh model each
time. We report the median, min and max train and predict seconds, the ratios against
naive_bayes (predict) and decision_tree (train) within the same setup, and the machine.

The forests use all cores (n_jobs=-1), so the timings depend on the core count. Run this
alone on the machine. Only a full run writes results/timing.csv; use --out to save a
partial run.
"""

import argparse
import os
import platform

import pandas as pd
import sklearn

from dosdetect.data import RESULTS_DIR, prepare
from dosdetect.evaluate import benchmark
from dosdetect.models import MODELS
from dosdetect.run import FULL_RUN
from dosdetect.run import parse_args as parse_run_args

MACHINE = {
    "cpu_count": os.cpu_count(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "sklearn": sklearn.__version__,
}


def parse_args(argv=None):
    """Same --models/--split/--n-features as dosdetect.run, plus --repeats."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--repeats", type=int, default=5, help="fits per model (default: 5)")
    own, rest = p.parse_known_args(argv)
    if own.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    args = parse_run_args(rest)
    args.repeats = own.repeats
    return args


def add_ratios(df):
    """Median ratios within each (split, n_features) setup; NaN if the reference didn't run."""
    df = df.copy()
    for column, ref, median in (
        ("predict_ratio_vs_naive_bayes", "naive_bayes", "predict_s_median"),
        ("train_ratio_vs_decision_tree", "decision_tree", "train_s_median"),
    ):
        base = df[df.model == ref].set_index(["split", "n_features"])[median]
        keys = pd.MultiIndex.from_frame(df[["split", "n_features"]])
        df[column] = df[median].to_numpy() / base.reindex(keys).to_numpy()
    return df


def main(argv=None):
    args = parse_args(argv)

    rows = []
    for split in args.split:
        for n in args.n_features:
            data = prepare(split, n)
            for name in args.models:
                r = benchmark(MODELS[name], data, args.repeats)
                rows.append({"split": split, "n_features": n, "model": name, **r})

    df = add_ratios(pd.DataFrame(rows))
    for column, value in MACHINE.items():
        df[column] = value

    cols = ["split", "n_features", "model", "train_s_median", "predict_s_median"]
    cols += ["predict_ratio_vs_naive_bayes", "train_ratio_vs_decision_tree"]
    print(df[cols].to_string(index=False, float_format="%.4f"))
    print(f"\n{args.repeats} repeats on {MACHINE['cpu_count']} cores, {MACHINE['platform']}")

    out = args.out
    chosen = (args.models, args.split, args.n_features)
    if out is None and all(set(a) == set(b) for a, b in zip(chosen, FULL_RUN, strict=True)):
        out = RESULTS_DIR / "timing.csv"
    if out is None:
        print("\nPartial run: not saved (use --out to save).")
        return df
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"\nSaved {out}")
    return df


if __name__ == "__main__":
    main()
