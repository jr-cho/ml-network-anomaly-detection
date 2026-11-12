"""Train and score every model on every setup; write results/metrics.csv.

python -m dosdetect.run                              # all models, all setups -> results/metrics.csv
python -m dosdetect.run --models knn naive_bayes     # a subset: printed only
python -m dosdetect.run --split official --n-features 8 --out my_run.csv

Only a full run (all models, splits and feature counts) writes results/metrics.csv,
so a partial run never overwrites the committed results. Use --out to save one.
"""

import argparse
from pathlib import Path

import pandas as pd

from dosdetect.data import FEATURE_COLUMNS, FEATURE_COUNTS, RESULTS_DIR, SPLITS, prepare
from dosdetect.evaluate import PAPER_RESULTS, evaluate
from dosdetect.models import MODELS

FULL_RUN = (list(MODELS), list(SPLITS), list(FEATURE_COUNTS))


def unique(values):
    return list(dict.fromkeys(values))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--split", nargs="+", choices=SPLITS, default=list(SPLITS))
    p.add_argument(
        "--n-features",
        nargs="+",
        type=int,
        choices=range(1, len(FEATURE_COLUMNS) + 1),
        default=list(FEATURE_COUNTS),
        metavar="N",
        help=f"features to keep, 1-{len(FEATURE_COLUMNS)} (default: 8 41)",
    )
    p.add_argument("--out", type=Path, help="CSV to write (default: metrics.csv on a full run)")
    args = p.parse_args(argv)
    args.models, args.split, args.n_features = map(
        unique, (args.models, args.split, args.n_features)
    )
    return args


def main(argv=None):
    args = parse_args(argv)

    rows = []
    for split in args.split:
        for n in args.n_features:
            data = prepare(split, n)
            for name in args.models:
                r = evaluate(MODELS[name](), data)
                r.pop("confusion_matrix")
                rows.append({"split": split, "n_features": n, "model": name, **r})

    df = pd.DataFrame(rows)
    paper = pd.DataFrame(PAPER_RESULTS).T.add_suffix(" (paper)")
    cols = ["split", "n_features", "model", "accuracy", "precision", "recall", "f1"]
    cols += [c for c in ("recall_unseen",) if c in df]
    print(df[cols].to_string(index=False, float_format="%.4f"))
    print("\nPaper (Section IV-A):\n" + paper.to_string(float_format="%.4f"))

    out = args.out
    chosen = (args.models, args.split, args.n_features)
    if out is None and all(set(a) == set(b) for a, b in zip(chosen, FULL_RUN, strict=True)):
        out = RESULTS_DIR / "metrics.csv"
    if out is None:
        print("\nPartial run: not saved (use --out to save).")
        return df
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved {out}")
    return df


if __name__ == "__main__":
    main()
