"""Train and score every model on every setup; write results/metrics.csv.

python -m dosdetect.run                              # all models, all four setups
python -m dosdetect.run --models knn naive_bayes     # a subset of models
python -m dosdetect.run --split official --n-features 8
"""

import argparse

import pandas as pd

from dosdetect.data import FEATURE_COUNTS, RESULTS_DIR, ROOT, SPLITS, prepare
from dosdetect.evaluate import PAPER_RESULTS, evaluate
from dosdetect.models import MODELS


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--split", nargs="+", choices=SPLITS, default=list(SPLITS))
    p.add_argument("--n-features", nargs="+", type=int, default=list(FEATURE_COUNTS))
    args = p.parse_args(argv)

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

    out = RESULTS_DIR / "metrics.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
