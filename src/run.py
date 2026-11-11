"""Run the full pipeline: load, preprocess, train, evaluate, compare to paper.

Usage:
    python -m src.run                         # official train/test files
    python -m src.run --split holdout         # 80/20 split of the training file
    python -m src.run --models knn --k 3
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import TEST_FILE, TRAIN_FILE, load
from src.evaluate import PAPER_RESULTS, evaluate
from src.models import MODELS
from src.preprocess import Preprocessor

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--split", choices=["official", "holdout"], default="official",
        help="official: train on KDDTrain+, test on KDDTest+ (includes DoS "
             "attacks unseen in training). holdout: stratified 80/20 split "
             "of KDDTrain+.",
    )
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--n-features", type=int, default=8, help="features to keep (paper: 8)")
    p.add_argument("--k", type=int, default=5, help="neighbours for KNN")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(split, seed):
    if split == "official":
        return load(TRAIN_FILE), load(TEST_FILE)
    df = load(TRAIN_FILE)
    return train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)


def main():
    args = parse_args()

    train_df, test_df = load_split(args.split, args.seed)
    print(f"Split: {args.split}")
    print(f"Train rows: {len(train_df):,} ({train_df['label'].mean():.1%} DoS)")
    print(f"Test rows:  {len(test_df):,} ({test_df['label'].mean():.1%} DoS)\n")

    prep = Preprocessor(n_features=args.n_features)
    X_train, y_train = prep.fit_transform(train_df)
    X_test, y_test = prep.transform(test_df)

    print(f"Top {args.n_features} features by |correlation| with label:")
    for name in prep.selected_features:
        print(f"  {name:<28} {prep.correlations[name]:.3f}")
    print()

    rows = []
    for name in args.models:
        model = MODELS[name](k=args.k) if name == "knn" else MODELS[name]()
        result = evaluate(model, X_train, y_train, X_test, y_test)
        cm = result.pop("confusion_matrix")
        rows.append({"model": name, **result})

        print(f"== {name} ==")
        print(f"  confusion matrix (rows = actual normal/DoS, cols = predicted):")
        print(f"    {cm[0]}\n    {cm[1]}")
        paper = PAPER_RESULTS.get(name)
        for metric in ("accuracy", "precision", "recall", "f1"):
            line = f"  {metric:<10} {result[metric]:.4f}"
            if paper and metric in paper:
                line += f"   (paper: {paper[metric]:.4f}, diff: {result[metric] - paper[metric]:+.4f})"
            print(line)
        print(f"  train {result['train_time_s']:.2f}s, predict {result['predict_time_s']:.2f}s\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"metrics_{args.split}_{args.n_features}feat.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    print(f"Saved {out.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
