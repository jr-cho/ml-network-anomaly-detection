"""Run the pipeline: load, preprocess, train, evaluate, compare to paper.

python -m dosdetect.run                  # official train/test files
python -m dosdetect.run --split holdout  # 80/20 split of KDDTrain+
python -m dosdetect.run --models knn --k 3
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from dosdetect.data import TEST_FILE, TRAIN_FILE, load
from dosdetect.evaluate import PAPER_RESULTS, evaluate
from dosdetect.models import MODELS
from dosdetect.paths import METRICS_DIR, ROOT
from dosdetect.preprocess import Preprocessor


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--split",
        choices=["official", "holdout"],
        default="official",
        help="official: train on KDDTrain+, test on KDDTest+ (includes DoS "
        "attacks unseen in training). holdout: stratified 80/20 split "
        "of KDDTrain+.",
    )
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--n-features", type=int, default=8, help="features to keep (paper: 8)")
    p.add_argument("--k", type=int, default=5, help="neighbours for KNN")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model(name, args):
    if name == "knn":
        return MODELS[name](k=args.k)
    if name in ("decision_tree", "random_forest", "random_forest_tuned"):
        return MODELS[name](seed=args.seed)
    return MODELS[name]()


def load_split(split, seed):
    if split == "official":
        return load(TRAIN_FILE), load(TEST_FILE)
    df = load(TRAIN_FILE)
    return train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)


def prepare(split="official", n_features=8, seed=42):
    """Load and preprocess a split; `unseen` = test DoS types absent from training."""
    train_df, test_df = load_split(split, seed)
    prep = Preprocessor(n_features=n_features)
    X_train, y_train = prep.fit_transform(train_df)
    X_test, y_test = prep.transform(test_df)
    unseen = (test_df["label"] == 1) & ~test_df["attack"].isin(train_df["attack"])
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "prep": prep,
        "train_df": train_df,
        "test_df": test_df,
        "unseen": unseen.values,
    }


def main():
    args = parse_args()

    d = prepare(args.split, args.n_features, args.seed)
    train_df, test_df, prep = d["train_df"], d["test_df"], d["prep"]
    X_train, y_train, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    unseen = d["unseen"]
    print(f"Split: {args.split}")
    print(f"Train rows: {len(train_df):,} ({train_df['label'].mean():.1%} DoS)")
    print(f"Test rows:  {len(test_df):,} ({test_df['label'].mean():.1%} DoS)\n")

    print(f"Top {args.n_features} features by |correlation| with label:")
    for name in prep.selected_features:
        print(f"  {name:<28} {prep.correlations[name]:.3f}")
    print()

    if unseen.any():
        names = ", ".join(sorted(test_df.loc[unseen, "attack"].unique()))
        print(f"Unseen DoS types in test: {names} ({unseen.sum():,} rows)\n")

    rows = []
    for name in args.models:
        model = build_model(name, args)
        result = evaluate(model, X_train, y_train, X_test, y_test, unseen=unseen)
        cm = result.pop("confusion_matrix")
        rows.append({"model": name, **result})

        print(f"== {name} ==")
        print("  confusion matrix (rows = actual normal/DoS, cols = predicted):")
        print(f"    {cm[0]}\n    {cm[1]}")
        paper = PAPER_RESULTS.get(name)
        for metric in ("accuracy", "precision", "recall", "f1"):
            line = f"  {metric:<10} {result[metric]:.4f}"
            if paper and metric in paper:
                line += (
                    f"   (paper: {paper[metric]:.4f}, diff: {result[metric] - paper[metric]:+.4f})"
                )
            print(line)
        if unseen.any():
            print(f"  recall on seen DoS types   {result['recall_seen']:.4f}")
            print(f"  recall on unseen DoS types {result['recall_unseen']:.4f}")
        if "oob_score" in result:
            print(f"  out-of-bag accuracy        {result['oob_score']:.4f}")
        print(f"  train {result['train_time_s']:.2f}s, predict {result['predict_time_s']:.2f}s\n")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"metrics_{args.split}_{args.n_features}feat.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
