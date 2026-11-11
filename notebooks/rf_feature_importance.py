"""Random Forest feature importance chart (BRAINSTORM.md step 4).

Trains the Random Forest on all 41 features of KDDTrain+ and compares two
importance measures:
  - Built-in (MDI): computed from the training data during fitting. Biased
    toward features with many unique values.
  - Permutation: shuffle one feature at a time on KDDTest+ and measure the F1
    drop. Shows what the model relies on for data it has not seen.

Bars are coloured by whether the paper's correlation filter (top 8 features)
keeps the feature.

Permutation importance under-rates features that duplicate each other
(shuffle one and its twin covers for it). So the script also shuffles each
group of highly correlated features (|r| > 0.9 on training data) together.

Writes:
    results/rf_feature_importance.png
    results/rf_feature_importance.csv   (all 41 features, the table view)
    results/rf_grouped_importance.csv   (correlated groups shuffled together)

Usage (from the repo root, ~1.5 min):
    python notebooks/rf_feature_importance.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `src` imports work

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from src.data import TEST_FILE, TRAIN_FILE, load
from src.models import random_forest
from src.preprocess import Preprocessor

SEED = 42
TOP_N = 12  # features shown: each measure's top 12, plus all of the paper's 8
GROUP_CORR = 0.9  # features this correlated are shuffled together

# Colours (validated for colour-blind separation and contrast on white).
KEPT = "#2a78d6"     # in the paper's 8 features
DROPPED = "#eb6834"  # dropped by the paper's filter
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def compute():
    train, test = load(TRAIN_FILE), load(TEST_FILE)
    prep = Preprocessor(n_features=41)
    X_train, y_train = prep.fit_transform(train)
    X_test, y_test = prep.transform(test)

    model = random_forest(seed=SEED).fit(X_train, y_train)
    perm = permutation_importance(
        model, X_test, y_test, scoring="f1", n_repeats=10, random_state=SEED
    )

    paper_8 = set(prep.correlations.index[:8])
    table = pd.DataFrame(
        {
            "mdi": model.feature_importances_,
            "perm_f1_drop": perm.importances_mean,
            "perm_f1_drop_std": perm.importances_std,
            "abs_corr_with_label": prep.correlations.reindex(X_train.columns).values,
            "unique_values": X_train.nunique().values,
        },
        index=X_train.columns,
    )
    table["in_paper_top8"] = table.index.isin(paper_8)
    table["rank_mdi"] = table["mdi"].rank(ascending=False).astype(int)
    table["rank_perm"] = table["perm_f1_drop"].rank(ascending=False).astype(int)

    groups = correlated_groups(X_train)
    table["group"] = [next(" + ".join(g) for g in groups if f in g) for f in table.index]
    grouped = grouped_permutation(model, X_test, y_test, groups, paper_8)
    return table.sort_values("perm_f1_drop", ascending=False), grouped


def correlated_groups(X):
    """Group features whose absolute correlation (training data) exceeds GROUP_CORR."""
    corr = X.corr().abs().fillna(0)
    groups, used = [], set()
    for f in corr.index:
        if f not in used:
            # Start with f itself: a constant column has NaN (-> 0) self-correlation.
            group = [f] + [c for c in corr.index
                           if c != f and corr.at[f, c] > GROUP_CORR and c not in used]
            used.update(group)
            groups.append(group)
    return groups


def grouped_permutation(model, X_test, y_test, groups, paper_8, n_repeats=10):
    """F1 drop when every feature in a group is shuffled with the same row order."""
    rng = np.random.default_rng(SEED)
    base = f1_score(y_test, model.predict(X_test))
    rows = []
    for group in groups:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[group] = X_test[group].values[rng.permutation(len(X_test))]
            drops.append(base - f1_score(y_test, model.predict(X_perm)))
        rows.append({"group": " + ".join(group), "size": len(group),
                     "f1_drop": np.mean(drops), "f1_drop_std": np.std(drops),
                     "has_paper_top8": any(f in paper_8 for f in group)})
    return pd.DataFrame(rows).sort_values("f1_drop", ascending=False)


def plot(table, out_path):
    shown = table[(table["rank_mdi"] <= TOP_N) | (table["rank_perm"] <= TOP_N)
                  | table["in_paper_top8"]]
    shown = shown.sort_values("perm_f1_drop")  # largest at the top of the chart
    colors = [KEPT if kept else DROPPED for kept in shown["in_paper_top8"]]
    y = range(len(shown))

    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT_MUTED,
                         "xtick.color": TEXT_MUTED, "ytick.color": TEXT})
    fig, (ax_mdi, ax_perm) = plt.subplots(
        1, 2, figsize=(9, 0.28 * len(shown) + 1.8), sharey=True
    )

    panels = [
        (ax_mdi, "mdi", None, "Built-in importance (MDI)",
         "Share of impurity reduction, training data"),
        (ax_perm, "perm_f1_drop", "perm_f1_drop_std", "Permutation importance",
         "F1 drop when shuffled, KDDTest+"),
    ]
    for ax, col, err, title, xlabel in panels:
        ax.barh(y, shown[col], height=0.62, color=colors, edgecolor="white", linewidth=1)
        if err:
            ax.errorbar(shown[col], y, xerr=shown[err], fmt="none",
                        ecolor=TEXT_MUTED, elinewidth=0.8, capsize=0)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel)
        # Permutation can dip below zero (shuffling helped by chance); keep those visible.
        low = (shown[col] - (shown[err] if err else 0)).min()
        ax.set_xlim(left=min(0, low * 1.3))
        ax.axvline(0, color=TEXT_MUTED, linewidth=0.8)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)

        # Direct labels on the top 3 bars of each panel only.
        top3 = shown[col].nlargest(3).index
        for i, (name, value) in enumerate(shown[col].items()):
            if name in top3:
                end = value + (shown.at[name, err] if err else 0)  # clear the error bar
                ax.text(end + ax.get_xlim()[1] * 0.015, i, f"{value:.3f}",
                        va="center", fontsize=8, color=TEXT_MUTED)

    ax_mdi.set_yticks(list(y), shown.index)
    fig.legend(
        handles=[Patch(color=KEPT, label="Kept by paper's 8-feature filter"),
                 Patch(color=DROPPED, label="Dropped by the filter")],
        loc="lower left", bbox_to_anchor=(0.01, 0.985), ncol=2, frameon=False, fontsize=9,
    )
    fig.suptitle("Random Forest feature importance (41 features, trained on KDDTrain+)",
                 x=0.01, y=1.045, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")


def main():
    table, grouped = compute()
    out_dir = ROOT / "results"
    table.to_csv(out_dir / "rf_feature_importance.csv", float_format="%.4f",
                 index_label="feature")
    grouped.to_csv(out_dir / "rf_grouped_importance.csv", float_format="%.4f", index=False)
    plot(table, out_dir / "rf_feature_importance.png")

    cols = ["mdi", "perm_f1_drop", "rank_mdi", "rank_perm", "abs_corr_with_label",
            "unique_values", "in_paper_top8"]
    print(table[cols].head(15).round(4).to_string())
    print("\nCorrelated groups shuffled together:")
    print(grouped.head(8).round(4).to_string(index=False))
    print("\nSaved results/rf_feature_importance.png, .csv and rf_grouped_importance.csv")


if __name__ == "__main__":
    main()
