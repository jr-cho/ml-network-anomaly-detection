"""Build the paper's figures as vector PDFs in docs/paper/figures/.

Run from the repo root: env -u PYTHONPATH .venv/bin/python docs/paper/make_figures.py [names...]
Data comes from results/*.csv; the feature ranking and confusion matrices refit models
through the dosdetect package.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dosdetect.data import prepare  # noqa: E402
from dosdetect.evaluate import evaluate, paper_table  # noqa: E402
from dosdetect.models import MODELS  # noqa: E402

RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

COL1, COL2 = 3.5, 7.16
BLUE, GRAY, ORANGE = "#2a78d6", "#8a8984", "#eb6834"
TEXT, GRID = "#222222", "#dddddd"
MODEL_COLORS = {
    "knn": "#2a78d6",
    "naive_bayes": "#eb6834",
    "decision_tree": "#1baf7a",
    "random_forest": "#eda100",
    "random_forest_tuned": "#e87ba4",
}
MODEL_LABELS = {
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "random_forest_tuned": "Tuned forest",
}
DATASET_LABELS = {"nsl-kdd": "NSL-KDD", "kddcup": "KDDCup"}
SPLIT_LABELS = {
    "official": "NSL-KDD\nofficial",
    "holdout": "NSL-KDD\nholdout",
    "kddcup": "KDDCup\n80/20",
}
BLUES = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#2a78d6", "#0d366b"])
NAMED_FEATURES = {"dst_bytes", "protocol_type", "count", "dst_host_count"}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.edgecolor": "#666666",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def cell_text_color(value, vmin=0.0, vmax=1.0):
    return "white" if (value - vmin) / (vmax - vmin) > 0.5 else TEXT


def fig_feature_ranking():
    """Top 12 features by |Spearman| on both training sets."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7), sharex=True)
    setups = [("official", "NSL-KDD (KDDTrain+)"), ("kddcup", "KDDCup (80% train split)")]
    for ax, (split, title) in zip(axes, setups, strict=True):
        corr = prepare(split, 41)["prep"].correlations.head(12)
        y = np.arange(len(corr))
        colors = [BLUE if f in NAMED_FEATURES else GRAY for f in corr.index]
        ax.barh(y, corr.values, color=colors, height=0.65)
        for yi, v in zip(y, corr.values, strict=True):
            ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=7)
        ax.axhline(7.5, color="#444444", ls="--", lw=0.7)
        ax.set_yticks(y, corr.index)
        ax.invert_yaxis()
        ax.set_xlim(0, 0.95)
        ax.set_xlabel("|Spearman correlation| with label")
        ax.set_title(title, loc="left")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=GRAY),
        Line2D([], [], color="#444444", ls="--", lw=0.7),
    ]
    fig.legend(
        handles,
        ["feature named in the reference paper", "other", "top-8 cut-off"],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.07),
    )
    fig.tight_layout()
    save(fig, "fig_feature_ranking")


def fig_paper_comparison():
    """Our NSL-KDD, KDDCup and mean vs the paper's reported numbers."""
    m = pd.read_csv(RESULTS / "metrics.csv")
    m = m[m["n_features"] == 8]
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 2.2), sharey=True)
    metrics = ["accuracy", "precision", "recall"]
    series = [
        ("NSL-KDD", "o", "#8a8984", "full"),
        ("KDDCup", "s", "#8a8984", "none"),
        ("two-dataset mean", "D", None, "full"),
        ("reported", "*", "black", "full"),
    ]
    for ax, name in zip(axes, ["knn", "naive_bayes"], strict=True):
        sub = m[m["model"] == name]
        results = {(r.split, 8): r._asdict() for r in sub.itertuples()}
        table = paper_table(name, results)
        mean_color = MODEL_COLORS[name]
        for label, marker, color, fill in series:
            col = {"two-dataset mean": "mean", "reported": "paper [4]"}.get(label, label)
            c = color or mean_color
            ax.plot(
                table.loc[metrics, col].values,
                np.arange(3),
                ls="",
                marker=marker,
                ms=7 if marker == "*" else 5,
                mfc=c if fill == "full" else "white",
                mec=c,
                mew=0.9,
            )
        ax.set_yticks(range(3), metrics)
        ax.set_ylim(2.5, -0.5)
        ax.set_xlabel("score")
        ax.set_title(MODEL_LABELS[name], loc="left")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_xlim(0.7, 1.012)
    axes[1].set_xlim(0.7, 1.012)
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=mk,
            ms=7 if mk == "*" else 5,
            mfc=(c or "#555555") if f == "full" else "white",
            mec=c or "#555555",
            label=lab,
        )
        for lab, mk, c, f in series
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    save(fig, "fig_paper_comparison")


def fig_seen_unseen():
    """Official split: recall on seen vs unseen DoS types, per model."""
    m = pd.read_csv(RESULTS / "metrics.csv")
    m = m[m["split"] == "official"]
    fig, axes = plt.subplots(2, 1, figsize=(COL1, 3.4), sharex=True)
    for ax, nf in zip(axes, [8, 41], strict=True):
        sub = m[m["n_features"] == nf].set_index("model").loc[list(MODEL_COLORS)]
        for i, (name, r) in enumerate(sub.iterrows()):
            c = MODEL_COLORS[name]
            ax.plot([r.recall_unseen, r.recall_seen], [i, i], color=c, lw=1.2, zorder=1)
            ax.plot(r.recall_unseen, i, "o", color=c, ms=5, mfc="white", mew=1.1, zorder=2)
            ax.plot(r.recall_seen, i, "o", color=c, ms=5, zorder=2)
            ax.text(r.recall_unseen - 0.025, i, f"{r.recall_unseen:.2f}", ha="right", va="center")
            ax.text(r.recall_seen + 0.025, i, f"{r.recall_seen:.2f}", ha="left", va="center")
        ax.set_yticks(range(5), [MODEL_LABELS[k] for k in MODEL_COLORS])
        ax.set_ylim(4.6, -0.6)
        ax.set_xlim(-0.14, 1.14)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"{nf} features", loc="left")
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("recall on DoS test rows")
    handles = [
        Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec="#555555", label="unseen types"),
        Line2D([], [], ls="", marker="o", ms=5, color="#555555", label="seen types"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig_seen_unseen")


def fig_loao_heatmap():
    """Held-out recall for every model and held-out group, in four panels."""
    a = pd.read_csv(RESULTS / "attack_out.csv")
    groups = list(dict.fromkeys(a["group"]))
    models = list(MODEL_COLORS)
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.3), sharey=True)
    im = None
    for ax, (ds, nf) in zip(
        axes, [("nsl-kdd", 8), ("nsl-kdd", 41), ("kddcup", 8), ("kddcup", 41)], strict=True
    ):
        sub = a[(a["dataset"] == ds) & (a["n_features"] == nf)]
        grid = sub.pivot(index="model", columns="group", values="recall_heldout")
        grid = grid.loc[models, groups]
        im = ax.imshow(grid.values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for i in range(len(models)):
            for j in range(len(groups)):
                v = grid.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=cell_text_color(v))  # fmt: skip
        ax.set_xticks(range(len(groups)), groups, rotation=40, ha="right")
        ax.set_yticks(range(len(models)), [MODEL_LABELS[k] for k in models])
        ax.set_title(f"{DATASET_LABELS[ds]}, {nf} features", loc="left")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.915, 0.3, 0.012, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("recall on held-out group")
    cb.outline.set_visible(False)
    save(fig, "fig_loao_heatmap")


def fig_loao_tradeoff():
    """False-alarm rate vs held-out recall for every leave-one-attack-out run."""
    a = pd.read_csv(RESULTS / "attack_out.csv")
    fig, ax = plt.subplots(figsize=(COL1, 3.0))
    for nf, marker in [(8, "o"), (41, "s")]:
        for is_nb in (False, True):
            sub = a[(a["n_features"] == nf) & ((a["model"] == "naive_bayes") == is_nb)]
            ax.scatter(
                sub["false_alarm_rate"],
                sub["recall_heldout"],
                marker=marker,
                s=22,
                color=ORANGE if is_nb else GRAY,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
                zorder=3 if is_nb else 2,
                clip_on=False,
            )
    pt = a[
        (a["dataset"] == "nsl-kdd")
        & (a["group"] == "smurf")
        & (a["model"] == "naive_bayes")
        & (a["n_features"] == 8)
    ].iloc[0]
    ax.annotate(
        "NSL-KDD, smurf,\nNaive Bayes, 8 features",
        (pt.false_alarm_rate, pt.recall_heldout),
        xytext=(0.4, 0.55),
        arrowprops={"arrowstyle": "-", "color": "#444444", "lw": 0.6},
        fontsize=7,
        va="center",
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("false-alarm rate on normal test rows")
    ax.set_ylabel("recall on held-out DoS group")
    ax.grid(True)
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], ls="", marker="o", ms=5, color=ORANGE, label="Naive Bayes"),
        Line2D([], [], ls="", marker="o", ms=5, color=GRAY, label="other models"),
        Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec="#555555", label="8 features"),
        Line2D([], [], ls="", marker="s", ms=5, mfc="white", mec="#555555", label="41 features"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    save(fig, "fig_loao_tradeoff")


def fig_timing():
    """Median predict and train time per setup, with min-max whiskers (log scale)."""
    t = pd.read_csv(RESULTS / "timing.csv")
    setups = list(dict.fromkeys(zip(t["split"], t["n_features"], strict=True)))
    models = list(MODEL_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8))
    offsets = np.linspace(-0.3, 0.3, len(models))
    for ax, kind, title in zip(
        axes, ["predict", "train"], ["Predict time", "Train time"], strict=True
    ):
        for off, name in zip(offsets, models, strict=True):
            sub = t[t["model"] == name].set_index(["split", "n_features"]).loc[setups]
            med, lo, hi = (sub[f"{kind}_s_{s}"].values for s in ("median", "min", "max"))
            ax.errorbar(
                np.arange(len(setups)) + off,
                med,
                yerr=[med - lo, hi - med],
                fmt="o",
                ms=4.5,
                color=MODEL_COLORS[name],
                elinewidth=0.9,
                capsize=1.5,
                label=MODEL_LABELS[name],
            )
        ax.set_yscale("log")
        ax.set_xticks(range(len(setups)), [f"{SPLIT_LABELS[s]}\n{n} feat." for s, n in setups])
        ax.set_xlim(-0.6, len(setups) - 0.4)
        ax.set_ylabel(f"{kind} time, seconds (log scale)")
        ax.set_title(title + " (median, min to max of 5 runs)", loc="left")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig_timing")


def fig_rf_tuning():
    """Random-forest tuning configs: mean F1 and unseen recall with +/- std."""
    r = pd.read_csv(RESULTS / "rf_tuning.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 2.5), sharey=True)
    y = np.arange(len(r))
    for ax, col, label in zip(
        axes,
        ["f1_mean", "recall_unseen_mean"],
        ["F1 (mean $\\pm$ std)", "recall on unseen types\n(mean $\\pm$ std)"],
        strict=True,
    ):
        std = r[col.replace("_mean", "_std")]
        ax.errorbar(
            r[col], y, xerr=std, fmt="o", ms=5, color="#52514e",  # neutral: these configs are not random_forest_tuned()
            ecolor="#555555", elinewidth=0.9, capsize=2,
        )  # fmt: skip
        for yi, v in zip(y, r[col], strict=True):
            ax.text(v, yi - 0.22, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(len(r) - 0.4, -0.7)
        ax.set_xlabel(label)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [c.replace(", ", ",\n").replace(" (", "\n(") for c in r["config"]])
    axes[0].set_xlim(0.8, 0.92)
    axes[1].set_xlim(0, 0.25)
    axes[0].set_xticks([0.80, 0.84, 0.88, 0.92])
    fig.tight_layout()
    save(fig, "fig_rf_tuning")


def fig_confusion():
    """Confusion matrices for KNN and NB at 8 features on the official split and KDDCup."""
    panels = [(s, m) for s in ("official", "kddcup") for m in ("knn", "naive_bayes")]
    data = {s: prepare(s, 8) for s in ("official", "kddcup")}
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.0))
    for ax, (split, name) in zip(axes, panels, strict=True):
        cm = evaluate(MODELS[name](), data[split])["confusion_matrix"]
        ax.imshow(cm, cmap=BLUES, vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=7,
                        color=cell_text_color(cm[i, j], 0, cm.max()))  # fmt: skip
        ax.set_xticks([0, 1], ["normal", "DoS"])
        ax.set_yticks([0, 1], ["normal", "DoS"], rotation=90, va="center")
        ax.set_xlabel("predicted")
        ax.set_title(f"{MODEL_LABELS[name]}, {'NSL-KDD official' if split == 'official' else 'KDDCup'}",
                     loc="left", fontsize=7)  # fmt: skip
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("true")
    fig.tight_layout()
    save(fig, "fig_confusion")


FIGURES = [
    fig_feature_ranking,
    fig_paper_comparison,
    fig_seen_unseen,
    fig_loao_heatmap,
    fig_loao_tradeoff,
    fig_timing,
    fig_rf_tuning,
    fig_confusion,
]

if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for fn in FIGURES:
        if not wanted or fn.__name__ in wanted:
            print(fn.__name__, flush=True)
            fn()
