"""Generate the paper's LaTeX tables and number macros from results/*.csv.

Run from the repository root:
    env -u PYTHONPATH .venv/bin/python docs/paper/make_tables.py

Writes docs/paper/tables/tab_*.tex (booktabs floats) and docs/paper/tables/numbers.tex,
a file of \\newcommand macros that main.tex uses for every number quoted in the prose, so
no result is typed by hand. Percentages are shown to 2 decimals.
"""

from pathlib import Path

import pandas as pd

from dosdetect.data import CATEGORICAL_COLUMNS, DOS_ATTACKS, load_split, prepare
from dosdetect.evaluate import PAPER_RESULTS, paper_table

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "tables"

MODEL_NAMES = {
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "random_forest_tuned": "RF (tuned)",
}
MODELS = list(MODEL_NAMES)
DATASET_NAMES = {"nsl-kdd": "NSL-KDD", "kddcup": "KDDCup"}
SPLIT_NAMES = {"official": "official", "holdout": "holdout", "kddcup": "KDDCup"}


def pct(x):
    """Fraction to a percentage string with 2 decimals (no percent sign)."""
    return f"{100 * x:.2f}"


def sig(x):
    """Three significant figures, for times and ratios."""
    if x >= 1000:
        return f"{x:,.0f}".replace(",", "{,}")
    return f"{x:.3g}"


def tt(name):
    """Feature name in \\texttt, with breakable escaped underscores."""
    return "\\texttt{" + name.replace("_", "\\_\\allowbreak ") + "}"


def write(name, text):
    (OUT / name).write_text(text)


def float_env(body, caption, label, wide=False, size="footnotesize"):
    env = "table*" if wide else "table"
    return (
        f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\{size}\n\\setlength{{\\tabcolsep}}{{2.5pt}}\n{body}\\end{{{env}}}\n"
    )


def tabular(spec, header_lines, rows):
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    lines += header_lines
    lines.append("\\midrule")
    lines += rows
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def row(cells):
    return " & ".join(cells) + " \\\\"


# ---------------------------------------------------------------- tables


def tab_paper(metrics):
    rows, mac = [], {}
    for model in ("knn", "naive_bayes"):
        results = {
            (r.split, r.n_features): r._asdict()
            for r in metrics[metrics.model == model].itertuples()
        }
        t = paper_table(model, results)
        for i, (metric, r) in enumerate(t.iterrows()):
            first = f"\\multirow{{3}}{{*}}{{{MODEL_NAMES[model]}}}" if i == 0 else ""
            rows.append(
                row(
                    [first, metric]
                    + [pct(r[c]) for c in ("NSL-KDD", "KDDCup", "mean", "paper [4]")]
                    + [f"{100 * r['difference']:+.2f}".replace("-", "$-$")]
                )
            )
            tag = {"knn": "Knn", "naive_bayes": "Nb"}[model] + metric.capitalize()
            mac[f"diff{tag}"] = f"{abs(100 * r['difference']):.2f}"
            mac[f"mean{tag}"] = pct(r["mean"])
            mac[f"paper{tag}"] = pct(r["paper [4]"])
            mac[f"nsl{tag}"] = pct(r["NSL-KDD"])
            mac[f"cup{tag}"] = pct(r["KDDCup"])
        if model == "knn":
            rows.append("\\midrule")
    head = [row(["Model", "Metric", "NSL-KDD", "KDDCup", "Mean", "Reported", "Diff."])]
    body = tabular("llrrrrr", head, rows)
    cap = (
        "Reproduction of the reference results with 8 features (percent). "
        "NSL-KDD is the official split and KDDCup our 80/20 split. The paper reports one value "
        "across both datasets, so the mean of our two datasets is the like-for-like column. "
        "Diff.\\ is our mean minus the paper, in percentage points."
    )
    write("tab_paper.tex", float_env(body, cap, "tab:paper"))
    return mac


def tab_official(metrics):
    df = metrics[metrics.split == "official"]
    rows = []
    for nf in (8, 41):
        for m in MODELS:
            r = df[(df.n_features == nf) & (df.model == m)].iloc[0]
            rows.append(
                row(
                    [MODEL_NAMES[m], str(nf)]
                    + [pct(r[c]) for c in ("accuracy", "precision", "recall", "f1")]
                    + [pct(r.recall_seen), pct(r.recall_unseen)]
                )
            )
        if nf == 8:
            rows.append("\\midrule")
    head = [row(["Model", "Feat.", "Acc.", "Prec.", "Rec.", "F1", "Rec.\\ seen", "Rec.\\ unseen"])]
    body = tabular("lrrrrrrr", head, rows)
    cap = (
        "NSL-KDD official split (train on KDDTrain+, test on KDDTest+), percent. "
        "Rec.\\ seen and Rec.\\ unseen are recall on DoS types present and absent in training."
    )
    write("tab_official.tex", float_env(body, cap, "tab:official", wide=True))


def tab_holdout_kddcup(metrics):
    rows = []
    for nf in (8, 41):
        for m in MODELS:
            cells = [MODEL_NAMES[m], str(nf)]
            for split in ("holdout", "kddcup"):
                r = metrics[
                    (metrics.split == split) & (metrics.n_features == nf) & (metrics.model == m)
                ].iloc[0]
                cells += [pct(r.accuracy), pct(r.f1)]
            rows.append(row(cells))
        if nf == 8:
            rows.append("\\midrule")
    head = [
        " & & \\multicolumn{2}{c}{NSL-KDD holdout} & \\multicolumn{2}{c}{KDDCup} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        row(["Model", "Feat.", "Acc.", "F1", "Acc.", "F1"]),
    ]
    body = tabular("lrrrrr", head, rows)
    cap = "Accuracy and F1 (percent) on the two in-distribution 80/20 splits."
    write("tab_holdout_kddcup.tex", float_env(body, cap, "tab:holdout"))


def tab_loao(ao):
    g = ao.groupby(["dataset", "n_features", "model"])[
        ["recall_heldout", "recall_seen", "false_alarm_rate"]
    ].mean()
    rows = []
    for nf in (8, 41):
        for m in MODELS:
            cells = [MODEL_NAMES[m], str(nf)]
            for ds in DATASET_NAMES:
                r = g.loc[(ds, nf, m)]
                cells += [pct(r.recall_heldout), pct(r.recall_seen), pct(r.false_alarm_rate)]
            rows.append(row(cells))
        if nf == 8:
            rows.append("\\midrule")
    head = [
        " & & \\multicolumn{3}{c}{NSL-KDD} & \\multicolumn{3}{c}{KDDCup} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        row(["Model", "Feat.", "Held-out", "Seen", "FA", "Held-out", "Seen", "FA"]),
    ]
    body = tabular("lrrrrrrr", head, rows)
    cap = (
        "Leave-one-attack-out, mean over the four held-out groups (percent, one seed). "
        "Held-out is recall on the withheld DoS types, Seen is recall on the DoS types "
        "kept in training, and FA is the false-alarm rate on normal traffic."
    )
    write("tab_loao.tex", float_env(body, cap, "tab:loao", wide=True))
    return g


def tab_timing(tm):
    rows = []
    for m in MODELS:
        cells = [MODEL_NAMES[m]]
        for split, nf in (("official", 8), ("kddcup", 41)):
            r = tm[(tm.split == split) & (tm.n_features == nf) & (tm.model == m)].iloc[0]
            cells += [
                sig(r.predict_s_median),
                sig(r.train_s_median),
                sig(r.predict_ratio_vs_naive_bayes),
                sig(r.train_ratio_vs_decision_tree),
            ]
        rows.append(row(cells))
    head = [
        " & \\multicolumn{4}{c}{NSL-KDD official, 8 features} & \\multicolumn{4}{c}{KDDCup, 41 features}"
        " \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        row(["Model"] + ["Pred.\\ (s)", "Train (s)", "Pred./NB", "Train/DT"] * 2),
    ]
    body = tabular("lrrrrrrrr", head, rows)
    cap = (
        "Median of 5 runs of prediction and training time in seconds, and the ratios of "
        "prediction time to Naive Bayes (Pred./NB) and of training time to the Decision Tree "
        "(Train/DT). Measured on an 8-core machine under background load."
    )
    write("tab_timing.tex", float_env(body, cap, "tab:timing", wide=True))


def count(n):
    return f"{n:,}".replace(",", "{,}")


def tab_data():
    rows, mac = [], {}
    for split in ("official", "holdout", "kddcup"):
        train, test = load_split(split)
        unseen = "--"
        if split == "official":
            d = prepare("official", 8)
            n_unseen = int(d["unseen"].sum())
            n_dos = int(test.label.sum())
            unseen = count(n_unseen)
            mac["unseenCount"] = count(n_unseen)
            mac["testDosCount"] = count(n_dos)
            mac["unseenShare"] = pct(n_unseen / n_dos)
            mac["unseenShareOne"] = f"{100 * n_unseen / n_dos:.1f}"
        rows.append(
            row(
                [SPLIT_NAMES[split]]
                + [count(int((df.label == v).sum())) for df in (train, test) for v in (0, 1)]
                + [unseen]
            )
        )
    head = [
        " & \\multicolumn{2}{c}{Train} & \\multicolumn{2}{c}{Test} & \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        row(["Setup", "Normal", "DoS", "Normal", "DoS", "Unseen DoS"]),
    ]
    body = tabular("lrrrrr", head, rows)
    cap = (
        "Number of connection records per setup after keeping only normal and DoS traffic. "
        "The last column counts test DoS records whose attack type never appears in training."
    )
    write("tab_data.tex", float_env(body, cap, "tab:data"))
    return mac


# ---------------------------------------------------------------- prose numbers


def rank_macros():
    """Spearman ranks of the features the reference paper names, on both training sets."""
    mac = {}
    for tag, split in (("Nsl", "official"), ("Cup", "kddcup")):
        prep = prepare(split, 8)["prep"]
        order = list(prep.correlations.index)
        for feat, key in (
            ("dst_bytes", "DstBytes"),
            ("protocol_type", "Protocol"),
            ("count", "Count"),
            ("dst_host_count", "DstHostCount"),
        ):
            mac[f"rank{tag}{key}"] = str(order.index(feat) + 1)
        mac[f"top{tag}"] = ", ".join(tt(f) for f in prep.selected_features)
    mac["nCategorical"] = str(len(CATEGORICAL_COLUMNS))
    mac["nDosTypes"] = str(len(DOS_ATTACKS))
    return mac


def prose_macros(metrics, ao, g, tm, rf):
    mac = {}
    off = metrics[metrics.split == "official"].set_index(["n_features", "model"])
    for nf, tag in ((8, "Eight"), (41, "All")):
        un = off.loc[nf].recall_unseen
        mac[f"unseenMin{tag}"] = pct(un.min())
        mac[f"unseenMax{tag}"] = pct(un.max())
        for m, mt in (("knn", "Knn"), ("naive_bayes", "Nb"), ("decision_tree", "Dt")):
            mac[f"unseen{mt}{tag}"] = pct(off.loc[(nf, m)].recall_unseen)
            mac[f"seen{mt}{tag}"] = pct(off.loc[(nf, m)].recall_seen)
            mac[f"offAcc{mt}{tag}"] = pct(off.loc[(nf, m)].accuracy)
    for m, mt in (("random_forest", "Rf"), ("random_forest_tuned", "Rft")):
        for nf, tag in ((8, "Eight"), (41, "All")):
            mac[f"unseen{mt}{tag}"] = pct(off.loc[(nf, m)].recall_unseen)
    # seen-vs-unseen gap for the strongest and weakest case
    for split, st in (("holdout", "Hold"), ("kddcup", "Cup")):
        d = metrics[metrics.split == split].set_index(["n_features", "model"])
        mac[f"nbAccEight{st}"] = pct(d.loc[(8, "naive_bayes")].accuracy)
        mac[f"nbAccAll{st}"] = pct(d.loc[(41, "naive_bayes")].accuracy)
        mac[f"minAccAll{st}"] = pct(d.loc[41].drop("naive_bayes").accuracy.min())
    mac["nbAccEightOff"] = pct(off.loc[(8, "naive_bayes")].accuracy)
    mac["nbAccAllOff"] = pct(off.loc[(41, "naive_bayes")].accuracy)
    for m, mt in (("decision_tree", "Dt"), ("random_forest", "Rf"), ("random_forest_tuned", "Rft")):
        mac[f"offAccAll{mt}"] = pct(off.loc[(41, m)].accuracy)

    # leave-one-attack-out
    non_nb = ao[ao.model != "naive_bayes"]
    mac["loaoN"] = str(len(non_nb))
    mac["loaoUnderHalf"] = str(int((non_nb.recall_heldout < 0.5).sum()))
    mac["loaoSeenMin"] = pct(non_nb.recall_seen.min())
    mac["loaoFaMax"] = pct(non_nb.false_alarm_rate.max())
    knn_means = g.xs("knn", level="model").recall_heldout
    mac["knnLoaoMax"] = pct(knn_means.max())
    nb = g.xs("naive_bayes", level="model")
    mac["nbLoaoNslAllRec"] = pct(nb.loc[("nsl-kdd", 41)].recall_heldout)
    mac["nbLoaoNslAllFa"] = pct(nb.loc[("nsl-kdd", 41)].false_alarm_rate)
    mac["nbLoaoNslEightRec"] = pct(nb.loc[("nsl-kdd", 8)].recall_heldout)
    mac["nbLoaoNslEightFa"] = pct(nb.loc[("nsl-kdd", 8)].false_alarm_rate)
    mac["nbFaMax"] = pct(ao[ao.model == "naive_bayes"].false_alarm_rate.max())
    sm = ao[
        (ao.dataset == "nsl-kdd")
        & (ao.n_features == 8)
        & (ao.model == "naive_bayes")
        & (ao.group == "smurf")
    ].iloc[0]
    mac["nbSmurfNslRec"] = pct(sm.recall_heldout)
    mac["nbSmurfNslFa"] = pct(sm.false_alarm_rate)
    sc = ao[
        (ao.dataset == "kddcup")
        & (ao.n_features == 8)
        & (ao.model == "naive_bayes")
        & (ao.group == "smurf")
    ].iloc[0]
    mac["nbSmurfCupRec"] = pct(sc.recall_heldout)
    mac["nbSmurfCupFa"] = pct(sc.false_alarm_rate)
    for ds, dt in (("nsl-kdd", "Nsl"), ("kddcup", "Cup")):
        for nf, ft in ((8, "Eight"), (41, "All")):
            for m, mt in (("random_forest", "Rf"), ("random_forest_tuned", "Rft")):
                mac[f"loao{dt}{ft}{mt}"] = pct(g.loc[(ds, nf, m)].recall_heldout)

    # timing
    tmx = tm.set_index(["split", "n_features", "model"])
    kp = tm[tm.model == "knn"].predict_ratio_vs_naive_bayes
    mac["knnPredMin"] = f"{kp.min():.1f}"
    mac["knnPredMax"] = f"{kp.max():.1f}"
    for m, mt in (("random_forest", "Rf"), ("random_forest_tuned", "Rft")):
        t = tm[tm.model == m]
        mac[f"train{mt}Min"] = f"{t.train_ratio_vs_decision_tree.min():.1f}"
        mac[f"train{mt}Max"] = f"{t.train_ratio_vs_decision_tree.max():.1f}"
    mac["knnPredCupAll"] = f"{tmx.loc[('kddcup', 41, 'knn')].predict_s_median:.2f}"
    mac["rfTrainCupAll"] = f"{tmx.loc[('kddcup', 41, 'random_forest')].train_s_median:.2f}"
    dtp = tm[tm.model == "decision_tree"].predict_ratio_vs_naive_bayes
    mac["dtPredMin"] = f"{dtp.min():.2f}"
    mac["dtPredMax"] = f"{dtp.max():.2f}"
    mac["cpuCount"] = str(int(tm.cpu_count.iloc[0]))
    mac["repeats"] = str(int(tm.repeats.iloc[0]))
    mac["pyVersion"] = str(tm.python.iloc[0])
    mac["skVersion"] = str(tm.sklearn.iloc[0])

    # rf tuning
    r = rf.set_index("config")
    for cfg, tag in (
        ("default, top 8", "DefEight"),
        ("default, 41", "DefAll"),
        ("tuned (standard CV)", "TunStd"),
        ("tuned (attack-out CV)", "TunAo"),
    ):
        mac[f"rf{tag}F"] = f"{r.loc[cfg].f1_mean:.4f}"
        mac[f"rf{tag}U"] = f"{r.loc[cfg].recall_unseen_mean:.4f}"
        mac[f"rf{tag}Cv"] = f"{r.loc[cfg].cv_f1:.4f}" if pd.notna(r.loc[cfg].cv_f1) else ""
    return mac


def write_numbers(mac):
    lines = ["% Generated by make_tables.py. Do not edit."]
    for k, v in sorted(mac.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    write("numbers.tex", "\n".join(lines) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    ao = pd.read_csv(RESULTS / "attack_out.csv")
    tm = pd.read_csv(RESULTS / "timing.csv")
    rf = pd.read_csv(RESULTS / "rf_tuning.csv")
    mac = tab_paper(metrics)
    tab_official(metrics)
    tab_holdout_kddcup(metrics)
    g = tab_loao(ao)
    tab_timing(tm)
    mac.update(tab_data())
    mac.update(rank_macros())
    mac.update(prose_macros(metrics, ao, g, tm, rf))
    mac["paperKnnAcc"] = pct(PAPER_RESULTS["knn"]["accuracy"])
    write_numbers(mac)
    print(f"wrote tables and {len(mac)} macros to {OUT}")


if __name__ == "__main__":
    main()
