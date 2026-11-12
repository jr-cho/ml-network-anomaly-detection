"""Check the Random Forest claims in docs/random_forest_notes.md (section 9).

Prints results only, writes no files. ~2 min.
Run: python experiments/random_forest/logic_test.py
"""

import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dosdetect.data import TEST_FILE, TRAIN_FILE, load
from dosdetect.models import knn, naive_bayes
from dosdetect.preprocess import Preprocessor

SEED = 42
UNSEEN = {"apache2", "processtable", "mailbomb", "udpstorm"}


def rf(**kw):
    p = dict(n_estimators=300, n_jobs=-1, oob_score=True, random_state=SEED)
    p.update(kw)
    return RandomForestClassifier(**p)


def splits():
    tr, te = load(TRAIN_FILE), load(TEST_FILE)
    h_tr, h_te = train_test_split(tr, test_size=0.2, stratify=tr["label"], random_state=SEED)
    return {"holdout": (h_tr, h_te), "official": (tr, te)}


def scores(y, p):
    return dict(
        acc=accuracy_score(y, p),
        prec=precision_score(y, p),
        rec=recall_score(y, p),
        f1=f1_score(y, p),
    )


def fmt(d):
    return "  ".join(f"{k}={v:.4f}" for k, v in d.items())


def main():
    S = splits()
    data = {}
    print("=== T1: default RF, 4 configs (+ KNN/NB for timing and recall breakdown) ===")
    for split, (tr, te) in S.items():
        for n in (8, 41):
            prep = Preprocessor(n_features=n)
            Xtr, ytr = prep.fit_transform(tr)
            Xte, yte = prep.transform(te)
            data[(split, n)] = (prep, Xtr, ytr, Xte, yte, te)
            for name, m in [("rf", rf()), ("knn", knn()), ("nb", naive_bayes())]:
                t = time.perf_counter()
                m.fit(Xtr, ytr)
                ft = time.perf_counter() - t
                t = time.perf_counter()
                p = m.predict(Xte)
                pt = time.perf_counter() - t
                line = (
                    f"{split:8} {n:2}f {name:3} {fmt(scores(yte, p))}  fit={ft:.2f}s pred={pt:.2f}s"
                )
                if name == "rf":
                    line += f"  oob={m.oob_score_:.4f}"
                if split == "official":
                    dos = te["label"].values == 1
                    un = te["attack"].isin(UNSEEN).values
                    seen_rec = p[dos & ~un].mean()
                    unseen_rec = p[dos & un].mean()
                    line += f"\n{'':15} recall seen={seen_rec:.4f} unseen={unseen_rec:.4f}"
                    if name == "rf" and n == 8:
                        per = (
                            te[dos]
                            .assign(pred=p[dos])
                            .groupby("attack")["pred"]
                            .agg(["mean", "size"])
                        )
                        line += "\n" + per.to_string()
                print(line, flush=True)

    print("\n=== T2: StandardScaler changes nothing for RF? (official, 8f) ===")
    _, Xtr, ytr, Xte, yte, _ = data[("official", 8)]
    a = rf(oob_score=False).fit(Xtr, ytr).predict(Xte)
    b = make_pipeline(StandardScaler(), rf(oob_score=False)).fit(Xtr, ytr).predict(Xte)
    print(f"identical predictions: {(a == b).mean():.4%}")

    print("\n=== T3: n_estimators plateau (OOB, official train, 8f) ===")
    m = RandomForestClassifier(warm_start=True, oob_score=True, n_jobs=-1, random_state=SEED)
    for n in (10, 25, 50, 100, 200, 300, 500):
        m.set_params(n_estimators=n)
        m.fit(Xtr, ytr)
        print(f"  trees={n:3}  oob={m.oob_score_:.5f}")

    print("\n=== T4: max_features='sqrt' with 8 / 41 features ===")
    for n in (8, 41):
        f = rf(n_estimators=5).fit(*data[("official", n)][1:3])
        print(f"  {n} features -> max_features_ used = {f.estimators_[0].max_features_}")

    print("\n=== T5: MDI vs permutation importance (41 features, official) ===")
    prep, Xtr, ytr, Xte, yte, _ = data[("official", 41)]
    f = rf().fit(Xtr, ytr)
    mdi = pd.Series(f.feature_importances_, Xtr.columns)
    _, hXtr, hytr, hXte, hyte, _ = data[("holdout", 41)]
    fh = rf().fit(hXtr, hytr)
    pi_h = permutation_importance(
        fh, hXte, hyte, scoring="recall", n_repeats=5, random_state=SEED, n_jobs=-1
    )
    pi_o = permutation_importance(
        f, Xte, yte, scoring="recall", n_repeats=5, random_state=SEED, n_jobs=-1
    )
    nunique = Xtr.nunique()
    tbl = pd.DataFrame(
        {
            "mdi": mdi,
            "perm_holdout": pd.Series(pi_h.importances_mean, hXte.columns),
            "perm_official": pd.Series(pi_o.importances_mean, Xte.columns),
            "abs_corr": prep.correlations.reindex(Xtr.columns),
            "n_unique": nunique,
        }
    )
    print(tbl.sort_values("mdi", ascending=False).head(12).round(4).to_string())
    print(
        "  top by perm_official:", list(tbl["perm_official"].sort_values(ascending=False).index[:6])
    )
    print(
        f"  Spearman(MDI rank, n_unique) = {tbl['mdi'].corr(tbl['n_unique'], method='spearman'):.3f}"
    )
    print(f"  Top-8 by correlation: {prep.selected_features[:8]}")
    print(f"  Top-8 by MDI:         {list(mdi.sort_values(ascending=False).index[:8])}")

    print("\n=== T6: class_weight effect (official, 8f) ===")
    _, Xtr, ytr, Xte, yte, _ = data[("official", 8)]
    for cw in (None, "balanced", "balanced_subsample"):
        print(
            f"  {str(cw):20} {fmt(scores(yte, rf(class_weight=cw, oob_score=False).fit(Xtr, ytr).predict(Xte)))}"
        )

    print(
        "\n=== T7: cheap RandomizedSearchCV on train only (scored by recall), then official test ==="
    )
    for n in (8, 41):
        _, Xtr, ytr, Xte, yte, _ = data[("official", n)]
        space = dict(
            max_features=["sqrt", "log2", 0.3, 0.5, 1.0],
            min_samples_leaf=[1, 2, 5, 10, 20],
            max_depth=[None, 10, 20, 30],
            max_samples=[None, 0.5, 0.8],
            class_weight=[None, "balanced", "balanced_subsample"],
            criterion=["gini", "entropy"],
        )
        t = time.perf_counter()
        s = RandomizedSearchCV(
            RandomForestClassifier(n_estimators=100, random_state=SEED),
            space,
            n_iter=15,
            scoring="recall",
            cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
            n_jobs=-1,
            random_state=SEED,
        ).fit(Xtr, ytr)
        print(
            f"  {n}f search {time.perf_counter() - t:.0f}s  best cv recall={s.best_score_:.5f}  {s.best_params_}"
        )
        tuned = rf(oob_score=False, **s.best_params_).fit(Xtr, ytr).predict(Xte)
        print(f"     tuned on official:   {fmt(scores(yte, tuned))}")


if __name__ == "__main__":
    main()
