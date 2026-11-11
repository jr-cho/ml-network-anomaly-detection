# Random Forest Brainstorm

Owner: Joshua. Goal: add a Random Forest (RF) to the KNN / Naive Bayes comparison in a way that is fast, fair, and easy to explain in the report.

Citations use `[n]` and point to the **References** list at the bottom.

---

## 1. Where the project stands

- Pipeline: `src/data.py` (normal + DoS rows, 0/1 label) → `src/preprocess.py` (ordinal-encode 3 text columns, keep the top 8 features by |Pearson correlation|) → `src/evaluate.py` (fit, time, score) → `src/run.py`.
- RF slot: `src/models.py` has a `TODO (Joshua): random_forest()`. `run.py` calls `MODELS[name]()` for anything that is not KNN, so a no-argument `random_forest()` drops straight in.
- Current baselines (8 features):

| Split | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| holdout (80/20 of KDDTrain+) | KNN | 0.957 | 0.974 | 0.919 | 0.945 |
| holdout | Naive Bayes | 0.936 | 0.944 | 0.895 | 0.919 |
| official (KDDTest+) | KNN | 0.860 | 0.964 | 0.704 | 0.814 |
| official | Naive Bayes | 0.867 | 0.967 | 0.719 | 0.825 |

### Data facts that shape the RF plan

Counted directly from `data/`:

| | Normal | DoS | DoS share |
|---|---|---|---|
| KDDTrain+ | 67,343 | 45,927 | ~40.5% |
| KDDTest+ | 9,711 | 7,458 | ~43.4% |

- **About 23% of test DoS rows (1,717 of 7,458) are attack types never seen in training**: `apache2` (737), `processtable` (685), `mailbomb` (293), `udpstorm` (2). NSL-KDD's test set was built this way on purpose [3][4].
- Unseen attacks explain only **part** of the recall drop on the official split (~0.70 for both baselines). With 8 features, RF misses ~1,070 *seen* DoS rows (mostly `smurf`, `back`, `pod`) as well as ~1,470 unseen ones. The 8-feature filter is the other cause (see §9). With all 41 features, seen recall is 99.6% and almost every miss is an unseen attack. **No supervised model, RF included, is guaranteed to catch attack types it was never trained on** [8][9]. That gap is a finding to report, not a bug.
- Classes are only mildly imbalanced (40/60), so class weighting is optional (see §4).

---

## 2. Why Random Forest fits this project

- **What it is:** many decision trees, each trained on a bootstrap sample of the rows and choosing each split from a random subset of features. Predictions are a majority vote [1][5].
- **No scaling needed.** Trees split on thresholds, so `StandardScaler` adds nothing (unlike KNN/SVM) [12][17]. The model can be a bare estimator, not a pipeline.
- **Ordinal encoding is fine for trees.** The existing `OrdinalEncoder` output works, though it gives nominal columns like `service` a made-up order. Trees handle `OrdinalEncoder` output fine as long as they are deep enough [17], but it is worth one sentence in the report [12].
- **Built-in explainability.** Feature importances give a forensic story ("which traffic features signal DoS?"), which ties into the paper's call for explainable AI. Read them with care (see §5).
- **Cost:** heavier than NB to train, but prediction is fast and parallel. The reference paper itself notes RF "reduces overfitting and improves robustness, but also increases the computational load" (reference paper, Sec. II).

---

## 3. The efficient default (start here)

The scikit-learn defaults are already strong for classification [5][6]. Recommended first version:

```python
from sklearn.ensemble import RandomForestClassifier

def random_forest(n_estimators=300, seed=42):
    """Random Forest. Trees don't need scaled features."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",   # default; recommended for classification [5]
        n_jobs=-1,             # build trees on all CPU cores [5][6]
        oob_score=True,        # free validation estimate, no extra split [5][6]
        random_state=seed,     # reproducible results
    )
```

Why each choice:

| Setting | Reason |
|---|---|
| `n_estimators` ≈ 200–500 | More trees never hurt much; gains level off past a point. **Tested:** OOB score is flat from ~50–100 trees on (0.9639 at 100, 0.9639 at 500), so 100–300 is plenty. Probst & Boulesteix argue the tree count should be set "large enough", not tuned [5][11]. Fix it and leave it out of the search. |
| `max_features="sqrt"` | sklearn's recommended default for classification. sklearn rounds down: with 8 features that means **2** candidates per split; with 41 it is 6 [5] (tested). |
| `max_depth=None`, `min_samples_split=2` | Fully grown trees often do well, but they are large in RAM. Limit them only if model size or predict time matters [5][6]. |
| `n_jobs=-1` | Parallel fit and predict. Speedup is good but less than linear [5]. |
| `oob_score=True` | Each tree skips about 1/3 of rows; scoring on those gives a free generalisation estimate [1][5][6]. Use it to sanity-check without touching the test set. |
| `random_state` | Reuse `--seed` from `run.py` so results match between runs. |

**Wiring change:** `run.py` currently passes only `k` to KNN. Either keep `random_forest()` argument-free (simplest), or extend the `run.py` call to pass `seed=args.seed`.

---

## 4. Tuning: worth it, but keep it cheap

Probst, Wright & Boulesteix find RF defaults are usually reasonable, but tuning can still help a little [10]. Plan:

1. **Tune on training data only.** Use cross-validation (`StratifiedKFold`, 5 folds) or OOB on KDDTrain+. **Never pick settings using KDDTest+**, or the official numbers leak test information.
2. **Use `RandomizedSearchCV`, not a full grid.** Random search sets the budget (`n_iter`) independently of the number of parameters, and parameters that don't matter don't waste runs [13]. `HalvingRandomSearchCV` can be faster still but is experimental and needs `from sklearn.experimental import enable_halving_search_cv` [13].
3. **Small search space** (n_estimators fixed at ~300):

   | Param | Candidates |
   |---|---|
   | `max_features` | `"sqrt"`, `"log2"`, 0.3, 0.5, 1.0 |
   | `min_samples_leaf` | 1, 2, 5, 10, 20 |
   | `max_depth` | None, 10, 20, 30 |
   | `max_samples` | None, 0.5, 0.8 (smaller = faster, more randomness) [6] |
   | `class_weight` | None, `"balanced"`, `"balanced_subsample"` [6] |
   | `criterion` | `"gini"`, `"entropy"` |

4. **Score with F1 or recall, not accuracy.** In forensics a missed attack (false negative) costs more than a false alarm, and recall is where the baselines are weakest.
5. **Limit (found in testing):** cross-validation on KDDTrain+ only contains *seen* attacks. With 41 features, CV recall is already ~0.9998 for almost every setting, so the search can't tell candidates apart and can't aim at the unseen attacks. Any official-split gain from tuning is partly luck; check it over several seeds before claiming it.
6. **Report both** the default RF and the tuned RF. If tuning gains < ~0.5%, say so. That is a useful result too.

### Speed knobs if training is slow

- `n_jobs=-1` on the forest **or** the search, not both nested (avoids oversubscribing cores).
- `max_samples=0.5` halves the data per tree.
- `warm_start=True` lets you add trees to an existing forest instead of refitting, handy for plotting OOB score vs. tree count [6].
- `min_samples_leaf` / `max_depth` / `max_leaf_nodes` shrink trees → less RAM, faster predict [5][6].

~113k rows × 8 features is small, so the defaults should train in seconds.

---

## 5. Feature importance (the forensics angle)

Two ways to measure importance; report both and compare:

- **MDI (`feature_importances_`)**: free, computed during training. **Known flaws:** it is calculated on training data, and it **favours high-cardinality features** (columns with many unique values, e.g. `src_bytes`, `dst_bytes`) [5][7].
- **Permutation importance** (`sklearn.inspection.permutation_importance`): shuffle one column at a time on **held-out data** and measure the score drop. Avoids both MDI flaws [5][7]. Caveat: with strongly correlated features, shuffling one may look harmless because its twin carries the same signal [7].

Ideas for the report:

- Compare the RF ranking to the paper's correlation ranking from `Preprocessor.correlations`. Do they agree? Correlation only captures linear, one-feature-at-a-time links; RF can catch interactions.
- **Run with `--n-features 41` as well.** RF handles many features well and does its own selection. If RF on 41 features beats RF on 8, the paper's correlation filter is throwing away useful signal. Several NSL-KDD studies report RF doing best with a small selected set (7–11 features) [14][15], so this is worth testing.
- Bar chart: top-10 permutation importances (holdout and official side by side).

---

## 6. Evaluation plan

Run the same 4 configurations for all models so the tables line up:

| | 8 features | 41 features |
|---|---|---|
| holdout | ✔ | ✔ |
| official | ✔ | ✔ |

Extra RF-specific outputs:

- OOB score vs. holdout accuracy (they should be close; if not, something is off).
- Train time and predict time vs. KNN (KNN prediction was ~2 s; RF should be well under that).
- **Recall on seen vs. unseen DoS types** on the official split. Group test rows by the `attack` column: `neptune`/`smurf`/`back`/... vs. `apache2`/`processtable`/`mailbomb`/`udpstorm`. This shows exactly where every model fails. It would strengthen the whole project, not only the RF section.
- Confusion matrix (already printed by `run.py`).

### What to expect

- Holdout: RF ≥ 99% **only with 41 features** (tested: 99.99%). With the paper's 8 features it was 96.0%, barely above KNN (95.7%). Published NSL-KDD RF results are often 99%+ [14][15][16], but most use random splits of one file, multiclass labels, or different features, so **don't compare those numbers directly to ours**.
- Official: expect a clear drop. In one KNIME study, RF fell to ~72% accuracy on KDDTest+ (full multiclass, so not directly comparable) [8]. Calibration research shows novel attacks often get confidently labelled "normal" [9].

---

## 7. Pitfalls to avoid

- Tuning or choosing features using KDDTest+ (leakage).
- Wrapping RF in `StandardScaler` (harmless but pointless; makes the code look like we didn't understand the model).
- Treating MDI importances as "truth" without the permutation cross-check.
- Comparing our numbers to papers with different splits and labels.
- Forgetting `random_state` → numbers change every run and the tables won't match the report.

---

## 8. Proposed order of work

1. [x] Add `random_forest()` to `src/models.py` with the §3 defaults; register `"random_forest"` in `MODELS`.
2. [x] Run all four configs (§6) and save to `results/`.
3. [x] Add OOB score, and seen/unseen DoS recall breakdown to the evaluation.
4. [x] Permutation importance + MDI chart: `notebooks/rf_feature_importance.py` → `results/rf_feature_importance.png` (+ `rf_feature_importance.csv`, `rf_grouped_importance.csv`). Robustness checks: `notebooks/rf_importance_checks.py`. Findings in §10.
5. [x] Tuning on KDDTrain+ only, standard CV vs. leave-attack-type-out CV, 5 seeds: `notebooks/rf_tuning.py` → `results/rf_tuning.csv`. Findings in §11.
6. [ ] Write up: RF vs. KNN/NB/SVM table, importance chart, unseen-attack discussion.

### Open questions

- Should tuning target F1 or recall? (Recall fits the forensics framing better.)
- Do we also try Extra Trees (`ExtraTreesClassifier`)? It is a one-line change and often faster [5]. It may fall outside project scope.
- Do we report probability thresholds (lower threshold → higher recall)? This could help on the unseen attacks.


---

## 9. Logic test results (run 2026-11-11)

Script: `notebooks/rf_logic_test.py` (run `python notebooks/rf_logic_test.py` from the repo root) (default RF: 300 trees, `n_jobs=-1`, seed 42; same preprocessing as `src/`).

### Default RF vs. baselines

| Split | Feat | Model | Acc | Prec | Recall | F1 | Fit | Predict |
|---|---|---|---|---|---|---|---|---|
| holdout | 8 | RF | 0.960 | 0.983 | 0.918 | 0.950 | 2.6 s | 0.17 s |
| holdout | 8 | KNN | 0.957 | 0.974 | 0.919 | 0.945 | 0.06 s | 1.6 s |
| holdout | 41 | RF | 0.9999 | 0.9999 | 0.9998 | 0.9998 | 3.1 s | 0.16 s |
| official | 8 | RF | 0.846 | 0.979 | 0.660 | 0.789 | 3.5 s | 0.17 s |
| official | 8 | KNN | 0.860 | 0.964 | 0.704 | 0.814 | 0.09 s | 1.7 s |
| official | 41 | RF | 0.903 | 0.986 | 0.787 | 0.876 | 4.6 s | 0.16 s |
| official | 41 | KNN | 0.895 | 0.990 | 0.767 | 0.864 | 0.13 s | 1.4 s |

### Official split: recall on seen vs. unseen DoS types

| Model | 8 feat seen | 8 feat unseen | 41 feat seen | 41 feat unseen |
|---|---|---|---|---|
| RF | 0.814 | **0.146** | 0.996 | **0.089** |
| KNN | 0.819 | 0.322 | 0.975 | 0.069 |
| Naive Bayes | 0.811 | 0.412 | 0.934 | 0.443 |

### Claim-by-claim

| Claim | Result |
|---|---|
| RF near-perfect on holdout | ❌ with 8 features (96.0%); ✅ with 41 (99.99%) |
| Unseen attacks explain most of the recall drop | ⚠️ Only with 41 features. With 8, RF misses `smurf` (97% missed), `back` (87%), `pod` (98%), all seen in training |
| RF on 41 beats RF on 8 → correlation filter discards signal | ✅ Strongly. `src_bytes` has \|corr\| = 0.018, so the filter drops it, yet it is the **#1 feature by permutation importance** on the official test |
| No scaling needed | ✅ 99.994% identical predictions with/without `StandardScaler` |
| Tree count plateaus | ✅ OOB flat after ~50–100 trees |
| `max_features="sqrt"` → 2–3 with 8 features | ❌ It is 2 (rounded down). Fixed above |
| OOB ≈ holdout | ✅ 0.9645 vs 0.9604 (8f); 0.9998 vs 0.9999 (41f). ⚠️ OOB says **nothing** about the official split (0.964 OOB vs 0.846 official) |
| MDI favours high-cardinality features | ✅ Spearman(MDI, # unique values) = 0.62. `diff_srv_rate` is MDI #4 but ~0 by permutation. `src_bytes` is high on both, so it really matters |
| Permutation importance struggles with correlated features | ✅ On holdout, almost every feature scores ~0 because the `*serror_rate` columns duplicate each other |
| Class weighting optional | ✅ ±1% recall/precision trade, F1 unchanged |
| Tuning gives small gains | ✅ 8f: none (F1 0.789 → 0.788). 41f: F1 0.876 → 0.899 on official, but CV couldn't distinguish settings (see §4), so treat this as unconfirmed |
| RF predicts faster than KNN | ✅ ~10× (0.17 s vs 1.6 s); training ~30× slower but still a few seconds |

### What this changes

1. **The 8-feature filter is the biggest problem.** It hurts RF more than any setting does. Lead the RF section with 8 vs. 41 features, not with tuning.
2. **RF is the worst model on unseen attacks** (9–15% recall). It fits the training attack types very tightly. NB generalises better here, despite lower overall scores. This makes a strong discussion point.
3. Tuning is low priority: no gain on 8 features, and unverifiable gains on 41.
4. New open question: should RF use the paper's 8 features (fair comparison) or 41 (its best case)? Suggest reporting both.


---

## 10. Feature importance: verification (run 2026-11-11)

Two passes over the step 4 results, so the report's claims rest on more than one run.

### Pass 1: did the chart hide anything?

| Check | Result |
|---|---|
| Chart's "paper's 8" matches `Preprocessor(n_features=8)` | ✅ |
| All 8 paper features shown | ❌ `dst_host_srv_count` was missing → **fixed** (chart now always shows all 8) |
| Negative permutation values | ❌ hidden by an axis starting at 0 → **fixed** (`srv_count` −0.0006 now visible; `duration` −0.0020 is outside the shown set) |
| Hidden features that matter | ✅ none: largest hidden permutation value is 0.0032 |
| Constant column | `num_outbound_cmds` is always 0; it crashed the new grouping code until fixed |

### Pass 2: do the conclusions hold?

**Seeds.** `src_bytes` is #1 and `flag` #2 under all 4 seeds tried. `src_bytes`'s score ranges 0.057–0.071. Overall rank agreement between seeds is 0.85–0.90 (Spearman), so ranks below #2 shuffle a little. **Report ranks 1–2 with confidence; treat the rest as approximate.**

**Metric.** Recall and accuracy give the same top feature (`src_bytes`). Precision barely changes for any feature (max 0.0014): the model makes few false alarms, and no single feature controls them.

**Correlated features shuffled together** (`results/rf_grouped_importance.csv`, groups with |r| > 0.9):

| Group | F1 drop | Paper's 8? |
|---|---|---|
| `src_bytes` | 0.072 | no |
| 4 × `*rerror_rate` | 0.023 | no |
| `flag` | 0.013 | yes |
| `dst_bytes` | 0.010 | no |
| 4 × `*serror_rate` | 0.009 | yes |

The `*rerror_rate` features look minor one at a time but rank **#2 as a group**, and the paper's filter drops them. The paper's `*serror_rate` group stays minor even shuffled together, so the earlier finding was not a duplicate-feature artefact.

**Retraining with different feature sets** (official split, seed 42):

| Features | F1 | Recall seen | Recall unseen |
|---|---|---|---|
| 41 | 0.876 | 0.996 | 0.089 |
| 41 minus `src_bytes` | 0.864 | 0.994 | 0.008 |
| 41 minus `src_bytes`, `dst_bytes` | 0.862 | 0.993 | 0.000 |
| paper 8 | 0.789 | 0.814 | 0.146 |
| paper 8 + `src_bytes` | 0.881 | 0.965 | 0.223 |
| **paper 8 + `src_bytes`, `dst_bytes`** | **0.891** | 0.974 | **0.258** |

- With all 41 features, removing `src_bytes` costs little F1 because other features fill in, **but unseen-attack recall collapses to ~0**. The byte counts are what lets the forest catch new attack types at all.
- **Adding just `src_bytes` to the paper's 8 beats all 41 features** (F1 0.881 vs 0.876). Adding `dst_bytes` as well gives the best F1 so far (0.891) and 3× the unseen recall of 41 features. Fewer, better-chosen features generalise better here.
- ⚠️ **Leakage caveat:** these sets were tried *after* seeing test-set importance. `src_bytes` and `dst_bytes` are also #2 and #5 by built-in importance, which uses training data only, so a training-only rule could have picked them. Still, present this as exploratory, not as a tuned result.

**Unseen attacks.** Detection of the 4 unseen types relies most on `src_bytes`, `dst_host_same_srv_rate`, `dst_host_diff_srv_rate`, `logged_in` and `dst_host_srv_rerror_rate`. Base recall there is only 0.089, so these are small effects.

### What this means for step 5

Choosing features moved F1 by up to 10 points; the tuning test in §9 moved it by ~2. Step 5 should treat **feature set as a tuning choice** chosen on training data only (e.g. top-k by built-in importance, with k chosen by cross-validation). It should also check across seeds.


---

## 11. Tuning results (step 5, run 2026-11-11)

Script: `notebooks/rf_tuning.py` (~5 min). Tuning uses KDDTrain+ only. `RandomizedSearchCV` tries 30 candidates, scored by F1. The search also picks the number of features to keep (top-k by built-in importance, ranked inside each fold so nothing leaks). Each final model is then trained with 5 seeds and scored on KDDTest+.

### Two ways to validate

- **Standard CV** (stratified 3-fold): every attack type appears in both training and validation.
- **Attack-out CV** (new): each fold hides one group of DoS types from training and validates on it (`neptune` | `smurf` | `back`+`land` | `teardrop`+`pod`), plus 1/4 of normal rows. It imitates KDDTest+'s unseen attacks using training data only.

| | Best CV F1 | Spread across 30 candidates | Picked |
|---|---|---|---|
| Standard | 0.9998 | 0.9956–0.9998 | 41 features, `max_features=0.3`, `max_samples=0.8`, entropy, full depth |
| Attack-out | 0.164 | 0.001–0.164 | **5 features**, `max_depth=10`, `min_samples_leaf=2`, `max_features=0.5`, `balanced_subsample` |

- Standard CV can't separate the candidates (all within 0.004), as §9 predicted.
- **Attack-out F1 is at most 0.16.** Even inside the training data, a Random Forest trained without one DoS type mostly fails to flag it. This is the same weakness the official test set shows, reproduced without using any test data. It is a strong point for the report.

### Official test set (mean ± std over 5 seeds)

| Config | F1 | Recall seen | Recall unseen |
|---|---|---|---|
| Default, paper's 8 | 0.786 ± 0.002 | 0.813 | 0.129 ± 0.012 |
| Default, 41 | 0.873 ± 0.004 | 0.995 | 0.073 ± 0.026 |
| Tuned, standard CV | 0.889 ± 0.004 | 0.996 | 0.188 ± 0.029 |
| Tuned, attack-out CV | 0.889 ± 0.013 | 0.961 | **0.297 ± 0.108** |

- **Tuning helps, and the gain is real:** both tuned models beat default-41 by ~1.6 F1, about 4× the seed-to-seed spread. Standard CV found its setting by luck, though, since its candidates were indistinguishable. Most of its gain likely comes from `max_features=0.3` (12 features per split instead of 6).
- **Attack-out tuning gives the best unseen-attack recall (0.30, 4× default-41)** and ties on F1, using only 5 features. The trade-off: seen recall drops (0.996 → 0.961) and results vary more by seed (unseen 0.17–0.45).
- The 5 features it keeps vary by seed. `src_bytes` and `same_srv_rate` are always picked, and `dst_bytes` and `flag` in 4 of 5 runs. The variation explains the higher variance.
- This matches §10's exploratory "paper 8 + `src_bytes`, `dst_bytes`" result (F1 0.891, unseen 0.258), **but reaches it with training data only**. Use this version in the report, not the §10 one.

### Recommendation

- Report **default (paper 8)**, **default (41)** and **tuned (attack-out CV)** in the Random Forest section. Add the attack-out CV score as evidence of the generalisation gap.
- Open question: should `src/run.py` gain a `random_forest_tuned` model with the attack-out settings? It needs the feature selection step inside the model, which `run.py` doesn't do today.

---

## References

1. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001. doi:10.1023/A:1010933404324. https://mlanthology.org/mlj/2001/breiman2001mlj-random · PDF: https://www.cise.ufl.edu/~anand/fa11/Breiman_Random_Forests.pdf
2. A. Tundis, F. Cauteruccio, "On Machine Learning for Digital Forensics Investigation in Network Traffic," DCOSS-IoT 2025. (local: `reference/`)
3. M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set," IEEE CISDA 2009. doi:10.1109/CISDA.2009.5356528. https://www.ee.torontomu.ca/~bagheri/papers/cisda.pdf
4. Canadian Institute for Cybersecurity, "NSL-KDD dataset." https://unb.ca/cic/datasets/nsl.html
5. scikit-learn User Guide, "Ensembles: Random forests and other randomized tree ensembles" (parameters, parallelization, feature importance). https://scikit-learn.org/stable/modules/ensemble.html
6. scikit-learn API, `RandomForestClassifier`. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
7. scikit-learn example, "Permutation Importance vs Random Forest Feature Importance (MDI)." https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
8. "Machine Learning in Network Security Using KNIME Analytics," arXiv:2001.11489. https://arxiv.org/pdf/2001.11489
9. "When Calibrated Detectors Meet New Attacks," *IJACSA* vol. 17 no. 8 (2026). https://thesai.org/Downloads/Volume17No8/Paper_12-When_Calibrated_Detectors_Meet_New_Attacks.pdf
10. P. Probst, M. N. Wright, A.-L. Boulesteix, "Hyperparameters and Tuning Strategies for Random Forest," *WIREs Data Mining and Knowledge Discovery*, 9(3), 2019. doi:10.1002/widm.1301. https://arxiv.org/abs/1804.03515
11. P. Probst, A.-L. Boulesteix, "To Tune or Not to Tune the Number of Trees in Random Forest," *JMLR* 18, 2018. https://www.jmlr.org/papers/v18/17-269.html
12. Machine Learning Mastery, "Decision Trees and Ordinal Encoding: A Practical Guide." https://machinelearningmastery.com/decision-trees-and-ordinal-encoding-a-practical-guide/
13. scikit-learn User Guide, "Tuning the hyper-parameters of an estimator" (randomized search, successive halving). https://scikit-learn.org/stable/modules/grid_search.html
14. "Reinforcing cyber security: Defensive machine learning based intrusion detection system for NSL-KDD" (RF, 7 features, 99.7%). https://ejmtc.journals.ekb.eg/article_414562.html
15. DergiPark article: RF with CFS-reduced 11 features, 99.97%. https://dergipark.org.tr/en/download/article-file/1613739
16. Avci & Koca, *Acta Polytechnica Hungarica* (RF on NSL-KDD). https://uni-obuda.hu/journal/Avci_Koca_136.pdf
17. Inria scikit-learn MOOC, "Exercise M1.05" (scaling and `OrdinalEncoder` with tree models). https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline_ex_02.html
