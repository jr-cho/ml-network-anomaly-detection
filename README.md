# DoS Detection on NSL-KDD and KDDCup'99: Reproducing and Extending Tundis & Cauteruccio (2025)

Undergraduate research project for the Digital Forensics course at Florida Polytechnic University.
Team: Joshua Gottus, Ethan Barron.

We reproduce the DoS detection method summarised in Section IV-A of the reference paper (originally from Kachavimath et al. [4]): KNN and Naive Bayes classifying network traffic as normal or DoS, after keeping 8 features chosen by their correlation with the label. [4] used two datasets, KDDCup and NSL-KDD, so we run both. See [How this differs from the paper](#how-this-differs-from-the-paper) for what we match and what we cannot. We then compare a Decision Tree, a Random Forest, a Random Forest tuned for DoS attack types not seen in training (a gain that does not hold up in leave-one-attack-out; see [Generalization](#generalization-leave-one-attack-out)), and an SVM (RBF kernel, C=50). A leave-one-attack-out test (each DoS type held out in turn) checks whether the main result on unseen attack types holds beyond the one official test file.

**Reference:** Tundis, A., Cauteruccio, F. "On Machine Learning for Digital Forensics Investigation in Network Traffic." *DCOSS-IoT 2025*. Full text: `reference/On_Machine_Learning_for_Digital_Forensics_Investigation_in_Network_Traffic.pdf`.

**Source study it summarises ([4] in the paper):** Kachavimath, A. V., Nazare, S. V., Akki, S. S. "Distributed Denial of Service Attack Detection using Naïve Bayes and K-Nearest Neighbor for Network Forensics." *ICIMIA 2020*, pp. 711–717.

## Layout

```
notebooks/            one notebook per model: the write-up
  01_knn.ipynb
  02_naive_bayes.ipynb
  03_decision_tree.ipynb
  04_random_forest.ipynb
  05_svm.ipynb
src/dosdetect/        shared code the notebooks use
  data.py             download, load, encode, feature selection, prepare() / prepare_all()
  models.py           MODELS registry: one function per model
  evaluate.py         evaluate(), tables, confusion-matrix plots, paper comparison
  run.py              CLI: every model on every setup -> results/metrics.csv
  attack_out.py       CLI: leave-one-attack-out experiment -> results/attack_out.csv
  timing.py           CLI: repeated-run timing benchmark -> results/timing.csv
results/              metrics.csv (all models), rf_tuning.csv (Random Forest search),
                      attack_out.csv (leave-one-attack-out), timing.csv (median train/predict times)
tests/                unit tests on synthetic data
reference/            source paper (PDF)
docs/                 project proposal
```

Every notebook follows the same structure: load the six setups with `prepare_all()`, score the model with `evaluate_all()`, show tables and confusion matrices, then a model-specific section and takeaways.

## Quickstart

Tested on Python 3.12 (pinned versions in `requirements.txt`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # pinned deps + the package (editable) + jupyter, pytest, ruff
python -m dosdetect.run             # all models, all setups -> results/metrics.csv
python -m dosdetect.attack_out      # leave-one-attack-out -> results/attack_out.csv
python -m dosdetect.timing          # repeated-run timings -> results/timing.csv
jupyter lab notebooks/
```

The NSL-KDD files are downloaded into `data/` on first use. KDDCup'99 (10% file) is downloaded automatically through scikit-learn (`fetch_kddcup99`) into `data/` on first use.

## Setups

- **Official** (NSL-KDD): train on `KDDTrain+`, test on `KDDTest+`. About 23% of the test DoS rows (1717 of 7458) are attack types that never appear in training ("unseen").
- **Holdout** (NSL-KDD): stratified 80/20 split of `KDDTrain+`. Every test attack type is also in training.
- **KDDCup** (KDDCup'99 10% file): stratified 80/20 split. UCI's labelled `corrected` test file can no longer be downloaded, so there is no official KDDCup train/test split and we make our own.
- **8 features** (top 8 by |Spearman correlation| with the label, our reading of the paper's filter) or **all 41**.

Only normal and DoS rows are kept (label: 0 = normal, 1 = DoS). The preprocessor is fit on training rows only, so the top 8 is chosen per training set and differs between datasets.

## How this differs from the paper

**Matched**

- Spearman feature ranking. On KDDCup it reproduces the features the paper names: `dst_bytes` ranks 1st, `protocol_type` 6th, and `count` and `dst_host_count` 3rd and 4th. The KDDCup top 8 is `dst_bytes`, `logged_in`, `count`, `dst_host_count`, `dst_host_srv_diff_host_rate`, `protocol_type`, `srv_count`, `srv_diff_host_rate`. [4] is paywalled, so its exact 8 features cannot be confirmed; the Spearman match on KDDCup is our evidence.
- Brute-force Euclidean KNN: each test row is compared with every training row, as the paper describes.
- Both datasets (NSL-KDD and KDDCup'99), and the comparison against the paper uses the mean over the two, because the paper reports its numbers "across the two datasets".

**Not matched**

- K is not given in the paper; we use 5 (scikit-learn's default).
- [4]'s exact train/test split is unknown.
- There is no official KDDCup test file (see Setups), so our KDDCup result uses a stratified 80/20 split.
- We add feature scaling (standardisation) for KNN, which the paper does not mention.
- On NSL-KDD the Spearman top 8 still lacks `dst_bytes` (rank 9th) and `protocol_type` (rank 19th). The connection-count features `count` and `dst_host_count` rank 8th and 15th.
- The paper's Naive Bayes accuracy of 93.95% differs from the 91.31% quoted by one secondary source; we compare against 93.95%.

Correlation for the text columns (`protocol_type`, `service`, `flag`) is computed on alphabetical integer codes, so their ranking depends on that arbitrary order.

## Key results

Generated from `results/metrics.csv` (see `python -m dosdetect.run`). Train and predict times are not in that file; see [Timing](#timing).

**Comparison with the paper** (8 features; NSL-KDD = official split, KDDCup = our 80/20 split; `difference` is our mean minus the paper). The paper's numbers are one figure "across the two datasets", so the mean is the like-for-like column.

KNN

| Metric | NSL-KDD | KDDCup | mean | paper [4] | difference (points) |
|---|---|---|---|---|---|
| accuracy | 86.66% | 99.88% | 93.27% | 98.51% | -5.24 |
| precision | 93.09% | 99.93% | 96.51% | 98.90% | -2.39 |
| recall | 74.83% | 99.92% | 87.38% | 97.80% | -10.42 |

Naive Bayes

| Metric | NSL-KDD | KDDCup | mean | paper [4] | difference (points) |
|---|---|---|---|---|---|
| accuracy | 87.12% | 98.72% | 92.92% | 93.95% | -1.03 |
| precision | 92.93% | 99.43% | 96.18% | 97.74% | -1.56 |
| recall | 76.15% | 98.96% | 87.56% | 95.54% | -7.98 |

**Official split (NSL-KDD)**

| Model | Features | Accuracy | Precision | Recall | F1 | Recall, unseen DoS types |
|---|---|---|---|---|---|---|
| KNN | 8 | 86.66% | 93.09% | 74.83% | 82.97% | 31.92% |
| Naive Bayes | 8 | 87.12% | 92.93% | 76.15% | 83.71% | 46.01% |
| Decision Tree | 8 | 84.97% | 92.79% | 70.92% | 80.39% | 18.75% |
| Random Forest | 8 | 85.43% | 92.99% | 71.87% | 81.08% | 19.86% |
| Random Forest (tuned) | 8 | 86.39% | 93.23% | 74.05% | 82.54% | 29.94% |
| SVM | 8 | 86.76% | 93.00% | 75.17% | 83.14% | 36.05% |
| KNN | 41 | 89.53% | 99.00% | 76.67% | 86.41% | 6.93% |
| Naive Bayes | 41 | 61.78% | 53.95% | 82.06% | 65.10% | 44.26% |
| Decision Tree | 41 | 91.33% | 96.51% | 83.05% | 89.28% | 30.81% |
| Random Forest | 41 | 90.12% | 98.45% | 78.48% | 87.34% | 8.21% |
| Random Forest (tuned) | 41 | 90.59% | 98.93% | 79.20% | 87.97% | 26.97% |
| SVM | 41 | 93.03% | 99.21% | 84.62% | 91.34% | 36.69% |

**Holdout split (NSL-KDD)**

| Model | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| KNN | 8 | 97.33% | 99.08% | 94.28% | 96.62% |
| Naive Bayes | 8 | 95.52% | 95.77% | 93.06% | 94.40% |
| Decision Tree | 8 | 97.28% | 99.20% | 94.04% | 96.55% |
| Random Forest | 8 | 97.34% | 99.13% | 94.26% | 96.63% |
| Random Forest (tuned) | 8 | 97.17% | 99.41% | 93.57% | 96.40% |
| SVM | 8 | 97.19% | 99.44% | 93.60% | 96.43% |
| KNN | 41 | 99.96% | 99.96% | 99.93% | 99.95% |
| Naive Bayes | 41 | 65.78% | 54.34% | 97.63% | 69.82% |
| Decision Tree | 41 | 99.97% | 99.96% | 99.97% | 99.96% |
| Random Forest | 41 | 99.99% | 100.00% | 99.98% | 99.99% |
| Random Forest (tuned) | 41 | 99.69% | 99.28% | 99.97% | 99.62% |
| SVM | 41 | 99.95% | 99.95% | 99.92% | 99.93% |

**KDDCup split**

| Model | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| KNN | 8 | 99.88% | 99.93% | 99.92% | 99.92% |
| Naive Bayes | 8 | 98.72% | 99.43% | 98.96% | 99.20% |
| Decision Tree | 8 | 99.94% | 99.98% | 99.94% | 99.96% |
| Random Forest | 8 | 99.94% | 99.99% | 99.94% | 99.96% |
| Random Forest (tuned) | 8 | 99.83% | 99.98% | 99.81% | 99.90% |
| SVM | 8 | 99.36% | 99.97% | 99.23% | 99.60% |
| KNN | 41 | 99.99% | 99.99% | 100.00% | 100.00% |
| Naive Bayes | 41 | 93.31% | 92.81% | 99.34% | 95.97% |
| Decision Tree | 41 | 99.99% | 99.99% | 100.00% | 99.99% |
| Random Forest | 41 | 100.00% | 100.00% | 100.00% | 100.00% |
| Random Forest (tuned) | 41 | 99.95% | 100.00% | 99.94% | 99.97% |
| SVM | 41 | 99.99% | 100.00% | 99.98% | 99.99% |

Findings:

- **The paper's numbers are approached on KDDCup but not on NSL-KDD's official file.** With 8 features KNN reaches 99.88% accuracy on KDDCup and 86.66% on the official file; Naive Bayes reaches 98.72% and 87.12%. The two-dataset mean is below the paper for both models, and the largest gap is recall (see the comparison tables).
- **Unseen DoS attack types are the main weakness.** With 8 features, recall on unseen types ranges from 18.75% to 46.01% across the six models, and Naive Bayes is the highest at 46.01% (with 41 features, 44.26%), followed by the SVM at 36.05% (with 41 features, 36.69%), but its recall is bought with false alarms (see the leave-one-attack-out findings). Leave-one-attack-out repeats the test for every DoS type and confirms it for KNN, the Decision Tree, the forests and the SVM. All 41 features raise KNN's recall on seen types (87.67% to 97.53%) but lower it on unseen ones (31.92% to 6.93%).
- **Naive Bayes depends on correlation-based feature selection.** With all 41 features its official accuracy falls from 87.12% to 61.78%, and the same direction shows on holdout (95.52% to 65.78%) and KDDCup (98.72% to 93.31%), because many near-duplicate features break its independence assumption.
- **Tree models.** On the official file with 41 features the Decision Tree (91.33% accuracy) is ahead of both the default Random Forest (90.12%) and the tuned one (90.59%). On KDDCup and holdout with 41 features all tree models are at or above 99.69% accuracy. `src_bytes`, which the top-8 correlation filter drops, is one of the most useful features for the forest (notebook 04).
- **SVM.** With 8 features the SVM (RBF, C=50) is level with KNN and Naive Bayes on the official file (86.76% accuracy, 83.14% F1), and its recall on unseen types, 36.05%, is second only to Naive Bayes. With all 41 features it has the best official-split accuracy (93.03%), precision (99.21%), recall (84.62%) and F1 (91.34%) of the six models, and unseen recall of 36.69%. On holdout and KDDCup with 41 features it is at 99.95% and 99.99% accuracy; with 8 features on KDDCup it is lower than the tree models (99.36% against 99.94%). In leave-one-attack-out it follows the other non-Naive-Bayes models (see [Generalization](#generalization-leave-one-attack-out)), and with 8 features it is the slowest model to train (see [Timing](#timing)).
- **Random Forest tuning.** On the official split the tuned forest `random_forest_tuned()` has the best forest recall on unseen types in `metrics.csv`: 29.94% with 8 features and 26.97% with 41, against 19.86% and 8.21% for the default forest. That gain is not general: in leave-one-attack-out (one seed) it beats the default forest on held-out recall only on KDDCup with 41 features, and is lower on the other three dataset and feature-count combinations (see [Generalization](#generalization-leave-one-attack-out)). Its settings are fixed in `models.py` and came from an earlier tuning run. The current attack-out search (notebook 04, `results/rf_tuning.csv`) picked different settings, because the Spearman ranking reorders the 41 columns and the forest's random draws depend on column order. Its best CV F1 is only 0.0797, against 0.9998 for standard CV, and over 5 seeds its forest reaches unseen-type recall 0.1406 versus 0.0684 for the default 41-feature forest. That is a modest gain, and it does not beat the fixed `random_forest_tuned()` on unseen recall.

Each notebook ends with a longer list of takeaways for its model.

## Generalization: leave-one-attack-out

The official split has only one set of unseen attack types, so we repeat the test with every DoS type held out in turn (`python -m dosdetect.attack_out`, results in `results/attack_out.csv`). For each dataset (NSL-KDD `KDDTrain+` only, and the KDDCup'99 10% file) and each held-out group (`neptune`; `smurf`; `back` + `land`; `teardrop` + `pod`), all rows outside the group are split 80/20 stratified by attack type. A model trains on the 80% (normal traffic and the other DoS types, with the feature selection also fit there) and is tested on the other 20% plus every row of the held-out group (capped at 20,000 rows). This is **one seed (42)** only. The tables show the mean over the four groups: held-out recall (share of the unseen-type rows flagged as DoS), recall on the seen DoS types in the test set, and the false-alarm rate (share of normal test rows flagged as DoS).

**NSL-KDD**

| Model | Features | Held-out recall | Seen recall | False-alarm rate |
|---|---|---|---|---|
| KNN | 8 | 1.79% | 97.99% | 0.32% |
| Naive Bayes | 8 | 36.46% | 92.44% | 30.87% |
| Decision Tree | 8 | 2.91% | 98.41% | 0.13% |
| Random Forest | 8 | 2.42% | 98.45% | 0.12% |
| Random Forest (tuned) | 8 | 0.47% | 98.23% | 0.21% |
| SVM | 8 | 0.41% | 97.17% | 0.37% |
| KNN | 41 | 0.64% | 99.62% | 0.05% |
| Naive Bayes | 41 | 48.67% | 98.68% | 45.61% |
| Decision Tree | 41 | 4.61% | 99.94% | 0.06% |
| Random Forest | 41 | 2.18% | 99.94% | 0.02% |
| Random Forest (tuned) | 41 | 0.93% | 99.80% | 0.22% |
| SVM | 41 | 0.76% | 99.89% | 0.06% |

**KDDCup**

| Model | Features | Held-out recall | Seen recall | False-alarm rate |
|---|---|---|---|---|
| KNN | 8 | 0.73% | 99.96% | 0.29% |
| Naive Bayes | 8 | 43.84% | 98.80% | 3.50% |
| Decision Tree | 8 | 25.71% | 99.97% | 0.07% |
| Random Forest | 8 | 23.78% | 99.97% | 0.05% |
| Random Forest (tuned) | 8 | 17.69% | 99.74% | 0.05% |
| SVM | 8 | 19.35% | 99.31% | 0.03% |
| KNN | 41 | 0.67% | 99.99% | 0.01% |
| Naive Bayes | 41 | 18.58% | 98.86% | 9.86% |
| Decision Tree | 41 | 4.63% | 100.00% | 0.01% |
| Random Forest | 41 | 0.97% | 100.00% | 0.00% |
| Random Forest (tuned) | 41 | 28.84% | 99.33% | 0.04% |
| SVM | 41 | 0.00% | 99.99% | 0.01% |

Findings:

- **Leave-one-attack-out confirms the main finding for KNN, the Decision Tree, the forests and the SVM: they almost never catch a DoS type they did not see.** Across these five models, 75 of 80 (dataset, features, model, group) results have held-out recall under 50%, while recall on seen types stays at or above 95.86% and false alarms at or below 0.64%. The five exceptions are all on KDDCup: the Decision Tree, default forest and SVM on `neptune` at 8 features (the SVM at 69.35%), the tuned forest on `neptune` at 41 features, and the tuned forest on `teardrop` + `pod` at 8 features. The SVM's mean held-out recall is 0.41% and 0.76% on NSL-KDD (8 and 41 features) and 19.35% and 0.00% on KDDCup. KNN is the weakest: its mean held-out recall is at most 1.79% in every setup.
- **Naive Bayes has the highest held-out recall, but only because it also flags normal traffic.** Its mean held-out recall is 48.67% on NSL-KDD with 41 features, at a false-alarm rate of 45.61%; with 8 features it is 36.46% at 30.87%. In the worst case, `smurf` held out of NSL-KDD with 8 features, it flags 100.00% of the held-out rows and 98.92% of the normal rows. It catches `smurf` on KDDCup (99.99% held-out recall, 6.98% false alarms) at 8 features, so it is a large false-alarm trade, not a reliable detector of new attacks.
- **The tuned forest's advantage on unseen types does not generalise.** `random_forest_tuned()` beats the default forest on mean held-out recall only on KDDCup with 41 features (28.84% against 0.97%). It is lower on KDDCup with 8 features (17.69% against 23.78%) and on NSL-KDD (0.47% against 2.42% at 8 features, 0.93% against 2.18% at 41).
- **The result depends on the dataset, the feature count and the group.** Held-out recall moves from near zero to near 99% between groups for the same model, so a one-seed number for a single group should not be read as a general rate.

## Timing

Generated from `results/timing.csv` (see `python -m dosdetect.timing`). Each model is fitted and used to predict `repeats` = 5 times with a fresh model each time; the tables show the **median of 5 runs on an 8-core machine with other background load**, in seconds. Machine: Linux-6.8.0-149-generic-x86_64-with-glibc2.39, Python 3.12.3, scikit-learn 1.9.1. The forests use all cores (`n_jobs=-1`), so absolute times depend on the core count; ratios are more portable. The CSV also holds the min and max of each timing. Columns are the setup (split and feature count).

Median predict time (s)

| Model | official 8 | official 41 | holdout 8 | holdout 41 | kddcup 8 | kddcup 41 |
|---|---|---|---|---|---|---|
| KNN | 0.7750 | 1.18 | 0.8149 | 1.23 | 14.51 | 21.87 |
| Naive Bayes | 0.0009 | 0.0023 | 0.0011 | 0.0025 | 0.0031 | 0.0135 |
| Decision Tree | 0.0012 | 0.0015 | 0.0014 | 0.0019 | 0.0025 | 0.0052 |
| Random Forest | 0.1333 | 0.1334 | 0.1338 | 0.1332 | 0.1614 | 0.1630 |
| Random Forest (tuned) | 0.1665 | 0.1665 | 0.1662 | 0.1682 | 0.1866 | 0.1899 |
| SVM | 2.58 | 0.2293 | 2.66 | 0.2592 | 7.55 | 1.96 |

Median train time (s)

| Model | official 8 | official 41 | holdout 8 | holdout 41 | kddcup 8 | kddcup 41 |
|---|---|---|---|---|---|---|
| KNN | 0.0181 | 0.0470 | 0.0058 | 0.0216 | 0.0223 | 0.1168 |
| Naive Bayes | 0.0108 | 0.0314 | 0.0089 | 0.0223 | 0.0370 | 0.1102 |
| Decision Tree | 0.0689 | 0.3801 | 0.0464 | 0.2952 | 0.1518 | 0.6064 |
| Random Forest | 2.88 | 3.85 | 2.05 | 2.58 | 7.49 | 10.32 |
| Random Forest (tuned) | 3.50 | 3.61 | 2.80 | 2.79 | 8.79 | 9.63 |
| SVM | 90.53 | 2.02 | 137.35 | 1.19 | 37.07 | 22.42 |

- **KNN is the slowest to predict in four of the six setups** (the SVM is slower in the two NSL-KDD setups with 8 features). Its median predict time is 498.9 to 4627.1 times Naive Bayes's, depending on the setup (the lower end is NSL-KDD holdout with 41 features, the upper end KDDCup with 8 features), because every test row is compared with every training row. Its train time is small, since it only stores the data. On KDDCup with 41 features its median predict time is 21.87 s.
- **The forests cost more to train and predict than a single tree or Naive Bayes.** The default forest's median train time is 8.8 to 49.3 times the Decision Tree's, and its predict time 12.1 to 141.1 times Naive Bayes's. For the tuned forest the ranges are 9.5 to 60.2 and 14.1 to 176.2.
- **The SVM is the most expensive to train with 8 features, and its cost depends on the feature count.** Its median train time is 90.53 s on the official split, 137.35 s on holdout and 37.07 s on KDDCup with 8 features, but 2.02 s, 1.19 s and 22.42 s with 41 features. Across the six setups it is 4.0 to 2957.7 times the Decision Tree's, and the slowest model to train in four of them (the forests are the slowest on the other two). `gamma="auto"` is 1/n_features, so the 8-feature kernel has a larger gamma (1/8 against 1/41); we have not tested whether this explains the gap. Its median predict time is 98.3 to 2731.5 times Naive Bayes's.
- **The Decision Tree predicts about as fast as Naive Bayes** (ratio 0.39 to 1.32) and **Naive Bayes trains much faster than the Decision Tree**: its median train time is 0.08 to 0.24 times the Decision Tree's.
- These timings come from one benchmark run on a machine that was not idle, so they are indicative; the largest min-to-max spread in any cell is 1.66 times, on KNN's millisecond-scale train time.

## Models

| Model | Registry name | Owner | Status |
|---|---|---|---|
| KNN | `knn` | Shared | Done (paper baseline) |
| Naive Bayes | `naive_bayes` | Shared | Done (paper baseline) |
| Decision Tree | `decision_tree` | Joshua | Done |
| Random Forest | `random_forest`, `random_forest_tuned` | Joshua | Done |
| SVM | `svm` | Ethan | Done (C=50, RBF, gamma auto, balanced; notebook `05_svm.ipynb`) |

To add a model: write a function in `src/dosdetect/models.py` that returns an untrained scikit-learn estimator, add it to `MODELS`, and copy one of the notebooks as a starting point. `python -m dosdetect.run` picks it up automatically.

## Development

```bash
pytest                                # synthetic data, no download needed
python -m dosdetect.attack_out        # leave-one-attack-out, full run -> results/attack_out.csv
python -m dosdetect.timing            # run alone on the machine -> results/timing.csv
ruff check . && ruff format --check . # lint (includes notebooks)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # re-run notebooks
```

CI (`.github/workflows/ci.yml`) runs ruff and pytest on Python 3.12 only, the version the results were produced with.
