# DoS Detection on NSL-KDD and KDDCup'99: Reproducing and Extending Tundis & Cauteruccio (2025)

Undergraduate research project for the Digital Forensics course at Florida Polytechnic University.
Team: Joshua Gottus, Ethan Barron.

We reproduce the DoS detection method summarised in Section IV-A of the reference paper (originally from Kachavimath et al. [4]): KNN and Naive Bayes classifying network traffic as normal or DoS, after keeping 8 features chosen by their correlation with the label. [4] used two datasets, KDDCup and NSL-KDD, so we run both. See [How this differs from the paper](#how-this-differs-from-the-paper) for what we match and what we cannot. We then compare a Decision Tree, a Random Forest, and a Random Forest tuned for DoS attack types not seen in training. An SVM is planned.

**Reference:** Tundis, A., Cauteruccio, F. "On Machine Learning for Digital Forensics Investigation in Network Traffic." *DCOSS-IoT 2025*. Full text: `reference/On_Machine_Learning_for_Digital_Forensics_Investigation_in_Network_Traffic.pdf`.

**Source study it summarises ([4] in the paper):** Kachavimath, A. V., Nazare, S. V., Akki, S. S. "Distributed Denial of Service Attack Detection using Naïve Bayes and K-Nearest Neighbor for Network Forensics." *ICIMIA 2020*, pp. 711–717.

## Layout

```
notebooks/            one notebook per model: the write-up
  01_knn.ipynb
  02_naive_bayes.ipynb
  03_decision_tree.ipynb
  04_random_forest.ipynb
src/dosdetect/        shared code the notebooks use
  data.py             download, load, encode, feature selection, prepare() / prepare_all()
  models.py           MODELS registry: one function per model
  evaluate.py         evaluate(), tables, confusion-matrix plots, paper comparison
  run.py              CLI: every model on every setup -> results/metrics.csv
results/              metrics.csv (all models), rf_tuning.csv (Random Forest search)
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

Generated from `results/metrics.csv` (see `python -m dosdetect.run`). Predict and train times are in that file but not quoted here, because they depend on the machine.

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
| KNN | 41 | 89.53% | 99.00% | 76.67% | 86.41% | 6.93% |
| Naive Bayes | 41 | 61.78% | 53.95% | 82.06% | 65.10% | 44.26% |
| Decision Tree | 41 | 91.33% | 96.51% | 83.05% | 89.28% | 30.81% |
| Random Forest | 41 | 90.12% | 98.45% | 78.48% | 87.34% | 8.21% |
| Random Forest (tuned) | 41 | 90.59% | 98.93% | 79.20% | 87.97% | 26.97% |

**Holdout split (NSL-KDD)**

| Model | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| KNN | 8 | 97.33% | 99.08% | 94.28% | 96.62% |
| Naive Bayes | 8 | 95.52% | 95.77% | 93.06% | 94.40% |
| Decision Tree | 8 | 97.28% | 99.20% | 94.04% | 96.55% |
| Random Forest | 8 | 97.34% | 99.13% | 94.26% | 96.63% |
| Random Forest (tuned) | 8 | 97.17% | 99.41% | 93.57% | 96.40% |
| KNN | 41 | 99.96% | 99.96% | 99.93% | 99.95% |
| Naive Bayes | 41 | 65.78% | 54.34% | 97.63% | 69.82% |
| Decision Tree | 41 | 99.97% | 99.96% | 99.97% | 99.96% |
| Random Forest | 41 | 99.99% | 100.00% | 99.98% | 99.99% |
| Random Forest (tuned) | 41 | 99.69% | 99.28% | 99.97% | 99.62% |

**KDDCup split**

| Model | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| KNN | 8 | 99.88% | 99.93% | 99.92% | 99.92% |
| Naive Bayes | 8 | 98.72% | 99.43% | 98.96% | 99.20% |
| Decision Tree | 8 | 99.94% | 99.98% | 99.94% | 99.96% |
| Random Forest | 8 | 99.94% | 99.99% | 99.94% | 99.96% |
| Random Forest (tuned) | 8 | 99.83% | 99.98% | 99.81% | 99.90% |
| KNN | 41 | 99.99% | 99.99% | 100.00% | 100.00% |
| Naive Bayes | 41 | 93.31% | 92.81% | 99.34% | 95.97% |
| Decision Tree | 41 | 99.99% | 99.99% | 100.00% | 99.99% |
| Random Forest | 41 | 100.00% | 100.00% | 100.00% | 100.00% |
| Random Forest (tuned) | 41 | 99.95% | 100.00% | 99.94% | 99.97% |

Findings:

- **The paper's numbers are approached on KDDCup but not on NSL-KDD's official file.** With 8 features KNN reaches 99.88% accuracy on KDDCup and 86.66% on the official file; Naive Bayes reaches 98.72% and 87.12%. The two-dataset mean is below the paper for both models, and the largest gap is recall (see the comparison tables).
- **Unseen DoS attack types are the main weakness.** With 8 features, recall on unseen types ranges from 18.75% to 46.01% across the five models, and Naive Bayes is the highest at 46.01% (with 41 features, 44.26%). All 41 features raise KNN's recall on seen types (87.67% to 97.53%) but lower it on unseen ones (31.92% to 6.93%).
- **Naive Bayes depends on correlation-based feature selection.** With all 41 features its official accuracy falls from 87.12% to 61.78%, and the same direction shows on holdout (95.52% to 65.78%) and KDDCup (98.72% to 93.31%), because many near-duplicate features break its independence assumption.
- **Tree models.** On the official file with 41 features the Decision Tree (91.33% accuracy) is ahead of both the default Random Forest (90.12%) and the tuned one (90.59%). On KDDCup and holdout with 41 features all tree models are at or above 99.69% accuracy. `src_bytes`, which the top-8 correlation filter drops, is one of the most useful features for the forest (notebook 04).
- **Random Forest tuning.** The tuned forest `random_forest_tuned()` has the best forest recall on unseen types in `metrics.csv`: 29.94% with 8 features and 26.97% with 41, against 19.86% and 8.21% for the default forest. Its settings are fixed in `models.py` and came from an earlier tuning run. The current attack-out search (notebook 04, `results/rf_tuning.csv`) picked different settings, because the Spearman ranking reorders the 41 columns and the forest's random draws depend on column order. Its best CV F1 is only 0.0797, against 0.9998 for standard CV, and over 5 seeds its forest reaches unseen-type recall 0.1406 versus 0.0684 for the default 41-feature forest. That is a modest gain, and it does not beat the fixed `random_forest_tuned()` on unseen recall.

Each notebook ends with a longer list of takeaways for its model.

## Models

| Model | Registry name | Owner | Status |
|---|---|---|---|
| KNN | `knn` | Shared | Done (paper baseline) |
| Naive Bayes | `naive_bayes` | Shared | Done (paper baseline) |
| Decision Tree | `decision_tree` | Joshua | Done |
| Random Forest | `random_forest`, `random_forest_tuned` | Joshua | Done |
| SVM | – | Ethan | Not started |

To add a model: write a function in `src/dosdetect/models.py` that returns an untrained scikit-learn estimator, add it to `MODELS`, and copy one of the notebooks as a starting point. `python -m dosdetect.run` picks it up automatically.

## Development

```bash
pytest                                # synthetic data, no download needed
ruff check . && ruff format --check . # lint (includes notebooks)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # re-run notebooks
```

CI (`.github/workflows/ci.yml`) runs ruff and pytest on Python 3.10 and 3.12.
