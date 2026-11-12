# DoS Detection on NSL-KDD: Reproducing and Extending Tundis & Cauteruccio (2025)

Undergraduate research project for the Digital Forensics course at Florida Polytechnic University.
Team: Joshua Gottus, Ethan Barron.

We reproduce the DoS detection method summarised in Section IV-A of the reference paper (originally from Kachavimath et al. [4]): KNN and Naive Bayes classifying NSL-KDD traffic as normal or DoS, after keeping 8 features chosen by their correlation with the label. Our reproduction differs from the paper in two ways; see [How this differs from the paper](#how-this-differs-from-the-paper). We then compare a Decision Tree, a Random Forest, and a Random Forest tuned for DoS attack types not seen in training. An SVM is planned.

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
  evaluate.py         evaluate(), tables, confusion-matrix plots, paper's reported numbers
  run.py              CLI: every model on every setup -> results/metrics.csv
results/              metrics.csv (all models), rf_tuning.csv (Random Forest search)
tests/                unit tests on synthetic data
reference/            source paper (PDF)
docs/                 project proposal
```

Every notebook follows the same structure: load the four setups with `prepare_all()`, score the model with `evaluate_all()`, show tables and confusion matrices, then a model-specific section and takeaways.

## Quickstart

Tested on Python 3.12 (pinned versions in `requirements.txt`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # pinned deps + the package (editable) + jupyter, pytest, ruff
python -m dosdetect.run             # all models, all setups -> results/metrics.csv
jupyter lab notebooks/
```

The NSL-KDD files are downloaded into `data/` on first use.

## Setups

- **Official**: train on `KDDTrain+`, test on `KDDTest+`. About 23% of the test DoS rows are attack types that never appear in training ("unseen").
- **Holdout**: stratified 80/20 split of `KDDTrain+`. Every test attack type is also in training.
- **8 features** (top 8 by |correlation| with the label, our reading of the paper's filter) or **all 41**.

Only normal and DoS rows are kept (label: 0 = normal, 1 = DoS). The preprocessor is fit on training rows only.

## How this differs from the paper

1. **Our 8 features are not the paper's 8.** The paper says [4] dropped features "which do not exhibit significant correlation with the label", leaving eight, and names three of them: the number of connections of one host, the bytes sent from destination to source, and the connection protocol. It does not list all eight. We keep the top 8 by absolute Pearson correlation on `KDDTrain+`, which gives `same_srv_rate`, `dst_host_srv_serror_rate`, `dst_host_serror_rate`, `srv_serror_rate`, `serror_rate`, `dst_host_same_srv_rate`, `flag`, and `dst_host_srv_count`. Two of the paper's named features are missing: `dst_bytes` ranks 27th and `protocol_type` 14th. (The connection-count feature, `count` or `dst_host_count`, ranks 9th or 11th.) Correlation for the text columns (`protocol_type`, `service`, `flag`) is computed on alphabetical integer codes, so their ranking depends on that arbitrary order.
2. **The paper's numbers are not NSL-KDD-only.** The paper reports [4]'s KNN and Naive Bayes results "across the two datasets", KDDCup and NSL-KDD, and does not say which train/test split was used. We test on NSL-KDD only. Rows marked "paper [4]" below are therefore a reference point, not a like-for-like comparison.

## Key results

From `results/metrics.csv`. The paper reports only accuracy, precision and recall, from [4] across KDDCup and NSL-KDD (see above).

**Official split, 8 features**

| Model | Accuracy | Precision | Recall | F1 | Recall, unseen DoS types |
|---|---|---|---|---|---|
| KNN (paper [4]) | 98.51% | 98.9% | 97.8% | – | – |
| Naive Bayes (paper [4]) | 93.95% | 97.74% | 95.54% | – | – |
| KNN | 86.00% | 96.37% | 70.43% | 81.39% | 32.15% |
| Naive Bayes | 86.74% | 96.74% | 71.91% | 82.50% | 41.23% |
| Decision Tree | 85.31% | 98.02% | 67.55% | 79.98% | 20.91% |
| Random Forest | 84.61% | 97.85% | 66.02% | 78.85% | 14.62% |
| Random Forest (tuned) | 86.36% | 99.06% | 69.25% | 81.52% | 27.37% |

**Official split, all 41 features**

| Model | Accuracy | Precision | Recall | F1 | Recall, unseen DoS types |
|---|---|---|---|---|---|
| KNN | 89.53% | 99.00% | 76.67% | 86.41% | 6.93% |
| Naive Bayes | 61.78% | 53.95% | 82.06% | 65.10% | 44.26% |
| Decision Tree | 90.54% | 97.84% | 79.99% | 88.02% | 17.36% |
| Random Forest | 90.27% | 98.59% | 78.73% | 87.55% | 8.91% |
| Random Forest (tuned) | 90.59% | 98.93% | 79.20% | 87.97% | 26.97% |

**Holdout split, 8 features**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN | 95.70% | 97.35% | 91.89% | 94.54% |
| Naive Bayes | 93.56% | 94.36% | 89.47% | 91.85% |
| Decision Tree | 96.02% | 98.50% | 91.58% | 94.92% |
| Random Forest | 96.04% | 98.34% | 91.79% | 94.95% |
| Random Forest (tuned) | 95.98% | 98.94% | 91.07% | 94.84% |

Findings:

- **The paper's reported numbers are approached only on the holdout split.** (They mix KDDCup and NSL-KDD, so this is a rough comparison.) There, Naive Bayes reaches 93.56% accuracy (paper: 93.95%) and KNN 95.70% (paper: 98.51%). On the official test file, accuracy with 8 features drops to 84.6–86.7% for every model, mainly because recall falls to 66–72%.
- **Unseen DoS attack types are the main weakness.** With 8 features, recall on unseen types ranges from 14.62% (Random Forest) to 41.23% (Naive Bayes). All 41 features raise recall on seen types (KNN: 81.88% to 97.53%) but lower it on unseen ones (KNN: 32.15% to 6.93%).
- **Naive Bayes depends on correlation-based feature selection.** With all 41 features its accuracy falls to 61.78%, because many near-duplicate features break its independence assumption.
- **Tree models.** A single Decision Tree is within 0.3 points of the default Random Forest. The tuned Random Forest has the best official accuracy (90.59%, 41 features) and triples unseen-type recall over the default forest with 41 features (8.91% to 26.97%). `src_bytes`, which the top-8 correlation filter drops, is one of the most useful features.

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
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # re-run notebooks (~3 min)
```

CI (`.github/workflows/ci.yml`) runs ruff and pytest on Python 3.10 and 3.12.
