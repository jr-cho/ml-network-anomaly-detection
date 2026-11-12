# DoS Detection on NSL-KDD: Reproducing and Extending Tundis & Cauteruccio (2025)

Undergraduate research project for the Digital Forensics course at Florida Polytechnic University.
Team: Joshua Gottus, Ethan Barron.

This project reproduces the DoS detection method in Section IV-A of the reference paper: K-nearest neighbours (KNN) and Naive Bayes classifying NSL-KDD network traffic as normal or DoS, after keeping the eight features most correlated with the label. We compare our results with the paper's reported figures and extend the comparison with a Decision Tree, a Random Forest, and a Random Forest tuned for DoS attack types not seen in training. An SVM is planned. All models share the same preprocessing and splits, and we report recall separately for DoS attack types that appear only in the test file.

## Reference

Tundis, A., Cauteruccio, F. "On Machine Learning for Digital Forensics Investigation in Network Traffic." *2025 21st International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)*, 2025.

Full text: `reference/On_Machine_Learning_for_Digital_Forensics_Investigation_in_Network_Traffic.pdf`

## Key results

Two evaluation splits are used:

- **Official**: train on `KDDTrain+`, test on `KDDTest+`. About 23% of the DoS rows in the test file belong to attack types that never appear in training ("unseen").
- **Holdout**: stratified 80/20 split of `KDDTrain+`. Every test attack type is also in training.

The paper reports only accuracy, recall, and precision. Its numbers are shown for reference; the paper does not say which split it used.

**Official split, 8 features** (`results/metrics/metrics_official_8feat.csv`)

| Model | Accuracy | Precision | Recall | F1 | Recall, unseen DoS types |
|---|---|---|---|---|---|
| KNN (paper) | 98.51% | 98.9% | 97.8% | – | – |
| Naive Bayes (paper) | 93.95% | 97.74% | 95.54% | – | – |
| KNN | 86.00% | 96.37% | 70.43% | 81.39% | 32.15% |
| Naive Bayes | 86.74% | 96.74% | 71.91% | 82.50% | 41.23% |
| Decision Tree | 85.31% | 98.02% | 67.55% | 79.98% | 20.91% |
| Random Forest | 84.61% | 97.85% | 66.02% | 78.85% | 14.62% |
| Random Forest (tuned) | 86.36% | 99.06% | 69.25% | 81.52% | 27.37% |

**Official split, all 41 features** (`results/metrics/metrics_official_41feat.csv`)

| Model | Accuracy | Precision | Recall | F1 | Recall, unseen DoS types |
|---|---|---|---|---|---|
| KNN | 89.53% | 99.00% | 76.67% | 86.41% | 6.93% |
| Naive Bayes | 61.78% | 53.95% | 82.06% | 65.10% | 44.26% |
| Decision Tree | 90.54% | 97.84% | 79.99% | 88.02% | 17.36% |
| Random Forest | 90.27% | 98.59% | 78.73% | 87.55% | 8.91% |
| Random Forest (tuned) | 90.59% | 98.93% | 79.20% | 87.97% | 26.97% |

**Holdout split, 8 features** (`results/metrics/metrics_holdout_8feat.csv`)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN (paper) | 98.51% | 98.9% | 97.8% | – |
| Naive Bayes (paper) | 93.95% | 97.74% | 95.54% | – |
| KNN | 95.70% | 97.35% | 91.89% | 94.54% |
| Naive Bayes | 93.56% | 94.36% | 89.47% | 91.85% |
| Decision Tree | 96.02% | 98.50% | 91.58% | 94.92% |
| Random Forest | 96.04% | 98.34% | 91.79% | 94.95% |
| Random Forest (tuned) | 95.98% | 98.94% | 91.07% | 94.84% |

Findings:

- **The paper's numbers are approached only on the holdout split.** There, Naive Bayes reaches 93.56% accuracy (paper: 93.95%) and KNN 95.70% (paper: 98.51%). On the official test file, accuracy with 8 features drops to 84.61–86.74% for every model, mainly because recall falls to 66–72%.
- **Unseen DoS attack types are the main weakness.** With 8 features, recall on unseen types ranges from 14.62% (Random Forest) to 41.23% (Naive Bayes). Using all 41 features raises recall on seen types (KNN: 81.88% to 97.53%) but lowers it on unseen ones (KNN: 32.15% to 6.93%).
- **Naive Bayes depends on the paper's feature selection.** With all 41 features its accuracy falls to 61.78% and precision to 53.95%, because many near-duplicate features break its independence assumption.
- **Tree models.** A single Decision Tree is within 0.3 points of the default Random Forest. The tuned Random Forest has the best official accuracy (90.59%, 41 features) and raises unseen-type recall from 8.91% to 26.97% over the default forest. Random Forest feature importances show that `src_bytes`, which the paper's correlation filter drops, is one of the most useful features (see `docs/random_forest_notes.md` §10).

Each notebook ends with a longer list of takeaways for its model.

## Approach

- **Data**: NSL-KDD (`KDDTrain+.txt`, `KDDTest+.txt`). Only normal and DoS rows are kept; the other attack categories (probe, R2L, U2R) are dropped. Label: 0 = normal, 1 = DoS.
- **Preprocessing**: categorical columns are encoded, then the top N features by absolute correlation with the label are kept (N = 8 as in the paper, or all 41). The preprocessor is fit on training rows only.
- **Models**: scikit-learn estimators defined in `src/dosdetect/models.py`. KNN (and the planned SVM) are wrapped with `StandardScaler`; tree models use unscaled features.
- **Evaluation**: accuracy, precision, recall, F1, training and prediction time, recall on seen vs. unseen DoS attack types, and the out-of-bag score for the forests.

### Models

| Model | Registry name | Owner | Status |
|---|---|---|---|
| KNN | `knn` | Shared | Done (paper baseline) |
| Naive Bayes | `naive_bayes` | Shared | Done (paper baseline) |
| Decision Tree | `decision_tree` | Joshua | Done |
| Random Forest | `random_forest`, `random_forest_tuned` | Joshua | Done |
| SVM | – | Ethan | Not started |

`random_forest_tuned` keeps the top 5 features by Random Forest importance and uses shallower, regularised trees. Its settings were chosen by cross-validation that hides one DoS attack type at a time from training (`experiments/random_forest/tuning.py`, `docs/random_forest_notes.md` §11).

## Repository layout

```
.
├── src/dosdetect/          Python package
│   ├── data.py             load NSL-KDD, keep normal + DoS rows, add 0/1 label
│   ├── preprocess.py       encoding and top-N correlation feature selection
│   ├── models.py           model definitions (MODELS registry)
│   ├── evaluate.py         train, time, and score a model; paper's reported numbers
│   ├── run.py              CLI pipeline; prepare() is also used by the notebooks
│   ├── report.py           tables and confusion-matrix plots for the notebooks
│   ├── download.py         download NSL-KDD into data/
│   └── paths.py            repository paths
├── notebooks/              one notebook per model (01_knn ... 04_random_forest)
├── experiments/
│   └── random_forest/      logic test, feature importance, importance checks, tuning
├── results/
│   ├── metrics/            metrics_<split>_<n>feat.csv from dosdetect.run
│   └── random_forest/      feature importance (CSV, PNG) and tuning results
├── tests/                  unit tests on synthetic data
├── docs/                   project proposal (LaTeX), Random Forest working notes
├── reference/              source paper
├── data/                   NSL-KDD files (downloaded, not committed)
├── pyproject.toml          package metadata, loose dependency ranges, ruff/pytest config
├── requirements.txt        exact versions used for the committed results
└── Makefile                common tasks (run `make help`)
```

## Quickstart

Tested on Python 3.12 (the pinned versions in `requirements.txt`). The package itself supports Python 3.10+, which CI also tests.

```bash
make setup                       # create .venv, install pinned deps + package with dev and notebook extras
source .venv/bin/activate
make data                        # download KDDTrain+.txt and KDDTest+.txt into data/
python -m dosdetect.run          # all models, official split, 8 features
```

`make setup` runs `pip install -r requirements.txt -e ".[dev,notebooks]"`. On Ubuntu, `python3 -m venv` may first need `sudo apt install python3.12-venv`.

CLI options (also available as `dosdetect-run`):

```bash
python -m dosdetect.run --split holdout                      # stratified 80/20 split of KDDTrain+
python -m dosdetect.run --n-features 41                      # skip feature selection
python -m dosdetect.run --models knn --k 3                   # one model, custom k
python -m dosdetect.run --models random_forest random_forest_tuned --n-features 41
```

Each run prints a comparison with the paper and writes `results/metrics/metrics_<split>_<n>feat.csv`.

## Reproducing results

The committed results were produced with the exact versions in `requirements.txt` (Python 3.12.3). After `make setup` and `make data`:

| Command | Output |
|---|---|
| `make results` | `results/metrics/*.csv` for both splits and both feature counts |
| `make notebooks` | re-executes all notebooks in place |
| `make experiments` | `results/random_forest/*` from the Random Forest analysis scripts (~10 min) |
| `make all` | lint, test, results, notebooks |

The notebooks are the main write-up: each trains its model on all four split/feature setups, compares with the paper where applicable, and ends with takeaways. Open them with `jupyter lab notebooks/`. Detailed Random Forest analysis (logic test, feature-importance verification, tuning) is in `docs/random_forest_notes.md`.

## Development

```bash
make test        # pytest; uses synthetic data, no dataset download needed
make lint        # ruff check + ruff format --check (includes notebooks)
make format      # auto-fix lint issues and format code
make clean       # remove caches
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, and `pytest` on Python 3.10 and 3.12 for pushes to `main` and for pull requests.

To add a model, write a function in `src/dosdetect/models.py` that returns an untrained estimator and register it in `MODELS`. It is then available to the CLI through `--models`.

## Status

- Done: KNN and Naive Bayes paper baselines, Decision Tree, Random Forest, and tuned Random Forest, each with its own notebook. Results for all four split/feature setups are in `results/metrics/`, including recall on seen vs. unseen DoS types.
- Not started: SVM (Ethan).
