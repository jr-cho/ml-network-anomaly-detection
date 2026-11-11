# Machine Learning for Digital Forensics Investigations in Network Traffic

Undergraduate project for the Digital Forensics course at Florida Polytechnic University.

Team: Joshua Gottus, Ethan Barron.

## Summary

This project implements the DDoS detection method described in Section IV-A of the reference paper below. The paper compares K-nearest neighbours (KNN) and Naive Bayes for classifying network traffic as normal or attack traffic. We build a working version of that method and test it against the paper's reported results. We also extend the comparison with two additional models, Random Forest and Support Vector Machine (SVM), trained on the same preprocessed data.

## Reference

Tundis, A., Cauteruccio, F. "On Machine Learning for Digital Forensics Investigation in Network Traffic." 2025 21st International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT).

Full text: `reference/On_Machine_Learning_for_Digital_Forensics_Investigation_in_Network_Traffic.pdf`

## Approach

- **Dataset**: NSL-KDD. Labels traffic as normal or one of four attack types (user-to-root, root-to-local, DoS, probing).
- **Preprocessing**: Keep DoS and normal traffic only. Drop other attack types. Score each feature's correlation with the label and keep the top eight, matching the paper's preprocessing step.
- **Models**: KNN and Naive Bayes, built with scikit-learn.
- **Extension models**: Random Forest and SVM, using the same preprocessing and train/test split so all four results are directly comparable.
- **Evaluation metrics**: Accuracy, precision, recall, F1-score, and training/prediction time for all four models. Compare KNN and Naive Bayes against the paper's reported figures (KNN: 98.51% accuracy, 97.8% recall, 98.9% precision; Naive Bayes: 93.95% accuracy, 95.54% recall, 97.74% precision).

### Model assignments

| Model | Owner | Notes |
|---|---|---|
| KNN | Shared | Paper baseline. Features must be scaled. |
| Naive Bayes | Shared | Paper baseline. |
| Random Forest | Joshua | Ensemble of decision trees. Feature importances help explain which traffic features signal an attack. |
| SVM | Ethan | Margin-based classifier. Features must be scaled. May need a subsample or `LinearSVC` for training time on the full dataset. |

## Layout

```
data/       NSL-KDD dataset files
src/        preprocessing, training, and evaluation scripts
results/    metrics CSVs written by src.run (one per split and feature count)
notebooks/  exploratory analysis
docs/       written project proposal, slides, and report (LaTeX)
reference/  source paper
```

## Setup

```bash
python3 -m venv .venv            # Ubuntu may need: sudo apt install python3.12-venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data      # fetches KDDTrain+.txt and KDDTest+.txt into data/
```

## Usage

```bash
python -m src.run                         # train on KDDTrain+, test on KDDTest+
python -m src.run --split holdout         # stratified 80/20 split of KDDTrain+
python -m src.run --models knn --k 3      # one model, custom K
python -m src.run --n-features 41         # skip feature selection
```

Code overview:

- `src/data.py`: loads NSL-KDD, keeps normal and DoS rows, adds a 0/1 label.
- `src/preprocess.py`: encodes the text columns and keeps the top N features by correlation with the label (fit on training data only).
- `src/models.py`: model definitions. Add new models to the `MODELS` dict.
- `src/evaluate.py`: trains, times, and scores a model. Holds the paper's reported numbers.
- `src/run.py`: runs the full pipeline and prints a comparison with the paper.

## Status

Paper baselines (KNN, Naive Bayes) and Random Forest implemented. Results for all four split/feature setups are in `results/`, including recall on seen vs. unseen DoS types. SVM not started.
