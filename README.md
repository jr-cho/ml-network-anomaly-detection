# Machine Learning for Digital Forensics Investigations in Network Traffic

Undergraduate project for the Digital Forensics course at Florida Polytechnic University.

Team: Joshua Gottus, Ethan Barron.

## Summary

This project implements the DDoS detection method described in Section IV-A of the reference paper below. The paper compares K-nearest neighbours (KNN) and Naive Bayes for classifying network traffic as normal or attack traffic. We build a working version of that method and test it against the paper's reported results.

## Reference

Tundis, A., Cauteruccio, F. "On Machine Learning for Digital Forensics Investigation in Network Traffic." 2025 21st International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT).

Full text: `reference/On_Machine_Learning_for_Digital_Forensics_Investigation_in_Network_Traffic.pdf`

## Approach

- **Dataset**: NSL-KDD. Labels traffic as normal or one of four attack types (user-to-root, root-to-local, DoS, probing).
- **Preprocessing**: Keep DoS and normal traffic only. Drop other attack types. Score each feature's correlation with the label and keep the top eight, matching the paper's preprocessing step.
- **Models**: KNN and Naive Bayes, built with scikit-learn.
- **Evaluation metrics**: Accuracy, precision, recall. Compare our results against the paper's reported figures (KNN: 98.51% accuracy, 97.8% recall, 98.9% precision; Naive Bayes: 93.95% accuracy, 95.54% recall, 97.74% precision).

## Planned layout

```
data/       NSL-KDD dataset files
src/        preprocessing, training, and evaluation scripts
notebooks/  exploratory analysis
docs/       written project proposal, slides, and report (LaTeX)
reference/  source paper
```

## Status

Proposal stage. Implementation has not started yet.
