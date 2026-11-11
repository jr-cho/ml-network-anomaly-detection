"""Model definitions. Each entry builds a fresh, untrained scikit-learn model.

To add a model, write a function that returns an estimator and register it in
MODELS. Models that rely on distances (KNN, SVM) need scaled features, so wrap
them in a pipeline with StandardScaler.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def knn(k=5, metric="euclidean"):
    """K-nearest neighbours, as in the paper (Euclidean distance, majority vote)."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, metric=metric))


def naive_bayes():
    """Gaussian Naive Bayes, as in the paper."""
    return GaussianNB()


def random_forest(n_estimators=300, seed=42):
    """Random Forest. Trees split on thresholds, so no feature scaling is needed.

    Accuracy stops improving after about 100 trees on this data, so 300 is
    plenty. oob_score gives a free accuracy estimate from the rows each tree
    skipped during training (see BRAINSTORM.md sections 3 and 9).
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        n_jobs=-1,
        oob_score=True,
        random_state=seed,
    )


# TODO (Ethan): svm() using sklearn.svm.SVC or LinearSVC, wrapped with StandardScaler

MODELS = {
    "knn": knn,
    "naive_bayes": naive_bayes,
    "random_forest": random_forest,
}
