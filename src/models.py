"""Model definitions. Each entry builds a fresh, untrained scikit-learn model.

To add a model, write a function that returns an estimator and register it in
MODELS. Models that rely on distances (KNN, SVM) need scaled features, so wrap
them in a pipeline with StandardScaler.
"""

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


# TODO (Joshua): random_forest() using sklearn.ensemble.RandomForestClassifier
# TODO (Ethan): svm() using sklearn.svm.SVC or LinearSVC, wrapped with StandardScaler

MODELS = {
    "knn": knn,
    "naive_bayes": naive_bayes,
}
