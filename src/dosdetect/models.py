"""Model factories; distance-based models need StandardScaler."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def knn(k=5, metric="euclidean"):
    """KNN as in the paper (scaled, Euclidean)."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, metric=metric))


def naive_bayes():
    """Gaussian Naive Bayes, as in the paper."""
    return GaussianNB()


def decision_tree(seed=42):
    """Single-tree baseline for Random Forest (no scaling needed)."""
    return DecisionTreeClassifier(criterion="gini", random_state=seed)


def random_forest(n_estimators=300, seed=42):
    """Random Forest; accuracy plateaus ~100 trees, OOB gives free estimate."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        n_jobs=-1,
        oob_score=True,
        random_state=seed,
    )


def random_forest_tuned(seed=42, n_keep=5):
    """RF tuned for unseen attacks; best with all 41 features (notebook 04)."""
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=seed),
        threshold=-float("inf"),  # rank only; keep exactly the top n_keep
        max_features=lambda X: min(n_keep, X.shape[1]),
    )
    forest = RandomForestClassifier(
        n_estimators=300,
        max_features=0.5,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        oob_score=True,
        random_state=seed,
    )
    return Pipeline([("select", selector), ("rf", forest)])


# TODO (Ethan): svm() with SVC/LinearSVC + StandardScaler

MODELS = {
    "knn": knn,
    "naive_bayes": naive_bayes,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
    "random_forest_tuned": random_forest_tuned,
}
