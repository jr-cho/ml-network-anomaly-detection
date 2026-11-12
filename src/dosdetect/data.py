"""Load NSL-KDD, keep normal/DoS rows, encode, and pick the top-N features.

`prepare(split, n_features)` is the one entry point the notebooks and the CLI use.
"""

import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"
DOWNLOAD_URL = "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/"

SPLITS = ("official", "holdout")
FEATURE_COUNTS = (8, 41)

# Raw files have no header: 41 features, attack label, difficulty.
FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURE_COLUMNS + ["attack", "difficulty"]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Last four appear only in the test file (unseen in training).
DOS_ATTACKS = {
    "back",
    "land",
    "neptune",
    "pod",
    "smurf",
    "teardrop",
    "apache2",
    "mailbomb",
    "processtable",
    "udpstorm",
}


def download(timeout=60):
    """Fetch KDDTrain+.txt and KDDTest+.txt into data/ if missing.

    Each file is written to a .part file and renamed only once complete, so an
    interrupted download never leaves a truncated file behind.
    """
    DATA_DIR.mkdir(exist_ok=True)
    for path in (TRAIN_FILE, TEST_FILE):
        if path.exists():
            continue
        print(f"Downloading {path.name}...")
        part = path.with_name(path.name + ".part")
        try:
            with urllib.request.urlopen(DOWNLOAD_URL + path.name, timeout=timeout) as r:
                with open(part, "wb") as f:
                    shutil.copyfileobj(r, f)
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)


def load(path):
    """Read one NSL-KDD file, keep normal/DoS rows, add label (1 = DoS)."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df[(df["attack"] == "normal") | df["attack"].isin(DOS_ATTACKS)].copy()
    df["label"] = (df["attack"] != "normal").astype(int)
    return df.reset_index(drop=True)


def load_split(split="official", seed=42):
    """official: KDDTrain+ vs KDDTest+. holdout: stratified 80/20 of KDDTrain+."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    download()
    if split == "official":
        return load(TRAIN_FILE), load(TEST_FILE)
    df = load(TRAIN_FILE)
    return train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)


class Preprocessor:
    """Ordinal-encode text columns, keep the top-N features by |correlation| (fit on train)."""

    def __init__(self, n_features=8):
        if not 1 <= n_features <= len(FEATURE_COLUMNS):
            raise ValueError(f"n_features must be 1-{len(FEATURE_COLUMNS)}, got {n_features}")
        self.n_features = n_features
        # Categories missing from training become -1 at test time.
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, df):
        X = self._encode(df, fit=True)
        # Constant columns give NaN correlation; treat as 0.
        with np.errstate(divide="ignore", invalid="ignore"):  # older pandas warns on them
            corr = X.corrwith(df["label"])
        self.correlations = corr.abs().fillna(0).sort_values(ascending=False)
        self.selected_features = list(self.correlations.index[: self.n_features])
        return self

    def transform(self, df):
        return self._encode(df, fit=False)[self.selected_features], df["label"]

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def _encode(self, df, fit):
        X = df[FEATURE_COLUMNS].copy()
        cats = X[CATEGORICAL_COLUMNS]
        X[CATEGORICAL_COLUMNS] = (
            self.encoder.fit_transform(cats) if fit else self.encoder.transform(cats)
        )
        return X.astype(float)


def prepare(split="official", n_features=8, seed=42):
    """Load and preprocess one setup; `unseen` marks test DoS types absent from training."""
    train_df, test_df = load_split(split, seed)
    prep = Preprocessor(n_features)
    X_train, y_train = prep.fit_transform(train_df)
    X_test, y_test = prep.transform(test_df)
    unseen = (test_df["label"] == 1) & ~test_df["attack"].isin(train_df["attack"])
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "unseen": unseen.to_numpy(),
        "prep": prep,
        "train_df": train_df,
        "test_df": test_df,
    }


def prepare_all(seed=42):
    """All four setups, keyed by (split, n_features)."""
    return {(s, n): prepare(s, n, seed) for s in SPLITS for n in FEATURE_COUNTS}


if __name__ == "__main__":
    download()
