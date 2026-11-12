"""Load NSL-KDD or KDDCup'99, keep normal/DoS rows, encode, and pick the top-N features.

`prepare(split, n_features)` is the one entry point the notebooks and the CLI use.
"""

import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"
DOWNLOAD_URL = "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/"

# official: NSL-KDD KDDTrain+ vs KDDTest+. holdout: 80/20 of KDDTrain+.
# kddcup: 80/20 of the KDDCup'99 10% training file (its labelled test file is no longer
# downloadable, so there is no official KDDCup split).
SPLITS = ("official", "holdout", "kddcup")
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


def keep_dos(df):
    """Keep normal/DoS rows and add label (1 = DoS)."""
    df = df[(df["attack"] == "normal") | df["attack"].isin(DOS_ATTACKS)].copy()
    df["label"] = (df["attack"] != "normal").astype(int)
    return df.reset_index(drop=True)


def load(path):
    """Read one NSL-KDD file, keep normal/DoS rows, add label (1 = DoS)."""
    return keep_dos(pd.read_csv(path, header=None, names=COLUMNS))


def load_kddcup():
    """KDDCup'99 10% training file (downloaded and checksum-verified by scikit-learn)."""
    df = fetch_kddcup99(percent10=True, as_frame=True, data_home=DATA_DIR).frame
    df.columns = FEATURE_COLUMNS + ["attack"]
    text = CATEGORICAL_COLUMNS + ["attack"]
    df[text] = df[text].apply(lambda c: c.str.decode("ascii"))
    df["attack"] = df["attack"].str.rstrip(".")  # "normal." -> "normal"
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    df[numeric] = df[numeric].astype(float)
    return keep_dos(df)


def load_split(split="official", seed=42):
    """Train and test frames for one split (see SPLITS); holdout splits are stratified 80/20."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "kddcup":
        df = load_kddcup()
        return train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    download()
    if split == "official":
        return load(TRAIN_FILE), load(TEST_FILE)
    df = load(TRAIN_FILE)
    return train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)


# Held-out attack groups for leave-one-attack-out (also used by notebook 04's attack-out CV).
ATTACK_GROUPS = [["neptune"], ["smurf"], ["back", "land"], ["teardrop", "pod"]]
DATASETS = ("nsl-kdd", "kddcup")


def load_dataset(name):
    """A whole dataset: "nsl-kdd" (KDDTrain+ only) or "kddcup" (the 10% file)."""
    if name not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {name!r}")
    if name == "kddcup":
        return load_kddcup()
    download()
    return load(TRAIN_FILE)


def attack_out_splits(df, seed=42, cap=20_000):
    """Yield (group_name, train_df, test_df, heldout_mask) for leave-one-attack-out.

    Rows outside the group are split 80/20, stratified by attack. Train is the 80%.
    Test is the 20% plus every held-out row (at most `cap`, sampled). `heldout_mask`
    marks the held-out rows of `test_df`.
    """
    for group in ATTACK_GROUPS:
        in_group = df["attack"].isin(group)
        train, seen_test = train_test_split(
            df[~in_group], test_size=0.2, stratify=df.loc[~in_group, "attack"], random_state=seed
        )
        heldout = df[in_group]
        if len(heldout) > cap:
            heldout = heldout.sample(n=cap, random_state=seed)
        test = pd.concat([seen_test, heldout])
        mask = np.r_[np.zeros(len(seen_test), bool), np.ones(len(heldout), bool)]
        yield "+".join(group), train, test, mask


class Preprocessor:
    """Ordinal-encode text columns, keep the top-N features by |Spearman correlation|.

    Spearman (rank) correlation is used because it reproduces the features the paper
    names (dst_bytes, protocol_type, connection counts) on KDDCup'99; Pearson does not.
    Everything is fit on the training data only.
    """

    def __init__(self, n_features=8):
        if not 1 <= n_features <= len(FEATURE_COLUMNS):
            raise ValueError(f"n_features must be 1-{len(FEATURE_COLUMNS)}, got {n_features}")
        self.n_features = n_features
        # Categories missing from training become -1 at test time.
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, df):
        X = self._encode(df, fit=True)
        # Spearman = Pearson on ranks. Constant columns give NaN; treat as 0.
        with np.errstate(divide="ignore", invalid="ignore"):  # older pandas warns on them
            corr = X.rank().corrwith(df["label"].rank())
        self.correlations = corr.abs().fillna(0).sort_values(ascending=False, kind="stable")
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
    """Every setup, keyed by (split, n_features)."""
    return {(s, n): prepare(s, n, seed) for s in SPLITS for n in FEATURE_COUNTS}


if __name__ == "__main__":
    download()
    load_kddcup()
