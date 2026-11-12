"""Load NSL-KDD and label each row as normal (0) or DoS (1)."""

from pathlib import Path

import pandas as pd

from dosdetect.paths import DATA_DIR

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

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


def load(path):
    """Read one NSL-KDD file, keep normal/DoS rows, add label (1 = DoS)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m dosdetect.download` first.")
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df[(df["attack"] == "normal") | df["attack"].isin(DOS_ATTACKS)].copy()
    df["label"] = (df["attack"] != "normal").astype(int)
    return df.reset_index(drop=True)
