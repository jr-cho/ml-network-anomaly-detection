"""Load NSL-KDD and label each row as normal (0) or DoS (1)."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

# The raw files have no header row. 41 features, then the attack label and a
# difficulty score added by the NSL-KDD authors.
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURE_COLUMNS + ["attack", "difficulty"]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Attack names that fall under the DoS category. The last four only appear in
# the test file, so the models never see them during training.
DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "mailbomb", "processtable", "udpstorm",
}


def load(path):
    """Read one NSL-KDD file and keep only normal and DoS rows.

    Adds a binary `label` column: 0 = normal, 1 = DoS.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` first."
        )
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df[(df["attack"] == "normal") | df["attack"].isin(DOS_ATTACKS)].copy()
    df["label"] = (df["attack"] != "normal").astype(int)
    return df.reset_index(drop=True)
