"""Synthetic NSL-KDD-shaped data so the tests run without the real dataset."""

import numpy as np
import pandas as pd
import pytest

from dosdetect.data import COLUMNS, FEATURE_COLUMNS


def make_raw(attacks, seed=0):
    """Raw rows for the given attacks; DoS rows get high serror_rate and count."""
    rng = np.random.default_rng(seed)
    n = len(attacks)
    df = pd.DataFrame(rng.random((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    is_attack = np.array([a != "normal" for a in attacks])
    df["serror_rate"] = np.where(is_attack, 0.9, 0.1) + rng.normal(0, 0.05, n)
    df["count"] = np.where(is_attack, 400, 20) + rng.integers(0, 50, n)
    df["num_outbound_cmds"] = 0.0  # constant, as in the real data
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    df["service"] = rng.choice(["http", "private", "ftp"], n)
    df["flag"] = np.where(is_attack, "S0", "SF")
    df["attack"] = attacks
    df["difficulty"] = 20
    return df[COLUMNS]


def with_label(df):
    df = df.copy()
    df["label"] = (df["attack"] != "normal").astype(int)
    return df


@pytest.fixture
def train_df():
    attacks = ["normal"] * 120 + ["neptune"] * 60 + ["smurf"] * 40
    return with_label(make_raw(attacks, seed=1))


@pytest.fixture
def test_df():
    # apache2 is unseen in training, like KDDTest+
    attacks = ["normal"] * 40 + ["neptune"] * 20 + ["apache2"] * 10
    df = with_label(make_raw(attacks, seed=2))
    df.loc[0, "service"] = "telnet"  # category unseen in training
    return df


@pytest.fixture
def raw_file(tmp_path):
    """Headerless NSL-KDD file with normal, DoS and non-DoS rows."""
    attacks = ["normal"] * 5 + ["neptune"] * 3 + ["portsweep"] * 2 + ["guess_passwd"] * 2
    path = tmp_path / "KDDFake.txt"
    make_raw(attacks).to_csv(path, header=False, index=False)
    return path
