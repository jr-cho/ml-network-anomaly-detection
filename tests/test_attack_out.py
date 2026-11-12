import numpy as np
import pytest
from conftest import make_raw, with_label

from dosdetect import attack_out
from dosdetect.data import ATTACK_GROUPS, attack_out_splits

TYPES = [t for g in ATTACK_GROUPS for t in g]


@pytest.fixture
def df():
    attacks = ["normal"] * 200 + [t for t in TYPES for _ in range(30)]
    return with_label(make_raw(attacks))


def splits(df, **kw):
    return {g: (tr, te, m) for g, tr, te, m in attack_out_splits(df, **kw)}


def test_group_names_cover_attack_groups(df):
    names = list(splits(df))
    assert names == ["+".join(g) for g in ATTACK_GROUPS]
    assert sorted(t for n in names for t in n.split("+")) == sorted(TYPES)


def test_heldout_not_in_train(df):
    for name, (train, _, _) in splits(df).items():
        assert not train["attack"].isin(name.split("+")).any()


def test_heldout_all_in_test_and_mask_matches(df):
    for name, (_, test, mask) in splits(df).items():
        group = name.split("+")
        assert len(mask) == len(test)
        assert (test["attack"].isin(group).to_numpy() == mask).all()
        assert mask.sum() == df["attack"].isin(group).sum()


def test_train_and_test_do_not_overlap(df):
    for train, test, _ in splits(df).values():
        assert train.index.intersection(test.index).empty


def test_cap_is_respected(df):
    for _, _, mask in splits(df, cap=10).values():
        assert mask.sum() == 10


def test_run_columns_and_row_count(df, monkeypatch):
    monkeypatch.setattr(attack_out, "load_dataset", lambda name: df)
    out = attack_out.run(("nsl-kdd",), (3, 8), ("naive_bayes", "decision_tree"))
    assert len(out) == 1 * 2 * 2 * len(ATTACK_GROUPS)
    assert list(out.columns) == [
        "dataset",
        "n_features",
        "model",
        "group",
        "n_heldout",
        "recall_heldout",
        "recall_seen",
        "false_alarm_rate",
    ]
    assert out[["recall_heldout", "recall_seen", "false_alarm_rate"]].stack().between(0, 1).all()
    assert np.isfinite(out["n_heldout"]).all()
