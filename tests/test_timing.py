import numpy as np
import pytest

from dosdetect import data, timing

FIELDS = [f"{t}_s_{s}" for t in ("train", "predict") for s in ("median", "min", "max")] + [
    "repeats"
]


@pytest.fixture
def fake_prepare(monkeypatch, train_df, test_df, tmp_path):
    monkeypatch.setattr(data, "load_split", lambda split, seed: (train_df, test_df))
    monkeypatch.setattr(timing, "RESULTS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def full_run(fake_prepare):
    return timing.main(["--repeats", "2"])


def test_full_run_has_expected_columns_and_writes_csv(full_run, fake_prepare):
    assert (fake_prepare / "timing.csv").exists()
    assert len(full_run) == len(timing.MODELS) * len(data.SPLITS) * len(data.FEATURE_COUNTS)
    for column in [*FIELDS, "cpu_count", "platform", "python", "sklearn"]:
        assert column in full_run
    assert (full_run["repeats"] == 2).all()


def test_min_median_max_are_ordered(full_run):
    for t in ("train_s", "predict_s"):
        assert (full_run[f"{t}_min"] <= full_run[f"{t}_median"]).all()
        assert (full_run[f"{t}_median"] <= full_run[f"{t}_max"]).all()


def test_ratios_are_medians_within_the_same_setup(full_run):
    for _, g in full_run.groupby(["split", "n_features"]):
        g = g.set_index("model")
        nb, dt = g.loc["naive_bayes"], g.loc["decision_tree"]
        assert nb["predict_ratio_vs_naive_bayes"] == pytest.approx(1)
        assert dt["train_ratio_vs_decision_tree"] == pytest.approx(1)
        assert g.loc["knn", "predict_ratio_vs_naive_bayes"] == pytest.approx(
            g.loc["knn", "predict_s_median"] / nb["predict_s_median"]
        )
        assert g.loc["knn", "train_ratio_vs_decision_tree"] == pytest.approx(
            g.loc["knn", "train_s_median"] / dt["train_s_median"]
        )


def test_ratio_is_nan_when_reference_model_not_run(fake_prepare):
    df = timing.main(["--repeats", "1", "--models", "knn", "--split", "official"])
    assert np.isnan(df["predict_ratio_vs_naive_bayes"]).all()
    assert np.isnan(df["train_ratio_vs_decision_tree"]).all()


def test_partial_run_does_not_write_csv(fake_prepare):
    timing.main(["--repeats", "1", "--models", "knn"])
    assert not (fake_prepare / "timing.csv").exists()


def test_partial_run_saves_with_out(fake_prepare):
    out = fake_prepare / "sub" / "t.csv"
    timing.main(["--repeats", "1", "--models", "knn", "--out", str(out)])
    assert out.exists()


def test_repeats_must_be_positive():
    with pytest.raises(SystemExit):
        timing.parse_args(["--repeats", "0"])
