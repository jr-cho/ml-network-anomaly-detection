import pytest

from dosdetect import data, run


@pytest.fixture
def fake_prepare(monkeypatch, train_df, test_df, tmp_path):
    """Run the CLI on synthetic data with results written to a temp folder."""
    monkeypatch.setattr(data, "load_split", lambda split, seed: (train_df, test_df))
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_full_run_writes_metrics_csv(fake_prepare):
    df = run.main([])
    assert (fake_prepare / "metrics.csv").exists()
    assert len(df) == len(run.MODELS) * len(data.SPLITS) * len(data.FEATURE_COUNTS)


def test_partial_run_does_not_overwrite_metrics_csv(fake_prepare):
    run.main(["--models", "knn"])
    assert not (fake_prepare / "metrics.csv").exists()


def test_partial_run_saves_with_out(fake_prepare):
    out = fake_prepare / "sub" / "knn.csv"
    run.main(["--models", "knn", "--split", "official", "--n-features", "8", "--out", str(out)])
    assert out.exists()


def test_reordered_full_run_still_counts_as_full(fake_prepare):
    run.main(["--split", "kddcup", "holdout", "official", "--n-features", "41", "8"])
    assert (fake_prepare / "metrics.csv").exists()


def test_repeated_values_are_ignored():
    args = run.parse_args(["--split", "official", "official", "--models", "knn", "knn"])
    assert args.split == ["official"] and args.models == ["knn"]


@pytest.mark.parametrize("n", ["0", "42"])
def test_n_features_out_of_range_is_rejected(n):
    with pytest.raises(SystemExit):
        run.parse_args(["--n-features", n])


def test_full_run_prints_two_dataset_comparison(fake_prepare, capsys):
    run.main([])
    out = capsys.readouterr().out
    for name in ("knn", "naive_bayes"):
        assert f"{name}: NSL-KDD + KDDCup vs paper [4]" in out
    assert "NSL-KDD" in out and "KDDCup" in out and "paper [4]" in out and "difference" in out


def test_partial_run_without_both_datasets_skips_comparison(fake_prepare, capsys):
    run.main(["--split", "official", "--n-features", "8"])
    assert "vs paper [4]" not in capsys.readouterr().out
