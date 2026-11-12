import io

import numpy as np
import pytest
from conftest import make_raw

from dosdetect import data
from dosdetect.data import DOS_ATTACKS, FEATURE_COLUMNS, Preprocessor, load


def test_load_keeps_only_normal_and_dos_with_label(raw_file):
    df = load(raw_file)
    assert set(df["attack"]) == {"normal", "neptune"}
    assert len(df) == 8
    assert df["label"].sum() == 3
    assert (df["label"] == (df["attack"] != "normal")).all()


def test_test_only_dos_types_are_listed():
    assert {"apache2", "mailbomb", "processtable", "udpstorm"} <= DOS_ATTACKS


def test_keeps_requested_number_of_features(train_df):
    X, y = Preprocessor(n_features=8).fit_transform(train_df)
    assert X.shape == (len(train_df), 8)
    assert (y == train_df["label"]).all()
    X, _ = Preprocessor(n_features=41).fit_transform(train_df)
    assert set(X.columns) == set(FEATURE_COLUMNS)


def test_picks_most_correlated_features(train_df):
    prep = Preprocessor(n_features=3).fit(train_df)
    assert set(prep.selected_features) == {"serror_rate", "count", "flag"}


def test_ranking_is_by_rank_correlation(train_df):
    # Zero for every DoS row, heavy-tailed for normal rows, like dst_bytes: perfectly
    # separable by rank, but a few huge values keep the Pearson correlation low.
    is_dos = train_df["label"] == 1
    heavy = np.where(is_dos, 0.0, np.geomspace(1, 1e9, len(train_df)))
    df = train_df.assign(dst_bytes=heavy)
    assert abs(np.corrcoef(df["dst_bytes"], df["label"])[0, 1]) < 0.3
    assert Preprocessor(n_features=41).fit(df).correlations["dst_bytes"] == pytest.approx(
        np.corrcoef(df["dst_bytes"].rank(), df["label"].rank())[0, 1].__abs__()
    )
    assert "dst_bytes" in Preprocessor(n_features=4).fit(df).selected_features


def test_constant_column_does_not_break_ranking(train_df):
    prep = Preprocessor(n_features=41).fit(train_df)
    assert prep.correlations["num_outbound_cmds"] == 0
    assert prep.correlations.notna().all()


def test_transform_does_not_refit_on_test_data(train_df, test_df):
    prep = Preprocessor(n_features=8).fit(train_df)
    selected, corr = list(prep.selected_features), prep.correlations.copy()
    prep.transform(test_df)
    assert prep.selected_features == selected
    assert prep.correlations.equals(corr)


def test_unseen_category_encodes_as_minus_one(train_df, test_df):
    X, _ = Preprocessor(n_features=41).fit(train_df).transform(test_df)
    assert X.loc[0, "service"] == -1
    assert (X.loc[1:, "service"] >= 0).all()


def test_prepare_marks_only_test_only_dos_types_as_unseen(monkeypatch, train_df, test_df):
    monkeypatch.setattr(data, "load_split", lambda split, seed: (train_df, test_df))
    d = data.prepare("official", n_features=8)
    assert (d["unseen"] == (test_df["attack"] == "apache2").to_numpy()).all()
    assert list(d["X_test"].columns) == d["prep"].selected_features


def test_prepare_has_no_unseen_rows_when_test_types_match(monkeypatch, train_df):
    monkeypatch.setattr(data, "load_split", lambda split, seed: (train_df, train_df))
    assert not data.prepare("holdout")["unseen"].any()


def test_load_kddcup_decodes_and_keeps_normal_and_dos(monkeypatch):
    raw = make_raw(["normal", "smurf", "neptune", "satan"]).drop(columns="difficulty")
    raw["attack"] = raw["attack"] + "."  # sklearn labels look like "normal."
    for col in data.CATEGORICAL_COLUMNS + ["attack"]:
        raw[col] = raw[col].str.encode("ascii")  # sklearn returns bytes
    raw.columns = [c.upper() for c in raw.columns]  # sklearn's names differ from ours
    fake = type("Bunch", (), {"frame": raw.astype(object)})
    monkeypatch.setattr(data, "fetch_kddcup99", lambda **kw: fake)
    df = data.load_kddcup()
    assert list(df["attack"]) == ["normal", "smurf", "neptune"]
    assert list(df["label"]) == [0, 1, 1]
    assert df["protocol_type"].isin(["tcp", "udp", "icmp"]).all()
    assert df["count"].dtype == float


def test_kddcup_split_is_stratified_80_20(monkeypatch, train_df):
    monkeypatch.setattr(data, "load_kddcup", lambda: train_df)
    train, test = data.load_split("kddcup")
    assert len(test) == len(train_df) // 5
    assert test["label"].mean() == pytest.approx(train_df["label"].mean(), abs=0.01)


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="split"):
        data.load_split("Official")


@pytest.mark.parametrize("n", [0, 42])
def test_n_features_out_of_range_is_rejected(n):
    with pytest.raises(ValueError, match="n_features"):
        Preprocessor(n_features=n)


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "TRAIN_FILE", tmp_path / "KDDTrain+.txt")
    monkeypatch.setattr(data, "TEST_FILE", tmp_path / "KDDTest+.txt")
    return tmp_path


def test_download_fetches_missing_files_and_skips_existing(monkeypatch, data_dir):
    (data_dir / "KDDTrain+.txt").write_text("already here")
    fetched = []

    def urlopen(url, timeout):
        fetched.append(url)
        return io.BytesIO(b"downloaded")

    monkeypatch.setattr(data.urllib.request, "urlopen", urlopen)
    data.download()
    assert fetched == [data.DOWNLOAD_URL + "KDDTest+.txt"]
    assert (data_dir / "KDDTest+.txt").read_bytes() == b"downloaded"
    assert (data_dir / "KDDTrain+.txt").read_text() == "already here"


def test_interrupted_download_leaves_no_file(monkeypatch, data_dir):
    class Broken(io.BytesIO):
        def read(self, *args):
            raise ConnectionResetError("dropped")

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda url, timeout: Broken())
    with pytest.raises(ConnectionResetError):
        data.download()
    assert list(data_dir.iterdir()) == []
