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
