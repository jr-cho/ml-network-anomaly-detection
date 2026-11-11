from dosdetect.data import FEATURE_COLUMNS
from dosdetect.preprocess import Preprocessor


def test_keeps_requested_number_of_features(train_df):
    X, y = Preprocessor(n_features=8).fit_transform(train_df)
    assert X.shape == (len(train_df), 8)
    assert (y == train_df["label"]).all()


def test_all_41_features(train_df):
    X, _ = Preprocessor(n_features=41).fit_transform(train_df)
    assert set(X.columns) == set(FEATURE_COLUMNS)


def test_picks_most_correlated_features(train_df):
    prep = Preprocessor(n_features=3).fit(train_df)
    # the fixture makes these track the label
    assert set(prep.selected_features) == {"serror_rate", "count", "flag"}


def test_constant_column_does_not_break_ranking(train_df):
    prep = Preprocessor(n_features=41).fit(train_df)
    assert prep.correlations["num_outbound_cmds"] == 0
    assert prep.correlations.notna().all()


def test_transform_does_not_refit_on_test_data(train_df, test_df):
    prep = Preprocessor(n_features=8).fit(train_df)
    before = (list(prep.selected_features), prep.correlations.copy())
    prep.transform(test_df)
    assert prep.selected_features == before[0]
    assert prep.correlations.equals(before[1])


def test_unseen_category_encodes_as_minus_one(train_df, test_df):
    prep = Preprocessor(n_features=41).fit(train_df)
    X, _ = prep.transform(test_df)
    assert X.loc[0, "service"] == -1
    assert (X.loc[1:, "service"] >= 0).all()
