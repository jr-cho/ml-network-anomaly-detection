from dosdetect import run


def test_prepare_marks_only_test_only_dos_types_as_unseen(monkeypatch, train_df, test_df):
    monkeypatch.setattr(run, "load_split", lambda split, seed: (train_df, test_df))
    d = run.prepare("official", n_features=8)

    expected = (test_df["attack"] == "apache2").to_numpy()
    assert (d["unseen"] == expected).all()
    assert d["X_train"].shape[1] == d["X_test"].shape[1] == 8
    assert list(d["X_test"].columns) == d["prep"].selected_features


def test_prepare_has_no_unseen_rows_when_test_types_match(monkeypatch, train_df):
    monkeypatch.setattr(run, "load_split", lambda split, seed: (train_df, train_df))
    assert not run.prepare("holdout").get("unseen").any()


def test_cli_defaults_match_the_paper(monkeypatch):
    monkeypatch.setattr("sys.argv", ["dosdetect-run"])
    args = run.parse_args()
    assert (args.split, args.n_features, args.k) == ("official", 8, 5)
    assert args.models == list(run.MODELS)


def test_build_model_passes_k_and_seed():
    class Args:
        k, seed = 3, 7

    assert run.build_model("knn", Args)[-1].n_neighbors == 3
    assert run.build_model("random_forest", Args).random_state == 7
    assert run.build_model("decision_tree", Args).random_state == 7
