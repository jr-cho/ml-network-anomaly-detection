import pytest

from dosdetect.data import DOS_ATTACKS, load


def test_load_keeps_only_normal_and_dos(raw_file):
    df = load(raw_file)
    assert set(df["attack"]) == {"normal", "neptune"}
    assert len(df) == 8


def test_load_labels_dos_as_one(raw_file):
    df = load(raw_file)
    assert (df["label"] == (df["attack"] != "normal")).all()
    assert df["label"].sum() == 3


def test_load_missing_file_explains_how_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="dosdetect.download"):
        load(tmp_path / "missing.txt")


def test_test_only_dos_types_are_listed():
    # only in KDDTest+, needed for unseen recall
    assert {"apache2", "mailbomb", "processtable", "udpstorm"} <= DOS_ATTACKS
