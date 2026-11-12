"""Download NSL-KDD train/test files into data/ (python -m dosdetect.download)."""

import urllib.request

from dosdetect.data import DATA_DIR, TEST_FILE, TRAIN_FILE

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for path in (TRAIN_FILE, TEST_FILE):
        if path.exists():
            print(f"{path.name} already exists, skipping")
            continue
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(BASE_URL + path.name, path)
    print("Done.")


if __name__ == "__main__":
    main()
