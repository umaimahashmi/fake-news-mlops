"""Download ISOT and LIAR datasets."""

import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def check_isot():
    isot_dir = os.path.join(DATA_DIR, "isot")
    fake = os.path.join(isot_dir, "Fake.csv")
    true = os.path.join(isot_dir, "True.csv")
    if os.path.exists(fake) and os.path.exists(true):
        print(f"ISOT already present at {isot_dir}")
        return True
    return False


def check_liar():
    liar_dir = os.path.join(DATA_DIR, "liar")
    train = os.path.join(liar_dir, "train.tsv")
    if os.path.exists(train):
        print(f"LIAR already present at {liar_dir}")
        return True
    return False


def download_liar():
    liar_dir = os.path.join(DATA_DIR, "liar")
    os.makedirs(liar_dir, exist_ok=True)
    base = "https://raw.githubusercontent.com/thunlp/LIAR-PLUS/master/dataset/tsv"
    for fname in ["train.tsv", "valid.tsv", "test.tsv"]:
        out = os.path.join(liar_dir, fname)
        if not os.path.exists(out):
            url = f"{base}/{fname}"
            print(f"Downloading {url}")
            try:
                urllib.request.urlretrieve(url, out)
                print(f"  Saved {out}")
            except Exception as e:
                print(f"  Failed: {e}")
                print(
                    "  Please manually download LIAR dataset from "
                    "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
                )
                print(f"  and place train.tsv, valid.tsv, test.tsv in {liar_dir}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "isot"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "liar"), exist_ok=True)

    if not check_isot():
        print("\nISOT dataset not found.")
        print(
            "Please download from Kaggle: "
            "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )
        print(f"Place Fake.csv and True.csv in: {os.path.join(DATA_DIR, 'isot')}")

    if not check_liar():
        print("\nDownloading LIAR dataset...")
        download_liar()

    print("\nData check complete.")
    print("Expected paths:")
    print(f"  {os.path.join(DATA_DIR, 'isot/Fake.csv')}")
    print(f"  {os.path.join(DATA_DIR, 'isot/True.csv')}")
    print(f"  {os.path.join(DATA_DIR, 'liar/train.tsv')}")


if __name__ == "__main__":
    main()
