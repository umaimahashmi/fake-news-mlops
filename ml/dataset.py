import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SEED = 42


class FakeNewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_loaders(train_df, val_df, test_df, tokenizer, max_len=128, batch_size=32):
    train_ds = FakeNewsDataset(train_df, tokenizer, max_len)
    val_ds = FakeNewsDataset(val_df, tokenizer, max_len)
    test_ds = FakeNewsDataset(test_df, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def load_isot(data_dir="data/isot"):
    dfs = []
    for fname, lbl in [
        ("Fake.csv", 0),
        ("True.csv", 1),
        ("fake.csv", 0),
        ("true.csv", 1),
    ]:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, on_bad_lines="skip")
            text_col = (
                "text"
                if "text" in df.columns
                else "title" if "title" in df.columns else df.columns[0]
            )
            df = df[[text_col]].dropna()
            df.columns = ["text"]
            df["label"] = lbl
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No ISOT files found in {data_dir}")
    combined = pd.concat(dfs, ignore_index=True)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 5]
    return combined


def load_liar(data_dir="data/liar"):
    splits = []
    for fname in ["train.tsv", "valid.tsv", "test.tsv"]:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, sep="\t", header=None, on_bad_lines="skip")
            splits.append(df)
    if not splits:
        raise FileNotFoundError(f"No LIAR files found in {data_dir}")
    combined = pd.concat(splits, ignore_index=True)
    text_col, label_col = (
        (2, 1) if combined.shape[1] >= 3 else (combined.shape[1] - 1, 0)
    )
    df = combined[[label_col, text_col]].copy()
    df.columns = ["label_str", "text"]
    real = {"true", "mostly-true", "half-true"}
    fake = {"barely-true", "false", "pants-fire", "pants on fire"}

    def map_label(x):
        x = str(x).lower().strip()
        if x in real:
            return 1
        if x in fake:
            return 0
        return None

    df["label"] = df["label_str"].apply(map_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 5][["text", "label"]]
    return df


def make_splits(df, train_ratio=0.7, val_ratio=0.15, seed=SEED, max_samples=None):
    df = df.dropna()
    if max_samples and len(df) > max_samples:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_samples // 2), random_state=seed)
        )
    df = df.reset_index(drop=True)
    try:
        train, temp = train_test_split(
            df, test_size=1 - train_ratio, stratify=df["label"], random_state=seed
        )
    except Exception:
        train, temp = train_test_split(df, test_size=1 - train_ratio, random_state=seed)
    val_size = val_ratio / (1 - train_ratio)
    val, test = train_test_split(temp, test_size=1 - val_size, random_state=seed)
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
