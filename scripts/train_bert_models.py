"""Train all 4 model variants on ISOT and save .pt weight files."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformers import BertTokenizer  # noqa: E402

from ml.dataset import load_isot, make_loaders, make_splits  # noqa: E402
from ml.model import MODEL_CONFIGS, OptimalFNDModel  # noqa: E402
from ml.train import set_seed, train_model  # noqa: E402

DATA_DIR = os.getenv("DATA_DIR", ".")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EPOCHS = int(os.getenv("TRAIN_EPOCHS", "5"))
MAX_LEN = 64
BATCH_SIZE = 16


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    set_seed(42)

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("Loading ISOT dataset...")
    isot = load_isot(os.path.join(DATA_DIR, "data/isot"))
    train_df, val_df, test_df = make_splits(isot, max_samples=5000)
    train_loader, val_loader, test_loader = make_loaders(
        train_df, val_df, test_df, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE
    )

    for model_name, cfg in MODEL_CONFIGS.items():
        out_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
        if os.path.exists(out_path):
            print(f"  {model_name}: already exists at {out_path}, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Training {model_name}: {cfg}")
        print(f"{'='*50}")

        model = OptimalFNDModel(**cfg)
        results = train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            num_epochs=EPOCHS,
            lr=3e-4,
            patience=3,
            seed=42,
            verbose=True,
            device="cpu",
            mlflow_run=False,
        )
        torch.save(model.state_dict(), out_path)
        print(f"  Saved {out_path} | Test Acc: {results['test_acc']:.4f}")

    print("\nAll models trained and saved!")


if __name__ == "__main__":
    main()
