"""Master experiment script: runs Setup A vs B vs C and saves comparison results."""

import json
import os
import sys

import mlflow
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.simulate_drift import create_drift_windows  # noqa: E402
from ml.dataset import load_isot, load_liar  # noqa: E402
from ml.retraining.setup_a import run_setup_a  # noqa: E402
from ml.retraining.setup_b import run_setup_b  # noqa: E402
from ml.retraining.setup_c import run_setup_c  # noqa: E402

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_DIR = os.getenv("DATA_DIR", ".")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)

    print("Loading datasets...")
    isot = load_isot(os.path.join(DATA_DIR, "data/isot"))
    liar = load_liar(os.path.join(DATA_DIR, "data/liar"))

    print("Creating 8-window drift simulation...")
    windows = create_drift_windows(isot, liar, window_size=2000)

    print("\n" + "=" * 60)
    print("Running Setup A: Static Model")
    print("=" * 60)
    results_a = run_setup_a(windows)

    print("\n" + "=" * 60)
    print("Running Setup B: Periodic Retraining (every 2 windows)")
    print("=" * 60)
    results_b = run_setup_b(windows, retrain_every=2)

    print("\n" + "=" * 60)
    print("Running Setup C: Drift-Triggered Retraining")
    print("=" * 60)
    results_c = run_setup_c(windows, drift_threshold=0.1)

    all_results = results_a + results_b + results_c
    df = pd.DataFrame(all_results)

    csv_path = os.path.join(RESULTS_DIR, "comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison results to {csv_path}")

    json_path = os.path.join(RESULTS_DIR, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    summary = df.groupby(["setup", "window"])["accuracy"].mean().unstack(level=0)
    print(summary.to_string())

    total_retrains = df.groupby("setup")["retrained"].sum()
    print("\nTotal retraining cycles:")
    print(total_retrains.to_string())

    avg_acc = df.groupby("setup")["accuracy"].mean()
    print("\nAverage accuracy under drift:")
    print(avg_acc.to_string())


if __name__ == "__main__":
    main()
