"""Import existing notebook results into MLflow BERT_Baselines experiment."""

import csv
import json
import os
import sys

import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def seed_optimal_results():
    json_path = os.path.join(RESULTS_DIR, "all_run_results.json")
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found, skipping optimal results seeding")
        return

    with open(json_path) as f:
        data = json.load(f)

    mlflow.set_experiment("BERT_Baselines_Optimal")
    total = 0
    for dataset_name, runs in data.items():
        for i, entry in enumerate(runs):
            with mlflow.start_run(run_name=f"optimal_{dataset_name}_run{i+1}"):
                mlflow.log_params(
                    {
                        "model": "OptimalFNDModel",
                        "dataset": dataset_name,
                        "seed": [42, 123, 456][i] if i < 3 else i,
                        "d_model": 192,
                        "num_heads": 6,
                        "num_layers": 4,
                    }
                )
                mlflow.log_metrics(
                    {
                        "test_acc": entry.get("test_acc", 0),
                        "test_f1": entry.get("test_f1", 0),
                        "test_precision": entry.get("test_prec", 0),
                        "test_recall": entry.get("test_rec", 0),
                        "val_acc": entry.get("val_acc", 0),
                    }
                )
            total += 1
    print(f"Seeded {total} optimal runs into MLflow")


def seed_ablation_results():
    for model_name in ["BERT", "RoBERTa", "DeBERTa", "DistilBERT"]:
        csv_path = os.path.join(RESULTS_DIR, f"ablation_{model_name}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue

        mlflow.set_experiment(f"BERT_Ablation_{model_name}")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            cfg = row.get("Configuration", "unknown")
            ds = row.get("Dataset", "")
            with mlflow.start_run(run_name=f"{model_name}_{cfg}_{ds}"):
                mlflow.log_params(
                    {
                        "model": model_name,
                        "configuration": row.get("Configuration", ""),
                        "dataset": row.get("Dataset", ""),
                    }
                )
                mlflow.log_metrics(
                    {
                        "test_acc": float(row.get("Test Accuracy", 0)),
                        "test_f1": float(row.get("Test F1", 0)),
                    }
                )
        print(f"Seeded {len(rows)} ablation runs for {model_name}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    print(f"Seeding results into MLflow at {MLFLOW_URI}")
    seed_optimal_results()
    seed_ablation_results()
    print("Done! Open MLflow UI at", MLFLOW_URI)
