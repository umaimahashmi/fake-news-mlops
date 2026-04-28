"""Setup B: Periodic retraining — retrain every N windows regardless of drift."""

import time

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def run_setup_b(
    windows: list, retrain_every: int = 2, experiment_name: str = "SetupB_Periodic"
) -> list:
    mlflow.set_experiment(experiment_name)
    vectorizer = None
    model = None
    retrain_count = 0
    results = []

    for i, window in enumerate(windows):
        retrained = i % retrain_every == 0
        if retrained:
            vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
            model = LogisticRegression(max_iter=1000, random_state=42)
            X_train = vectorizer.fit_transform(window["text"])
            model.fit(X_train, window["label"])
            retrain_count += 1

        t0 = time.time()
        X = vectorizer.transform(window["text"])
        preds = model.predict(X)
        latency_ms = (time.time() - t0) * 1000

        acc = accuracy_score(window["label"], preds)
        f1 = f1_score(window["label"], preds, average="weighted", zero_division=0)

        with mlflow.start_run(run_name=f"setup_b_window_{i}"):
            mlflow.log_params(
                {
                    "setup": "B",
                    "window_id": i,
                    "retrain_every": retrain_every,
                    "retrained": retrained,
                    "retrain_count": retrain_count,
                    "model_type": "tfidf_lr",
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "inference_latency_ms": latency_ms,
                    "cumulative_retrains": retrain_count,
                }
            )

        results.append(
            {
                "setup": "B",
                "window": i,
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "retrained": retrained,
                "retrain_count": retrain_count,
                "latency_ms": round(latency_ms, 2),
            }
        )
        print(f"  [B] Window {i}: acc={acc:.4f}, f1={f1:.4f}, retrained={retrained}")

    return results
