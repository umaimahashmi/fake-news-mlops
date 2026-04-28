"""Setup A: Static model — trained once on window[0], never retrained."""

import time

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def run_setup_a(windows: list, experiment_name: str = "SetupA_Static") -> list:
    mlflow.set_experiment(experiment_name)
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    model = LogisticRegression(max_iter=1000, random_state=42)

    X_train = vectorizer.fit_transform(windows[0]["text"])
    model.fit(X_train, windows[0]["label"])

    results = []
    for i, window in enumerate(windows):
        t0 = time.time()
        X = vectorizer.transform(window["text"])
        preds = model.predict(X)
        latency_ms = (time.time() - t0) * 1000

        acc = accuracy_score(window["label"], preds)
        f1 = f1_score(window["label"], preds, average="weighted", zero_division=0)

        with mlflow.start_run(run_name=f"setup_a_window_{i}"):
            mlflow.log_params(
                {
                    "setup": "A",
                    "window_id": i,
                    "retrained": False,
                    "model_type": "tfidf_lr",
                }
            )
            mlflow.log_metrics(
                {"accuracy": acc, "f1_score": f1, "inference_latency_ms": latency_ms}
            )

        results.append(
            {
                "setup": "A",
                "window": i,
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "retrained": False,
                "latency_ms": round(latency_ms, 2),
            }
        )
        print(f"  [A] Window {i}: acc={acc:.4f}, f1={f1:.4f}, retrained=False")

    return results
