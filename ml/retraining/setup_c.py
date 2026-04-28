"""Setup C: Drift-triggered retraining — retrain only when drift is detected."""

import time

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ml.drift_detector import DriftDetector


def run_setup_c(
    windows: list,
    drift_threshold: float = 0.1,
    experiment_name: str = "SetupC_DriftTriggered",
) -> list:
    mlflow.set_experiment(experiment_name)
    detector = DriftDetector(windows[0]["text"].tolist(), threshold=drift_threshold)
    vectorizer = None
    model = None
    retrain_count = 0
    results = []

    for i, window in enumerate(windows):
        drift_info = detector.check_drift(window["text"].tolist())
        retrained = drift_info["drift_detected"] or i == 0

        if retrained:
            vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
            model = LogisticRegression(max_iter=1000, random_state=42)
            X_train = vectorizer.fit_transform(window["text"])
            model.fit(X_train, window["label"])
            detector.update_reference(window["text"].tolist())
            retrain_count += 1

        t0 = time.time()
        X = vectorizer.transform(window["text"])
        preds = model.predict(X)
        latency_ms = (time.time() - t0) * 1000

        acc = accuracy_score(window["label"], preds)
        f1 = f1_score(window["label"], preds, average="weighted", zero_division=0)

        with mlflow.start_run(run_name=f"setup_c_window_{i}"):
            mlflow.log_params(
                {
                    "setup": "C",
                    "window_id": i,
                    "drift_threshold": drift_threshold,
                    "retrained": retrained,
                    "retrain_count": retrain_count,
                    "model_type": "tfidf_lr",
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "drift_score_ks": drift_info["drift_score"],
                    "inference_latency_ms": latency_ms,
                    "cumulative_retrains": retrain_count,
                }
            )

        results.append(
            {
                "setup": "C",
                "window": i,
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "drift_score": round(drift_info["drift_score"], 4),
                "retrained": retrained,
                "retrain_count": retrain_count,
                "latency_ms": round(latency_ms, 2),
            }
        )
        print(
            f"  [C] Window {i}: acc={acc:.4f}, f1={f1:.4f}, "
            f"drift={drift_info['drift_score']:.4f}, retrained={retrained}"
        )

    return results
