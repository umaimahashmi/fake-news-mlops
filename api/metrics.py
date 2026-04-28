from prometheus_client import Counter, Gauge, Histogram

PREDICTION_COUNTER = Counter(
    "fakenews_predictions_total",
    "Total predictions made",
    ["model", "label"],
)

INFERENCE_LATENCY = Histogram(
    "fakenews_inference_duration_seconds",
    "Inference latency per request",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

DRIFT_SCORE = Gauge(
    "fakenews_drift_score",
    "Current KS drift score vs training distribution",
)

RETRAINING_COUNTER = Counter(
    "fakenews_retraining_total",
    "Number of retraining cycles completed",
    ["setup", "trigger"],
)

MODEL_ACCURACY = Gauge(
    "fakenews_model_accuracy",
    "Latest test accuracy",
    ["model", "setup", "dataset"],
)

ACTIVE_MODELS = Gauge(
    "fakenews_active_models",
    "Number of currently loaded models",
)

REQUEST_ERRORS = Counter(
    "fakenews_request_errors_total",
    "Total failed prediction requests",
    ["model", "error_type"],
)
