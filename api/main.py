from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from api.inference import registry
from api.metrics import MODEL_ACCURACY
from api.routers import admin, health, predict


def _seed_accuracy_metrics():
    """Seed MODEL_ACCURACY gauge with experiment results at startup."""
    # 3-setup comparison results (TF-IDF+LR, averaged across 8 drift windows)
    setup_results = {
        "static":          0.8251,   # Setup A — train once, never retrain
        "periodic":        0.8848,   # Setup B — retrain every 2 windows
        "drift_triggered": 0.8681,   # Setup C — retrain when KS > 0.1
    }
    for setup, acc in setup_results.items():
        MODEL_ACCURACY.labels(
            model="tfidf_lr",
            setup=setup,
            dataset="isot_liar_mixed",
        ).set(acc)

    # Transformer baseline results (OptimalFNDModel, from notebook)
    baseline_results = {
        "isot": 0.9973,   # 99.73% on clean ISOT dataset
        "liar": 0.5526,   # 55.26% on LIAR dataset
    }
    for dataset, acc in baseline_results.items():
        MODEL_ACCURACY.labels(
            model="bert_transformer",
            setup="baseline",
            dataset=dataset,
        ).set(acc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if registry.tokenizer is None:
        registry.load_tokenizer()
    if registry.drift_detector is None:
        registry.initialize_drift_detector()
    _seed_accuracy_metrics()
    yield


app = FastAPI(
    title="Fake News Detection API",
    description="Real-time fake news detection with drift monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(admin.router)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    return {
        "service": "Fake News Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "metrics": "/metrics",
    }
