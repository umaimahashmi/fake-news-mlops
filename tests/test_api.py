import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tiny_model, mock_tokenizer):
    from api import inference as inf_module
    from api import main

    registry = inf_module.registry
    registry.tokenizer = mock_tokenizer
    registry._models = {"bert": tiny_model}
    registry.drift_detector = None

    from api.metrics import ACTIVE_MODELS

    ACTIVE_MODELS.set(1)

    with TestClient(main.app) as c:
        yield c


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_predict_returns_valid_schema(client):
    resp = client.post(
        "/predict", json={"text": "Scientists discover vaccine", "model": "bert"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in ("fake", "real")
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_used"] == "bert"
    assert data["inference_time_ms"] >= 0


def test_predict_missing_text_returns_422(client):
    resp = client.post("/predict", json={"model": "bert"})
    assert resp.status_code == 422


def test_predict_invalid_model_returns_400(client):
    resp = client.post("/predict", json={"text": "test article", "model": "gpt4"})
    assert resp.status_code == 422


def test_ready_endpoint(client):
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_models_list(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    assert "bert" in data["available"]


def test_drift_status(client):
    resp = client.get("/admin/drift/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "drift_score" in data
    assert "drift_detected" in data
