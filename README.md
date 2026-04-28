# Real-Time Fake News Detection MLOps Pipeline

[![CI](https://github.com/umaimahashmi/fake-news-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/umaimahashmi/fake-news-mlops/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-orange.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)

> **Research Question:** Does drift-aware retraining improve fake-news detection accuracy when misinformation patterns evolve over time?

This repository accompanies the paper *"Drift-Aware MLOps for Fake News Detection: A Three-Setup Comparison Under Distributional Shift"*. It extends the transformer-based fake news detector from Rout et al. (2025) with a full production-grade MLOps pipeline including real-time inference, data drift monitoring, automated retraining, and experiment tracking.

---

## Key Results

| Setup | Strategy | Avg Accuracy | Retraining Cycles |
|-------|----------|:------------:|:-----------------:|
| A | Static (train once, never retrain) | ~65% | 0 |
| B | Periodic (every 2 windows) | ~82% | 4 |
| **C** | **Drift-triggered (KS test > 0.1)** | **~88%** | **2** |

**Setup C maintains ~88% accuracy under distributional shift with 2× fewer retraining cycles than Setup B.**

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               Docker Compose Stack                  │
│                                                     │
│  [ISOT + LIAR Data]                                 │
│        ↓                                            │
│  [TF-IDF Vectorizer] → [KS Drift Detector]         │
│                               ↓                    │
│  [Setup A/B/C Retraining] → [ModelRegistry]        │
│                               ↓                    │
│              [FastAPI :8000] (/predict /health)     │
│                    ↓                               │
│  [Prometheus :9090] → [Grafana :3000]              │
│  [MLflow :5000]                                    │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
fake-news-mlops/
├── api/                        # FastAPI service
│   ├── main.py                 # App entry point + lifespan
│   ├── inference.py            # ModelRegistry (lazy-load 4 models)
│   ├── metrics.py              # Prometheus counters/histograms/gauges
│   ├── models.py               # Pydantic request/response schemas
│   └── routers/
│       ├── predict.py          # POST /predict
│       ├── health.py           # GET /health, /ready, GET /models
│       └── admin.py            # GET /admin/drift/status, POST /admin/retrain
├── ml/
│   ├── model.py                # OptimalFNDModel transformer architecture
│   ├── dataset.py              # FakeNewsDataset, data loaders
│   ├── train.py                # Training loop with MLflow logging
│   ├── drift_detector.py       # TF-IDF + Kolmogorov-Smirnov drift detection
│   └── retraining/
│       ├── setup_a.py          # Static model (no retraining)
│       ├── setup_b.py          # Periodic retraining (every N windows)
│       └── setup_c.py          # Drift-triggered retraining
├── experiments/
│   ├── run_comparison.py       # Master script: A vs B vs C → CSV + MLflow
│   ├── simulate_drift.py       # 8-window ISOT→LIAR distribution mixing
│   └── plot_results.py         # Generate paper figures from results CSV
├── monitoring/
│   ├── prometheus.yml          # Scrape config
│   └── grafana/                # Dashboard provisioning JSON
├── scripts/
│   ├── download_data.py        # Dataset download helper
│   ├── train_bert_models.py    # Train all 4 model variants, save .pt files
│   └── seed_mlflow.py          # Import notebook results into MLflow
├── tests/
│   ├── conftest.py             # tiny_model + mock_tokenizer fixtures
│   ├── test_api.py             # FastAPI endpoint tests
│   ├── test_drift.py           # DriftDetector unit tests
│   └── test_model.py           # Forward pass smoke tests
├── results/                    # Generated figures + CSVs (gitignored binaries)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Docker Desktop (running)
- Python 3.11+
- ISOT dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) → place `Fake.csv` and `True.csv` in `data/isot/`

### 1. Start the full stack

```bash
docker compose up -d
```

| Service | URL | Description |
|---------|-----|-------------|
| FastAPI | http://localhost:8000 | Inference API + Swagger docs |
| MLflow | http://localhost:5000 | Experiment tracking UI |
| Prometheus | http://localhost:9090 | Metrics scraping |
| Grafana | http://localhost:3000 | Live dashboard (admin/admin) |

### 2. Run a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "COVID vaccine causes autism according to leaked documents", "model": "bert"}'
```

**Response:**
```json
{
  "label": "fake",
  "confidence": 0.94,
  "model_used": "bert",
  "inference_time_ms": 142,
  "drift_score": 0.07
}
```

### 3. Check API health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/models
```

### 4. View live drift status

```bash
curl http://localhost:8000/admin/drift/status
```

---

## Running Experiments

### Download data

```bash
python scripts/download_data.py
```

Place `Fake.csv` and `True.csv` (ISOT) in `data/isot/` manually from Kaggle.

### Train model weights

```bash
python scripts/train_bert_models.py
```

Trains all 4 variants (`bert`, `roberta`, `deberta`, `distilbert`) and saves `.pt` files to `models/`. Requires ISOT data. Takes ~30 min on CPU.

### Run the 3-setup comparison

```bash
python experiments/run_comparison.py
```

Simulates 8 data windows (ISOT→LIAR distribution shift) and runs Setup A, B, C. Saves results to `results/comparison_results.csv` and logs all runs to MLflow.

### Generate paper figures

```bash
python experiments/plot_results.py
```

Outputs `fig1_accuracy_over_windows.png`, `fig2_drift_vs_accuracy.png`, `fig3_retraining_efficiency.png` to `results/`.

### Seed MLflow with baseline results

```bash
python scripts/seed_mlflow.py
```

Imports BERT baseline results from `results/all_run_results.json` into MLflow `BERT_Baselines_Optimal` experiment.

---

## Model Architecture

The **OptimalFNDModel** is a custom lightweight transformer (7.28M parameters) — it does **not** use pre-trained HuggingFace weights. Only the `BertTokenizer` is borrowed for tokenization.

```
Input Text
    ↓
BertTokenizer (vocab_size=30,522)
    ↓
TokenEmbedding + PositionalEncoding
    ↓
N × FNDTransformerLayer
    ├── EnhancedMultiHeadAttention
    └── LightweightFFN
    ↓
CLS pooling (or mean pooling)
    ↓
Linear → Softmax → {fake, real}
```

**Four variants** share the same class with different configs:

| Variant | d_model | Heads | Layers | d_ff |
|---------|--------:|------:|-------:|-----:|
| bert | 192 | 6 | 4 | 512 |
| roberta | 128 | 4 | 3 | 384 |
| deberta | 128 | 4 | 3 | 384 |
| distilbert | 128 | 4 | 3 | 384 |

---

## Drift Detection

```
Reference corpus (window 0)
    ↓
TfidfVectorizer(max_features=50)
    ↓
For each new window:
    KS statistic per TF-IDF feature → mean KS score
    drift_detected = mean_ks > 0.1 (threshold)
```

**Drift simulation** — 8 windows of ISOT→LIAR mixing:

| Windows | ISOT % | LIAR % | Drift Level |
|---------|-------:|-------:|-------------|
| 1–3 | 100% | 0% | Stable baseline |
| 4–5 | 75% | 25% | Mild drift |
| 6–7 | 50% | 50% | Moderate drift |
| 8 | 25% | 75% | Severe drift |

---

## Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fakenews_predictions_total` | Counter | Predictions by label + model |
| `fakenews_inference_latency_seconds` | Histogram | Latency by model |
| `fakenews_drift_score` | Gauge | Current KS drift score |
| `fakenews_retraining_total` | Counter | Retraining events by setup |
| `fakenews_model_accuracy` | Gauge | Accuracy by model + setup |
| `fakenews_active_models` | Gauge | Number of loaded models |
| `fakenews_request_errors_total` | Counter | Errors by model + type |

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use a tiny model (d_model=16) so they run in seconds without GPU.

---

## CI/CD

**CI** runs on every push and pull request:
1. `black` + `isort` — auto-format check
2. `flake8` — lint (max line length 100)
3. `pytest` — full test suite
4. Docker Compose smoke test — `GET /health` must return 200

**CD** runs on merge to `main`:
- Builds and pushes Docker image to GitHub Container Registry (`ghcr.io`)

---

## Datasets

| Dataset | Samples | Classes | Source |
|---------|--------:|---------|--------|
| ISOT | ~44,000 | fake / real | [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| LIAR | ~12,800 | 6-class → binarized | [UCSB](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) |

LIAR binarization: `pants-fire / false / barely-true → fake`, `half-true / mostly-true / true → real`

---

## References

- Rout, J. K., et al. (2025). *Fake News Detection Using Transformer-Based Models*. Journal of Cybersecurity and Privacy.
- Ahmed, H., et al. (2017). *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques*. INISTA.
- Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection*. ACL.

---

## Author

**Umaima Hashmi**
Fake News Detection MLOps Pipeline — Track-II Research Paper, 2026
