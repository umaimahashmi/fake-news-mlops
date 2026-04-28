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
