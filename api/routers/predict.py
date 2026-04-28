import time

from fastapi import APIRouter, HTTPException

from api.inference import registry
from api.metrics import INFERENCE_LATENCY, PREDICTION_COUNTER, REQUEST_ERRORS
from api.models import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    model_name = req.model
    try:
        t0 = time.perf_counter()
        label, confidence = registry.predict(req.text, model_name)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        PREDICTION_COUNTER.labels(model=model_name, label=label).inc()
        INFERENCE_LATENCY.labels(model=model_name).observe(elapsed_ms / 1000)
        drift_score = registry.get_drift_score()

        return PredictResponse(
            label=label,
            confidence=round(confidence, 4),
            model_used=model_name,
            inference_time_ms=round(elapsed_ms, 2),
            drift_score=round(drift_score, 4),
        )
    except ValueError as e:
        REQUEST_ERRORS.labels(model=model_name, error_type="invalid_model").inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        REQUEST_ERRORS.labels(model=model_name, error_type="inference_error").inc()
        raise HTTPException(status_code=500, detail=str(e))
