from fastapi import APIRouter, BackgroundTasks

from api.inference import registry
from api.metrics import RETRAINING_COUNTER
from api.models import DriftStatusResponse, RetrainResponse

router = APIRouter(prefix="/admin")


@router.get("/drift/status", response_model=DriftStatusResponse)
async def drift_status():
    detector = registry.drift_detector
    if detector is None:
        return DriftStatusResponse(
            drift_score=0.0, drift_detected=False, threshold=0.1, checked_at=None
        )
    return DriftStatusResponse(
        drift_score=round(detector.last_score, 4),
        drift_detected=detector.drift_detected,
        threshold=detector.threshold,
        checked_at=detector.last_check,
    )


@router.post("/retrain", response_model=RetrainResponse)
async def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_task)
    return RetrainResponse(
        status="accepted", message="Retraining scheduled in background"
    )


def _retrain_task():
    RETRAINING_COUNTER.labels(setup="manual", trigger="api").inc()
