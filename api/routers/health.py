from fastapi import APIRouter, Response

from api.inference import registry
from api.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", loaded_models=registry.loaded_models)


@router.get("/ready")
async def ready(response: Response):
    if registry.tokenizer is None:
        response.status_code = 503
        return {"status": "not_ready", "reason": "tokenizer not loaded"}
    return {"status": "ready", "loaded_models": registry.loaded_models}


@router.get("/models")
async def list_models():
    from ml.model import MODEL_CONFIGS

    return {
        "available": list(MODEL_CONFIGS.keys()),
        "loaded": registry.loaded_models,
    }
