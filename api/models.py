from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="News article text to classify")
    model: Literal["bert", "roberta", "deberta", "distilbert"] = Field(
        "bert", description="Model variant to use for inference"
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    label: Literal["fake", "real"]
    confidence: float
    model_used: str
    inference_time_ms: float
    drift_score: float


class HealthResponse(BaseModel):
    status: str
    loaded_models: list


class DriftStatusResponse(BaseModel):
    drift_score: float
    drift_detected: bool
    threshold: float
    checked_at: Optional[str]


class RetrainResponse(BaseModel):
    status: str
    message: str
