"""HTTP request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import LABELS


class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw support message to route.")

    model_config = {
        "json_schema_extra": {
            "example": {"text": "Someone withdrew 2 BTC from my account and I never authorised it"}
        }
    }


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)
    skip_invalid: bool = Field(
        default=False,
        description=(
            "Return null at the position of any unusable message instead of failing the "
            "whole request. Positions always align with the input list."
        ),
    )


class PredictResponse(BaseModel):
    label: str = Field(..., description=f"One of: {', '.join(LABELS)}")
    confidence: float | None = Field(
        default=None,
        description=(
            "Max class probability. Null when the model has no `predict_proba` "
            "(e.g. LinearSVC) - absent rather than faked from a decision function."
        ),
    )
    scores: dict[str, float] | None = None


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse | None]
    n_invalid: int = Field(..., description="Count of positions returned as null.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: str | None = None


class InfoResponse(BaseModel):
    labels: list[str]
    model_name: str | None = None
    trained_at: str | None = None
    metrics: dict | None = None
    metadata: dict | None = None
