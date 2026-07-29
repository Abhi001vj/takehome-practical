"""FastAPI service for single-message and batch classification."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import LABELS
from ..data import DataValidationError
from ..inference import (
    ModelNotTrainedError,
    Prediction,
    load_model,
    model_info,
    predict,
    predict_batch,
)
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    InfoResponse,
    PredictRequest,
    PredictResponse,
)

log = logging.getLogger("support_router.api")

#: Set by the container so one image can serve an artifact mounted anywhere.
MODEL_DIR_ENV = "SUPPORT_ROUTER_MODEL_DIR"

_state: dict[str, object] = {"loaded": False, "error": None}


def _model_dir() -> Path | None:
    raw = os.environ.get(MODEL_DIR_ENV)
    return Path(raw) if raw else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model once so the first request is not the one that pays for it."""
    try:
        load_model(_model_dir())
        # `load_model` only unpickles the trained head. An embedding artifact loads its
        # external encoder on the first prediction, so make one real call before marking
        # readiness true.
        predict("How are transaction fees calculated?", model_path=_model_dir())
        _state["loaded"] = True
        _state["error"] = None
        log.info("model loaded; ready to serve")
    except ModelNotTrainedError as exc:
        # Deliberately not fatal - see module docstring.
        _state["loaded"] = False
        _state["error"] = str(exc)
        log.warning("starting without a model: %s", exc)
    yield


def _as_response(p: Prediction) -> PredictResponse:
    return PredictResponse(label=p.label, confidence=p.confidence, scores=p.scores)


def _require_model() -> None:
    if not _state["loaded"]:
        raise HTTPException(status_code=503, detail=str(_state["error"] or "model not loaded"))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Support router",
        description=(
            "Routes a support message to one of four queues: "
            + ", ".join(LABELS)
            + ". See /docs for the interactive schema."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(DataValidationError)
    async def _bad_text(request: Request, exc: DataValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.middleware("http")
    async def _timing(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        """Liveness + readiness in one. `model_loaded=false` means do not send traffic."""
        return HealthResponse(
            status="ok" if _state["loaded"] else "degraded",
            model_loaded=bool(_state["loaded"]),
            detail=str(_state["error"]) if _state["error"] else None,
        )

    @app.get("/info", response_model=InfoResponse, tags=["ops"])
    async def info() -> InfoResponse:
        """What is actually deployed: model name, training date, and its CV numbers."""
        _require_model()
        meta = model_info(_model_dir())
        return InfoResponse(
            labels=list(LABELS),
            model_name=meta.get("model_name"),
            trained_at=meta.get("trained_at"),
            metrics={
                "cv_macro_f1": meta.get("cv_macro_f1"),
                "cv_macro_f1_std": meta.get("cv_macro_f1_std"),
                "cv_critical_recall": meta.get("cv_critical_recall"),
                "cv_scheme": meta.get("cv_scheme"),
            },
            metadata=meta,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict_one(req: PredictRequest) -> PredictResponse:
        """Route a single message."""
        _require_model()
        result = predict(req.text, model_path=_model_dir(), with_scores=True)
        return _as_response(result)

    @app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
    def predict_many(req: BatchPredictRequest) -> BatchPredictResponse:
        """Route many messages in one pass.

        Batching is not just convenience: the vectoriser and the embedding encoder are
        both markedly faster over a batch than over a loop of single calls.
        """
        _require_model()
        results = predict_batch(
            req.texts,
            model_path=_model_dir(),
            with_scores=True,
            skip_invalid=req.skip_invalid,
        )
        payload = [_as_response(r) if r is not None else None for r in results]
        return BatchPredictResponse(
            predictions=payload,
            n_invalid=sum(1 for r in payload if r is None),
        )

    return app


app = create_app()
