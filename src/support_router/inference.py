"""The serving interface: `predict(text) -> label`.

The model is loaded once per process and cached. `predict` is safe to call from multiple
threads once loaded (the underlying estimators are read-only at inference time), which is
what lets the FastAPI app serve it without a lock.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import ARTIFACTS, LABELS
from .data import DataValidationError, validate_text
from .train import METADATA_FILENAME, MODEL_FILENAME

_LOCK = threading.Lock()
_MODEL: Any | None = None
_META: dict | None = None
_LOADED_FROM: Path | None = None


class ModelNotTrainedError(RuntimeError):
    """Raised when no artifact exists yet - actionable rather than a bare KeyError."""


@dataclass(frozen=True)
class Prediction:
    """A routing decision, with confidence when the model can supply one."""

    label: str
    confidence: float | None = None
    scores: dict[str, float] | None = None

    def as_dict(self) -> dict:
        return {"label": self.label, "confidence": self.confidence, "scores": self.scores}


def load_model(path: Path | None = None, force: bool = False) -> tuple[Any, dict]:
    """Load and cache the trained artifact."""
    global _MODEL, _META, _LOADED_FROM

    directory = Path(path) if path is not None else ARTIFACTS
    model_path = directory / MODEL_FILENAME

    with _LOCK:
        if _MODEL is not None and not force and model_path == _LOADED_FROM:
            return _MODEL, _META or {}

        if not model_path.exists():
            raise ModelNotTrainedError(
                f"no trained model at {model_path}. Train one first:\n"
                "    support-router train --model embedding_logreg"
            )

        _MODEL = joblib.load(model_path)
        meta_path = directory / METADATA_FILENAME
        _META = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        _LOADED_FROM = model_path
        return _MODEL, _META


def _score_texts(model: Any, texts: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (labels, probabilities-or-None).

    Not every model can produce probabilities - LinearSVC has no `predict_proba` and the
    generative classifier has no meaningful notion of one - so confidence is optional
    throughout rather than faked from a decision function.
    """
    labels = np.asarray(model.predict(texts))
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(texts))
        except Exception:
            proba = None
    return labels, proba


def _classes(model: Any) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("clf"), "classes_", None)
    return list(classes) if classes is not None else list(LABELS)


def predict(text: str, model_path: Path | None = None, with_scores: bool = False):
    """Classify a single support message into one of the four routes.

    Args:
        text: the raw message.
        model_path: directory holding `model.joblib`; defaults to `artifacts/`.
        with_scores: return a `Prediction` with confidence instead of a bare string.

    Returns:
        The route name, or a `Prediction` when `with_scores=True`.

    Raises:
        DataValidationError: the text is empty, not a string, or too long.
        ModelNotTrainedError: no artifact has been trained yet.
    """
    cleaned = validate_text(text)
    model, _ = load_model(model_path)

    labels, proba = _score_texts(model, [cleaned])
    label = str(labels[0])

    if not with_scores:
        return label

    if proba is None:
        return Prediction(label=label)

    classes = _classes(model)
    scores = {str(c): float(p) for c, p in zip(classes, proba[0], strict=True)}
    return Prediction(label=label, confidence=float(np.max(proba[0])), scores=scores)


def predict_batch(
    texts: Sequence[str],
    model_path: Path | None = None,
    with_scores: bool = False,
    skip_invalid: bool = False,
) -> list:
    """Classify many messages in one pass.

    Batching matters: the vectoriser and the embedding encoder are both far more
    efficient over a batch than over a loop of single calls.

    With `skip_invalid=True`, unusable rows yield `None` in the output at their original
    position, so the caller can align results with inputs. Otherwise the first bad row
    raises.
    """
    cleaned: list[str] = []
    positions: list[int] = []
    results: list[Any] = [None] * len(texts)

    for i, raw in enumerate(texts):
        try:
            cleaned.append(validate_text(raw))
            positions.append(i)
        except DataValidationError:
            if not skip_invalid:
                raise

    if not cleaned:
        return results

    model, _ = load_model(model_path)
    labels, proba = _score_texts(model, cleaned)
    classes = _classes(model) if proba is not None else None

    for j, position in enumerate(positions):
        if not with_scores:
            results[position] = str(labels[j])
        elif proba is None:
            results[position] = Prediction(label=str(labels[j]))
        else:
            results[position] = Prediction(
                label=str(labels[j]),
                confidence=float(np.max(proba[j])),
                scores={str(c): float(p) for c, p in zip(classes, proba[j], strict=True)},
            )
    return results


def model_info(model_path: Path | None = None) -> dict:
    """Metadata for the loaded model - used by the API's /info endpoint."""
    _, meta = load_model(model_path)
    return meta


def reset_cache() -> None:
    """Drop the cached model. Used by tests that train into a temp directory."""
    global _MODEL, _META, _LOADED_FROM
    with _LOCK:
        _MODEL = _META = _LOADED_FROM = None
