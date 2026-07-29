"""Public package interface for support-ticket classification."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

from .config import CRITICAL_LABEL, LABELS

__all__ = ["LABELS", "CRITICAL_LABEL", "__version__", "predict", "predict_batch"]

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .inference import predict, predict_batch


def __getattr__(name: str):
    """Expose inference functions without importing optional model backends eagerly."""
    if name in {"predict", "predict_batch"}:
        func = getattr(importlib.import_module(".inference", __name__), name)
        globals()[name] = func  # cache: __getattr__ only fires on a miss
        return func
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
