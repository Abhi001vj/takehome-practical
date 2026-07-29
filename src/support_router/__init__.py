"""Route a customer support message to one of four support queues.

    >>> from support_router import predict
    >>> predict("Someone withdrew funds I never authorised")
    'fraud-report'

The serving functions live in `support_router.inference`, deliberately *not* in a module called
`support_router.predict`. A module and a function of the same name inside one package cannot
coexist: importing the submodule binds the module object onto the package, clobbering the
function, so `from support_router import predict` would return a module or a function depending
on which import ran first. Naming the module `inference` removes the ambiguity rather
than arbitrating it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

from .config import CRITICAL_LABEL, LABELS

__all__ = ["LABELS", "CRITICAL_LABEL", "__version__", "predict", "predict_batch"]

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .inference import predict, predict_batch


def __getattr__(name: str):
    """Expose the serving functions at package level, lazily.

    Lazy because `support_router.inference` imports `support_router.train`, which imports the model
    registry, which imports lightgbm/xgboost/catboost. `import support_router` must work without
    those installed - the CLI's `--help` path and the API both depend on it, and the
    serving image installs no tree extras at all.
    """
    if name in {"predict", "predict_batch"}:
        func = getattr(importlib.import_module(".inference", __name__), name)
        globals()[name] = func  # cache: __getattr__ only fires on a miss
        return func
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
