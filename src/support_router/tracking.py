"""MLflow experiment tracking, dataset lineage, artifacts, and model registry helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import LABELS, PROJECT_ROOT, REPORTS
from .metrics import format_confusion

DEFAULT_EXPERIMENT = "support-routing"
log = logging.getLogger(__name__)


def _tracking_uri() -> str:
    """Use the configured tracking server or a repository-local SQLite store."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def git_metadata() -> dict[str, str]:
    """Commit and dirty-state, so a run can be traced back to reproducible source."""
    def _run(*args: str) -> str:
        try:
            return subprocess.run(
                args, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=False
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "git_commit": _run("git", "rev-parse", "HEAD"),
        "git_branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(_run("git", "status", "--porcelain"))),
    }


def setup(experiment: str = DEFAULT_EXPERIMENT):
    """Configure MLflow, returning None when tracking is unavailable.

    Model evaluation must remain usable when an optional local tracking server is down.
    The exception is recorded once in logs instead of manufacturing a successful MLflow
    run or aborting the scientific workflow.
    """
    if not mlflow_available():
        return None
    import mlflow

    try:
        mlflow.set_tracking_uri(_tracking_uri())
        mlflow.set_experiment(experiment)
    except Exception as exc:
        log.warning("MLflow unavailable at %s: %s", _tracking_uri(), exc)
        return None
    return mlflow


@contextmanager
def run(name: str, experiment: str = DEFAULT_EXPERIMENT, nested: bool = False, tags=None):
    """Start an MLflow run, degrading to a no-op if MLflow is not installed.

    The no-op path keeps the scientific workflow independent from the tracking extra.
    """
    mlflow = setup(experiment)
    if mlflow is None:
        yield None
        return

    with mlflow.start_run(run_name=name, nested=nested) as active:
        mlflow.set_tags({**git_metadata(), "python": platform.python_version(), **(tags or {})})
        yield active


def log_cv_result(result, extra_params: dict[str, Any] | None = None) -> None:
    """Log one `CVResult`: params, aggregate metrics, per-class metrics, confusion matrix."""
    mlflow = setup()
    if mlflow is None:
        return

    mlflow.log_params(
        {
            "model": result.name,
            "cv_scheme": result.scheme,
            "n_folds": result.scores.n_folds,
            **{f"p_{k}": v for k, v in (result.params or {}).items()},
            **(extra_params or {}),
        }
    )

    for key, value in result.scores.mean.items():
        mlflow.log_metric(key, value)
    for key, value in result.scores.std.items():
        mlflow.log_metric(f"{key}__std", value)
    mlflow.log_metric("fit_seconds", result.fit_seconds)
    mlflow.log_metric("predict_seconds", result.predict_seconds)
    mlflow.log_metric("leakage_rows", result.leakage_rows)
    mlflow.log_metric("parse_failures", result.parse_failures)
    mlflow.log_metric(
        "parse_failure_rate",
        result.parse_failures / result.prediction_rows if result.prediction_rows else 0.0,
    )
    mlflow.log_metric("prediction_rows", result.prediction_rows)

    mlflow.log_text(format_confusion(result.scores.confusion), f"confusion_{result.scheme}.txt")
    mlflow.log_dict(
        {
            "labels": list(LABELS),
            "matrix": result.scores.confusion.tolist(),
            "per_fold": [f.as_dict() for f in result.scores.per_fold],
        },
        f"fold_detail_{result.scheme}.json",
    )
    if result.grouping is not None:
        mlflow.log_dict(
            {
                "summary": result.grouping.summary(),
                "n_groups": result.grouping.n_groups,
                "n_exact_groups": result.grouping.n_exact_groups,
                "largest_group": result.grouping.largest_group,
                "compression": result.grouping.compression,
            },
            "grouping.json",
        )


def log_feature_spec(spec: dict[str, Any], name: str = "feature_spec.json") -> None:
    """Record the featurisation contract for this run. See the module docstring."""
    mlflow = setup()
    if mlflow is None:
        (REPORTS / name).parent.mkdir(parents=True, exist_ok=True)
        (REPORTS / name).write_text(json.dumps(spec, indent=2))
        return
    mlflow.log_dict(spec, name)


def log_dataset(path: Path, name: str = "train") -> None:
    """Attach dataset lineage to the active run.

    This is what `mlflow.data` genuinely provides: a content digest and source pointer,
    which makes the data side of a run inspectable and reproducible.
    """
    mlflow = setup()
    if mlflow is None:
        return
    try:
        import pandas as pd

        resolved = path.resolve()
        try:
            source = str(resolved.relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            source = resolved.name
        dataset = mlflow.data.from_pandas(pd.read_csv(path), name=name, targets="label")
        mlflow.log_input(dataset, context="training")
        mlflow.set_tag("dataset_path", source)
    except Exception as exc:  # pragma: no cover - lineage is best-effort, never fatal
        mlflow.set_tag("dataset_logging_error", str(exc))


def register_model(model_uri: str, registered_name: str, description: str = "") -> str | None:
    """Register a model version and return its version string."""
    mlflow = setup()
    if mlflow is None:
        return None
    from mlflow.tracking import MlflowClient

    result = mlflow.register_model(model_uri=model_uri, name=registered_name)
    if description:
        MlflowClient().update_model_version(
            name=registered_name, version=result.version, description=description
        )
    return result.version


def set_champion(registered_name: str, version: str) -> None:
    """Mark a version as the champion using an alias.

    Aliases replace the deprecated stage API in MLflow 2.9+; `@champion` is what the
    serving path and the CI gate both resolve.
    """
    mlflow = setup()
    if mlflow is None:
        raise RuntimeError("MLflow is unavailable; the champion alias was not changed")
    from mlflow.tracking import MlflowClient

    MlflowClient().set_registered_model_alias(registered_name, "champion", version)


def get_champion_metrics(registered_name: str, metric_keys: tuple[str, ...]) -> dict | None:
    """Fetch the current champion's metrics, or None if there is no champion yet.

    A checked-in grouped-CV benchmark is the fallback on clean CI runners, whose local
    MLflow registry starts empty. An actual registry alias takes precedence when present.
    """
    mlflow = setup()
    if mlflow is None:
        return _benchmark_fallback()
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=_tracking_uri())
    try:
        version = client.get_model_version_by_alias(registered_name, "champion")
    except Exception:
        return _benchmark_fallback()

    try:
        run_data = client.get_run(version.run_id).data
    except Exception:
        return _benchmark_fallback()

    return {
        "version": version.version,
        "run_id": version.run_id,
        "name": run_data.params.get("model"),
        **{k: run_data.metrics.get(k) for k in metric_keys},
    }


def _benchmark_fallback() -> dict | None:
    """Load the reviewed champion benchmark when registry state is unavailable."""
    path = PROJECT_ROOT / "conf" / "champion.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
