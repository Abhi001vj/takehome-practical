"""Fit, evaluate, serialize, and optionally register a classifier."""

from __future__ import annotations

import contextlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn

from . import __version__
from .config import ARTIFACTS, LABELS, PROJECT_ROOT, load_params
from .cv import cross_validate, make_splits
from .data import load_training_data
from .features import feature_spec
from .grouping import assign_groups
from .models import get_builder
from .tracking import log_cv_result, log_dataset, log_feature_spec, register_model, run

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "model_meta.json"


@dataclass
class ModelMetadata:
    """Travels with the artifact so a served model can always explain itself."""

    model_name: str
    labels: list[str]
    trained_at: str
    n_training_rows: int
    n_template_groups: int
    cv_macro_f1: float
    cv_macro_f1_std: float
    cv_critical_recall: float
    cv_scheme: str
    support_router_version: str
    sklearn_version: str
    python_version: str
    params: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def train(
    model: str = "embedding_logreg",
    model_params: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    data_path: Path | None = None,
    params_path: Path | None = None,
    evaluate: bool = True,
    metrics_path: Path | None = None,
    track: bool = True,
    register: bool = False,
) -> tuple[Any, ModelMetadata]:
    """Fit, evaluate, persist, and optionally register a model."""
    params = load_params(params_path)
    out_dir = out_dir or ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(data_path)
    texts, labels = df["text"].tolist(), df["label"].tolist()

    grouping = assign_groups(
        texts,
        similarity_threshold=params.grouping["similarity_threshold"],
        char_ngram_range=tuple(params.grouping["char_ngram_range"]),
    )

    builder = get_builder(model)
    kwargs: dict[str, Any] = {"seed": params["seed"], **(model_params or {})}
    if model.startswith("embedding_"):
        kwargs.setdefault("model_name", params.llm["embedding_model"])
    elif model.startswith("llm_"):
        from .models.llm import load_llm_config

        kwargs.update(load_llm_config(params.llm))

    if evaluate and model.startswith("llm_"):
        health = builder(**kwargs).health_check()
        if not health.get("ok"):
            raise RuntimeError(
                "LLM endpoint preflight failed: " + str(health.get("error", health))
            )

    # Evaluate before the final refit, so the reported numbers come from held-out folds
    # and never from the model we are about to ship.
    cv_macro_f1 = cv_std = cv_recall = float("nan")
    cv_result = None
    if evaluate:
        splits, _ = make_splits(
            texts, labels, scheme="grouped", n_splits=params.cv["n_splits"],
            n_repeats=params.cv["n_repeats"], seed=params["seed"], grouping=grouping,
        )
        cv_result = cross_validate(
            lambda: builder(**kwargs), texts, labels, name=model,
            scheme="grouped", splits=splits, grouping=grouping, params=_jsonable(kwargs),
        )
        cv_macro_f1 = cv_result.macro_f1
        cv_std = cv_result.macro_f1_std
        cv_recall = cv_result.critical_recall
    elif metrics_path is not None:
        measured = _load_grouped_metrics(metrics_path, model)
        cv_macro_f1 = measured["macro_f1"]
        cv_std = measured["macro_f1_std"]
        cv_recall = measured["critical_recall"]

    estimator = builder(**kwargs)
    estimator.fit(texts, labels)

    metadata = ModelMetadata(
        model_name=model,
        labels=list(LABELS),
        trained_at=datetime.now(UTC).isoformat(),
        n_training_rows=len(df),
        n_template_groups=grouping.n_groups,
        cv_macro_f1=cv_macro_f1,
        cv_macro_f1_std=cv_std,
        cv_critical_recall=cv_recall,
        cv_scheme="grouped",
        support_router_version=__version__,
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
        params=_jsonable(kwargs),
    )

    model_path = out_dir / MODEL_FILENAME
    joblib.dump(estimator, model_path)
    (out_dir / METADATA_FILENAME).write_text(metadata.to_json())

    if track:
        with run(f"train-{model}", tags={"model": model, "phase": "train"}):
            import mlflow

            if mlflow.active_run() is not None:
                mlflow.log_params(
                    {
                        "model": model,
                        **{
                            f"p_{key}": value
                            for key, value in _jsonable(kwargs).items()
                        },
                    }
                )
                if metrics_path is not None and not evaluate:
                    mlflow.log_param("evaluation_source", Path(metrics_path).name)
                    mlflow.set_tag("evaluation_reused", "true")
                mlflow.log_metric("cv_macro_f1", cv_macro_f1)
                mlflow.log_metric("cv_macro_f1_std", cv_std)
                mlflow.log_metric("cv_critical_recall", cv_recall)
                # The gate reads `macro_f1`; log under that name too so a training run
                # and a CV run are directly comparable.
                mlflow.log_metric("macro_f1", cv_macro_f1)
                mlflow.log_metric("critical_recall", cv_recall)
                mlflow.log_metric("macro_f1__std", cv_std)

                if cv_result is not None:
                    log_cv_result(cv_result)
                log_dataset(data_path or _default_data_path())
                _log_feature_spec_if_available(estimator)

                model_info = mlflow.sklearn.log_model(
                    estimator,
                    name="model",
                    serialization_format="cloudpickle",
                    code_paths=[str(PROJECT_ROOT / "src")],
                )
                mlflow.log_artifact(str(model_path), artifact_path="model_files")
                mlflow.log_artifact(
                    str(out_dir / METADATA_FILENAME), artifact_path="model_files"
                )

                if register:
                    version = register_model(
                        model_info.model_uri,
                        params.promotion["registered_model_name"],
                        description=(
                            f"{model}: grouped-CV macro-F1 {cv_macro_f1:.4f} "
                            f"± {cv_std:.4f}, fraud recall {cv_recall:.4f}"
                        ),
                    )
                    mlflow.set_tag("registered_version", version)

    return estimator, metadata


def _load_grouped_metrics(path: Path, model: str) -> dict[str, float]:
    """Read an already-produced grouped result for registry-only final fitting.

    This is useful for direct LLMs: their CV evaluation can take minutes, while fitting
    the deployable wrapper only selects the few-shot examples. The source is explicit and
    is logged as both a parameter and a tag so reused measurements cannot be mistaken for
    a fresh evaluation.
    """
    payload = json.loads(Path(path).read_text())
    results = payload.get("results", [])
    matches = [
        row for row in results
        if row.get("model") == model and row.get("scheme") == "grouped"
    ]
    if not matches:
        raise ValueError(f"{path} contains no grouped result for {model!r}")
    row = matches[0]
    return {
        "macro_f1": float(row["macro_f1"]),
        "macro_f1_std": float(row.get("macro_f1_std", 0.0)),
        "critical_recall": float(row["critical_recall"]),
    }


def _default_data_path() -> Path:
    from .config import DATA_RAW

    return DATA_RAW


def _log_feature_spec_if_available(estimator: Any) -> None:
    """Record the featurisation contract when the model exposes a vectoriser."""
    vec = None
    if hasattr(estimator, "named_steps") and "features" in getattr(estimator, "named_steps", {}):
        vec = estimator.named_steps["features"]
    elif getattr(estimator, "_vec", None) is not None:
        vec = estimator._vec

    if vec is not None:
        # Feature-spec logging is documentation, not part of the artifact: a tracking
        # outage must not fail a training run that otherwise succeeded.
        with contextlib.suppress(Exception):
            log_feature_spec(feature_spec(vec))
    elif hasattr(estimator, "model_name"):
        # Embedding models have no vocabulary; the encoder id *is* the feature contract.
        log_feature_spec(
            {"type": "sentence_embedding", "encoder": estimator.model_name, "normalised": True}
        )


def _jsonable(obj: dict) -> dict:
    return {
        k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
        for k, v in obj.items()
    }
