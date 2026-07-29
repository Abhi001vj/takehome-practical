"""Run comparable cross-validation experiments and persist their results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CRITICAL_LABEL, DATA_RAW, REPORTS, load_params
from .cv import CVResult, cross_validate, leakage_report, make_splits
from .data import class_distribution, load_training_data
from .features import feature_spec
from .grouping import assign_groups
from .metrics import format_confusion
from .models import LLM_MODELS, get_builder, resolve_names
from .tracking import log_cv_result, log_dataset, log_feature_spec, run, setup


@dataclass
class ComparisonReport:
    results: list[CVResult] = field(default_factory=list)
    leakage: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame([r.to_row() for r in self.results])
        ordered = df.sort_values(["scheme", "macro_f1"], ascending=[True, False])
        return ordered.reset_index(drop=True)

    def best(self, scheme: str = "grouped") -> CVResult | None:
        candidates = [r for r in self.results if r.scheme == scheme]
        return max(candidates, key=lambda r: r.macro_f1) if candidates else None


def run_comparison(
    models: str | list[str] = "all",
    schemes: list[str] | None = None,
    n_splits: int | None = None,
    n_repeats: int | None = None,
    seed: int | None = None,
    data_path: Path | None = None,
    params_path: Path | None = None,
    track: bool = True,
) -> ComparisonReport:
    """Score every requested model under every requested scheme.

    Folds are built once per scheme and shared by all models, so any difference in the
    table is attributable to the model.
    """
    params = load_params(params_path)
    schemes = schemes or params.cv["schemes"]
    n_splits = n_splits or params.cv["n_splits"]
    n_repeats = n_repeats or params.cv["n_repeats"]
    seed = seed if seed is not None else params["seed"]
    tracking_enabled = track and setup() is not None

    df = load_training_data(data_path)
    texts, labels = df["text"].tolist(), df["label"].tolist()
    names = resolve_names(models)

    grouping = assign_groups(
        texts,
        similarity_threshold=params.grouping["similarity_threshold"],
        char_ngram_range=tuple(params.grouping["char_ngram_range"]),
    )

    report = ComparisonReport(
        leakage=leakage_report(texts, labels, n_splits=n_splits, seed=seed),
        dataset={
            "n_rows": len(df),
            "class_distribution": class_distribution(df).to_dict(),
            "n_template_groups": grouping.n_groups,
            "grouping_summary": grouping.summary(),
            # The row-level imbalance overstates the real one: `general` has 160 rows but
            # only ~23 distinct templates. At the template level the classes are far
            # closer to balanced, which is why aggressive resampling is not warranted.
            "group_class_distribution": _group_class_distribution(df, grouping),
        },
    )

    llm_config = params.llm
    for scheme in schemes:
        splits, _ = make_splits(
            texts, labels, scheme=scheme, n_splits=n_splits,
            n_repeats=n_repeats, seed=seed, grouping=grouping,
        )
        for name in names:
            try:
                builder = get_builder(name)
            except (KeyError, ImportError) as exc:
                report.errors[name] = str(exc)
                continue

            kwargs: dict[str, Any] = {"seed": seed}
            if name in LLM_MODELS and name.startswith("llm_"):
                from .models.llm import load_llm_config

                kwargs.update(load_llm_config(llm_config))
            elif name.startswith("embedding_"):
                kwargs["model_name"] = llm_config["embedding_model"]

            if name.startswith("llm_"):
                health = builder(**kwargs).health_check()
                if not health.get("ok"):
                    report.errors[f"{name}[{scheme}]"] = (
                        "LLM endpoint preflight failed: " + str(health.get("error", health))
                    )
                    continue

            trace_llm = tracking_enabled and name.startswith("llm_")
            if trace_llm:
                # A full repeated-CV sweep can emit thousands of request traces before
                # the asynchronous exporter drains its default queue.
                os.environ.setdefault("MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE", "10000")
                os.environ.setdefault("MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS", "4")
                setup()
                import mlflow.openai

                mlflow.openai.autolog(log_traces=True, silent=True)

            try:
                result = cross_validate(
                    lambda b=builder, k=kwargs: b(**k),
                    texts, labels, name=name, scheme=scheme,
                    splits=splits, grouping=grouping, params=_jsonable(kwargs),
                )
            except Exception as exc:
                report.errors[f"{name}[{scheme}]"] = f"{type(exc).__name__}: {exc}"
                continue
            finally:
                if trace_llm:
                    mlflow.openai.autolog(disable=True, silent=True)

            report.results.append(result)
            if tracking_enabled:
                tags = {"model": name, "cv_scheme": scheme, "phase": "cv"}
                if name.startswith("llm_"):
                    tags.update(
                        {
                            "llm_model": str(kwargs.get("model")),
                            "llm_endpoint": str(kwargs.get("base_url")),
                            "inference_backend": os.environ.get(
                                "LLM_BACKEND", "openai-compatible"
                            ),
                        }
                    )
                with run(f"{name}-{scheme}", tags=tags):
                    log_cv_result(result)
                    log_dataset(Path(data_path) if data_path is not None else DATA_RAW)
                    log_feature_spec(_feature_spec_for_run(name, builder, kwargs))

    return report


def _feature_spec_for_run(name: str, builder, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Describe the input representation used by a comparison run."""
    if name.startswith("embedding_"):
        return {
            "type": "sentence_embedding",
            "encoder": kwargs.get("model_name"),
            "normalised": True,
            "head": name.removeprefix("embedding_"),
        }
    if name.startswith("llm_"):
        return {
            "type": "generative_llm",
            "model": kwargs.get("model"),
            "endpoint": kwargs.get("base_url"),
            "mode": name.removeprefix("llm_"),
        }

    estimator = builder(**kwargs)
    if hasattr(estimator, "named_steps") and "features" in estimator.named_steps:
        spec = feature_spec(estimator.named_steps["features"])
        spec["fitted_vocabulary_logged_on"] = "final_training_run"
        return spec
    return {"type": "none", "reason": "reference baseline has no text features"}


def _group_class_distribution(df: pd.DataFrame, grouping) -> dict[str, int]:
    """Class counts at the template level rather than the row level."""
    seen: dict[int, str] = {}
    for label, group in zip(df["label"], grouping.groups, strict=True):
        seen.setdefault(int(group), label)
    counts: dict[str, int] = {}
    for label in seen.values():
        counts[label] = counts.get(label, 0) + 1
    return counts


def _jsonable(obj: Any) -> dict:
    out = {}
    for k, v in obj.items():
        out[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    return out


def write_report(
    report: ComparisonReport,
    out_dir: Path | None = None,
    append: bool = False,
) -> Path:
    """Emit comparison.md / comparison.csv / comparison.json."""
    out_dir = out_dir or REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = report.to_frame()
    errors = dict(report.errors)
    if append and (out_dir / "comparison.json").exists():
        existing = json.loads((out_dir / "comparison.json").read_text())
        existing_frame = pd.DataFrame(existing.get("results", []))
        if not existing_frame.empty:
            frame = pd.concat([existing_frame, frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=["model", "scheme"], keep="last")
            frame = frame.sort_values(
                ["scheme", "macro_f1"], ascending=[True, False]
            ).reset_index(drop=True)
        errors = {**existing.get("errors", {}), **errors}
    frame.to_csv(out_dir / "comparison.csv", index=False)

    payload = {
        "dataset": report.dataset,
        "leakage": report.leakage,
        "results": frame.to_dict(orient="records"),
        "errors": errors,
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, default=str))

    md = _render_markdown(report, frame)
    path = out_dir / "comparison.md"
    path.write_text(md)
    return path


def _render_markdown(report: ComparisonReport, frame: pd.DataFrame) -> str:
    lines: list[str] = ["# Model comparison", ""]

    ds = report.dataset
    lines += [
        "## Dataset",
        "",
        f"- Rows: **{ds.get('n_rows')}**",
        f"- Independent template groups: **{ds.get('n_template_groups')}**",
        f"- {ds.get('grouping_summary', '')}",
        "",
        "| label | rows | templates |",
        "|---|---:|---:|",
    ]
    rows = ds.get("class_distribution", {})
    groups = ds.get("group_class_distribution", {})
    for label in rows:
        lines.append(f"| {label} | {rows[label]} | {groups.get(label, 0)} |")
    lines += [
        "",
        "The row counts imply a 3.2:1 imbalance; the template counts imply roughly "
        "1.6:1. Most of the apparent imbalance is duplication, not class rarity.",
        "",
    ]

    leak = report.leakage
    if leak:
        lines += [
            "## Leakage",
            "",
            "| scheme | test rows with a near-duplicate in train | rate |",
            "|---|---:|---:|",
        ]
        for scheme in ("naive", "grouped"):
            if scheme in leak:
                entry = leak[scheme]
                lines.append(
                    f"| {scheme} | {entry['leaked_rows']} / {entry['total_test_rows']} "
                    f"| {entry['leak_rate']:.1%} |"
                )
        lines += [
            "",
            "Under `naive` (plain StratifiedKFold) almost every test row has a template "
            "sibling in train, so its scores are inflated and do not transfer to the "
            "hidden holdout. `grouped` is zero by construction. **Select models on the "
            "grouped numbers.**",
            "",
        ]

    if not frame.empty:
        for scheme in frame["scheme"].unique():
            sub = frame[frame["scheme"] == scheme]
            lines += [
                f"## Results ({scheme} CV)",
                "",
                "| model | macro-F1 | ±std | fraud recall | accuracy | bal. acc | fit s | pred s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for _, r in sub.iterrows():
                lines.append(
                    f"| `{r['model']}` | **{r['macro_f1']:.3f}** | {r['macro_f1_std']:.3f} "
                    f"| {r['critical_recall']:.3f} | {r['accuracy']:.3f} "
                    f"| {r['balanced_accuracy']:.3f} | {r['fit_seconds']:.2f} "
                    f"| {r['predict_seconds']:.3f} |"
                )
            lines.append("")

    best = report.best("grouped")
    if best is not None:
        lines += [
            "## Best model (grouped CV)",
            "",
            f"`{best.name}` - macro-F1 {best.macro_f1:.3f} ± {best.macro_f1_std:.3f}, "
            f"{CRITICAL_LABEL} recall {best.critical_recall:.3f}",
            "",
            "Pooled confusion matrix (rows = truth, columns = prediction):",
            "",
            "```",
            format_confusion(best.scores.confusion),
            "```",
            "",
        ]

    if report.errors:
        lines += ["## Skipped", ""]
        for name, err in report.errors.items():
            lines.append(f"- `{name}`: {err}")
        lines.append("")

    return "\n".join(lines)
