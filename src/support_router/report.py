"""Generate one evidence-based report from the pipeline's persisted artifacts."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ARTIFACTS, PROJECT_ROOT, REPORTS


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _tracking_summary() -> dict[str, Any]:
    """Return local MLflow inventory without making report generation depend on it."""
    summary: dict[str, Any] = {
        "uri": os.environ.get(
            "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
        ),
        "experiment": "support-routing",
        "runs": None,
        "traces": None,
        "versions": [],
        "champion_version": None,
    }
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=summary["uri"])
        experiment = client.get_experiment_by_name(summary["experiment"])
        if experiment is not None:
            summary["experiment_id"] = experiment.experiment_id
            summary["runs"] = len(client.search_runs([experiment.experiment_id]))
        for version in client.search_model_versions("name='support-router'"):
            model_name = None
            with suppress(Exception):
                model_name = client.get_run(version.run_id).data.params.get("model")
            summary["versions"].append(
                {
                    "version": str(version.version),
                    "run_id": version.run_id,
                    "model": model_name,
                    "description": version.description or "",
                }
            )
        try:
            champion = client.get_model_version_by_alias("support-router", "champion")
            summary["champion_version"] = str(champion.version)
        except Exception:
            pass
    except Exception as exc:  # tracking is informative, not required to read artifacts
        summary["error"] = f"{type(exc).__name__}: {exc}"

    db = PROJECT_ROOT / "mlflow.db"
    if db.exists():
        try:
            with sqlite3.connect(db) as connection:
                summary["traces"] = connection.execute(
                    "select count(*) from trace_info"
                ).fetchone()[0]
        except (sqlite3.Error, TypeError):
            pass
    return summary


def generate_report(out_path: Path | None = None) -> Path:
    """Assemble EDA, evaluation, timing, gate, and MLflow evidence into Markdown."""
    comparison = _read_json(REPORTS / "comparison.json", {"results": []})
    eda = _read_json(REPORTS / "eda_stats.json", {})
    latency = _read_json(REPORTS / "latency.json", {})
    gate = _read_json(REPORTS / "gate.json", {})
    meta = _read_json(ARTIFACTS / "model_meta.json", {})
    tracking = _tracking_summary()
    tracking_uri = str(tracking.get("uri", ""))
    display_tracking_uri = (
        "local SQLite store (`mlflow.db`)"
        if tracking_uri.startswith("sqlite:///")
        else f"`{tracking_uri}`"
    )

    results = comparison.get("results", [])
    grouped = sorted(
        (row for row in results if row.get("scheme") == "grouped"),
        key=lambda row: row.get("macro_f1", -1),
        reverse=True,
    )
    naive = {row["model"]: row for row in results if row.get("scheme") == "naive"}
    winner = grouped[0] if grouped else {}
    llm_rows = [row for row in grouped if row.get("model", "").startswith("llm_")]
    leakage = comparison.get("leakage", {})

    lines = [
        "# Support routing — end-to-end report",
        "",
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`support-router report`._",
        "",
        "This report is assembled from the artifacts used by evaluation and promotion; "
        "it does not recompute or hand-edit model scores.",
        "",
        "## Headline",
        "",
        "| measure | result |",
        "|---|---:|",
        f"| selected model | `{winner.get('model', meta.get('model_name', '—'))}` |",
        f"| grouped macro-F1 | **{_fmt(winner.get('macro_f1'))} ± "
        f"{_fmt(winner.get('macro_f1_std'))}** |",
        f"| `fraud-report` recall | **{_fmt(winner.get('critical_recall'))}** |",
        f"| rows / independent template groups | {eda.get('n_rows', '—')} / "
        f"{eda.get('n_template_groups', '—')} |",
        f"| local MLflow runs / traces | "
        f"{tracking.get('runs') if tracking.get('runs') is not None else '—'} / "
        f"{tracking.get('traces') if tracking.get('traces') is not None else '—'} |",
        "",
        "The selection score is repeated grouped cross-validation. A random row split is "
        "reported only as a leakage diagnostic, never as the model-selection estimate.",
        "",
        "## Data and leakage",
        "",
        eda.get("grouping_summary", "EDA artifacts were not found."),
        "",
        "| route | rows | template groups |",
        "|---|---:|---:|",
    ]
    for label, count in eda.get("row_counts", {}).items():
        lines.append(f"| `{label}` | {count} | {eda.get('template_counts', {}).get(label, '—')} |")

    naive_leak = leakage.get("naive", {})
    grouped_leak = leakage.get("grouped", {})
    lines += [
        "",
        f"Row imbalance is **{eda.get('row_imbalance_ratio', '—')}:1**, but template-level "
        f"imbalance is **{eda.get('template_imbalance_ratio', '—')}:1**. This is why class "
        "weighting is preferable to synthesizing more versions of duplicated templates.",
        "",
        "| split | rows with a near-duplicate in training | leak rate |",
        "|---|---:|---:|",
        f"| naive | {naive_leak.get('leaked_rows', '—')} / "
        f"{naive_leak.get('total_test_rows', '—')} | "
        f"{_pct(naive_leak.get('leak_rate'))} |",
        f"| grouped | {grouped_leak.get('leaked_rows', '—')} / "
        f"{grouped_leak.get('total_test_rows', '—')} | "
        f"{_pct(grouped_leak.get('leak_rate'))} |",
        "",
        "Full EDA: [EDA report](eda.md) · [class distribution](class_distribution.png) · "
        "[message lengths](length_distribution.png) · "
        "[template groups](template_group_sizes.png) · "
        "[word clouds](wordclouds.png) · [distinctive terms](distinctive_terms.png)",
        "",
        "## Model comparison",
        "",
        "| rank | model | grouped macro-F1 | std | fraud recall | accuracy | naive macro-F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(grouped, 1):
        naive_row = naive.get(row["model"], {})
        lines.append(
            f"| {rank} | `{row['model']}` | **{_fmt(row.get('macro_f1'))}** | "
            f"{_fmt(row.get('macro_f1_std'))} | {_fmt(row.get('critical_recall'))} | "
            f"{_fmt(row.get('accuracy'))} | {_fmt(naive_row.get('macro_f1'))} |"
        )
    lines += [
        "",
        "Detailed fold results and confusion matrices: [comparison.md](comparison.md) · "
        "[comparison.csv](comparison.csv) · [comparison.json](comparison.json)",
        "",
        "### Direct Qwen experiment",
        "",
        "Qwen 2.5 1.5B Instruct was evaluated through a local OpenAI-compatible Ollama "
        "endpoint using Apple Metal. It was evaluated by the same grouped folds; it was "
        "not used to select or generate labels for the training data.",
        "",
        "| prompt | macro-F1 | std | fraud recall | parse failures | CV predict time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in llm_rows:
        lines.append(
            f"| `{row['model']}` | {_fmt(row.get('macro_f1'))} | "
            f"{_fmt(row.get('macro_f1_std'))} | {_fmt(row.get('critical_recall'))} | "
            f"{row.get('parse_failures', 0)} "
            f"({_pct(row.get('parse_failure_rate'))}) | "
            f"{row.get('predict_seconds', 0):.3f}s |"
        )
    lines += [
        "",
        "The Qwen timing above is aggregate CV request time with deterministic response "
        "caching; it is not a clean online latency benchmark. The deployment decision is "
        "therefore based on accuracy, operating cost, and measured CPU winner latency.",
        "",
        "## Inference performance",
        "",
        "Measured on 60 unique messages after one warm-up:",
        "",
        "| measure | value |",
        "|---|---:|",
        f"| one-off load/warm-up | {_milliseconds(latency.get('warmup_seconds'))} |",
        f"| median / p95 / max | {_milliseconds_value(latency.get('median_ms'))} / "
        f"{_milliseconds_value(latency.get('p95_ms'))} / "
        f"{_milliseconds_value(latency.get('max_ms'))} |",
        f"| single-message throughput | {_rate(latency.get('throughput_per_second'))} |",
        f"| batched throughput | {_rate(latency.get('batch_throughput_per_second'))} |",
        "",
        "At 10,000 requests/minute (about 167/s), the selected embedding-plus-linear model "
        "fits comfortably in a horizontally scaled CPU service. See "
        "[DEPLOYMENT.md](../DEPLOYMENT.md) "
        "and the editable [architecture.drawio](../architecture.drawio).",
        "",
        "## MLflow evidence and registry",
        "",
        f"Tracking backend used for this report: {display_tracking_uri}. Experiment: "
        f"`{tracking.get('experiment')}` (ID `{tracking.get('experiment_id', '—')}`).",
        "",
        "| registered version | measured model | role |",
        "|---:|---|---|",
    ]
    for version in sorted(tracking.get("versions", []), key=lambda row: int(row["version"])):
        description = version["description"] or "model artifact"
        described_model = description.split(":", 1)[0]
        model_name = (
            described_model
            if described_model.startswith(("embedding_", "llm_"))
            else version.get("model") or described_model
        )
        role = (
            "champion"
            if version["version"] == tracking.get("champion_version")
            else "comparison"
        )
        lines.append(f"| {version['version']} | `{model_name}` | {role} |")

    gate_passed = gate.get("passed")
    verdict = "PASS" if gate_passed is True else "REJECTED" if gate_passed is False else "NOT RUN"
    gate_explanation = (
        "The configured candidate cleared the grouped-CV quality guardrails."
        if gate_passed is True
        else "The candidate must not replace the champion until every grouped-CV guardrail passes."
        if gate_passed is False
        else "Run `support-router gate` to produce a promotion verdict."
    )
    lines += [
        "",
        f"The latest promotion-gate verdict is **{verdict}**. {gate_explanation}",
        "",
        "## Scope and trade-offs",
        "",
        "Prioritized: leakage-resistant evaluation, macro-F1 plus an explicit fraud-recall "
        "guardrail, simple reproducible baselines, a callable prediction interface, batch "
        "holdout scoring, meaningful tests, and traceable experiment artifacts.",
        "",
        "Deliberately left out: transformer fine-tuning on only 80 independent templates, "
        "SMOTE over duplicated text, a broad hyperparameter search, production auth and "
        "multi-tenancy, and automatic deployment from the model registry.",
        "",
        "With more time: collect genuine tickets, calibrate confidence thresholds, add a "
        "human-review path for uncertain or high-risk decisions, monitor drift by route and "
        "confidence, and target labeling at the fraud/dispute boundary.",
        "",
        "The complete practice build took approximately **10–12 focused hours**. A strict "
        "three-hour version would stop after grouped CV, TF-IDF linear baselines, the "
        "prediction/batch-scoring interfaces, validation, and tests.",
        "",
        "## Required reasoning questions",
        "",
        "### 1. Why macro-F1 when `fraud-report` is highest-stakes?",
        "",
        "Accuracy is misleading here: predicting the majority `general` class gives 40% "
        "accuracy while finding no fraud. Macro-F1 gives each route equal weight and "
        "penalizes both missed tickets and bad routing. Because one average still cannot "
        "encode asymmetric harm, fraud recall is reported separately and enforced as a "
        "hard promotion floor. The operational extension is a calibrated threshold that "
        "sends ambiguous fraud-like cases to human review.",
        "",
        "### 2. How was class imbalance handled, and how would harm be detected?",
        "",
        "Linear classifiers use balanced class weights, folds preserve route balance while "
        "keeping template groups intact, and selection uses macro-F1. Resampling was avoided "
        "because most row imbalance comes from repeated templates. Harm would appear as low "
        "per-class recall/F1, especially fraud recall, a skewed confusion matrix, unstable "
        "fold results, or a changed prediction distribution in production.",
        "",
        "### 3. Which decision was uncertain?",
        "",
        "The uncertain decision was how aggressively to merge near-duplicate templates. Too "
        "low a similarity threshold can join genuinely different intents; too high a threshold "
        "leaks paraphrases across folds. Character 3–5-gram cosine similarity at 0.85 was a "
        "transparent compromise, followed by inspection of group size and the measured zero "
        "grouped leakage rate. With more data, the threshold would be sensitivity-tested.",
        "",
        "### 4. What changes at 10,000 requests/minute or with an LLM?",
        "",
        "At roughly 167 requests/s, the measured embedding-linear winner remains a CPU-first "
        "service: keep workers warm, batch embeddings briefly, cache normalized repeated text, "
        "and autoscale behind a load balancer. Pure TF-IDF linear/tree models are even cheaper "
        "but less accurate here. A generative LLM is appropriate when labels require broader "
        "context or change too quickly for retraining; it needs GPU serving, continuous batching, "
        "bounded tokens, prefix/KV caching, strict output validation, timeouts, and a fallback. "
        "A confidence-based cascade can reserve that cost for uncertain cases.",
    ]

    out_path = out_path or REPORTS / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _milliseconds(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds * 1000:.1f} ms"


def _milliseconds_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} ms"


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}/s"
