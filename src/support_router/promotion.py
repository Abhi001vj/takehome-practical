"""Evaluate candidate metrics against the champion and release guardrails."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CRITICAL_LABEL, load_params

METRIC_KEYS = ("macro_f1", "critical_recall", "macro_f1__std")


def _load_candidate(
    metrics_path: Path | None,
    model: str | None,
    params_path: Path | None = None,
) -> dict[str, Any]:
    """Get candidate metrics from a JSON file, or by running grouped CV on `model`."""
    if metrics_path is not None:
        data = json.loads(Path(metrics_path).read_text())
        # Accept either a flat metrics dict or a `support-router cv` comparison.json.
        if "results" in data:
            grouped = [r for r in data["results"] if r.get("scheme") == "grouped"]
            if not grouped:
                raise ValueError(f"{metrics_path} contains no grouped-CV results")
            candidate_name = model or load_params(params_path)["train"]["model"]
            matches = [row for row in grouped if row.get("model") == candidate_name]
            if model is None and not matches and len(grouped) == 1:
                # A focused one-model comparison is unambiguous. Multi-model CI always
                # resolves the explicitly configured candidate instead of choosing the
                # highest row and accidentally gating a different model.
                matches = grouped
            if not matches:
                raise ValueError(
                    f"{metrics_path} contains no grouped-CV result for {candidate_name!r}"
                )
            best = matches[0]
            return {
                "name": best["model"],
                "macro_f1": best["macro_f1"],
                "critical_recall": best["critical_recall"],
                "macro_f1__std": best.get("macro_f1_std", 0.0),
            }
        return {
            "name": data.get("model", data.get("model_name", "candidate")),
            "macro_f1": data.get("macro_f1", data.get("cv_macro_f1")),
            "critical_recall": data.get("critical_recall", data.get("cv_critical_recall")),
            "macro_f1__std": data.get("macro_f1__std", data.get("cv_macro_f1_std", 0.0)),
        }

    if model is None:
        raise ValueError("provide either --candidate-metrics or --model")

    from .cv import cross_validate, make_splits
    from .data import load_training_data
    from .grouping import assign_groups
    from .models import get_builder

    params = load_params()
    df = load_training_data()
    texts, labels = df["text"].tolist(), df["label"].tolist()
    grouping = assign_groups(
        texts,
        similarity_threshold=params.grouping["similarity_threshold"],
        char_ngram_range=tuple(params.grouping["char_ngram_range"]),
    )
    splits, _ = make_splits(
        texts, labels, scheme="grouped", n_splits=params.cv["n_splits"],
        n_repeats=params.cv["n_repeats"], seed=params["seed"], grouping=grouping,
    )
    builder = get_builder(model)
    kwargs = {"seed": params["seed"]}
    if model.startswith("embedding_"):
        kwargs["model_name"] = params.llm["embedding_model"]

    result = cross_validate(
        lambda: builder(**kwargs), texts, labels, name=model,
        scheme="grouped", splits=splits, grouping=grouping,
    )
    return {
        "name": model,
        "macro_f1": result.macro_f1,
        "critical_recall": result.critical_recall,
        "macro_f1__std": result.macro_f1_std,
    }


def evaluate_candidate(
    metrics_path: Path | None = None,
    model: str | None = None,
    params_path: Path | None = None,
) -> dict[str, Any]:
    """Run every gate check and return a structured verdict."""
    from .tracking import get_champion_metrics

    params = load_params(params_path)
    promo = params.promotion

    candidate = _load_candidate(metrics_path, model, params_path)
    if candidate.get("macro_f1") is None:
        raise ValueError("candidate metrics contain no macro_f1")

    champion = get_champion_metrics(promo["registered_model_name"], METRIC_KEYS)

    checks: list[dict[str, Any]] = []

    # Check 1: the critical-route floor applies whether or not a champion exists.
    critical = candidate.get("critical_recall")
    floor = promo["critical_recall_floor"]
    checks.append(
        {
            "name": f"{CRITICAL_LABEL}_recall_floor",
            "passed": critical is not None and critical >= floor,
            "detail": (
                f"{CRITICAL_LABEL} recall {critical:.4f} vs floor {floor:.4f}"
                if critical is not None
                else "candidate reported no critical recall"
            ),
        }
    )

    if champion is None or champion.get("macro_f1") is None:
        checks.append(
            {
                "name": "champion_comparison",
                "passed": True,
                "detail": "no registered champion - first model promotes automatically",
            }
        )
    else:
        # Ordinary code changes may reproduce the current champion. A different model
        # must clear the configured practical-improvement margin.
        delta = candidate["macro_f1"] - champion["macro_f1"]
        if champion.get("name") == candidate.get("name"):
            checks.append(
                {
                    "name": "champion_non_regression",
                    "passed": delta >= -1e-9,
                    "detail": (
                        f"same model macro-F1 {candidate['macro_f1']:.4f} vs benchmark "
                        f"{champion['macro_f1']:.4f} (delta {delta:+.4f})"
                    ),
                }
            )
        else:
            checks.append(
                {
                    "name": "macro_f1_improvement",
                    "passed": delta > promo["min_improvement"],
                    "detail": (
                        f"macro-F1 {candidate['macro_f1']:.4f} vs champion "
                        f"{champion['macro_f1']:.4f} (delta {delta:+.4f}, "
                        f"required > {promo['min_improvement']:.4f})"
                    ),
                }
            )

        # Check 3: do not regress the critical route relative to the champion.
        champion_recall = champion.get("critical_recall")
        if champion_recall is not None and critical is not None:
            checks.append(
                {
                    "name": "no_critical_regression",
                    "passed": critical >= champion_recall - 1e-9,
                    "detail": (
                        f"{CRITICAL_LABEL} recall {critical:.4f} vs champion "
                        f"{champion_recall:.4f}"
                    ),
                }
            )

        # Check 4: the win must not be an artifact of a much noisier model.
        champion_std = champion.get("macro_f1__std")
        candidate_std = candidate.get("macro_f1__std")
        if champion_std and candidate_std is not None:
            ratio = candidate_std / champion_std if champion_std > 0 else 1.0
            checks.append(
                {
                    "name": "variance_not_inflated",
                    "passed": ratio <= promo["max_std_ratio"],
                    "detail": (
                        f"macro-F1 std {candidate_std:.4f} vs champion {champion_std:.4f} "
                        f"(ratio {ratio:.2f}, max {promo['max_std_ratio']:.2f})"
                    ),
                }
            )

    return {
        "passed": all(c["passed"] for c in checks),
        "candidate": candidate,
        "champion": champion,
        "checks": checks,
        "policy": dict(promo),
    }


def render_verdict(verdict: dict[str, Any]) -> str:
    """Human- and CI-readable summary (rendered into the PR comment)."""
    lines: list[str] = []
    status = "PASS" if verdict["passed"] else "FAIL"
    lines.append(f"## Model promotion gate: {status}")
    lines.append("")

    cand = verdict["candidate"]
    lines.append(
        f"**Candidate** `{cand.get('name')}` - macro-F1 {cand['macro_f1']:.4f} "
        f"(± {cand.get('macro_f1__std', 0.0):.4f}), "
        f"{CRITICAL_LABEL} recall {cand.get('critical_recall', float('nan')):.4f}"
    )

    champ = verdict.get("champion")
    if champ:
        lines.append(
            f"**Champion** v{champ.get('version')} - macro-F1 "
            f"{champ.get('macro_f1', float('nan')):.4f}, "
            f"{CRITICAL_LABEL} recall {champ.get('critical_recall', float('nan')):.4f}"
        )
    else:
        lines.append("**Champion** none registered yet")

    lines += ["", "| check | result | detail |", "|---|---|---|"]
    for check in verdict["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"| `{check['name']}` | {mark} | {check['detail']} |")

    if not verdict["passed"]:
        lines += [
            "",
            "The candidate did not clear the gate, so it must not replace the champion. "
            "Selecting on grouped CV is deliberate: a naive split would show a much "
            "higher number that does not survive contact with the holdout.",
        ]
    return "\n".join(lines)
