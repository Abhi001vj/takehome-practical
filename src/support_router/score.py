"""Score message CSVs while preserving input order and rejected rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import LABEL_COL, TEXT_COL
from .data import DataValidationError, load_messages_for_scoring
from .inference import predict_batch
from .metrics import format_confusion, pooled_confusion, score_fold

#: Checked in order when `--text-column` is not given.
CANDIDATE_TEXT_COLUMNS = (TEXT_COL, "message", "body", "ticket", "content", "description")


def detect_text_column(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise DataValidationError(
                f"column {explicit!r} not found. Available: {sorted(df.columns)}"
            )
        return explicit

    for candidate in CANDIDATE_TEXT_COLUMNS:
        if candidate in df.columns:
            return candidate

    # A single-column file is unambiguous whatever it is called.
    if len(df.columns) == 1:
        return str(df.columns[0])

    raise DataValidationError(
        f"could not find a text column among {sorted(df.columns)}. "
        f"Pass --text-column explicitly."
    )


def score_file(
    input_path: Path | str,
    output_path: Path | str,
    model_path: Path | None = None,
    text_column: str | None = None,
    with_confidence: bool = False,
    rejects_path: Path | str | None = None,
) -> dict:
    """Score a CSV and write predictions. Returns a summary dict."""
    input_path, output_path = Path(input_path), Path(output_path)

    # See the note in data.load_messages_for_scoring: blank rows must be preserved so
    # the output has exactly as many rows as the input.
    raw = pd.read_csv(input_path, skip_blank_lines=False)
    if raw.empty:
        raise DataValidationError(f"{input_path} contains no rows")
    column = detect_text_column(raw, text_column)

    scoreable, rejected = load_messages_for_scoring(input_path, text_column=column)
    if scoreable.empty:
        raise DataValidationError(
            f"none of the {len(raw)} rows in {input_path} contained usable text"
        )

    predictions = predict_batch(
        scoreable[TEXT_COL].tolist(), model_path=model_path, with_scores=with_confidence
    )

    out = raw.copy()
    out["predicted_label"] = pd.NA
    if with_confidence:
        out["confidence"] = pd.NA

    for source_index, prediction in zip(scoreable["source_index"], predictions, strict=True):
        if with_confidence:
            out.at[source_index, "predicted_label"] = prediction.label
            out.at[source_index, "confidence"] = prediction.confidence
        else:
            out.at[source_index, "predicted_label"] = prediction

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "rows_in": int(len(raw)),
        "rows_scored": int(len(scoreable)),
        "rows_rejected": int(len(rejected)),
        "text_column": column,
        "label_counts": out["predicted_label"].value_counts(dropna=True).to_dict(),
    }

    if not rejected.empty:
        rejects = Path(rejects_path) if rejects_path else output_path.with_suffix(".rejects.csv")
        rejected.to_csv(rejects, index=False)
        summary["rejects_file"] = str(rejects)

    # If the file is labelled, this doubles as an evaluation run.
    if LABEL_COL in raw.columns:
        truth = out.loc[scoreable["source_index"], LABEL_COL]
        pred = out.loc[scoreable["source_index"], "predicted_label"]
        mask = truth.notna() & pred.notna()
        if mask.any():
            fold = score_fold(truth[mask].to_numpy(), pred[mask].to_numpy())
            confusion = pooled_confusion(truth[mask].to_numpy(), pred[mask].to_numpy())
            summary["evaluation"] = {
                "macro_f1": fold.macro_f1,
                "accuracy": fold.accuracy,
                "balanced_accuracy": fold.balanced_accuracy,
                "critical_recall": fold.critical_recall,
                "fraud_leak_rate": fold.fraud_leak_rate,
                "per_class_f1": fold.per_class_f1,
                "per_class_recall": fold.per_class_recall,
                "support": fold.support,
            }
            summary["confusion_matrix"] = format_confusion(confusion)

            metrics_path = output_path.with_suffix(".metrics.json")
            metrics_path.write_text(json.dumps(summary["evaluation"], indent=2))
            summary["metrics_file"] = str(metrics_path)

    return summary
