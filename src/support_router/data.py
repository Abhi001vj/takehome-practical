"""Loading and validation for the ticket dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_RAW, LABEL_COL, LABELS, TEXT_COL

#: Rejects messages that carry no usable signal. Chosen from the training distribution:
#: the shortest genuine message is 8 words / 34 characters, so a 3-character floor
#: rejects junk without touching anything resembling a real ticket.
MIN_TEXT_CHARS = 3
#: Guards the API and batch scorer against pathological input. The longest training
#: message is 24 words; 10k characters is far above any real ticket and exists to stop a
#: single request from monopolising the vectoriser.
MAX_TEXT_CHARS = 10_000


class DataValidationError(ValueError):
    """Raised when an input file cannot be used at all."""


def validate_text(text: object) -> str:
    """Normalise and validate a single message.

    Returns the cleaned text, or raises `DataValidationError`. This is the one place
    that defines what "a scoreable message" means; the API, the CLI and the batch
    scorer all route through it so they cannot disagree.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        raise DataValidationError("text is missing")
    if not isinstance(text, str):
        raise DataValidationError(f"text must be a string, got {type(text).__name__}")

    cleaned = " ".join(text.split())
    if len(cleaned) < MIN_TEXT_CHARS:
        raise DataValidationError(
            f"text must contain at least {MIN_TEXT_CHARS} non-whitespace characters"
        )
    if len(cleaned) > MAX_TEXT_CHARS:
        raise DataValidationError(f"text exceeds the {MAX_TEXT_CHARS} character limit")
    return cleaned


def load_training_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the labelled training set, failing loudly on anything unexpected.

    Returns a frame with columns `text` and `label`, text normalised, index reset.
    """
    path = Path(path) if path is not None else DATA_RAW
    if not path.exists():
        raise DataValidationError(
            f"training data not found at {path}. The repository must include "
            "data/raw/train.csv."
        )

    df = pd.read_csv(path)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise DataValidationError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )

    unknown = set(df[LABEL_COL].unique()) - set(LABELS)
    if unknown:
        raise DataValidationError(
            f"{path} contains labels outside the four known routes: {sorted(unknown)}"
        )

    bad_rows: list[tuple[int, str]] = []
    cleaned: list[str] = []
    for idx, value in df[TEXT_COL].items():
        try:
            cleaned.append(validate_text(value))
        except DataValidationError as exc:
            bad_rows.append((int(idx), str(exc)))
            cleaned.append("")
    if bad_rows:
        preview = "; ".join(f"row {i}: {msg}" for i, msg in bad_rows[:5])
        raise DataValidationError(
            f"{path} has {len(bad_rows)} unusable text value(s). First few: {preview}"
        )

    df[TEXT_COL] = cleaned
    return df[[TEXT_COL, LABEL_COL]].reset_index(drop=True)


def load_messages_for_scoring(
    path: Path | str, text_column: str = TEXT_COL
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a CSV of messages to score.

    Unlike the training path this does not raise on individual bad rows, because a
    holdout file we did not author may contain them and dropping the whole file would be
    worse than reporting them. Returns `(scoreable, rejected)`; `rejected` carries a
    `reason` column and the original index so nothing disappears silently.
    """
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"input file not found: {path}")

    # skip_blank_lines=False is load-bearing: pandas drops wholly-empty rows by default,
    # which would shrink the output and silently misalign every row after the
    # gap. An empty row must survive to be reported as a reject.
    df = pd.read_csv(path, skip_blank_lines=False)
    if text_column not in df.columns:
        raise DataValidationError(
            f"{path} has no '{text_column}' column. Found: {sorted(df.columns)}. "
            "Pass --text-column to point at the right one."
        )
    if df.empty:
        raise DataValidationError(f"{path} contains no rows")

    rows, reasons = [], []
    for idx, value in df[text_column].items():
        try:
            rows.append((int(idx), validate_text(value)))
        except DataValidationError as exc:
            reasons.append((int(idx), value, str(exc)))

    scoreable = pd.DataFrame(rows, columns=["source_index", TEXT_COL])
    rejected = pd.DataFrame(reasons, columns=["source_index", "raw_value", "reason"])
    return scoreable, rejected


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Label counts in the fixed `LABELS` order (missing classes reported as 0)."""
    return df[LABEL_COL].value_counts().reindex(LABELS).fillna(0).astype(int)
