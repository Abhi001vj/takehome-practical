"""Classification metrics with fixed label ordering and fraud-specific guardrails."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import CRITICAL_LABEL, LABELS

PRIMARY_METRIC = "macro_f1"


@dataclass
class FoldScores:
    """Metrics for a single fold. Everything downstream aggregates these."""

    macro_f1: float
    weighted_f1: float
    accuracy: float
    balanced_accuracy: float
    critical_recall: float
    critical_precision: float
    critical_f1: float
    fraud_leak_rate: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    per_class_precision: dict[str, float] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def score_fold(y_true: np.ndarray, y_pred: np.ndarray) -> FoldScores:
    """Compute every metric for one fold.

    `labels=LABELS` is passed explicitly everywhere: with 50 fraud rows spread over 5
    folds a class can be absent from a fold's predictions, and without it sklearn would
    silently return a shorter array and misalign the per-class values.
    """
    labels = list(LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    idx = labels.index(CRITICAL_LABEL)

    return FoldScores(
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        critical_recall=float(recall[idx]),
        critical_precision=float(precision[idx]),
        critical_f1=float(f1[idx]),
        fraud_leak_rate=float(1.0 - recall[idx]),
        per_class_f1={lab: float(v) for lab, v in zip(labels, f1, strict=True)},
        per_class_recall={lab: float(v) for lab, v in zip(labels, recall, strict=True)},
        per_class_precision={lab: float(v) for lab, v in zip(labels, precision, strict=True)},
        support={lab: int(v) for lab, v in zip(labels, support, strict=True)},
    )


@dataclass
class AggregateScores:
    """Mean/std across folds, plus the pooled confusion matrix.

    Std matters as much as mean here: with ~80 independent template groups, a 0.02
    macro-F1 difference between two models is usually inside the noise. The promotion
    gate reads both.
    """

    mean: dict[str, float]
    std: dict[str, float]
    n_folds: int
    confusion: np.ndarray
    per_fold: list[FoldScores] = field(default_factory=list)

    def summary_row(self) -> dict[str, float]:
        return {
            "macro_f1": self.mean["macro_f1"],
            "macro_f1_std": self.std["macro_f1"],
            "critical_recall": self.mean["critical_recall"],
            "critical_recall_std": self.std["critical_recall"],
            "accuracy": self.mean["accuracy"],
            "balanced_accuracy": self.mean["balanced_accuracy"],
            "fraud_leak_rate": self.mean["fraud_leak_rate"],
        }


_SCALAR_FIELDS = (
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "balanced_accuracy",
    "critical_recall",
    "critical_precision",
    "critical_f1",
    "fraud_leak_rate",
)


def aggregate(folds: list[FoldScores], confusion: np.ndarray) -> AggregateScores:
    mean, std = {}, {}
    for name in _SCALAR_FIELDS:
        values = np.array([getattr(f, name) for f in folds], dtype=float)
        mean[name] = float(values.mean())
        std[name] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    for label in LABELS:
        for prefix, attr in (("f1", "per_class_f1"), ("recall", "per_class_recall")):
            values = np.array([getattr(f, attr)[label] for f in folds], dtype=float)
            mean[f"{prefix}__{label}"] = float(values.mean())
            std[f"{prefix}__{label}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    return AggregateScores(
        mean=mean, std=std, n_folds=len(folds), confusion=confusion, per_fold=folds
    )


def pooled_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(LABELS))


def format_confusion(matrix: np.ndarray) -> str:
    """Readable confusion matrix, rows = truth, cols = prediction."""
    width = max(len(lab) for lab in LABELS) + 2
    header = " " * width + "".join(f"{lab[:10]:>12}" for lab in LABELS)
    lines = [header, " " * width + "".join(f"{'':>12}" for _ in LABELS)]
    for i, label in enumerate(LABELS):
        row = f"{label:<{width}}" + "".join(f"{matrix[i, j]:>12d}" for j in range(len(LABELS)))
        lines.append(row)
    return "\n".join(lines)
