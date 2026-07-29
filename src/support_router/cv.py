"""Shared repeated cross-validation for grouped and row-level evaluation."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .grouping import GroupingResult, assign_groups
from .metrics import AggregateScores, FoldScores, aggregate, pooled_confusion, score_fold


@runtime_checkable
class Estimator(Protocol):
    """The whole contract a candidate approach must satisfy.

    Deliberately narrower than sklearn's: no `get_params`, no `score`. The direct-LLM
    classifier has no trainable parameters at all and would not survive a stricter
    interface, but it must still be benchmarked on identical folds as everything else.
    """

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Estimator: ...

    def predict(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class CVResult:
    """Everything needed to report on, compare, and reproduce one CV run."""

    name: str
    scheme: str
    scores: AggregateScores
    fit_seconds: float
    predict_seconds: float
    grouping: GroupingResult | None = None
    leakage_rows: int = 0
    params: dict = field(default_factory=dict)
    parse_failures: int = 0
    prediction_rows: int = 0

    @property
    def macro_f1(self) -> float:
        return self.scores.mean["macro_f1"]

    @property
    def macro_f1_std(self) -> float:
        return self.scores.std["macro_f1"]

    @property
    def critical_recall(self) -> float:
        return self.scores.mean["critical_recall"]

    def to_row(self) -> dict:
        row = {"model": self.name, "scheme": self.scheme}
        row.update(self.scores.summary_row())
        row["fit_seconds"] = self.fit_seconds
        row["predict_seconds"] = self.predict_seconds
        row["leakage_rows"] = self.leakage_rows
        row["parse_failures"] = self.parse_failures
        row["parse_failure_rate"] = (
            self.parse_failures / self.prediction_rows if self.prediction_rows else 0.0
        )
        return row


def make_splits(
    texts: Sequence[str],
    labels: Sequence[str],
    scheme: str = "grouped",
    n_splits: int = 5,
    n_repeats: int = 4,
    seed: int = 0,
    grouping: GroupingResult | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], GroupingResult | None]:
    """Build the fold list once, so every model sees exactly the same folds.

    `grouped` uses `StratifiedGroupKFold`: it keeps a template's renderings together
    while still balancing labels across folds. `naive` uses plain `StratifiedKFold` and
    exists only to quantify what grouping is protecting us from.
    """
    y = np.asarray(labels)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    if scheme == "grouped":
        if grouping is None:
            grouping = assign_groups(list(texts))
        if grouping.n_groups < n_splits:
            raise ValueError(
                f"only {grouping.n_groups} groups for {n_splits} folds; "
                "lower n_splits or raise the similarity threshold"
            )

        # A class with fewer template groups than folds can be stranded: every one of
        # its groups lands in a single test fold, leaving the training side with no
        # examples of that class at all. The resulting fold scores 0 for it and drags
        # macro-F1 down for a reason that has nothing to do with the model. Warn rather
        # than raise - it is a legitimate configuration for a quick look, just not one
        # whose numbers should be reported.
        groups_per_label: dict[str, set[int]] = {}
        for label, group in zip(y, grouping.groups, strict=True):
            groups_per_label.setdefault(str(label), set()).add(int(group))
        thin = {
            label: len(gs) for label, gs in groups_per_label.items() if len(gs) < n_splits
        }
        if thin:
            warnings.warn(
                f"these labels have fewer template groups than folds ({n_splits}): {thin}. "
                "Some folds will train without any examples of them, depressing macro-F1 "
                "for reasons unrelated to the model. Reduce n_splits.",
                UserWarning,
                stacklevel=2,
            )
        for repeat in range(n_repeats):
            # Permuting the row order before splitting is what actually yields a
            # different partition per repeat: StratifiedGroupKFold's assignment is a
            # deterministic greedy fill, so reseeding alone moves it very little.
            rng = np.random.default_rng(seed + repeat)
            perm = rng.permutation(len(y))
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed + repeat
            )
            for tr, te in splitter.split(perm.reshape(-1, 1), y[perm], grouping.groups[perm]):
                # Map fold positions back to original row indices.
                splits.append((np.sort(perm[tr]), np.sort(perm[te])))
    elif scheme == "naive":
        for repeat in range(n_repeats):
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
            for tr, te in splitter.split(np.zeros(len(y)), y):
                splits.append((tr, te))
    else:
        raise ValueError(f"unknown cv scheme: {scheme!r} (expected 'grouped' or 'naive')")

    return splits, grouping


def cross_validate(
    build: Callable[[], Estimator],
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    name: str,
    scheme: str = "grouped",
    n_splits: int = 5,
    n_repeats: int = 4,
    seed: int = 0,
    grouping: GroupingResult | None = None,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    params: dict | None = None,
) -> CVResult:
    """Run one approach through the harness.

    `build` is a factory, not an instance: every fold gets a fresh, unfitted estimator.
    Reusing one instance across folds is the classic way to leak state (a vectoriser
    fitted on fold 1's vocabulary quietly carrying into fold 2).
    """
    texts = list(texts)
    y = np.asarray(labels)

    if splits is None:
        splits, grouping = make_splits(
            texts,
            y,
            scheme=scheme,
            n_splits=n_splits,
            n_repeats=n_repeats,
            seed=seed,
            grouping=grouping,
        )

    fold_scores: list[FoldScores] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    fit_seconds = predict_seconds = 0.0
    leakage_rows = 0
    parse_failures = 0
    prediction_rows = 0

    for train_idx, test_idx in splits:
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        model = build()
        t0 = time.perf_counter()
        model.fit(train_texts, y[train_idx])
        fit_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = np.asarray(model.predict(test_texts))
        predict_seconds += time.perf_counter() - t0
        parse_failures += int(getattr(model, "parse_failures_", 0))
        prediction_rows += len(test_idx)

        if len(y_pred) != len(test_idx):
            raise RuntimeError(
                f"{name} returned {len(y_pred)} predictions for {len(test_idx)} inputs"
            )

        fold_scores.append(score_fold(y[test_idx], y_pred))
        all_true.append(y[test_idx])
        all_pred.append(y_pred)

        if grouping is not None:
            seen = set(grouping.groups[train_idx].tolist())
            leakage_rows += sum(1 for g in grouping.groups[test_idx].tolist() if g in seen)

    confusion = pooled_confusion(np.concatenate(all_true), np.concatenate(all_pred))
    n_folds = len(splits)

    return CVResult(
        name=name,
        scheme=scheme,
        scores=aggregate(fold_scores, confusion),
        fit_seconds=fit_seconds / n_folds,
        predict_seconds=predict_seconds / n_folds,
        grouping=grouping,
        leakage_rows=leakage_rows,
        params=params or {},
        parse_failures=parse_failures,
        prediction_rows=prediction_rows,
    )


def leakage_report(
    texts: Sequence[str], labels: Sequence[str], n_splits: int = 5, seed: int = 0
) -> dict:
    """Quantify how many test rows have a template sibling in train, per scheme.

    This is the evidence for the central modelling claim of the project. It reports
    counts, not opinions: under `naive` a large share of every test fold has already been
    seen in near-identical form; under `grouped` it is exactly zero by construction.
    """
    grouping = assign_groups(list(texts))
    out: dict = {"grouping": grouping.summary(), "n_groups": grouping.n_groups}

    for scheme in ("naive", "grouped"):
        splits, _ = make_splits(
            texts,
            labels,
            scheme=scheme,
            n_splits=n_splits,
            n_repeats=1,
            seed=seed,
            grouping=grouping,
        )
        leaked = total = 0
        for train_idx, test_idx in splits:
            seen = set(grouping.groups[train_idx].tolist())
            leaked += sum(1 for g in grouping.groups[test_idx].tolist() if g in seen)
            total += len(test_idx)
        out[scheme] = {
            "leaked_rows": leaked,
            "total_test_rows": total,
            "leak_rate": leaked / total if total else 0.0,
        }
    return out
