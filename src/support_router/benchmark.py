"""Measured inference benchmark used by the consolidated report."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from statistics import median

import numpy as np

from .config import ARTIFACTS, REPORTS
from .data import load_training_data
from .inference import load_model, predict, predict_batch, reset_cache


def _clear_embedding_vectors() -> None:
    """Remove exact-text cache hits without unloading the warmed encoder."""
    from .models.embeddings import clear_embedding_cache

    clear_embedding_cache()


def benchmark_inference(
    model_path: Path | None = None,
    out_path: Path | None = None,
    sample_size: int = 60,
    batch_repeats: int = 5,
) -> dict:
    """Benchmark the persisted model after a cold process-level cache reset.

    The first prediction is reported separately because embedding models must load and
    warm their external encoder. Steady-state single-row latency and batched throughput
    are then measured over distinct messages from the checked-in dataset.
    """
    if sample_size < 1 or batch_repeats < 1:
        raise ValueError("sample_size and batch_repeats must be positive")

    directory = Path(model_path) if model_path is not None else ARTIFACTS
    frame = load_training_data()
    texts = frame["text"].drop_duplicates().tolist()[:sample_size]
    if not texts:
        raise ValueError("the training dataset contains no benchmarkable messages")

    reset_cache()
    started = time.perf_counter()
    load_model(directory, force=True)
    predict(texts[0], model_path=directory)
    warmup_seconds = time.perf_counter() - started

    _clear_embedding_vectors()
    samples_ms: list[float] = []
    for text in texts:
        started = time.perf_counter()
        predict(text, model_path=directory)
        samples_ms.append((time.perf_counter() - started) * 1000)

    batch_seconds = 0.0
    for _ in range(batch_repeats):
        _clear_embedding_vectors()
        batch_started = time.perf_counter()
        predict_batch(texts, model_path=directory)
        batch_seconds += time.perf_counter() - batch_started

    mean_ms = float(np.mean(samples_ms))
    result = {
        "model_path": str(directory.relative_to(directory.parent)),
        "sample_size": len(texts),
        "batch_repeats": batch_repeats,
        "warmup_seconds": warmup_seconds,
        "median_ms": median(samples_ms),
        "mean_ms": mean_ms,
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "max_ms": max(samples_ms),
        "throughput_per_second": 1000.0 / mean_ms if mean_ms else None,
        "batch_seconds": batch_seconds,
        "batch_throughput_per_second": (
            len(texts) * batch_repeats / batch_seconds if batch_seconds else None
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    destination = Path(out_path) if out_path is not None else REPORTS / "latency.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result
