"""Frozen sentence embeddings with logistic-regression or LightGBM heads."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Process-level cache: {(model_name, text_hash): vector}. Encoding 400 short messages
#: takes a few seconds on CPU; repeated CV would otherwise redo it 20 times per model.
_CACHE: dict[tuple[str, str], np.ndarray] = {}
_MODELS: dict[str, object] = {}


def clear_embedding_cache(drop_encoder: bool = False) -> None:
    """Clear cached text vectors while optionally retaining the loaded encoder."""
    _CACHE.clear()
    if drop_encoder:
        _MODELS.clear()


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _get_encoder(model_name: str):
    if model_name not in _MODELS:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                'sentence-transformers is required. Install with: uv pip install -e ".[llm]"'
            ) from exc
        # HF_HUB_OFFLINE lets CI run against a pre-baked cache with no network.
        _MODELS[model_name] = SentenceTransformer(
            model_name, device=os.environ.get("EMBEDDING_DEVICE") or None
        )
    return _MODELS[model_name]


def encode(
    texts: Sequence[str], model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 64
) -> np.ndarray:
    """Embed messages, reusing cached vectors where available."""
    texts = list(texts)
    keys = [(model_name, _text_key(t)) for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in _CACHE]

    if missing:
        encoder = _get_encoder(model_name)
        fresh = encoder.encode(
            [texts[i] for i in missing],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine geometry; keeps the linear head well-conditioned
            show_progress_bar=False,
        )
        for i, vec in zip(missing, fresh, strict=True):
            _CACHE[keys[i]] = vec

    return np.vstack([_CACHE[k] for k in keys])


class EmbeddingClassifier:
    """Frozen sentence encoder + a trained head.

    `head` is a string rather than an estimator instance so the whole configuration
    stays JSON-serialisable for MLflow params and Optuna trials.
    """

    def __init__(
        self,
        head: str = "logreg",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        seed: int = 0,
        class_weight: str | None = "balanced",
        C: float = 8.0,
        **head_kwargs: object,
    ) -> None:
        self.head = head
        self.model_name = model_name
        self.seed = seed
        self.class_weight = class_weight
        self.C = C
        self.head_kwargs = head_kwargs
        self._model = None
        self._encoder_labels = LabelEncoder()

    def _build_head(self):
        if self.head == "logreg":
            # Dense 384-d input: lbfgs converges fine and is faster than saga here.
            return LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=3000,
                random_state=self.seed,
            )
        if self.head == "lightgbm":
            from lightgbm import LGBMClassifier

            kw = dict(self.head_kwargs)
            return LGBMClassifier(
                n_estimators=kw.pop("n_estimators", 300),
                learning_rate=kw.pop("learning_rate", 0.08),
                num_leaves=kw.pop("num_leaves", 15),
                min_child_samples=kw.pop("min_child_samples", 5),
                colsample_bytree=kw.pop("colsample_bytree", 0.7),
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
                **kw,
            )
        raise ValueError(f"unknown head: {self.head!r}")

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> EmbeddingClassifier:
        X = encode(texts, self.model_name)
        y = np.asarray(labels)
        self._model = self._build_head()

        if self.head == "lightgbm" and self.class_weight:
            from sklearn.utils.class_weight import compute_sample_weight

            y_enc = self._encoder_labels.fit_transform(y)
            self._model.fit(
                X, y_enc, sample_weight=compute_sample_weight(self.class_weight, y_enc)
            )
        else:
            self._model.fit(X, y)
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        X = encode(texts, self.model_name)
        pred = self._model.predict(X)
        if self.head == "lightgbm":
            return self._encoder_labels.inverse_transform(np.asarray(pred).ravel().astype(int))
        return np.asarray(pred)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.predict_proba(encode(texts, self.model_name))

    @property
    def classes_(self) -> np.ndarray:
        if self.head == "lightgbm":
            return self._encoder_labels.classes_
        return self._model.classes_


def build_embedding_logreg(seed: int = 0, **kw: object) -> EmbeddingClassifier:
    return EmbeddingClassifier(head="logreg", seed=seed, **kw)


def build_embedding_lightgbm(seed: int = 0, **kw: object) -> EmbeddingClassifier:
    return EmbeddingClassifier(head="lightgbm", seed=seed, **kw)
