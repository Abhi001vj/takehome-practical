"""Gradient-boosted and bagged tree ensembles over TF-IDF features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ..features import build_vectoriser

#: Trees choke on the full char+word union (100k+ columns for 400 rows). SVD to a dense
#: low-rank space is the standard remedy and is applied inside the fold, never before it.
DEFAULT_SVD_COMPONENTS = 120


def _maybe_svd(n_components: int | None):
    if not n_components:
        return None
    from sklearn.decomposition import TruncatedSVD

    return TruncatedSVD(n_components=n_components, random_state=0)


class TreeTextClassifier:
    """TF-IDF (optionally SVD-reduced) + a gradient-boosted or bagged ensemble.

    Written as a small explicit class rather than an sklearn `Pipeline` because the
    boosting libraries want a dense array and a label-encoded target, and expressing that
    through pipeline steps obscures more than it saves.
    """

    def __init__(
        self,
        backend: str,
        seed: int = 0,
        class_weight: str | None = "balanced",
        svd_components: int | None = DEFAULT_SVD_COMPONENTS,
        use_char: bool = True,
        **model_kwargs: object,
    ) -> None:
        self.backend = backend
        self.seed = seed
        self.class_weight = class_weight
        self.svd_components = svd_components
        self.use_char = use_char
        self.model_kwargs = model_kwargs
        self._vec = None
        self._svd = None
        self._model = None
        self._encoder = LabelEncoder()

    def _build_model(self, n_classes: int):
        kw = dict(self.model_kwargs)
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=kw.pop("n_estimators", 400),
                learning_rate=kw.pop("learning_rate", 0.08),
                num_leaves=kw.pop("num_leaves", 15),
                # 400 rows means the defaults (min_child_samples=20) would refuse to
                # split at all in the minority class; lowered deliberately.
                min_child_samples=kw.pop("min_child_samples", 5),
                subsample=kw.pop("subsample", 0.9),
                colsample_bytree=kw.pop("colsample_bytree", 0.7),
                reg_lambda=kw.pop("reg_lambda", 1.0),
                objective="multiclass",
                num_class=n_classes,
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
                **kw,
            )
        if self.backend == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=kw.pop("n_estimators", 400),
                learning_rate=kw.pop("learning_rate", 0.08),
                max_depth=kw.pop("max_depth", 4),
                min_child_weight=kw.pop("min_child_weight", 1),
                subsample=kw.pop("subsample", 0.9),
                colsample_bytree=kw.pop("colsample_bytree", 0.7),
                reg_lambda=kw.pop("reg_lambda", 1.0),
                objective="multi:softprob",
                num_class=n_classes,
                random_state=self.seed,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0,
                **kw,
            )
        if self.backend == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=kw.pop("iterations", 400),
                learning_rate=kw.pop("learning_rate", 0.08),
                depth=kw.pop("depth", 4),
                l2_leaf_reg=kw.pop("l2_leaf_reg", 3.0),
                loss_function="MultiClass",
                random_seed=self.seed,
                verbose=False,
                allow_writing_files=False,
                **kw,
            )
        if self.backend == "random_forest":
            return RandomForestClassifier(
                n_estimators=kw.pop("n_estimators", 600),
                max_depth=kw.pop("max_depth", None),
                min_samples_leaf=kw.pop("min_samples_leaf", 1),
                max_features=kw.pop("max_features", "sqrt"),
                class_weight=self.class_weight if self.class_weight else None,
                random_state=self.seed,
                n_jobs=-1,
                **kw,
            )
        raise ValueError(f"unknown tree backend: {self.backend!r}")

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> TreeTextClassifier:
        y = self._encoder.fit_transform(np.asarray(labels))
        self._vec = build_vectoriser(use_char=self.use_char)
        X = self._vec.fit_transform(list(texts))

        self._svd = _maybe_svd(self.svd_components)
        if self._svd is not None:
            # n_components cannot exceed the rank available in this fold.
            self._svd.n_components = min(self.svd_components, X.shape[1] - 1, X.shape[0] - 1)
            X = self._svd.fit_transform(X)
        else:
            X = X.toarray()

        self._model = self._build_model(len(self._encoder.classes_))

        # RandomForest takes class_weight directly; the boosters take sample weights.
        if self.backend == "random_forest" or not self.class_weight:
            self._model.fit(X, y)
        else:
            weights = compute_sample_weight(class_weight=self.class_weight, y=y)
            self._model.fit(X, y, sample_weight=weights)
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        X = self._vec.transform(list(texts))
        X = self._svd.transform(X) if self._svd is not None else X.toarray()
        codes = np.asarray(self._model.predict(X)).ravel().astype(int)
        return self._encoder.inverse_transform(codes)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self._vec.transform(list(texts))
        X = self._svd.transform(X) if self._svd is not None else X.toarray()
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self._encoder.classes_


def build_lightgbm(seed: int = 0, **kw: object) -> TreeTextClassifier:
    return TreeTextClassifier("lightgbm", seed=seed, **kw)


def build_xgboost(seed: int = 0, **kw: object) -> TreeTextClassifier:
    return TreeTextClassifier("xgboost", seed=seed, **kw)


def build_catboost(seed: int = 0, **kw: object) -> TreeTextClassifier:
    return TreeTextClassifier("catboost", seed=seed, **kw)


def build_random_forest(seed: int = 0, **kw: object) -> TreeTextClassifier:
    # RF handles wide sparse input acceptably, so SVD is off by default here; the
    # boosters get it because dense 100k-column input is not tractable for them.
    kw.setdefault("svd_components", None)
    return TreeTextClassifier("random_forest", seed=seed, **kw)
