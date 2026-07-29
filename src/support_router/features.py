"""Word- and character-level TF-IDF feature construction."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

#: Defaults mirror `conf/params.yaml`; the params file wins when the CLI passes one.
WORD_DEFAULTS: dict[str, Any] = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "lowercase": True,
}

CHAR_DEFAULTS: dict[str, Any] = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "sublinear_tf": True,
    "lowercase": True,
}


def build_vectoriser(
    word: dict[str, Any] | None = None,
    char: dict[str, Any] | None = None,
    use_char: bool = True,
) -> FeatureUnion | TfidfVectorizer:
    """Assemble the feature extractor.

    `use_char=False` exists because char n-grams roughly quadruple the feature count for
    a small gain; the tuner treats it as a hyperparameter rather than a fixed choice.
    """
    word_cfg = {**WORD_DEFAULTS, **(word or {})}
    word_cfg["ngram_range"] = tuple(word_cfg["ngram_range"])
    word_vec = TfidfVectorizer(**word_cfg)
    if not use_char:
        return word_vec

    char_cfg = {**CHAR_DEFAULTS, **(char or {})}
    char_cfg["ngram_range"] = tuple(char_cfg["ngram_range"])
    return FeatureUnion([("word", word_vec), ("char", TfidfVectorizer(**char_cfg))])


def feature_spec(vectoriser: FeatureUnion | TfidfVectorizer) -> dict[str, Any]:
    """Describe a *fitted* vectoriser for logging to MLflow.

    This is the artifact that stands in for a feature-store entry: the exact
    featurisation contract a served model expects, recorded next to the run that
    produced it. See the note in `tracking.py` on why this is not a real feature store.
    """
    spec: dict[str, Any] = {"type": type(vectoriser).__name__}
    parts = (
        vectoriser.transformer_list
        if isinstance(vectoriser, FeatureUnion)
        else [("word", vectoriser)]
    )
    total = 0
    for name, vec in parts:
        size = len(getattr(vec, "vocabulary_", {}) or {})
        total += size
        spec[name] = {
            "analyzer": vec.analyzer,
            "ngram_range": list(vec.ngram_range),
            "min_df": vec.min_df,
            "sublinear_tf": vec.sublinear_tf,
            "vocabulary_size": size,
        }
    spec["total_features"] = total
    return spec
