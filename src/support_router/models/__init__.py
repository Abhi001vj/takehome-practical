"""Model builders and named model-family registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .baselines import (
    build_bernoulli_nb,
    build_complement_nb,
    build_decision_tree,
    build_extra_trees,
    build_knn,
    build_linear_svc,
    build_logistic_regression,
    build_most_frequent,
    build_multinomial_nb,
    build_passive_aggressive,
    build_ridge,
    build_sgd,
    build_stratified_random,
    build_uniform_random,
)
from .trees import (
    build_catboost,
    build_lightgbm,
    build_random_forest,
    build_xgboost,
)

Builder = Callable[..., Any]

REGISTRY: dict[str, Builder] = {
    # Reference points
    "most_frequent": build_most_frequent,
    "stratified_random": build_stratified_random,
    "uniform_random": build_uniform_random,
    # Classical linear / probabilistic
    "logistic_regression": build_logistic_regression,
    "multinomial_nb": build_multinomial_nb,
    "complement_nb": build_complement_nb,
    "bernoulli_nb": build_bernoulli_nb,
    "linear_svc": build_linear_svc,
    "sgd": build_sgd,
    "ridge": build_ridge,
    "passive_aggressive": build_passive_aggressive,
    # Instance- and tree-based
    "knn": build_knn,
    "decision_tree": build_decision_tree,
    "extra_trees": build_extra_trees,
    # Boosted / bagged ensembles
    "lightgbm": build_lightgbm,
    "xgboost": build_xgboost,
    "catboost": build_catboost,
    "random_forest": build_random_forest,
}

FAMILIES: dict[str, tuple[str, ...]] = {
    "dummy": ("most_frequent", "stratified_random", "uniform_random"),
    "linear": (
        "logistic_regression",
        "linear_svc",
        "sgd",
        "ridge",
        "passive_aggressive",
    ),
    "naive_bayes": ("multinomial_nb", "complement_nb", "bernoulli_nb"),
    "simple": ("knn", "decision_tree", "extra_trees"),
    "trees": ("lightgbm", "xgboost", "catboost", "random_forest"),
    "llm": ("embedding_logreg", "embedding_lightgbm", "llm_zero_shot", "llm_few_shot"),
}
FAMILIES["classical"] = FAMILIES["linear"] + FAMILIES["naive_bayes"] + FAMILIES["simple"]
FAMILIES["baselines"] = FAMILIES["dummy"] + FAMILIES["classical"]
#: Everything that trains in seconds - the default for CI.
FAMILIES["fast"] = FAMILIES["dummy"] + FAMILIES["linear"] + FAMILIES["naive_bayes"]
FAMILIES["all"] = FAMILIES["dummy"] + FAMILIES["classical"] + FAMILIES["trees"]
#: Every approach including the ones needing the LLM extra and a live endpoint.
FAMILIES["everything"] = FAMILIES["all"] + FAMILIES["llm"]

#: Names that need the `llm` extra and (for the generative ones) a live endpoint.
LLM_MODELS: tuple[str, ...] = FAMILIES["llm"]


def _load_llm_builders() -> dict[str, Builder]:
    """Import the LLM builders on demand.

    Kept behind a function so `import support_router.models` never pulls in torch.
    """
    from .embeddings import build_embedding_lightgbm, build_embedding_logreg
    from .llm import build_llm_few_shot, build_llm_zero_shot

    return {
        "embedding_logreg": build_embedding_logreg,
        "embedding_lightgbm": build_embedding_lightgbm,
        "llm_zero_shot": build_llm_zero_shot,
        "llm_few_shot": build_llm_few_shot,
    }


def get_builder(name: str) -> Builder:
    """Resolve a model name to its factory, importing LLM deps only if asked for."""
    if name in REGISTRY:
        return REGISTRY[name]
    if name in LLM_MODELS:
        try:
            builder = _load_llm_builders()[name]
        except ImportError as exc:
            raise ImportError(
                f"model {name!r} needs the optional LLM dependencies. "
                f'Install them with: uv pip install -e ".[llm]"  (original error: {exc})'
            ) from exc
        REGISTRY[name] = builder
        return builder
    raise KeyError(f"unknown model {name!r}. Known: {sorted(set(REGISTRY) | set(LLM_MODELS))}")


def resolve_names(selection: str | list[str]) -> list[str]:
    """Expand a family name, a comma-separated list, or explicit names into model names."""
    if isinstance(selection, str):
        selection = [s.strip() for s in selection.split(",") if s.strip()]

    names: list[str] = []
    for item in selection:
        if item in FAMILIES:
            names.extend(FAMILIES[item])
        else:
            names.append(item)

    seen: set[str] = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


__all__ = ["REGISTRY", "FAMILIES", "LLM_MODELS", "get_builder", "resolve_names"]
