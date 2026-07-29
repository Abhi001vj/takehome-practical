"""Optuna hyperparameter search, logged to MLflow.

Two rules this module enforces, both of which are easy to get wrong:

1. **Tune on the grouped folds.** Tuning against the naive split would optimise the
   hyperparameters for memorising templates - the search would happily pick a
   high-capacity configuration because leakage rewards it.

2. **Optimise the metric the gate reads.** The objective is macro-F1, the same metric the
   promotion gate compares. A tuner optimising accuracy while the gate reads macro-F1
   produces models that get rejected for reasons the tuner never saw.

Search spaces are deliberately narrow. With ~80 independent groups, a wide search over
400 trials would find a configuration that fits the CV folds rather than the problem;
the number of effective samples does not support fine-grained tuning, and saying so is
more useful than a large trial count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_params
from .cv import cross_validate, make_splits
from .data import load_training_data
from .grouping import assign_groups
from .models import get_builder
from .tracking import run, setup


def _suggest(trial, model: str) -> dict[str, Any]:
    """Per-model search space."""
    if model == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.1, 50.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "use_char": trial.suggest_categorical("use_char", [True, False]),
        }
    if model == "linear_svc":
        return {
            "C": trial.suggest_float("C", 0.01, 20.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "use_char": trial.suggest_categorical("use_char", [True, False]),
        }
    if model in {"multinomial_nb", "complement_nb"}:
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 3.0, log=True),
            "use_char": trial.suggest_categorical("use_char", [True, False]),
        }
    if model in {"lightgbm", "xgboost"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "svd_components": trial.suggest_categorical("svd_components", [80, 120, 200]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            **(
                {"num_leaves": trial.suggest_int("num_leaves", 7, 63)}
                if model == "lightgbm"
                else {"max_depth": trial.suggest_int("max_depth", 2, 8)}
            ),
        }
    if model == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 150, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "depth": trial.suggest_int("depth", 2, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
            "svd_components": trial.suggest_categorical("svd_components", [80, 120, 200]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    if model == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]
            ),
        }
    if model.startswith("embedding_"):
        space: dict[str, Any] = {
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None])
        }
        if model == "embedding_logreg":
            space["C"] = trial.suggest_float("C", 0.1, 100.0, log=True)
        else:
            space["n_estimators"] = trial.suggest_int("n_estimators", 100, 600, step=50)
            space["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.3, log=True)
            space["num_leaves"] = trial.suggest_int("num_leaves", 7, 63)
        return space
    raise ValueError(f"no search space defined for {model!r} (tuning it is not supported)")


def tune(
    model: str,
    n_trials: int | None = None,
    timeout: int | None = None,
    n_splits: int | None = None,
    n_repeats: int = 2,
    seed: int | None = None,
    data_path: Path | None = None,
    params_path: Path | None = None,
    track: bool = True,
) -> dict[str, Any]:
    """Search hyperparameters for one model and return the best configuration.

    `n_repeats` defaults to 2 rather than the report's 4: tuning runs the CV loop once
    per trial, and the extra repeats buy precision that a 40-trial search cannot exploit.
    The winning configuration is then re-scored at full repeats by `support-router cv`.
    """
    import optuna

    params = load_params(params_path)
    n_trials = n_trials or params["tuning"]["n_trials"]
    timeout = timeout or params["tuning"]["timeout_seconds"]
    n_splits = n_splits or params.cv["n_splits"]
    seed = seed if seed is not None else params["seed"]

    df = load_training_data(data_path)
    texts, labels = df["text"].tolist(), df["label"].tolist()
    grouping = assign_groups(
        texts,
        similarity_threshold=params.grouping["similarity_threshold"],
        char_ngram_range=tuple(params.grouping["char_ngram_range"]),
    )
    # Grouped folds only - see rule 1 in the module docstring.
    splits, _ = make_splits(
        texts, labels, scheme="grouped", n_splits=n_splits,
        n_repeats=n_repeats, seed=seed, grouping=grouping,
    )

    builder = get_builder(model)
    if model.startswith("embedding_"):
        base_kwargs = {"model_name": params.llm["embedding_model"]}
    else:
        base_kwargs = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    history: list[dict[str, Any]] = []

    def objective(trial) -> float:
        suggested = _suggest(trial, model)
        kwargs = {**base_kwargs, **suggested, "seed": seed}
        result = cross_validate(
            lambda: builder(**kwargs), texts, labels, name=f"{model}-trial{trial.number}",
            scheme="grouped", splits=splits, grouping=grouping,
        )
        trial.set_user_attr("critical_recall", result.critical_recall)
        trial.set_user_attr("macro_f1_std", result.macro_f1_std)
        history.append(
            {
                "trial": trial.number,
                "macro_f1": result.macro_f1,
                "critical_recall": result.critical_recall,
                **suggested,
            }
        )
        return result.macro_f1

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"{model}-macro_f1",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = {
        "model": model,
        "best_params": study.best_params,
        "best_macro_f1": study.best_value,
        "best_critical_recall": study.best_trial.user_attrs.get("critical_recall"),
        "n_trials": len(study.trials),
    }

    if track and setup() is not None:
        import mlflow

        with run(f"tune-{model}", tags={"model": model, "phase": "tuning"}):
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_macro_f1", study.best_value)
            if best["best_critical_recall"] is not None:
                mlflow.log_metric("best_critical_recall", best["best_critical_recall"])
            mlflow.log_metric("n_trials", len(study.trials))
            mlflow.log_dict({"history": history}, "tuning_history.json")
            # Importances explain *why* a configuration won, which matters more than the
            # configuration itself when the effective sample size is this small.
            try:
                importances = optuna.importance.get_param_importances(study)
                mlflow.log_dict(
                    {k: float(v) for k, v in importances.items()}, "param_importances.json"
                )
            except Exception:
                pass

    return best


def tune_all(models: list[str], **kwargs: Any) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for model in models:
        try:
            out[model] = tune(model, **kwargs)
        except Exception as exc:
            out[model] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def best_of(results: dict[str, dict]) -> tuple[str, dict] | None:
    valid = {k: v for k, v in results.items() if "best_macro_f1" in v}
    if not valid:
        return None
    name = max(valid, key=lambda k: valid[k]["best_macro_f1"])
    return name, valid[name]


def summarise(results: dict[str, dict]) -> str:
    lines = ["model                  macro_f1   fraud_recall  trials"]
    for name, res in sorted(
        results.items(), key=lambda kv: kv[1].get("best_macro_f1", -1), reverse=True
    ):
        if "error" in res:
            lines.append(f"{name:22s} ERROR: {res['error']}")
        else:
            recall = res.get("best_critical_recall")
            recall_s = f"{recall:.3f}" if isinstance(recall, (int, float)) else "n/a"
            lines.append(
                f"{name:22s} {res['best_macro_f1']:.4f}     {recall_s:>8s}  "
                f"{res['n_trials']:>5d}"
            )
    return "\n".join(lines)
