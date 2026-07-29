"""Reference, linear, probabilistic, and simple classical classifiers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ..features import build_vectoriser


class DummyEstimator:
    """Wraps sklearn's DummyClassifier to the harness's text-in/label-out contract."""

    def __init__(self, strategy: str = "most_frequent", seed: int = 0) -> None:
        self.strategy = strategy
        self.seed = seed
        self._model = DummyClassifier(strategy=strategy, random_state=seed)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> DummyEstimator:
        self._model.fit(np.zeros((len(texts), 1)), np.asarray(labels))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.predict(np.zeros((len(texts), 1)))


def build_most_frequent(seed: int = 0, **_: object) -> DummyEstimator:
    return DummyEstimator("most_frequent", seed)


def build_stratified_random(seed: int = 0, **_: object) -> DummyEstimator:
    return DummyEstimator("stratified", seed)


def build_uniform_random(seed: int = 0, **_: object) -> DummyEstimator:
    return DummyEstimator("uniform", seed)


def build_logistic_regression(
    seed: int = 0,
    C: float = 4.0,
    class_weight: str | None = "balanced",
    use_char: bool = True,
    **_: object,
) -> Pipeline:
    """TF-IDF with multinomial logistic regression.

    `class_weight="balanced"` is the default but is exposed so the ablation in the
    report can show what it actually buys. On this dataset the row-level imbalance is
    mostly template duplication rather than genuine class rarity, so the effect is
    smaller than the 3.2:1 row ratio suggests.

    `saga` handles the sparse high-dimensional union without the convergence warnings
    `lbfgs` produces at this feature count.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=5000,
                    solver="saga",
                    random_state=seed,
                ),
            ),
        ]
    )


def build_multinomial_nb(
    seed: int = 0, alpha: float = 0.3, use_char: bool = False, **_: object
) -> Pipeline:
    """Multinomial naive Bayes.

    Char n-grams are off by default here: MultinomialNB's independence assumption is
    already badly violated by overlapping char n-grams, and including them measurably
    hurts. This is the one model where the shared feature default is wrong.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            ("clf", MultinomialNB(alpha=alpha)),
        ]
    )


def build_complement_nb(
    seed: int = 0, alpha: float = 0.3, use_char: bool = False, **_: object
) -> Pipeline:
    """Complement NB - the imbalance-aware NB variant.

    Included specifically as the naive-Bayes answer to class imbalance: it estimates
    parameters from the complement of each class, which is designed for exactly the
    skewed-prior situation MultinomialNB handles badly.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            ("clf", ComplementNB(alpha=alpha)),
        ]
    )


def build_bernoulli_nb(
    seed: int = 0, alpha: float = 0.3, use_char: bool = False, **_: object
) -> Pipeline:
    """Bernoulli NB - models term presence rather than frequency.

    Included because these messages are 16 words long: a term almost never repeats, so
    presence/absence loses very little information and the binarised model is a genuinely
    reasonable fit rather than a box-ticking entry.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            ("clf", BernoulliNB(alpha=alpha)),
        ]
    )


def build_linear_svc(
    seed: int = 0,
    C: float = 1.0,
    class_weight: str | None = "balanced",
    use_char: bool = True,
    **_: object,
) -> Pipeline:
    """Linear SVM. Frequently the strongest classical model on short-text TF-IDF.

    Note it has no `predict_proba`, so it cannot be used where the API needs calibrated
    confidence - a real trade-off, recorded in the README.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            (
                "clf",
                LinearSVC(C=C, class_weight=class_weight, random_state=seed, max_iter=5000),
            ),
        ]
    )


def build_sgd(
    seed: int = 0,
    alpha: float = 1e-4,
    loss: str = "modified_huber",
    class_weight: str | None = "balanced",
    use_char: bool = True,
    **_: object,
) -> Pipeline:
    """SGD-trained linear model.

    `modified_huber` rather than `hinge` specifically because it is the one SGD loss that
    supports `predict_proba`, which the API needs for a confidence score. It is also the
    model for high-throughput workloads: partial_fit
    allows online updates without a full retrain.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            (
                "clf",
                SGDClassifier(
                    loss=loss,
                    alpha=alpha,
                    class_weight=class_weight,
                    max_iter=5000,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_ridge(
    seed: int = 0,
    alpha: float = 1.0,
    class_weight: str | None = "balanced",
    use_char: bool = True,
    **_: object,
) -> Pipeline:
    """Ridge classifier - least-squares on one-hot targets.

    Cheapest thing here that is still competitive: it has a closed-form solution, so it
    trains in milliseconds and is a useful latency reference point.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            ("clf", RidgeClassifier(alpha=alpha, class_weight=class_weight, random_state=seed)),
        ]
    )


def build_passive_aggressive(
    seed: int = 0,
    C: float = 0.5,
    class_weight: str | None = "balanced",
    use_char: bool = True,
    **_: object,
) -> Pipeline:
    """Passive-aggressive - an online-learning linear model.

    Like SGD, it supports `partial_fit`, so it belongs in the comparison as a candidate
    for the streaming/high-throughput scenario rather than for peak accuracy.

    Built from `SGDClassifier` rather than `PassiveAggressiveClassifier`: the latter is
    deprecated in scikit-learn 1.8 and removed in 1.10, and this parameterisation is the
    exact equivalent the deprecation notice prescribes. `eta0=C` reproduces the PA-I
    aggressiveness parameter, which is what the old `C` argument controlled.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            (
                "clf",
                SGDClassifier(
                    loss="hinge",
                    penalty=None,
                    learning_rate="pa1",
                    eta0=C,
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_knn(
    seed: int = 0, n_neighbors: int = 5, use_char: bool = True, **_: object
) -> Pipeline:
    """k-NN with cosine distance.

    Worth including as a diagnostic rather than a candidate: because the corpus is
    template-generated, k-NN is essentially a template-retrieval system. Its score under
    *naive* CV is near-perfect and under *grouped* CV it collapses - which makes it the
    single clearest illustration in the report of what the leakage was hiding.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            ("clf", KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)),
        ]
    )


def build_decision_tree(
    seed: int = 0,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced",
    use_char: bool = False,
    **_: object,
) -> Pipeline:
    """A single decision tree - the interpretable floor for the tree family.

    Its gap to the ensembles is the clearest measure of how much of their performance
    comes from averaging rather than from the split criterion.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_extra_trees(
    seed: int = 0,
    n_estimators: int = 600,
    class_weight: str | None = "balanced",
    use_char: bool = True,
    **_: object,
) -> Pipeline:
    """Extremely randomised trees - random forest's higher-variance-reduction cousin.

    Handles wide sparse input directly, so unlike the boosters it needs no SVD step.
    """
    return Pipeline(
        [
            ("features", build_vectoriser(use_char=use_char)),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    class_weight=class_weight,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
