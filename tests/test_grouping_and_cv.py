"""Tests for the central claim of the project: the grouped split does not leak.

"""

from __future__ import annotations

import numpy as np
import pytest

from support_router.cv import cross_validate, leakage_report, make_splits
from support_router.grouping import assign_groups, normalise_for_grouping
from support_router.models import get_builder


class TestNormalisation:
    def test_strips_opener_and_closer(self):
        assert normalise_for_grouping("Hi, How do I reset my password? Thanks.") == (
            normalise_for_grouping("Urgent: How do I reset my password? Please advise.")
        )

    def test_masks_assets_and_amounts(self):
        a = normalise_for_grouping("My withdrawal of 2 ETH shows completed")
        b = normalise_for_grouping("My withdrawal of 0.5 BTC shows completed")
        assert a == b
        assert "<asset>" in a and "<num>" in a

    def test_keeps_genuinely_different_messages_apart(self):
        login = normalise_for_grouping("I can't log into my account")
        fraud = normalise_for_grouping("Someone withdrew funds I never authorized")
        assert login != fraud

    def test_stacked_openers(self):
        assert normalise_for_grouping("Urgent: Hello team, my account is locked") == (
            normalise_for_grouping("my account is locked")
        )


class TestGrouping:
    def test_template_variants_share_a_group(self, synthetic_frame):
        result = assign_groups(synthetic_frame["text"].tolist())
        # 16 templates x 5 renderings; grouping must recover roughly the templates, not
        # the rows. Allowing a little slack keeps this from being a brittle exact-match.
        assert result.n_groups <= 20, result.summary()
        assert result.n_rows == 80

    def test_groups_are_label_pure(self, synthetic_frame):
        """A group spanning two labels would make stratification incoherent."""
        result = assign_groups(synthetic_frame["text"].tolist())
        labels = synthetic_frame["label"].to_numpy()
        for group_id in range(result.n_groups):
            assert len(set(labels[result.groups == group_id])) == 1

    def test_deterministic(self, synthetic_frame):
        texts = synthetic_frame["text"].tolist()
        assert np.array_equal(assign_groups(texts).groups, assign_groups(texts).groups)

    def test_empty_input(self):
        assert assign_groups([]).n_groups == 0

    def test_threshold_of_one_still_groups_exact_normalised_matches(self, synthetic_frame):
        """Similarity clustering is a second line of defence, not the only one."""
        result = assign_groups(synthetic_frame["text"].tolist(), similarity_threshold=1.0)
        assert result.n_groups < result.n_rows


class TestSplitLeakage:
    """The property that justifies the whole CV design."""

    def test_grouped_split_leaks_nothing(self, synthetic_frame):
        texts = synthetic_frame["text"].tolist()
        labels = synthetic_frame["label"].tolist()
        splits, grouping = make_splits(
            texts, labels, scheme="grouped", n_splits=4, n_repeats=2, seed=0
        )
        assert splits
        for train_idx, test_idx in splits:
            train_groups = set(grouping.groups[train_idx].tolist())
            test_groups = set(grouping.groups[test_idx].tolist())
            assert train_groups.isdisjoint(test_groups), (
                "a template appeared on both sides of a grouped split"
            )

    def test_naive_split_does_leak(self, synthetic_frame):
        """Guards the comparison: if naive stopped leaking, the report's premise is void."""
        texts = synthetic_frame["text"].tolist()
        labels = synthetic_frame["label"].tolist()
        report = leakage_report(texts, labels, n_splits=4)
        assert report["naive"]["leak_rate"] > 0.5
        assert report["grouped"]["leak_rate"] == 0.0

    def test_every_row_tested_exactly_once_per_repeat(self, synthetic_frame):
        texts = synthetic_frame["text"].tolist()
        labels = synthetic_frame["label"].tolist()
        n_splits, n_repeats = 4, 3
        splits, _ = make_splits(
            texts, labels, scheme="grouped", n_splits=n_splits, n_repeats=n_repeats, seed=0
        )
        counts = np.zeros(len(texts), dtype=int)
        for _, test_idx in splits:
            counts[test_idx] += 1
        assert (counts == n_repeats).all()

    def test_train_and_test_are_disjoint(self, synthetic_frame):
        splits, _ = make_splits(
            synthetic_frame["text"].tolist(), synthetic_frame["label"].tolist(),
            scheme="grouped", n_splits=4, n_repeats=1, seed=0,
        )
        for train_idx, test_idx in splits:
            assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())

    def test_unknown_scheme_rejected(self, synthetic_frame):
        with pytest.raises(ValueError, match="unknown cv scheme"):
            make_splits(
                synthetic_frame["text"].tolist(), synthetic_frame["label"].tolist(),
                scheme="random",
            )

    def test_too_few_groups_for_folds_is_an_error(self):
        texts = ["I cannot log in at all"] * 6
        labels = ["account-access"] * 6
        with pytest.raises(ValueError, match="groups"):
            make_splits(texts, labels, scheme="grouped", n_splits=5)


class TestHarness:
    def test_fresh_estimator_per_fold(self, synthetic_frame):
        """Reusing one estimator across folds is a leak; the harness must refit.

        References are retained rather than `id()`s: CPython reuses addresses once an
        object is collected, so comparing ids would pass even if a single instance were
        reused.
        """
        fitted: list[object] = []

        class Counting:
            def fit(self, texts, labels):
                fitted.append(self)
                self._label = labels[0]
                return self

            def predict(self, texts):
                return np.array([self._label] * len(texts), dtype=object)

        cross_validate(
            Counting, synthetic_frame["text"].tolist(), synthetic_frame["label"].tolist(),
            name="counting", scheme="grouped", n_splits=4, n_repeats=1,
        )
        assert len(fitted) == 4
        assert len({id(obj) for obj in fitted}) == 4

    def test_mismatched_prediction_length_is_caught(self, synthetic_frame):
        class Truncating:
            def fit(self, texts, labels):
                return self

            def predict(self, texts):
                return np.array(["general"] * (len(texts) - 1), dtype=object)

        with pytest.raises(RuntimeError, match="predictions"):
            cross_validate(
                Truncating, synthetic_frame["text"].tolist(),
                synthetic_frame["label"].tolist(), name="bad",
                scheme="grouped", n_splits=4, n_repeats=1,
            )

    def test_identical_folds_across_models(self, synthetic_frame):
        texts, labels = synthetic_frame["text"].tolist(), synthetic_frame["label"].tolist()
        a, _ = make_splits(texts, labels, scheme="grouped", n_splits=4, n_repeats=2, seed=11)
        b, _ = make_splits(texts, labels, scheme="grouped", n_splits=4, n_repeats=2, seed=11)
        for (tr1, te1), (tr2, te2) in zip(a, b, strict=True):
            assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)

    def test_records_generative_parse_failures(self, synthetic_frame):
        class FailingParser:
            def fit(self, texts, labels):
                self.parse_failures_ = 0
                return self

            def predict(self, texts):
                self.parse_failures_ = len(texts)
                return np.array(["general"] * len(texts), dtype=object)

        result = cross_validate(
            FailingParser,
            synthetic_frame["text"].tolist(),
            synthetic_frame["label"].tolist(),
            name="failing-parser",
            scheme="grouped",
            n_splits=4,
            n_repeats=1,
        )

        assert result.parse_failures == len(synthetic_frame)
        assert result.prediction_rows == len(synthetic_frame)
        assert result.to_row()["parse_failure_rate"] == 1.0

    @pytest.mark.slow
    def test_real_model_beats_majority_baseline(self, synthetic_frame):
        """An end-to-end guard that the pipeline actually learns something."""
        texts, labels = synthetic_frame["text"].tolist(), synthetic_frame["label"].tolist()
        splits, grouping = make_splits(
            texts, labels, scheme="grouped", n_splits=4, n_repeats=1, seed=0
        )
        kwargs = dict(splits=splits, grouping=grouping, scheme="grouped")
        dummy = cross_validate(
            lambda: get_builder("most_frequent")(), texts, labels, name="dummy", **kwargs
        )
        model = cross_validate(
            lambda: get_builder("logistic_regression")(), texts, labels, name="lr", **kwargs
        )
        assert model.macro_f1 > dummy.macro_f1
