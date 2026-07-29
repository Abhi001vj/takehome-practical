"""Input validation and metric behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from support_router.config import CRITICAL_LABEL, LABELS
from support_router.data import (
    MAX_TEXT_CHARS,
    DataValidationError,
    class_distribution,
    load_messages_for_scoring,
    load_training_data,
    validate_text,
)
from support_router.metrics import aggregate, pooled_confusion, score_fold


class TestValidateText:
    def test_normalises_whitespace(self):
        assert validate_text("  I  cannot\n\tlog in  ") == "I cannot log in"

    @pytest.mark.parametrize("bad", ["", "   ", "\n\t", "a"])
    def test_rejects_empty_and_too_short(self, bad):
        with pytest.raises(DataValidationError):
            validate_text(bad)

    @pytest.mark.parametrize("bad", [None, 42, 3.5, [], {}, True])
    def test_rejects_non_strings(self, bad):
        with pytest.raises(DataValidationError):
            validate_text(bad)

    def test_rejects_nan(self):
        with pytest.raises(DataValidationError, match="missing"):
            validate_text(float("nan"))

    def test_rejects_oversized(self):
        with pytest.raises(DataValidationError, match="limit"):
            validate_text("x " * MAX_TEXT_CHARS)

    def test_accepts_a_realistic_message(self):
        text = "Hello team, The app won't let me sign in, it just spins. Thanks."
        assert validate_text(text) == text


class TestLoadTrainingData:
    def test_missing_file_explains_expected_location(self, tmp_path):
        with pytest.raises(DataValidationError, match="data/raw/train.csv"):
            load_training_data(tmp_path / "nope.csv")

    def test_rejects_unknown_label(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["I cannot log in"], "label": ["billing"]}).to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="outside the four known routes"):
            load_training_data(path)

    def test_rejects_missing_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"message": ["I cannot log in"], "label": ["general"]}).to_csv(
            path, index=False
        )
        with pytest.raises(DataValidationError, match="missing required column"):
            load_training_data(path)

    def test_rejects_blank_text_rows(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame(
            {"text": ["I cannot log in", "   "], "label": ["account-access", "general"]}
        ).to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="unusable text"):
            load_training_data(path)

    def test_real_data_shape(self, real_frame):
        assert list(real_frame.columns) == ["text", "label"]
        assert len(real_frame) > 0
        assert set(real_frame["label"]) <= set(LABELS)

    def test_class_distribution_uses_fixed_order(self, real_frame):
        assert list(class_distribution(real_frame).index) == list(LABELS)


class TestLoadForScoring:
    def test_separates_good_and_bad_rows(self, tmp_path):
        path = tmp_path / "in.csv"
        pd.DataFrame({"text": ["I cannot log in", "  ", None, "What are the fees?"]}).to_csv(
            path, index=False
        )
        good, bad = load_messages_for_scoring(path)
        assert len(good) == 2
        assert len(bad) == 2
        # Original positions must survive so results can be realigned.
        assert good["source_index"].tolist() == [0, 3]
        assert bad["source_index"].tolist() == [1, 2]

    def test_wholly_empty_rows_survive_the_read(self, tmp_path):
        """Regression: pandas drops blank lines by default.

        A single-column CSV containing an empty row loses that row on read, so the
        output would have fewer rows than the input and every row after the gap would
        be misaligned against the source file.
        """
        path = tmp_path / "in.csv"
        path.write_text("text\nI cannot log in\n\nWhat are the fees?\n")
        good, bad = load_messages_for_scoring(path)
        assert len(good) + len(bad) == 3
        assert good["source_index"].tolist() == [0, 2]

    def test_missing_column_error_is_actionable(self, tmp_path):
        path = tmp_path / "in.csv"
        pd.DataFrame({"body": ["I cannot log in"]}).to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="--text-column"):
            load_messages_for_scoring(path)

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "in.csv"
        pd.DataFrame({"text": []}).to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="no rows"):
            load_messages_for_scoring(path)


class TestMetrics:
    def test_majority_classifier_exposes_the_accuracy_trap(self):
        """The argument for macro-F1, encoded as a test.

        A classifier that always answers `general` scores respectable accuracy and
        catastrophic macro-F1, and misses every single fraud report.
        """
        y_true = np.array(
            ["general"] * 40 + ["account-access"] * 25 + ["transaction-dispute"] * 23
            + ["fraud-report"] * 12
        )
        y_pred = np.array(["general"] * len(y_true))
        scores = score_fold(y_true, y_pred)

        assert scores.accuracy == pytest.approx(0.40)
        assert scores.macro_f1 < 0.15
        assert scores.critical_recall == 0.0
        assert scores.fraud_leak_rate == 1.0

    def test_perfect_prediction(self):
        y = np.array(list(LABELS) * 5)
        scores = score_fold(y, y.copy())
        assert scores.macro_f1 == pytest.approx(1.0)
        assert scores.critical_recall == pytest.approx(1.0)
        assert scores.fraud_leak_rate == pytest.approx(0.0)

    def test_absent_class_does_not_shift_per_class_values(self):
        """With 50 fraud rows over 5 folds a class can vanish from a fold."""
        y_true = np.array(["general", "general", "account-access"])
        y_pred = np.array(["general", "general", "account-access"])
        scores = score_fold(y_true, y_pred)
        assert set(scores.per_class_f1) == set(LABELS)
        assert scores.per_class_f1["fraud-report"] == 0.0
        assert scores.support["fraud-report"] == 0

    def test_fraud_leak_rate_is_complement_of_recall(self):
        y_true = np.array(["fraud-report"] * 10)
        y_pred = np.array(["fraud-report"] * 7 + ["general"] * 3)
        with pytest.warns(UserWarning, match="classes not in y_true"):
            scores = score_fold(y_true, y_pred)
        assert scores.critical_recall == pytest.approx(0.7)
        assert scores.fraud_leak_rate == pytest.approx(0.3)

    def test_confusion_matrix_orientation(self):
        """Rows are truth, columns are prediction - the report depends on this."""
        y_true = np.array([CRITICAL_LABEL, CRITICAL_LABEL])
        y_pred = np.array([CRITICAL_LABEL, "general"])
        matrix = pooled_confusion(y_true, y_pred)
        fraud, general = LABELS.index(CRITICAL_LABEL), LABELS.index("general")
        assert matrix[fraud, fraud] == 1
        assert matrix[fraud, general] == 1

    def test_aggregate_reports_spread(self):
        y = np.array(list(LABELS))
        good = score_fold(y, y.copy())
        bad = score_fold(y, np.array(["general"] * 4))
        agg = aggregate([good, bad], pooled_confusion(y, y.copy()))
        assert agg.n_folds == 2
        assert 0 < agg.mean["macro_f1"] < 1
        assert agg.std["macro_f1"] > 0
