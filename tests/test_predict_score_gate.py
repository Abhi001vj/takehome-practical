"""The serving contract, the holdout scorer, and the promotion gate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from support_router.config import LABELS
from support_router.data import DataValidationError
from support_router.inference import ModelNotTrainedError, Prediction, predict, predict_batch
from support_router.models import FAMILIES, get_builder, resolve_names
from support_router.models.llm import FALLBACK_LABEL, load_llm_config, parse_label
from support_router.promotion import render_verdict
from support_router.score import detect_text_column, score_file


class TestPredictContract:
    def test_returns_a_valid_route(self, trained_model_dir):
        label = predict("I cannot log into my account", model_path=trained_model_dir)
        assert label in LABELS
        assert isinstance(label, str)

    def test_rejects_invalid_input_before_touching_the_model(self, trained_model_dir):
        for bad in ["", "  ", None, 123]:
            with pytest.raises(DataValidationError):
                predict(bad, model_path=trained_model_dir)

    def test_with_scores_returns_probability_distribution(self, trained_model_dir):
        result = predict(
            "Someone withdrew funds I never authorized",
            model_path=trained_model_dir,
            with_scores=True,
        )
        assert isinstance(result, Prediction)
        assert result.label in LABELS
        assert 0.0 <= result.confidence <= 1.0
        assert set(result.scores) == set(LABELS)
        assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-6)
        # The reported confidence must be the winning class's probability.
        assert result.scores[result.label] == pytest.approx(result.confidence)

    def test_untrained_model_gives_actionable_error(self, tmp_path):
        from support_router import inference as predict_module

        predict_module.reset_cache()
        with pytest.raises(ModelNotTrainedError, match="support-router train"):
            predict("I cannot log in", model_path=tmp_path / "empty")

    def test_whitespace_is_normalised_not_rejected(self, trained_model_dir):
        a = predict("I cannot log in", model_path=trained_model_dir)
        b = predict("  I   cannot\n log in  ", model_path=trained_model_dir)
        assert a == b

    def test_selected_embedding_model_round_trips_without_network(
        self, tmp_path, synthetic_frame, monkeypatch
    ):
        """Exercise the selected artifact type while replacing only its external encoder."""
        from support_router import inference as predict_module
        from support_router.models import embeddings
        from support_router.train import train

        vocabulary = ("login", "password", "withdraw", "fraud", "fee", "tax", "pending", "phishing")

        def fake_encode(texts, model_name=None, batch_size=64):
            return np.asarray(
                [[text.lower().count(token) for token in vocabulary] for text in texts],
                dtype=float,
            )

        monkeypatch.setattr(embeddings, "encode", fake_encode)
        data_path = tmp_path / "train.csv"
        synthetic_frame.to_csv(data_path, index=False)
        out_dir = tmp_path / "embedding-artifact"
        _, meta = train(
            model="embedding_logreg",
            out_dir=out_dir,
            data_path=data_path,
            evaluate=False,
            track=False,
        )

        predict_module.reset_cache()
        label = predict("Someone made a withdrawal I did not authorize", model_path=out_dir)
        assert meta.model_name == "embedding_logreg"
        assert label in LABELS
        predict_module.reset_cache()


class TestPredictBatch:
    def test_batch_matches_single_calls(self, trained_model_dir):
        texts = [
            "I cannot log into my account",
            "How long do withdrawals take?",
            "Someone withdrew funds I never authorized",
        ]
        batch = predict_batch(texts, model_path=trained_model_dir)
        singles = [predict(t, model_path=trained_model_dir) for t in texts]
        assert batch == singles

    def test_skip_invalid_preserves_positions(self, trained_model_dir):
        results = predict_batch(
            ["I cannot log in", "  ", "How do fees work?"],
            model_path=trained_model_dir,
            skip_invalid=True,
        )
        assert len(results) == 3
        assert results[1] is None
        assert results[0] in LABELS and results[2] in LABELS

    def test_raises_on_invalid_by_default(self, trained_model_dir):
        with pytest.raises(DataValidationError):
            predict_batch(["I cannot log in", ""], model_path=trained_model_dir)

    def test_empty_batch(self, trained_model_dir):
        assert predict_batch([], model_path=trained_model_dir) == []


class TestScoreFile:
    def test_writes_predictions_and_preserves_columns(self, tmp_path, trained_model_dir):
        source = tmp_path / "holdout.csv"
        pd.DataFrame(
            {
                "ticket_id": [1, 2, 3],
                "text": [
                    "I cannot log into my account",
                    "How long do withdrawals take?",
                    "Someone withdrew funds I never authorized",
                ],
            }
        ).to_csv(source, index=False)

        out = tmp_path / "preds.csv"
        summary = score_file(source, out, model_path=trained_model_dir)

        assert summary["rows_scored"] == 3
        result = pd.read_csv(out)
        # The id column must survive so a submission can be joined back.
        assert list(result.columns) == ["ticket_id", "text", "predicted_label"]
        assert result["ticket_id"].tolist() == [1, 2, 3]
        assert set(result["predicted_label"]) <= set(LABELS)

    def test_bad_rows_are_reported_not_dropped(self, tmp_path, trained_model_dir):
        source = tmp_path / "messy.csv"
        pd.DataFrame({"text": ["I cannot log in", "  ", None, "What are the fees?"]}).to_csv(
            source, index=False
        )
        out = tmp_path / "preds.csv"
        summary = score_file(source, out, model_path=trained_model_dir)

        assert summary["rows_scored"] == 2
        assert summary["rows_rejected"] == 2
        # Row count is preserved so predictions remain aligned to their source rows.
        assert len(pd.read_csv(out)) == 4
        assert pd.read_csv(summary["rejects_file"]).shape[0] == 2

    def test_evaluates_when_labels_present(self, tmp_path, trained_model_dir, synthetic_frame):
        source = tmp_path / "labelled.csv"
        synthetic_frame.to_csv(source, index=False)
        out = tmp_path / "preds.csv"
        summary = score_file(source, out, model_path=trained_model_dir)

        assert "evaluation" in summary
        assert 0.0 <= summary["evaluation"]["macro_f1"] <= 1.0
        metrics = json.loads((out.with_suffix(".metrics.json")).read_text())
        assert "critical_recall" in metrics

    def test_detects_alternative_text_columns(self):
        assert detect_text_column(pd.DataFrame({"message": ["x"], "id": [1]})) == "message"
        assert detect_text_column(pd.DataFrame({"anything": ["x"]})) == "anything"
        with pytest.raises(DataValidationError, match="--text-column"):
            detect_text_column(pd.DataFrame({"a": ["x"], "b": ["y"]}))

    def test_explicit_column_wins(self, tmp_path, trained_model_dir):
        source = tmp_path / "in.csv"
        pd.DataFrame({"text": ["ignored"], "body": ["I cannot log in"]}).to_csv(
            source, index=False
        )
        summary = score_file(
            source, tmp_path / "o.csv", model_path=trained_model_dir, text_column="body"
        )
        assert summary["text_column"] == "body"


class TestRegistry:
    def test_families_expand(self):
        assert "logistic_regression" in resolve_names("baselines")
        assert set(resolve_names("trees")) == set(FAMILIES["trees"])

    def test_names_deduplicated_and_ordered(self):
        assert resolve_names("logistic_regression,logistic_regression,most_frequent") == [
            "logistic_regression",
            "most_frequent",
        ]

    def test_unknown_model_lists_options(self):
        with pytest.raises(KeyError, match="unknown model"):
            get_builder("gradient_boosting_9000")


class TestLLMParsing:
    """The generative arm can emit anything; parsing must not be optimistic."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("fraud-report", "fraud-report"),
            ("Fraud Report", "fraud-report"),
            ("  GENERAL  ", "general"),
            ("route: account-access", "account-access"),
            ("The answer is transaction-dispute.", "transaction-dispute"),
            ('"fraud-report"', "fraud-report"),
        ],
    )
    def test_parses_known_forms(self, raw, expected):
        assert parse_label(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "banana", "I am not sure"])
    def test_returns_none_rather_than_guessing(self, raw):
        assert parse_label(raw) is None

    def test_four_label_compatibility_fallback_is_general(self):
        """Invalid output is visible and deterministic until human review exists."""
        assert FALLBACK_LABEL == "general"

    def test_endpoint_and_served_model_can_be_overridden(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "qwen2.5:1.5b")
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")

        config = load_llm_config(
            {
                "generative_model": "Qwen/Qwen2.5-1.5B-Instruct",
                "base_url": "http://localhost:8001/v1",
            }
        )

        assert config["model"] == "qwen2.5:1.5b"
        assert config["base_url"] == "http://localhost:11434/v1"


class TestPromotionGate:
    def _verdict(self, candidate, champion, policy=None):
        """Build a verdict the same way `evaluate_candidate` would, without MLflow."""
        from support_router.promotion import evaluate_candidate

        return evaluate_candidate, candidate, champion, policy

    def test_first_model_passes_when_it_clears_the_fraud_floor(self, tmp_path, monkeypatch):
        import support_router.promotion as promo

        monkeypatch.setattr(promo, "get_champion_metrics", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(
            "support_router.tracking.get_champion_metrics", lambda *a, **k: None, raising=False
        )
        path = tmp_path / "cand.json"
        path.write_text(
            json.dumps({"model": "lr", "macro_f1": 0.85, "critical_recall": 0.9,
                        "macro_f1__std": 0.03})
        )
        verdict = promo.evaluate_candidate(metrics_path=path)
        assert verdict["passed"] is True

    def test_fails_when_fraud_recall_below_floor(self, tmp_path, monkeypatch):
        import support_router.promotion as promo

        monkeypatch.setattr(
            "support_router.tracking.get_champion_metrics", lambda *a, **k: None, raising=False
        )
        path = tmp_path / "cand.json"
        path.write_text(
            json.dumps({"model": "lr", "macro_f1": 0.95, "critical_recall": 0.4,
                        "macro_f1__std": 0.02})
        )
        verdict = promo.evaluate_candidate(metrics_path=path)
        assert verdict["passed"] is False
        failed = [c["name"] for c in verdict["checks"] if not c["passed"]]
        assert "fraud-report_recall_floor" in failed

    def test_marginal_gain_does_not_count_as_beating_the_champion(self, tmp_path, monkeypatch):
        import support_router.promotion as promo

        monkeypatch.setattr(
            "support_router.tracking.get_champion_metrics",
            lambda *a, **k: {"version": "1", "macro_f1": 0.900,
                             "critical_recall": 0.85, "macro_f1__std": 0.03},
            raising=False,
        )
        path = tmp_path / "cand.json"
        path.write_text(
            json.dumps({"model": "lr", "macro_f1": 0.9005, "critical_recall": 0.86,
                        "macro_f1__std": 0.03})
        )
        verdict = promo.evaluate_candidate(metrics_path=path)
        assert verdict["passed"] is False
        assert "macro_f1_improvement" in [c["name"] for c in verdict["checks"] if not c["passed"]]

    def test_macro_f1_win_that_regresses_fraud_recall_is_blocked(self, tmp_path, monkeypatch):
        """The trade the gate exists to prevent."""
        import support_router.promotion as promo

        monkeypatch.setattr(
            "support_router.tracking.get_champion_metrics",
            lambda *a, **k: {"version": "1", "macro_f1": 0.900,
                             "critical_recall": 0.95, "macro_f1__std": 0.03},
            raising=False,
        )
        path = tmp_path / "cand.json"
        path.write_text(
            json.dumps({"model": "lr", "macro_f1": 0.94, "critical_recall": 0.86,
                        "macro_f1__std": 0.03})
        )
        verdict = promo.evaluate_candidate(metrics_path=path)
        assert verdict["passed"] is False
        assert "no_critical_regression" in [
            c["name"] for c in verdict["checks"] if not c["passed"]
        ]

    def test_genuine_improvement_passes(self, tmp_path, monkeypatch):
        import support_router.promotion as promo

        monkeypatch.setattr(
            "support_router.tracking.get_champion_metrics",
            lambda *a, **k: {"version": "1", "macro_f1": 0.80,
                             "critical_recall": 0.82, "macro_f1__std": 0.04},
            raising=False,
        )
        path = tmp_path / "cand.json"
        path.write_text(
            json.dumps({"model": "emb", "macro_f1": 0.93, "critical_recall": 0.94,
                        "macro_f1__std": 0.03})
        )
        verdict = promo.evaluate_candidate(metrics_path=path)
        assert verdict["passed"] is True
        assert "PASS" in render_verdict(verdict)

    def test_unchanged_champion_passes_non_regression_check(self, tmp_path, monkeypatch):
        import support_router.promotion as promo

        monkeypatch.setattr(
            "support_router.tracking.get_champion_metrics",
            lambda *a, **k: {
                "version": "1",
                "name": "embedding_logreg",
                "macro_f1": 0.91,
                "critical_recall": 0.90,
                "macro_f1__std": 0.03,
            },
            raising=False,
        )
        path = tmp_path / "cand.json"
        path.write_text(
            json.dumps(
                {
                    "model": "embedding_logreg",
                    "macro_f1": 0.91,
                    "critical_recall": 0.90,
                    "macro_f1__std": 0.03,
                }
            )
        )
        verdict = promo.evaluate_candidate(metrics_path=path)
        assert verdict["passed"] is True
        assert "champion_non_regression" in [check["name"] for check in verdict["checks"]]

    def test_reads_a_comparison_report(self, tmp_path, monkeypatch):
        """The gate must consume `support-router cv` output directly, picking grouped results."""
        import support_router.promotion as promo

        monkeypatch.setattr(
            "support_router.tracking.get_champion_metrics", lambda *a, **k: None, raising=False
        )
        path = tmp_path / "comparison.json"
        path.write_text(
            json.dumps(
                {
                    "results": [
                        # The naive row scores higher and must be ignored.
                        {"model": "lr", "scheme": "naive", "macro_f1": 0.99,
                         "critical_recall": 0.99, "macro_f1_std": 0.01},
                        {"model": "lr", "scheme": "grouped", "macro_f1": 0.86,
                         "critical_recall": 0.88, "macro_f1_std": 0.05},
                    ]
                }
            )
        )
        verdict = promo.evaluate_candidate(metrics_path=path)
        assert verdict["candidate"]["macro_f1"] == pytest.approx(0.86)

    def test_multi_model_comparison_gates_configured_candidate(self, tmp_path, monkeypatch):
        import support_router.promotion as promo

        monkeypatch.setattr(
            "support_router.tracking.get_champion_metrics", lambda *a, **k: None, raising=False
        )
        comparison = tmp_path / "comparison.json"
        comparison.write_text(
            json.dumps(
                {
                    "results": [
                        {"model": "good", "scheme": "grouped", "macro_f1": 0.95,
                         "critical_recall": 0.95, "macro_f1_std": 0.02},
                        {"model": "candidate", "scheme": "grouped", "macro_f1": 0.70,
                         "critical_recall": 0.40, "macro_f1_std": 0.04},
                    ]
                }
            )
        )
        params = tmp_path / "params.yaml"
        params.write_text(
            """seed: 1
grouping: {similarity_threshold: 0.85, char_ngram_range: [3, 5]}
cv: {n_splits: 2, n_repeats: 1, schemes: [grouped]}
train: {model: candidate}
promotion:
  registered_model_name: support-router
  min_improvement: 0.005
  critical_recall_floor: 0.80
  max_std_ratio: 1.5
llm: {}
"""
        )

        verdict = promo.evaluate_candidate(metrics_path=comparison, params_path=params)
        assert verdict["candidate"]["name"] == "candidate"
        assert verdict["candidate"]["macro_f1"] == pytest.approx(0.70)
        assert verdict["passed"] is False
