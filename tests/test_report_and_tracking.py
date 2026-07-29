"""Report and clean-runner MLflow behavior."""

from __future__ import annotations

import json


def test_benchmark_writes_measured_latency(
    monkeypatch, tmp_path, trained_model_dir, synthetic_frame
):
    import support_router.benchmark as benchmark

    monkeypatch.setattr(benchmark, "load_training_data", lambda: synthetic_frame)
    destination = tmp_path / "latency.json"
    result = benchmark.benchmark_inference(
        model_path=trained_model_dir,
        out_path=destination,
        sample_size=4,
        batch_repeats=1,
    )

    assert result["sample_size"] == 4
    assert result["median_ms"] > 0
    assert result["batch_throughput_per_second"] > 0
    assert json.loads(destination.read_text())["sample_size"] == 4


def test_checked_in_champion_fallback(monkeypatch, tmp_path):
    import support_router.tracking as tracking

    benchmark = {
        "name": "embedding_logreg",
        "macro_f1": 0.91,
        "critical_recall": 0.90,
        "macro_f1__std": 0.03,
    }
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "champion.json").write_text(json.dumps(benchmark))
    monkeypatch.setattr(tracking, "PROJECT_ROOT", tmp_path)

    assert tracking._benchmark_fallback() == benchmark


def test_report_marks_missing_latency_as_unavailable(monkeypatch, tmp_path):
    import support_router.report as report

    reports = tmp_path / "reports"
    artifacts = tmp_path / "artifacts"
    reports.mkdir()
    artifacts.mkdir()
    (reports / "comparison.json").write_text("{\"results\": []}")

    monkeypatch.setattr(report, "REPORTS", reports)
    monkeypatch.setattr(report, "ARTIFACTS", artifacts)
    monkeypatch.setattr(
        report,
        "_tracking_summary",
        lambda: {
            "uri": "sqlite:////private/machine/path/mlflow.db",
            "experiment": "support-routing",
            "runs": None,
            "traces": None,
            "versions": [],
            "champion_version": None,
        },
    )

    destination = report.generate_report(reports / "REPORT.md")
    content = destination.read_text()
    assert "0.0 ms" not in content
    assert "local SQLite store (`mlflow.db`)" in content
    assert "**NOT RUN**" in content
