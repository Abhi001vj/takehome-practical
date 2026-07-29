"""Command line interface.

    support-router eda         dataset stats, distributions, wordclouds, plots
    support-router cv          compare every approach on identical folds
    support-router leakage     quantify the template-leakage problem
    support-router test        run the test suite
    support-router experiment  EDA + full comparison + trained model, end to end
    support-router tune        Optuna search, logged to MLflow
    support-router train       fit and persist the chosen model
    support-router predict     classify one message
    support-router score       score a CSV (the holdout entry point)
    support-router gate        promotion check used by CI
    support-router benchmark   measure persisted-model inference latency
    support-router promote     move the MLflow champion alias
    support-router report      assemble the end-to-end Markdown report
    support-router serve       run the API
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import ARTIFACTS, PROJECT_ROOT, REPORTS, load_params

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Route a customer support message to one of four support queues.",
)
console = Console()


@app.command()
def cv(
    models: str = typer.Option("all", help="Family (dummy/classical/trees/llm/all) or names."),
    schemes: str = typer.Option("grouped,naive", help="CV schemes to run."),
    n_splits: int = typer.Option(None, help="Folds per repeat."),
    n_repeats: int = typer.Option(None, help="CV repeats."),
    seed: int = typer.Option(None),
    out: Path = typer.Option(None, help="Report directory."),
    track: bool = typer.Option(True, help="Log runs to MLflow."),
    append: bool = typer.Option(False, help="Merge these rows into an existing comparison."),
) -> None:
    """Compare approaches on identical folds and write reports/comparison.md."""
    from .experiment import run_comparison, write_report

    scheme_list = [s.strip() for s in schemes.split(",") if s.strip()]
    console.print(f"[bold]Comparing[/bold] {models} under {scheme_list}")

    report = run_comparison(
        models=models, schemes=scheme_list, n_splits=n_splits,
        n_repeats=n_repeats, seed=seed, track=track,
    )

    leak = report.leakage
    console.print(
        f"\n[bold]Leakage[/bold]  naive: {leak['naive']['leak_rate']:.1%} of test rows "
        f"have a near-duplicate in train   |   grouped: {leak['grouped']['leak_rate']:.1%}"
    )
    console.print(f"[dim]{leak['grouping']}[/dim]\n")

    frame = report.to_frame()
    for scheme in scheme_list:
        subset = frame[frame["scheme"] == scheme]
        if subset.empty:
            continue
        table = Table(title=f"{scheme} CV", show_lines=False)
        table.add_column("model", style="cyan")
        table.add_column("macro-F1", justify="right")
        table.add_column("±std", justify="right")
        table.add_column("fraud recall", justify="right")
        table.add_column("accuracy", justify="right")
        table.add_column("fit s", justify="right")
        for _, row in subset.iterrows():
            table.add_row(
                row["model"],
                f"{row['macro_f1']:.3f}",
                f"{row['macro_f1_std']:.3f}",
                f"{row['critical_recall']:.3f}",
                f"{row['accuracy']:.3f}",
                f"{row['fit_seconds']:.2f}",
            )
        console.print(table)

    if report.errors:
        console.print("\n[yellow]Skipped:[/yellow]")
        for name, err in report.errors.items():
            console.print(f"  {name}: {err}")

    path = write_report(report, out, append=append)
    console.print(f"\n[green]Report written to {path}[/green]")


@app.command()
def eda(
    out: Path = typer.Option(None, help="Report directory."),
    plots: bool = typer.Option(True, help="Render figures (needs the eda extra)."),
) -> None:
    """Profile the dataset: counts, distributions, wordclouds, discriminative terms."""
    from .eda import run_eda

    path, stats = run_eda(out_dir=out, plots=plots)

    table = Table(title="Class balance: rows vs templates")
    table.add_column("route", style="cyan")
    table.add_column("rows", justify="right")
    table.add_column("templates", justify="right")
    table.add_column("rows/template", justify="right")
    for label, rows in stats["row_counts"].items():
        templates = stats["template_counts"][label]
        table.add_row(
            label, str(rows), str(templates), f"{rows / templates:.1f}" if templates else "-"
        )
    console.print(table)
    console.print(f"[dim]{stats['grouping_summary']}[/dim]")
    console.print(
        f"Row-level imbalance [bold]{stats['row_imbalance_ratio']:.1f}:1[/bold]  |  "
        f"template-level [bold]{stats['template_imbalance_ratio']:.1f}:1[/bold]"
    )
    console.print(f"\n[green]EDA written to {path}[/green]")


@app.command()
def test(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    fast: bool = typer.Option(False, "--fast", help="Skip tests marked slow."),
    coverage: bool = typer.Option(False, help="Report coverage for the support_router package."),
    k: str = typer.Option(None, "-k", help="Only run tests matching this expression."),
) -> None:
    """Run the test suite."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "pytest"]
    cmd.append("-v" if verbose else "-q")
    markers = ["not llm"]  # the LLM arm needs a live endpoint; never required locally
    if fast:
        markers.append("not slow")
    cmd += ["-m", " and ".join(markers)]
    if k:
        cmd += ["-k", k]
    if coverage:
        cmd += ["--cov=support_router", "--cov-report=term-missing"]

    console.print(f"[dim]{' '.join(cmd)}[/dim]")
    raise typer.Exit(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))


@app.command()
def experiment(
    models: str = typer.Option(
        "all,embedding_logreg,embedding_lightgbm",
        help="Family or comma-separated names.",
    ),
    train_best: bool = typer.Option(True, help="Train and persist the winner."),
    plots: bool = typer.Option(True),
    track: bool = typer.Option(True, help="Log to MLflow."),
    register_winner: bool = typer.Option(False, help="Register the final winner in MLflow."),
) -> None:
    """Run EDA, comparison, training, benchmark, promotion gate, and final report."""
    from .eda import run_eda
    from .experiment import run_comparison, write_report
    from .train import train as train_model

    console.print("[bold cyan]1/6  EDA[/bold cyan]")
    eda_path, _ = run_eda(plots=plots)
    console.print(f"  -> {eda_path}")

    console.print("\n[bold cyan]2/6  Model comparison[/bold cyan]")
    report = run_comparison(models=models, track=track)
    comparison_path = write_report(report)
    console.print(f"  -> {comparison_path}")

    best = report.best("grouped")
    if best is None:
        console.print("[red]No model produced a grouped-CV result.[/red]")
        raise typer.Exit(1)

    console.print(
        f"\n  Best on grouped CV: [bold]{best.name}[/bold] "
        f"macro-F1 {best.macro_f1:.4f} ± {best.macro_f1_std:.4f}, "
        f"fraud recall {best.critical_recall:.4f}"
    )

    if train_best:
        console.print("\n[bold cyan]3/6  Training the winner[/bold cyan]")
        _, meta = train_model(model=best.name, track=track, register=register_winner)
        console.print(f"  -> artifacts/ ({meta.model_name})")

        console.print("\n[bold cyan]4/6  Inference benchmark[/bold cyan]")
        from .benchmark import benchmark_inference

        latency = benchmark_inference()
        console.print(
            f"  -> median {latency['median_ms']:.2f} ms, "
            f"p95 {latency['p95_ms']:.2f} ms"
        )
    else:
        console.print("\n[dim]3/6 and 4/6  skipped (--no-train-best)[/dim]")

    console.print("\n[bold cyan]5/6  Promotion gate[/bold cyan]")
    from .promotion import evaluate_candidate, render_verdict

    verdict = evaluate_candidate(metrics_path=REPORTS / "comparison.json", model=best.name)
    (REPORTS / "gate.json").write_text(json.dumps(verdict, indent=2, default=str))
    console.print(render_verdict(verdict))

    console.print("\n[bold cyan]6/6  Consolidated report[/bold cyan]")
    from .report import generate_report

    final_report = generate_report()
    console.print(f"  -> {final_report}")

    console.print("\n[green]Done.[/green]")


@app.command()
def leakage(seed: int = typer.Option(None), n_splits: int = typer.Option(5)) -> None:
    """Quantify how much of a naive split is already memorised."""
    from .cv import leakage_report
    from .data import load_training_data

    params = load_params()
    df = load_training_data()
    result = leakage_report(
        df["text"].tolist(), df["label"].tolist(),
        n_splits=n_splits, seed=seed if seed is not None else params["seed"],
    )
    console.print_json(json.dumps(result, indent=2))


@app.command()
def tune(
    models: str = typer.Option("logistic_regression", help="Comma-separated model names."),
    n_trials: int = typer.Option(None),
    timeout: int = typer.Option(None, help="Seconds per model."),
    seed: int = typer.Option(None),
) -> None:
    """Search hyperparameters on grouped folds, optimising macro-F1."""
    from .models import resolve_names
    from .tuning import summarise, tune_all

    names = resolve_names(models)
    console.print(f"[bold]Tuning[/bold] {names}")
    results = tune_all(names, n_trials=n_trials, timeout=timeout, seed=seed)
    console.print(summarise(results))

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "tuning.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    console.print(f"[green]Wrote {path}[/green]")


@app.command()
def train(
    model: str = typer.Option("embedding_logreg", help="Model name from the registry."),
    params: str = typer.Option(None, help='JSON overrides, e.g. \'{"C": 8.0}\'.'),
    out: Path = typer.Option(None, help="Artifact directory."),
    evaluate: bool = typer.Option(True, help="Run grouped CV before the final refit."),
    metrics_from: Path = typer.Option(
        None,
        help="Reuse this comparison JSON's grouped metrics with --no-evaluate.",
    ),
    track: bool = typer.Option(True, help="Log to MLflow."),
    register: bool = typer.Option(False, help="Register in the MLflow model registry."),
) -> None:
    """Fit on all data and persist to artifacts/."""
    from .train import train as train_model

    overrides = json.loads(params) if params else None
    console.print(f"[bold]Training[/bold] {model}")
    _, meta = train_model(
        model=model, model_params=overrides, out_dir=out,
        evaluate=evaluate, metrics_path=metrics_from, track=track, register=register,
    )
    console.print(
        f"[green]Trained[/green] {meta.model_name}  "
        f"grouped-CV macro-F1 {meta.cv_macro_f1:.4f} ± {meta.cv_macro_f1_std:.4f}  "
        f"fraud recall {meta.cv_critical_recall:.4f}"
    )
    console.print(f"[dim]Artifacts in {out or ARTIFACTS}[/dim]")


@app.command()
def predict(
    text: str = typer.Argument(..., help="The support message to route."),
    model_path: Path = typer.Option(None, help="Artifact directory."),
    scores: bool = typer.Option(False, "--scores", help="Include per-class probabilities."),
) -> None:
    """Classify a single message."""
    from .data import DataValidationError
    from .inference import ModelNotTrainedError
    from .inference import predict as predict_one

    try:
        result = predict_one(text, model_path=model_path, with_scores=scores)
    except DataValidationError as exc:
        console.print(f"[red]Invalid input:[/red] {exc}")
        raise typer.Exit(2) from exc
    except ModelNotTrainedError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc

    if scores:
        console.print_json(json.dumps(result.as_dict(), indent=2))
    else:
        console.print(result)


@app.command()
def score(
    input_csv: Path = typer.Argument(..., help="CSV of messages to score."),
    output_csv: Path = typer.Option(Path("predictions.csv"), "--output", "-o"),
    model_path: Path = typer.Option(None, help="Artifact directory."),
    text_column: str = typer.Option(None, help="Text column name (auto-detected)."),
    confidence: bool = typer.Option(False, help="Include a confidence column."),
) -> None:
    """Score a CSV of messages - the holdout entry point."""
    from .data import DataValidationError
    from .inference import ModelNotTrainedError
    from .score import score_file

    try:
        summary = score_file(
            input_csv, output_csv, model_path=model_path,
            text_column=text_column, with_confidence=confidence,
        )
    except (DataValidationError, ModelNotTrainedError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    console.print(
        f"[green]Scored[/green] {summary['rows_scored']}/{summary['rows_in']} rows "
        f"-> {summary['output']}"
    )
    if summary["rows_rejected"]:
        console.print(
            f"[yellow]{summary['rows_rejected']} row(s) could not be scored[/yellow] "
            f"-> {summary.get('rejects_file')}"
        )
    console.print(f"Predicted distribution: {summary['label_counts']}")

    if "evaluation" in summary:
        ev = summary["evaluation"]
        console.print(
            f"\n[bold]Evaluation[/bold]  macro-F1 {ev['macro_f1']:.4f}  "
            f"accuracy {ev['accuracy']:.4f}  fraud recall {ev['critical_recall']:.4f}"
        )
        console.print(summary["confusion_matrix"])


@app.command()
def gate(
    candidate_metrics: Path = typer.Option(None, help="JSON of candidate metrics."),
    model: str = typer.Option(None, help="Train this model and gate on its CV result."),
    fail_on_regression: bool = typer.Option(True),
) -> None:
    """Compare a candidate against the registered champion. Used by CI."""
    from .promotion import evaluate_candidate, render_verdict

    verdict = evaluate_candidate(metrics_path=candidate_metrics, model=model)
    # Plain print, not `console.print`: the verdict is a markdown table destined for a CI
    # log and a PR comment, and rich would soft-wrap the rows at terminal width, breaking
    # the table for anything that parses it.
    print(render_verdict(verdict))

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "gate.json").write_text(json.dumps(verdict, indent=2, default=str))

    if not verdict["passed"] and fail_on_regression:
        raise typer.Exit(1)


@app.command()
def benchmark(
    model_path: Path = typer.Option(None, help="Artifact directory."),
    out: Path = typer.Option(None, help="Output JSON path."),
    sample_size: int = typer.Option(60, min=1),
) -> None:
    """Measure warm-up, steady-state latency, and batch throughput."""
    from .benchmark import benchmark_inference

    result = benchmark_inference(model_path=model_path, out_path=out, sample_size=sample_size)
    console.print_json(json.dumps(result, indent=2))


@app.command()
def report(
    out: Path = typer.Option(None, help="Output Markdown path."),
) -> None:
    """Assemble the persisted EDA, model, timing, gate, and MLflow artifacts."""
    from .report import generate_report

    path = generate_report(out)
    console.print(f"[green]Report written to {path}[/green]")


@app.command()
def promote(
    version: str = typer.Option(..., help="Registered model version to alias as champion."),
    name: str = typer.Option(None, help="Registered model name; defaults to configuration."),
) -> None:
    """Move MLflow's champion alias after a candidate passes review."""
    from .tracking import set_champion

    registered_name = name or load_params().promotion["registered_model_name"]
    set_champion(registered_name, version)
    console.print(f"[green]{registered_name}@champion -> version {version}[/green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("support_router.api.service:app", host=host, port=port, reload=reload)


@app.command()
def info(model_path: Path = typer.Option(None)) -> None:
    """Show metadata for the trained model."""
    from .inference import ModelNotTrainedError, model_info

    try:
        console.print_json(json.dumps(model_info(model_path), indent=2))
    except ModelNotTrainedError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc


if __name__ == "__main__":
    app()
