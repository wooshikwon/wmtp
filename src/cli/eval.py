"""
Evaluation entrypoint for WMTP Fine-Tuning Framework.

This module provides the main CLI interface for evaluating trained models
on MBPP and CodeContests benchmarks following Meta MTP protocols.
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.traceback import install

install(show_locals=False)
console = Console()
app = typer.Typer(
    name="wmtp-eval",
    help="WMTP Evaluation CLI - Evaluate models on coding benchmarks",
    pretty_exceptions_show_locals=False,
)


@app.command()
def evaluate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to environment configuration YAML file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    recipe: Path = typer.Option(
        ...,
        "--recipe",
        "-r",
        help="Path to evaluation recipe YAML file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        help="Path to model checkpoint to evaluate",
        exists=True,
        dir_okay=False,
    ),
    datasets: str | None = typer.Option(
        None,
        "--datasets",
        "-d",
        help="Comma-separated list of datasets to evaluate (default: all in recipe)",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save evaluation results",
    ),
    save_predictions: bool = typer.Option(
        False,
        "--save-predictions",
        help="Save model predictions to file",
    ),
    save_report: bool = typer.Option(
        False,
        "--save-report",
        help="Generate and save detailed evaluation report",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Override batch size from recipe",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """
    Evaluate a trained model on coding benchmarks.

    This command runs evaluation following Meta MTP protocols:
    - MBPP: Exact match evaluation
    - CodeContests: Pass@k evaluation
    - Generates detailed metrics and reports
    - Logs results to MLflow
    """
    console.print("[bold blue]WMTP Evaluation Framework[/bold blue]")
    console.print(f"Config: {config}")
    console.print(f"Recipe: {recipe}")
    console.print(f"Checkpoint: {checkpoint}")

    # Parse datasets if provided
    dataset_list = []
    if datasets:
        dataset_list = [d.strip() for d in datasets.split(",")]
        console.print(f"Datasets: {dataset_list}")
    else:
        console.print("Datasets: [all from recipe]")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Output directory: {output_dir}")

    try:
        from src.pipelines import EvaluationPipeline
        from src.settings import load_config, load_recipe

        # Load configuration
        cfg = load_config(config, verbose=verbose)
        rcp = load_recipe(recipe, verbose=verbose)

        # Create and run evaluation pipeline
        pipeline = EvaluationPipeline(cfg, rcp)

        # Parse tags if provided
        tag_list = []
        if output_dir:  # Use output_dir as tag indicator
            tag_list.append("eval")

        evaluation_results = pipeline.run(
            checkpoint=checkpoint,
            datasets=dataset_list,
            run_name=f"eval_{rcp.run.name}",
            tags=tag_list,
            save_predictions=save_predictions,
            save_report=save_report,
        )

        # Extract metrics for display
        results_metrics = evaluation_results.get("results", {}).get("metrics", {})

        # Display results table
        table = Table(title="Evaluation Results")
        table.add_column("Dataset", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Score", style="green")

        # Group metrics by dataset for better display
        dataset_metrics = {}
        for metric_name, metric_value in results_metrics.items():
            if isinstance(metric_value, int | float):
                # Parse metric name to extract dataset
                if metric_name.startswith("mbpp"):
                    dataset = "MBPP"
                    metric_key = metric_name.replace("mbpp_", "").replace(
                        "mbpp", "exact_match"
                    )
                elif "contest" in metric_name:
                    dataset = "CodeContests"
                    metric_key = metric_name.replace("contest_", "").replace(
                        "codecontests_", ""
                    )
                else:
                    dataset = "Other"
                    metric_key = metric_name

                if dataset not in dataset_metrics:
                    dataset_metrics[dataset] = {}
                dataset_metrics[dataset][metric_key] = metric_value

        # Add rows to table
        for dataset, metrics in dataset_metrics.items():
            for metric, score in metrics.items():
                if isinstance(score, float) and 0 <= score <= 1:
                    score_str = f"{score:.2%}"
                else:
                    score_str = f"{score:.4f}"
                table.add_row(dataset, metric, score_str)

        console.print(table)

        # Summary information
        console.print("\n[green]Evaluation completed successfully![/green]")
        console.print(f"Algorithm: {rcp.train.algo}")
        console.print(f"Model: {rcp.model.base_id}")
        console.print(f"Checkpoint: {checkpoint}")

        # Save results to output directory if specified
        if output_dir:
            import json

            results_file = output_dir / "evaluation_results.json"
            with open(results_file, "w") as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            console.print(f"[green]Results saved to: {results_file}[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    app()
