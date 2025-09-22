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
        # TODO: Import and call actual evaluation pipeline
        # from src.pipelines import get_evaluator
        # from src.settings import load_config, load_recipe

        # config_dict = load_config(config)
        # recipe_dict = load_recipe(recipe)

        # evaluator = get_evaluator(recipe_dict["eval"]["protocol"])
        # results = evaluator.run(
        #     config=config_dict,
        #     recipe=recipe_dict,
        #     checkpoint=checkpoint,
        #     datasets=dataset_list or recipe_dict["eval"]["datasets"],
        #     batch_size=batch_size or recipe_dict["eval"]["batch_size"],
        #     verbose=verbose,
        # )

        # Mock results for demonstration
        results = {
            "mbpp": {"exact_match": 0.0, "pass@1": 0.0},
            "codecontests": {"pass@1": 0.0, "pass@5": 0.0},
        }

        # Display results table
        table = Table(title="Evaluation Results")
        table.add_column("Dataset", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Score", style="green")

        for dataset, metrics in results.items():
            for metric, score in metrics.items():
                table.add_row(dataset, metric, f"{score:.2%}")

        console.print(table)

        if save_predictions:
            console.print("[yellow]Prediction saving not yet implemented[/yellow]")

        if save_report:
            console.print("[yellow]Report generation not yet implemented[/yellow]")

        console.print(
            "[yellow]Note: This is a stub implementation. "
            "Actual evaluation pipeline not yet implemented.[/yellow]"
        )

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
