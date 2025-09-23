"""
Training entrypoint for WMTP Fine-Tuning Framework.

This module provides the main CLI interface for training models using either
critic-weighted or Rho-1 weighted Multi-Token Prediction approaches.
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.traceback import install

install(show_locals=False)
console = Console()
app = typer.Typer(
    name="wmtp-train",
    help="WMTP Training CLI - Train models with weighted Multi-Token Prediction",
    pretty_exceptions_show_locals=False,
)


@app.command()
def train(
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
        help="Path to training recipe YAML file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    run_name: str | None = typer.Option(
        None,
        "--run-name",
        help="Optional run name for MLflow tracking",
    ),
    resume: Path | None = typer.Option(
        None,
        "--resume",
        help="Path to checkpoint to resume training from",
        exists=True,
        dir_okay=False,
    ),
    tags: str | None = typer.Option(
        None,
        "--tags",
        help="Comma-separated tags for MLflow (e.g., 'exp1,critic,mbpp')",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration without starting training",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
):
    """
    Train a model using weighted Multi-Token Prediction.

    This command orchestrates the complete training pipeline including:
    - Loading and validating configuration
    - Setting up distributed training environment
    - Initializing models and datasets
    - Running the training loop with MLflow tracking
    - Saving checkpoints and artifacts
    """
    console.print("[bold blue]WMTP Training Framework[/bold blue]")
    console.print(f"Config: {config}")
    console.print(f"Recipe: {recipe}")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - Validating configuration only[/yellow]")

    # Parse tags if provided
    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        console.print(f"Tags: {tag_list}")

    if resume:
        console.print(f"[green]Resuming from checkpoint: {resume}[/green]")

    try:
        from src.pipelines import run_training_pipeline
        from src.settings import load_config, load_recipe

        cfg = load_config(config, verbose=verbose)
        rcp = load_recipe(recipe, verbose=verbose)

        outputs = run_training_pipeline(
            cfg,
            rcp,
            run_name=run_name,
            tags=tag_list,
            dry_run=dry_run,
            max_steps=10 if dry_run else None,
        )

        if dry_run:
            console.print("[green]Configuration validation passed![/green]")
        else:
            console.print(
                f"[green]Training finished. Metrics: {outputs.trainer_metrics}[/green]"
            )

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    app()
