"""
Evaluation pipeline for WMTP framework.

Orchestrates the complete evaluation process including checkpoint loading,
dataset preparation, model evaluation, and results logging to MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from src.factory.component_factory import ComponentFactory
from src.settings import Config, Recipe
from src.utils import create_mlflow_manager, set_seed

console = Console()


class EvaluationPipeline:
    """
    Pipeline for evaluating trained models on coding benchmarks.

    Follows the same pattern as training_pipeline.py for consistency:
    - ComponentFactory for component creation
    - setup() → run() pattern for components
    - MLflow integration for experiment tracking
    """

    def __init__(self, config: Config, recipe: Recipe):
        """
        Initialize evaluation pipeline.

        Args:
            config: Environment configuration
            recipe: Recipe configuration
        """
        self.config = config
        self.recipe = recipe
        self.mlflow = None

        # Set seed for reproducible evaluation
        set_seed(config.seed)

    def run(
        self,
        checkpoint: Path,
        datasets: list[str] | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            checkpoint: Path to model checkpoint to evaluate
            datasets: List of datasets to evaluate on (None = use recipe default)
            run_name: MLflow run name (None = use recipe default)
            tags: Additional tags for MLflow run

        Returns:
            Dictionary containing evaluation results and metrics
        """
        console.print("[bold blue]Starting WMTP Evaluation Pipeline[/bold blue]")
        console.print(f"Checkpoint: {checkpoint}")

        # Validate checkpoint exists
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        # Initialize MLflow
        self.mlflow = create_mlflow_manager(self.config.model_dump())
        tag_map = {str(i): t for i, t in enumerate(tags or [])}
        self.mlflow.start_run(
            run_name=run_name or f"eval_{self.recipe.run.name}",
            tags=tag_map
        )

        try:
            # Load model from checkpoint
            console.print("[cyan]Loading model from checkpoint...[/cyan]")
            model, tokenizer = self._load_checkpoint(checkpoint)

            # Prepare datasets
            console.print("[cyan]Preparing datasets...[/cyan]")
            dataset_sources = datasets or self.recipe.data.eval.sources
            loaded_datasets = self._load_datasets(dataset_sources)

            # Create and run evaluator
            console.print("[cyan]Running evaluation...[/cyan]")
            evaluator = ComponentFactory.create_evaluator(self.recipe, self.config)
            evaluator.setup({
                "sampling": self.recipe.eval.sampling.model_dump(),
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            })

            # Prepare evaluation context
            eval_ctx = {
                "model": model,
                "tokenizer": tokenizer,
                **loaded_datasets  # mbpp_dataset, contest_dataset etc.
            }

            # Run evaluation
            results = evaluator.run(eval_ctx)

            # Log results to MLflow
            self._log_results(results)

            # Create summary
            summary = {
                "checkpoint": str(checkpoint),
                "datasets": dataset_sources,
                "algorithm": self.recipe.train.algo,
                "results": results,
                "config": {
                    "model_id": self.recipe.model.base_id,
                    "mtp_heads": self.recipe.model.mtp.n_heads,
                    "horizon": self.recipe.model.mtp.horizon,
                    "eval_protocol": self.recipe.eval.protocol,
                    "sampling": self.recipe.eval.sampling.model_dump(),
                }
            }

            console.print("[green]Evaluation completed successfully![/green]")
            self.mlflow.end_run("FINISHED")
            return summary

        except Exception as e:
            console.print(f"[red]Evaluation failed: {e}[/red]")
            if self.mlflow:
                self.mlflow.end_run("FAILED")
            raise

    def _load_checkpoint(self, checkpoint_path: Path) -> tuple[Any, Any]:
        """
        Load model and tokenizer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (model, tokenizer)
        """
        # Load base model and tokenizer using existing infrastructure
        model_loader = ComponentFactory.create_model_loader(self.config)
        model_loader.setup({})

        # Load all models (base, ref, rm if needed)
        models = model_loader.run({"load_all_models": True})["models"]
        base_model = models["base"]["model"]
        tokenizer = models["base"]["tokenizer"]

        # Load checkpoint state
        console.print(f"Loading checkpoint state from {checkpoint_path}")
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")

        # Load model weights
        if "model_state_dict" in checkpoint_state:
            base_model.load_state_dict(checkpoint_state["model_state_dict"])
        elif "state_dict" in checkpoint_state:
            base_model.load_state_dict(checkpoint_state["state_dict"])
        else:
            # Assume the checkpoint is just the model state dict
            base_model.load_state_dict(checkpoint_state)

        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = base_model.to(device)
        base_model.eval()

        console.print(f"[green]Model loaded on {device}[/green]")
        return base_model, tokenizer

    def _load_datasets(self, dataset_sources: list[str]) -> dict[str, Any]:
        """
        Load evaluation datasets.

        Args:
            dataset_sources: List of dataset names to load

        Returns:
            Dictionary mapping dataset names to loaded datasets
        """
        loaded_datasets = {}

        for source in dataset_sources:
            console.print(f"Loading dataset: {source}")

            try:
                # Create data loader for this source
                data_loader = ComponentFactory.create_data_loader(source, self.config)
                data_loader.setup({})

                # Load evaluation split
                dataset = data_loader.run({
                    "split": "test",  # Use test split for evaluation
                    "max_length": getattr(self.recipe.data.eval, "max_length", 2048),
                    "add_solution": False,  # Don't need solutions for evaluation
                })["dataset"]

                # Map to expected context keys for evaluator
                if source == "mbpp":
                    loaded_datasets["mbpp_dataset"] = dataset
                elif source == "contest":
                    loaded_datasets["contest_dataset"] = dataset
                else:
                    loaded_datasets[f"{source}_dataset"] = dataset

                console.print(f"[green]Loaded {len(dataset)} samples from {source}[/green]")

            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load {source}: {e}[/yellow]")
                # Continue with other datasets

        return loaded_datasets

    def _log_results(self, results: dict[str, Any]) -> None:
        """
        Log evaluation results to MLflow.

        Args:
            results: Results dictionary from evaluator
        """
        if not self.mlflow:
            return

        metrics = results.get("metrics", {})

        # Log all metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.mlflow.log_metric(metric_name, metric_value)

        # Log evaluation config as parameters
        eval_params = {
            "eval_protocol": self.recipe.eval.protocol,
            "eval_temperature": self.recipe.eval.sampling.temperature,
            "eval_top_p": self.recipe.eval.sampling.top_p,
            "eval_n_samples": self.recipe.eval.sampling.n,
            "batch_size": self.recipe.data.eval.batch_size,
        }

        for param_name, param_value in eval_params.items():
            self.mlflow.log_param(param_name, param_value)

        # Save results as artifact
        import json
        results_path = Path("evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.mlflow.log_artifact(str(results_path), "evaluation")
        results_path.unlink()  # Clean up temporary file

        console.print("[green]Results logged to MLflow[/green]")