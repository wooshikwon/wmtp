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
        save_predictions: bool = False,
        save_report: bool = False,
    ) -> dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            checkpoint: Path to model checkpoint to evaluate
            datasets: List of datasets to evaluate on (None = use recipe default)
            run_name: MLflow run name (None = use recipe default)
            tags: Additional tags for MLflow run
            save_predictions: Whether to save prediction samples as artifacts
            save_report: Whether to generate and save evaluation report

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
            run_name=run_name or f"eval_{self.recipe.run.name}", tags=tag_map
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
            evaluator.setup(
                {
                    "sampling": self.recipe.eval.sampling.model_dump(),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                }
            )

            # Prepare evaluation context
            eval_ctx = {
                "model": model,
                "tokenizer": tokenizer,
                **loaded_datasets,  # mbpp_dataset, contest_dataset etc.
            }

            # Run evaluation
            results = evaluator.run(eval_ctx)

            # Log results to MLflow with enhanced options
            self._log_results(results, save_predictions, save_report, checkpoint)

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
                },
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
                dataset = data_loader.run(
                    {
                        "split": "test",  # Use test split for evaluation
                        "max_length": getattr(
                            self.recipe.data.eval, "max_length", 2048
                        ),
                        "add_solution": False,  # Don't need solutions for evaluation
                    }
                )["dataset"]

                # Map to expected context keys for evaluator
                if source == "mbpp":
                    loaded_datasets["mbpp_dataset"] = dataset
                elif source == "contest":
                    loaded_datasets["contest_dataset"] = dataset
                else:
                    loaded_datasets[f"{source}_dataset"] = dataset

                console.print(
                    f"[green]Loaded {len(dataset)} samples from {source}[/green]"
                )

            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load {source}: {e}[/yellow]")
                # Continue with other datasets

        return loaded_datasets

    def _log_results(
        self,
        results: dict[str, Any],
        save_predictions: bool,
        save_report: bool,
        checkpoint: Path,
    ) -> None:
        """
        Log evaluation results to MLflow with enhanced artifacts.

        Args:
            results: Results dictionary from evaluator
            save_predictions: Whether to save prediction samples
            save_report: Whether to generate evaluation report
            checkpoint: Path to checkpoint being evaluated
        """
        if not self.mlflow:
            return

        import json
        from datetime import datetime

        metrics = results.get("metrics", {})

        # Log all metrics with enhanced categorization
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, int | float):
                self.mlflow.log_metric(metric_name, metric_value)

        # Add summary metrics
        if metrics:
            # Calculate average performance across all metrics
            numeric_metrics = [
                v for v in metrics.values() if isinstance(v, int | float)
            ]
            if numeric_metrics:
                avg_performance = sum(numeric_metrics) / len(numeric_metrics)
                self.mlflow.log_metric("avg_performance", avg_performance)

        # Log evaluation config as parameters
        eval_params = {
            "eval_protocol": self.recipe.eval.protocol,
            "eval_temperature": self.recipe.eval.sampling.temperature,
            "eval_top_p": self.recipe.eval.sampling.top_p,
            "eval_n_samples": self.recipe.eval.sampling.n,
            "batch_size": self.recipe.data.eval.batch_size,
            "checkpoint_path": str(checkpoint),
            "algorithm": self.recipe.train.algo,
            "model_id": self.recipe.model.base_id,
            "mtp_heads": self.recipe.model.mtp.n_heads,
        }

        for param_name, param_value in eval_params.items():
            self.mlflow.log_param(param_name, param_value)

        # Save enhanced results as artifact
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Save complete evaluation results
        results_file = Path(f"evaluation_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(
                {
                    "checkpoint": str(checkpoint),
                    "timestamp": timestamp,
                    "algorithm": self.recipe.train.algo,
                    "metrics": metrics,
                    "config": eval_params,
                },
                f,
                indent=2,
            )
        self.mlflow.log_artifact(str(results_file), "evaluation")
        results_file.unlink()

        # 2. Save prediction samples if requested
        if save_predictions:
            self._save_prediction_samples(results, timestamp)

        # 3. Save weight distribution statistics if available
        self._save_weight_statistics(results, timestamp)

        # 4. Generate and save evaluation report if requested
        if save_report:
            self._generate_evaluation_report(results, metrics, checkpoint, timestamp)

        console.print("[green]Results and artifacts logged to MLflow[/green]")

    def _save_prediction_samples(self, results: dict[str, Any], timestamp: str) -> None:
        """
        Save prediction samples as MLflow artifacts.

        Args:
            results: Evaluation results dictionary
            timestamp: Timestamp for file naming
        """
        import json

        # Extract predictions if available
        predictions = results.get("predictions", [])
        references = results.get("references", [])

        if not predictions:
            console.print("[yellow]No predictions available to save[/yellow]")
            return

        # Save sample predictions (first 10)
        samples = []
        for i, (pred, ref) in enumerate(zip(predictions[:10], references[:10])):
            samples.append(
                {
                    "sample_id": i,
                    "prediction": pred if isinstance(pred, str) else str(pred),
                    "reference": ref if isinstance(ref, str) else str(ref),
                }
            )

        if samples:
            samples_file = Path(f"prediction_samples_{timestamp}.json")
            with open(samples_file, "w") as f:
                json.dump(
                    {
                        "total_samples": len(predictions),
                        "shown_samples": len(samples),
                        "samples": samples,
                    },
                    f,
                    indent=2,
                )

            self.mlflow.log_artifact(str(samples_file), "predictions")
            samples_file.unlink()
            console.print(f"[green]Saved {len(samples)} prediction samples[/green]")

    def _save_weight_statistics(self, results: dict[str, Any], timestamp: str) -> None:
        """
        Save weight distribution statistics as MLflow artifacts.

        Args:
            results: Evaluation results dictionary
            timestamp: Timestamp for file naming
        """
        import json

        import numpy as np

        # Extract weight statistics if available
        weight_stats = results.get("weight_statistics", {})

        if not weight_stats and hasattr(self, "_last_scorer_output"):
            weight_stats = getattr(self, "_last_scorer_output", {}).get(
                "statistics", {}
            )

        if weight_stats:
            # Calculate additional statistics if raw weights are available
            weights = results.get("weights", [])
            if weights:
                try:
                    weights_np = np.array(weights)
                    weight_stats.update(
                        {
                            "percentiles": {
                                "p10": float(np.percentile(weights_np, 10)),
                                "p25": float(np.percentile(weights_np, 25)),
                                "p50": float(np.percentile(weights_np, 50)),
                                "p75": float(np.percentile(weights_np, 75)),
                                "p90": float(np.percentile(weights_np, 90)),
                                "p95": float(np.percentile(weights_np, 95)),
                                "p99": float(np.percentile(weights_np, 99)),
                            },
                            "distribution": {
                                "mean": float(np.mean(weights_np)),
                                "std": float(np.std(weights_np)),
                                "min": float(np.min(weights_np)),
                                "max": float(np.max(weights_np)),
                                "variance": float(np.var(weights_np)),
                            },
                        }
                    )
                except Exception:
                    pass

            # Save weight statistics
            stats_file = Path(f"weight_statistics_{timestamp}.json")
            with open(stats_file, "w") as f:
                json.dump(
                    {
                        "algorithm": str(self.recipe.train.algo)
                        if hasattr(self.recipe.train, "algo")
                        else "unknown",
                        "statistics": weight_stats,
                        "config": {
                            "temperature": float(
                                getattr(self.recipe.loss, "temperature", 0.7)
                            ),
                            "lambda": float(getattr(self.recipe.loss, "lambda", 0.3)),
                        },
                    },
                    f,
                    indent=2,
                )

            self.mlflow.log_artifact(str(stats_file), "weights")
            stats_file.unlink()
            console.print("[green]Saved weight distribution statistics[/green]")

    def _generate_evaluation_report(
        self,
        results: dict[str, Any],
        metrics: dict[str, float],
        checkpoint: Path,
        timestamp: str,
    ) -> None:
        """
        Generate comprehensive evaluation report.

        Args:
            results: Complete evaluation results
            metrics: Extracted metrics
            checkpoint: Checkpoint path
            timestamp: Timestamp for file naming
        """
        from datetime import datetime

        # Create markdown report
        report_lines = [
            "# WMTP Evaluation Report",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Checkpoint**: `{checkpoint}`",
            f"\n**Algorithm**: {self.recipe.train.algo}",
            f"\n**Model**: {self.recipe.model.base_id}",
            "\n## Configuration",
            f"- **MTP Heads**: {self.recipe.model.mtp.n_heads}",
            f"- **Horizon**: {self.recipe.model.mtp.horizon}",
            f"- **Evaluation Protocol**: {self.recipe.eval.protocol}",
            f"- **Sampling Temperature**: {self.recipe.eval.sampling.temperature}",
            f"- **Top-p**: {self.recipe.eval.sampling.top_p}",
            "\n## Results Summary",
        ]

        # Add metrics table
        if metrics:
            report_lines.append("\n### Performance Metrics\n")
            report_lines.append("| Metric | Score |")
            report_lines.append("|--------|-------|")

            # Group by dataset
            mbpp_metrics = {}
            contest_metrics = {}
            other_metrics = {}

            for name, value in metrics.items():
                if "mbpp" in name.lower():
                    mbpp_metrics[name] = value
                elif "contest" in name.lower():
                    contest_metrics[name] = value
                else:
                    other_metrics[name] = value

            # Add MBPP metrics
            if mbpp_metrics:
                report_lines.append("| **MBPP** | |")
                for name, value in mbpp_metrics.items():
                    if isinstance(value, float) and 0 <= value <= 1:
                        report_lines.append(f"| {name} | {value:.2%} |")
                    else:
                        report_lines.append(f"| {name} | {value:.4f} |")

            # Add CodeContests metrics
            if contest_metrics:
                report_lines.append("| **CodeContests** | |")
                for name, value in contest_metrics.items():
                    if isinstance(value, float) and 0 <= value <= 1:
                        report_lines.append(f"| {name} | {value:.2%} |")
                    else:
                        report_lines.append(f"| {name} | {value:.4f} |")

            # Add other metrics
            if other_metrics:
                report_lines.append("| **Other** | |")
                for name, value in other_metrics.items():
                    if isinstance(value, float) and 0 <= value <= 1:
                        report_lines.append(f"| {name} | {value:.2%} |")
                    else:
                        report_lines.append(f"| {name} | {value:.4f} |")

        # Add weight statistics if available
        weight_stats = results.get("weight_statistics", {})
        if weight_stats:
            report_lines.extend(
                [
                    "\n### Weight Distribution Statistics",
                    f"- **Mean Weight**: {weight_stats.get('mean_weight', 'N/A')}",
                    f"- **Std Weight**: {weight_stats.get('std_weight', 'N/A')}",
                    f"- **Min Weight**: {weight_stats.get('min_weight', 'N/A')}",
                    f"- **Max Weight**: {weight_stats.get('max_weight', 'N/A')}",
                ]
            )

        # Add algorithm-specific details
        if self.recipe.train.algo == "critic-wmtp":
            report_lines.extend(
                [
                    "\n### Critic-WMTP Details",
                    "- **Value Head**: Trained with GAE",
                    "- **Delta Mode**: TD",
                    f"- **Temperature**: {getattr(self.recipe.loss, 'temperature', 0.7)}",
                ]
            )
        elif self.recipe.train.algo == "rho1-wmtp":
            report_lines.extend(
                [
                    "\n### Rho-1-WMTP Details",
                    "- **Scoring Method**: Absolute CE Excess",
                    f"- **Top Percentile**: {getattr(self.recipe, 'rho1', {}).get('percentile_top_p', 0.2) * 100}%",
                    f"- **Temperature**: {getattr(self.recipe.loss, 'temperature', 0.7)}",
                ]
            )

        # Save report
        report_content = "\n".join(report_lines)
        report_file = Path(f"evaluation_report_{timestamp}.md")
        with open(report_file, "w") as f:
            f.write(report_content)

        self.mlflow.log_artifact(str(report_file), "reports")
        report_file.unlink()
        console.print("[green]Generated comprehensive evaluation report[/green]")
