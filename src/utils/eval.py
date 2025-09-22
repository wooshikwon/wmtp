"""
Base evaluation protocol for WMTP framework.

This module provides the base class and helper functions for evaluation.
"""

import re
from collections import defaultdict
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from ..components.base import Component

console = Console()


class EvaluationProtocol(Component):
    """
    Base evaluation protocol for code generation benchmarks.

    Implements common evaluation logic and metrics computation.
    """

    def __init__(
        self,
        sampling_config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        """
        Initialize evaluation protocol.

        Args:
            sampling_config: Sampling configuration (temperature, top_p, etc.)
            device: Device for evaluation
        """
        self.sampling_config = sampling_config or {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_length": 2048,
            "n": 1,
        }
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.initialized = False

    def setup(self, config: dict[str, Any]) -> None:
        """Setup evaluator from configuration."""
        # Update sampling config if provided
        if "sampling" in config:
            self.sampling_config.update(config["sampling"])

        # Update device if provided
        if "device" in config:
            self.device = torch.device(config["device"])

        self.initialized = True

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run evaluation.

        Args:
            inputs: Dictionary containing:
                - model: Model to evaluate
                - tokenizer: Tokenizer
                - dataset: Optional dataset
                - batch_size: Batch size
                - num_samples: Number of samples

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} must be initialized with setup()"
            )

        return self.evaluate(**inputs)

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        batch_size: int = 8,
        num_samples: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate model on dataset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            dataset: Evaluation dataset
            batch_size: Batch size
            num_samples: Optional number of samples to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def generate_completion(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate code completion for a prompt.

        Args:
            model: Language model
            tokenizer: Tokenizer
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Generated completion
        """
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.sampling_config["temperature"],
                top_p=self.sampling_config["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from completion
        if completion.startswith(prompt):
            completion = completion[len(prompt) :]

        return completion.strip()

    def compute_metrics(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs,
    ) -> dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Model predictions
            references: Ground truth references
            **kwargs: Additional metric parameters

        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError("Subclasses must implement compute_metrics()")

    def extract_code(self, completion: str) -> str:
        """
        Extract code from model completion.

        Args:
            completion: Model output

        Returns:
            Extracted code
        """
        # Try to extract code block
        code_pattern = r"```python\n(.*?)```"
        match = re.search(code_pattern, completion, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Try to extract function definition
        func_pattern = r"def\s+\w+.*?(?=\n\n|\Z)"
        match = re.search(func_pattern, completion, re.DOTALL)

        if match:
            return match.group(0).strip()

        # Return full completion as fallback
        return completion.strip()

    def display_results(
        self, metrics: dict[str, float], title: str = "Evaluation Results"
    ) -> None:
        """Display evaluation results."""
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")

        for metric, score in metrics.items():
            table.add_row(metric, f"{score:.2%}")

        console.print(table)


def aggregate_metrics(
    metrics_list: list[dict[str, float]],
) -> dict[str, float]:
    """
    Aggregate metrics from multiple evaluations.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Aggregated metrics with mean and std
    """
    aggregated = defaultdict(list)

    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated[key].append(value)

    result = {}
    for key, values in aggregated.items():
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        result[f"{key}_mean"] = mean
        result[f"{key}_std"] = std

    return result


# Export main classes and functions
__all__ = [
    "EvaluationProtocol",
    "aggregate_metrics",
]
