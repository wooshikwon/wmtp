"""
CodeContests evaluator component for WMTP framework.

Implements pass@k evaluation following Meta MTP protocol.
"""

from collections import defaultdict
from typing import Any

from rich.console import Console
from tqdm import tqdm

from ...utils.eval import EvaluationProtocol
from ..registry import evaluator_registry

console = Console()


@evaluator_registry.register("codecontests-v1", category="evaluator", version="1.0.0")
class CodeContestsEvaluator(EvaluationProtocol):
    """
    Evaluator for CodeContests benchmark.

    Implements pass@k evaluation following Meta MTP protocol.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize CodeContests evaluator."""
        super().__init__(
            sampling_config=config.get("sampling", None) if config else None,
            device=config.get("device", None) if config else None,
        )
        self.config = config or {}
        self.dataset_name = "codecontests"

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any | None = None,
        batch_size: int = 8,
        num_samples: int | None = None,
        k_values: list[int] = [1, 5],
    ) -> dict[str, float]:
        """
        Evaluate model on CodeContests dataset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            dataset: CodeContests dataset
            batch_size: Batch size
            num_samples: Number of problems to evaluate
            k_values: Values of k for pass@k metrics

        Returns:
            Dictionary with pass@k metrics
        """
        if dataset is None:
            dataset = self.load_codecontests_dataset()

        metrics = defaultdict(list)

        # Sample subset if specified
        if num_samples:
            dataset = dataset[:num_samples]

        console.print(
            f"[cyan]Evaluating on {len(dataset)} CodeContests problems...[/cyan]"
        )

        for problem in tqdm(dataset, desc="Generating solutions"):
            # Generate multiple solutions for pass@k
            solutions = []
            for _ in range(max(k_values)):
                prompt = self.format_codecontests_prompt(problem)
                completion = self.generate_completion(
                    model, tokenizer, prompt, max_new_tokens=512
                )
                code = self.extract_code(completion)
                solutions.append(code)

            # Evaluate solutions
            results = self.evaluate_solutions(solutions, problem["test_cases"])

            # Compute pass@k for this problem
            for k in k_values:
                passed = any(results[:k])
                metrics[f"pass@{k}"].append(passed)

        # Average metrics
        final_metrics = {}
        for metric, values in metrics.items():
            final_metrics[metric] = sum(values) / len(values)

        self.display_results(final_metrics, title="CodeContests Evaluation Results")
        return final_metrics

    def load_codecontests_dataset(self) -> list[dict[str, Any]]:
        """Load CodeContests dataset."""
        # Mock implementation for demonstration
        # In production, this would load the actual CodeContests dataset
        return [
            {
                "description": "Given two integers, return their sum.",
                "test_cases": [
                    {"input": "1 2", "output": "3"},
                    {"input": "-1 1", "output": "0"},
                ],
            },
            {
                "description": "Find the maximum element in an array.",
                "test_cases": [
                    {"input": "5\n1 3 5 2 4", "output": "5"},
                    {"input": "3\n-1 -5 -3", "output": "-1"},
                ],
            },
        ]

    def format_codecontests_prompt(self, problem: dict[str, Any]) -> str:
        """Format CodeContests problem as prompt."""
        prompt = f"""Problem: {problem['description']}

Write a complete Python solution:

```python
"""
        return prompt

    def evaluate_solutions(
        self,
        solutions: list[str],
        test_cases: list[dict[str, str]],
    ) -> list[bool]:
        """
        Evaluate multiple solutions against test cases.

        Args:
            solutions: List of generated solutions
            test_cases: Test cases with input/output

        Returns:
            List of pass/fail for each solution
        """
        results = []

        for solution in solutions:
            passed = self.run_test_cases(solution, test_cases)
            results.append(passed)

        return results

    def run_test_cases(
        self,
        code: str,
        test_cases: list[dict[str, str]],
    ) -> bool:
        """
        Run code against test cases.

        Args:
            code: Solution code
            test_cases: Test cases with input/output

        Returns:
            True if all tests pass
        """
        try:
            for test in test_cases:
                # Create execution namespace
                namespace = {}

                # Mock stdin for input
                import io
                import sys

                old_stdin = sys.stdin
                sys.stdin = io.StringIO(test["input"])

                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()

                try:
                    # Execute the code
                    exec(code, namespace)

                    # Get output
                    output = sys.stdout.getvalue().strip()

                    # Check if output matches expected
                    if output != test["output"]:
                        return False

                finally:
                    # Restore stdin/stdout
                    sys.stdin = old_stdin
                    sys.stdout = old_stdout

            return True

        except Exception:
            return False

    def compute_pass_at_k(
        self,
        n: int,
        c: int,
        k: int,
    ) -> float:
        """
        Compute pass@k metric.

        Args:
            n: Total number of samples generated
            c: Number of correct samples
            k: k value for pass@k

        Returns:
            pass@k score
        """
        if n - c < k:
            return 1.0

        return 1.0 - float(self._comb(n - c, k)) / float(self._comb(n, k))

    def _comb(self, n: int, k: int) -> int:
        """Compute binomial coefficient."""
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1

        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
