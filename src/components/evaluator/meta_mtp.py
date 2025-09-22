"""
Meta MTP evaluator orchestrator.

Routes evaluation requests to MBPP and/or CodeContests evaluators based on
requested metrics, consolidating results under the Meta MTP protocol.
"""

from typing import Any

from rich.console import Console

from ...components.base import BaseComponent
from ..registry import evaluator_registry

console = Console()


@evaluator_registry.register(
    "meta-mtp-evaluator", category="evaluator", version="1.0.0"
)
class MetaMTPEvaluator(BaseComponent):
    """
    Orchestrates MBPP and CodeContests evaluators according to metrics list.

    Config schema (as passed by factory):
      - sampling: dict (temperature, top_p, n, ...)
      - metrics: list[str]
      - batch_size: int | None
      - device: optional device string
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.metrics = self.config.get("metrics") or []
        self.sampling = self.config.get("sampling", {})
        self.batch_size = self.config.get("batch_size", 8)
        self.device = self.config.get("device")

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        self.validate_initialized()

        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")

        results: dict[str, float] = {}

        wants_mbpp = any(m.startswith("mbpp") for m in self.metrics)
        wants_contest = any("contest" in m for m in self.metrics) or any(
            m.startswith("contest_pass@") for m in self.metrics
        )

        if wants_mbpp:
            try:
                from .mbpp_eval import MBPPEvaluator  # local import to avoid cycles

                mbpp_eval = MBPPEvaluator(
                    {
                        "sampling": self.sampling,
                        "device": self.device,
                    }
                )
                mbpp_eval.setup({"sampling": self.sampling, "device": self.device})
                mbpp_ds = ctx.get("mbpp_dataset")  # optional external dataset
                mbpp_metrics = mbpp_eval.evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=mbpp_ds,
                    batch_size=int(self.batch_size or 8),
                )
                # namespacing consistency
                for k, v in mbpp_metrics.items():
                    key = (
                        k
                        if k.startswith("mbpp")
                        else f"mbpp_{k}"
                        if k in {"exact_match"}
                        else k
                    )
                    results[key] = v
            except Exception as e:
                console.print(f"[red]MBPP evaluation failed: {e}[/red]")

        if wants_contest:
            try:
                from .codecontests import CodeContestsEvaluator

                cc_eval = CodeContestsEvaluator(
                    {
                        "sampling": self.sampling,
                        "device": self.device,
                    }
                )
                cc_eval.setup({"sampling": self.sampling, "device": self.device})
                cc_ds = ctx.get("contest_dataset")  # optional external dataset
                # choose k_values from requested metrics
                requested = [m for m in self.metrics if m.startswith("contest_pass@")]
                k_values = (
                    sorted({int(m.split("@")[1]) for m in requested})
                    if requested
                    else [1, 5]
                )
                cc_metrics = cc_eval.evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=cc_ds,
                    batch_size=int(self.batch_size or 8),
                    k_values=k_values,
                )
                for k, v in cc_metrics.items():
                    # keep codecontests metric names as-is
                    results[k] = v
            except Exception as e:
                console.print(f"[red]CodeContests evaluation failed: {e}[/red]")

        return {"metrics": results}
