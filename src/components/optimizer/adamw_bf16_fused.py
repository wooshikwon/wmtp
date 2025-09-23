"""
AdamW optimizer component with optional fused kernel and scheduler support.

This component wraps torch.optim.AdamW and optionally enables the fused
implementation when available. It also provides an LR scheduler created via
transformers.get_scheduler using the configured policy and warmup ratio.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import AdamW
from transformers import get_scheduler

from src.components.base import BaseComponent
from src.components.registry import optimizer_registry


@optimizer_registry.register(
    "adamw-bf16-fused", category="optimizer", version="1.0.0"
)
class AdamWBF16FusedOptimizer(BaseComponent):
    """
    AdamW optimizer component with optional fused kernel and LR scheduler.

    Expected configuration keys:
      - params: Iterable[Tensor] (model parameters)
      - lr: float
      - weight_decay: float
      - betas: list[float] of length 2
      - scheduler: str in {"cosine","linear","constant"}
      - warmup_ratio: float in [0, 0.5]
      - grad_clip: float (stored for trainer usage)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._last_lr: float = 0.0

    def setup(self, ctx: dict[str, Any]) -> None:
        super().setup(ctx)

        params = self.config.get("params")
        if params is None:
            raise ValueError("'params' must be provided to optimizer config")

        lr: float = float(self.config.get("lr", 1e-5))
        weight_decay: float = float(self.config.get("weight_decay", 0.0))
        betas = self.config.get("betas", [0.9, 0.95])

        # Enable fused AdamW when available (PyTorch CUDA build)
        fused_supported = hasattr(AdamW, "__init__") and torch.cuda.is_available()
        fused_flag = False
        if fused_supported:
            try:
                self.optimizer = AdamW(
                    params,
                    lr=lr,
                    betas=(betas[0], betas[1]),
                    weight_decay=weight_decay,
                    fused=True,
                )
                fused_flag = True
            except TypeError:
                # Older builds do not support fused argument
                self.optimizer = AdamW(
                    params, lr=lr, betas=(betas[0], betas[1]), weight_decay=weight_decay
                )
        else:
            self.optimizer = AdamW(
                params, lr=lr, betas=(betas[0], betas[1]), weight_decay=weight_decay
            )

        # Scheduler setup via transformers
        scheduler_type: str = self.config.get("scheduler", "cosine")
        warmup_ratio: float = float(self.config.get("warmup_ratio", 0.0))
        num_training_steps: int = int(ctx.get("num_training_steps", 0))
        num_warmup_steps = int(num_training_steps * warmup_ratio) if num_training_steps > 0 else 0

        if num_training_steps > 0:
            self.scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        # Cache initial LR for reporting
        if len(self.optimizer.param_groups) > 0:
            self._last_lr = float(self.optimizer.param_groups[0]["lr"])

        # Store fused flag and grad clip for trainer reference
        self.fused = fused_flag
        self.grad_clip = float(self.config.get("grad_clip", 1.0))

    def step(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call setup() first.")

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if len(self.optimizer.param_groups) > 0:
            self._last_lr = float(self.optimizer.param_groups[0]["lr"])

    def zero_grad(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call setup() first.")
        self.optimizer.zero_grad(set_to_none=True)

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Report current optimizer state (e.g., learning rate).
        """
        return {
            "lr": self._last_lr,
            "has_scheduler": self.scheduler is not None,
            "fused": bool(getattr(self, "fused", False)),
        }


