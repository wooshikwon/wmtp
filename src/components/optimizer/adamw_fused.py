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


@optimizer_registry.register("adamw", category="optimizer", version="1.0.0")
class AdamWFusedOptimizer(BaseComponent):
    """
    AdamW optimizer with CUDA fused kernel optimization and LR scheduler.
    
    Mixed precision (BF16/FP16) is handled by trainer autocast context,
    not by this optimizer. This component focuses on:
    - Fused kernel optimization for CUDA acceleration
    - Learning rate scheduling
    - Gradient clipping support

    Expected configuration keys:
      - params: Iterable[Tensor] (model parameters)
      - lr: float (learning rate)
      - weight_decay: float (L2 regularization)
      - betas: list[float] of length 2 (Adam momentum coefficients)
      - scheduler: str in {"cosine","linear","constant"}
      - warmup_ratio: float in [0, 0.5]
      - grad_clip: float (gradient clipping threshold)
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

        # Recipe에서 검증된 값들을 사용 (Pydantic으로 이미 검증됨)
        lr: float = float(self.config["lr"])  # Recipe optim.lr에서 제공
        weight_decay: float = float(self.config["weight_decay"])  # Recipe optim.weight_decay에서 제공
        betas = self.config["betas"]  # Recipe optim.betas에서 제공

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

        # Scheduler setup via transformers (Recipe에서 값 제공)
        scheduler_type: str = self.config["scheduler"]  # Recipe optim.scheduler에서 제공
        warmup_ratio: float = float(self.config["warmup_ratio"])  # Recipe optim.warmup_ratio에서 제공
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
        self.grad_clip = float(self.config["grad_clip"])  # Recipe optim.grad_clip에서 제공

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

    # Added to support adding param groups (e.g., Critic value head)
    def add_param_group(self, group: dict[str, Any]) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call setup() first.")
        self.optimizer.add_param_group(group)

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Report current optimizer state (e.g., learning rate).
        """
        return {
            "lr": self._last_lr,
            "has_scheduler": self.scheduler is not None,
            "fused": bool(getattr(self, "fused", False)),
        }
    
    def state_dict(self) -> dict[str, Any]:
        """Return the state dictionary for checkpointing.
        
        Returns:
            Dictionary containing optimizer and scheduler states
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call setup() first.")
            
        state = {
            "optimizer": self.optimizer.state_dict(),
            "last_lr": self._last_lr,
        }
        
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
            
        return state
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from checkpoint.
        
        Args:
            state_dict: State dictionary from checkpointing
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call setup() first.")
            
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            
        if "last_lr" in state_dict:
            self._last_lr = float(state_dict["last_lr"])
            
        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])


