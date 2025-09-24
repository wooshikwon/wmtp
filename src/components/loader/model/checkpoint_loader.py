"""Checkpoint loader for resuming training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.components.loader.base_loader import ModelLoader
from src.components.registry import loader_registry
from src.utils.s3 import S3Utils


@loader_registry.register(
    "checkpoint", version="1.0.0", description="Training checkpoint loader"
)
class CheckpointLoader(ModelLoader):
    """Load training checkpoints for resuming or evaluation."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.s3_utils = S3Utils()
        self.device = (
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            if config
            else "cuda"
        )
        self.load_optimizer = config.get("load_optimizer", True) if config else True
        self.load_scheduler = config.get("load_scheduler", True) if config else True

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        local_path = Path(path)

        # Handle S3 paths
        if path.startswith("s3://"):
            local_path = self.s3_utils.download_checkpoint(path)
            if not local_path.exists():
                raise FileNotFoundError(f"Failed to download checkpoint from {path}")

        if not local_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {local_path}")

        # Load checkpoint
        checkpoint = torch.load(local_path, map_location=self.device)

        return checkpoint

    def load_model_state(self, checkpoint: dict[str, Any], model: Any) -> None:
        """
        Load model state from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary
            model: Model to load state into
        """
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint)

    def load_optimizer_state(self, checkpoint: dict[str, Any], optimizer: Any) -> None:
        """
        Load optimizer state from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary
            optimizer: Optimizer to load state into
        """
        if not self.load_optimizer:
            return

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    def load_scheduler_state(self, checkpoint: dict[str, Any], scheduler: Any) -> None:
        """
        Load scheduler state from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary
            scheduler: Scheduler to load state into
        """
        if not self.load_scheduler:
            return

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    def get_training_state(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """
        Extract training state from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary

        Returns:
            Training state dictionary
        """
        state = {}

        # Extract common training state
        state["epoch"] = checkpoint.get("epoch", 0)
        state["global_step"] = checkpoint.get("global_step", 0)
        state["best_loss"] = checkpoint.get("best_loss", float("inf"))
        state["best_metric"] = checkpoint.get("best_metric", 0.0)

        # Extract metrics history if available
        if "metrics" in checkpoint:
            state["metrics"] = checkpoint["metrics"]

        # Extract config if available
        if "config" in checkpoint:
            state["config"] = checkpoint["config"]

        return state

    def save_checkpoint(
        self,
        path: str,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        **kwargs,
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            path: Path to save checkpoint
            model: Model to save
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            **kwargs: Additional state to save
        """
        checkpoint = {"model_state_dict": model.state_dict(), **kwargs}

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Create parent directory if needed
        local_path = Path(path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        torch.save(checkpoint, local_path)

        # Upload to S3 if path is S3
        if path.startswith("s3://"):
            self.s3_utils.upload_checkpoint(local_path, path)

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Run the checkpoint loader.

        Args:
            ctx: Context containing checkpoint_path and optionally model/optimizer/scheduler

        Returns:
            Dictionary with checkpoint data and extracted state
        """
        checkpoint_path = ctx.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required in context")

        # Load checkpoint
        checkpoint = self.load_checkpoint(checkpoint_path)

        result = {
            "checkpoint": checkpoint,
            "training_state": self.get_training_state(checkpoint),
        }

        # Load into provided model/optimizer/scheduler if given
        if "model" in ctx:
            self.load_model_state(checkpoint, ctx["model"])
            result["model_loaded"] = True

        if "optimizer" in ctx and self.load_optimizer:
            self.load_optimizer_state(checkpoint, ctx["optimizer"])
            result["optimizer_loaded"] = True

        if "scheduler" in ctx and self.load_scheduler:
            self.load_scheduler_state(checkpoint, ctx["scheduler"])
            result["scheduler_loaded"] = True

        return result
