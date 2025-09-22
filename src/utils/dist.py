"""
Distributed training utility functions for WMTP framework.

This module handles FSDP initialization, distributed setup, and
training utilities for multi-GPU environments.
"""

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from rich.console import Console
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

console = Console()


@dataclass
class FSDPConfig:
    """Configuration for Fully Sharded Data Parallel."""

    enabled: bool = True
    auto_wrap: bool = True
    activation_ckpt: bool = True
    sharding: str = "full"  # full, shard_grad_op, no_shard
    cpu_offload: bool = False
    backward_prefetch: bool = True
    mixed_precision: str = "bf16"
    sync_module_states: bool = True
    use_orig_params: bool = True


class DistributedManager:
    """
    Manager for distributed training setup and utilities.

    Handles FSDP initialization, process group setup, and
    distributed training helpers.
    """

    def __init__(self):
        """Initialize distributed manager."""
        self.initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = None
        self.accelerator = None

    def setup(
        self,
        backend: str = "nccl",
        timeout: int = 1800,
        use_accelerate: bool = False,
    ) -> None:
        """
        Set up distributed training environment.

        Args:
            backend: Distributed backend ('nccl', 'gloo')
            timeout: Timeout in seconds
            use_accelerate: Use HuggingFace Accelerate
        """
        if use_accelerate:
            self._setup_with_accelerate()
        else:
            self._setup_pytorch_dist(backend, timeout)

        self.initialized = True
        self._log_setup_info()

    def _setup_pytorch_dist(self, backend: str, timeout: int) -> None:
        """Set up PyTorch distributed."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        elif torch.cuda.is_available():
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
        else:
            # CPU-only mode
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        if self.world_size > 1:
            dist.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size,
                timeout=torch.distributed.default_pg_timeout(timeout),
            )

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

    def _setup_with_accelerate(self) -> None:
        """Set up with HuggingFace Accelerate."""
        self.accelerator = Accelerator()
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.local_rank = self.accelerator.local_process_index
        self.device = self.accelerator.device

    def _log_setup_info(self) -> None:
        """Log distributed setup information."""
        if self.is_main_process():
            console.print("[green]Distributed Training Setup:[/green]")
            console.print(f"  World Size: {self.world_size}")
            console.print(
                f"  Backend: {'Accelerate' if self.accelerator else 'PyTorch'}"
            )
            console.print(f"  Device: {self.device}")

    def setup_fsdp(
        self,
        model: torch.nn.Module,
        config: FSDPConfig | dict[str, Any],
    ) -> FSDP:
        """
        Wrap model with FSDP.

        Args:
            model: Model to wrap
            config: FSDP configuration

        Returns:
            FSDP-wrapped model
        """
        if isinstance(config, dict):
            config = FSDPConfig(**config)

        # Set up mixed precision
        mixed_precision_policy = self._get_mixed_precision(config.mixed_precision)

        # Set up sharding strategy
        sharding_strategy = self._get_sharding_strategy(config.sharding)

        # Set up auto wrap policy
        auto_wrap_policy = None
        if config.auto_wrap:
            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls={LlamaDecoderLayer},
            )

        # Set up CPU offload
        cpu_offload_config = None
        if config.cpu_offload:
            cpu_offload_config = CPUOffload(offload_params=True)

        # Wrap model with FSDP
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=cpu_offload_config,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE
            if config.backward_prefetch
            else None,
            sync_module_states=config.sync_module_states,
            use_orig_params=config.use_orig_params,
            device_id=torch.cuda.current_device()
            if torch.cuda.is_available()
            else None,
        )

        if config.activation_ckpt:
            self._enable_activation_checkpointing(model)

        console.print("[green]Model wrapped with FSDP[/green]")
        return model

    def _get_mixed_precision(self, dtype_str: str) -> MixedPrecision:
        """Get mixed precision configuration."""
        if dtype_str == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif dtype_str == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:  # fp32
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

    def _get_sharding_strategy(self, strategy: str) -> ShardingStrategy:
        """Get FSDP sharding strategy."""
        strategy_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
        }
        return strategy_map.get(strategy, ShardingStrategy.FULL_SHARD)

    def _enable_activation_checkpointing(self, model: FSDP) -> None:
        """Enable activation checkpointing for memory efficiency."""
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )

            check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=check_fn,
            )
            console.print("[green]Activation checkpointing enabled[/green]")
        except ImportError:
            console.print("[yellow]Activation checkpointing not available[/yellow]")

    def save_checkpoint(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
        epoch: int,
        step: int,
        **kwargs,
    ) -> None:
        """
        Save FSDP checkpoint.

        Args:
            model: FSDP-wrapped model
            optimizer: Optimizer
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            step: Current step
            **kwargs: Additional checkpoint data
        """
        if self.is_main_process():
            # Configure state dict
            save_policy = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )

            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                state_dict = model.state_dict()

                checkpoint = {
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    **kwargs,
                }

                torch.save(checkpoint, checkpoint_path)
                console.print(f"[green]Checkpoint saved to {checkpoint_path}[/green]")

        self.barrier()

    def load_checkpoint(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """
        Load FSDP checkpoint.

        Args:
            model: FSDP-wrapped model
            optimizer: Optimizer
            checkpoint_path: Path to checkpoint

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        # Configure state dict
        load_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=False,
        )

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        console.print(f"[green]Checkpoint loaded from {checkpoint_path}[/green]")
        return checkpoint

    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.world_size > 1:
            dist.barrier()

    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
    ) -> torch.Tensor:
        """
        All-reduce tensor across processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ('sum', 'mean', 'max', 'min')

        Returns:
            Reduced tensor
        """
        if self.world_size == 1:
            return tensor

        op_map = {
            "sum": dist.ReduceOp.SUM,
            "mean": dist.ReduceOp.SUM,  # Will divide by world_size after
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }

        dist.all_reduce(tensor, op=op_map[op])

        if op == "mean":
            tensor = tensor / self.world_size

        return tensor

    def cleanup(self) -> None:
        """Clean up distributed process group."""
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
            console.print("[green]Distributed process group destroyed[/green]")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
        deterministic: Use deterministic algorithms (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster but less reproducible
        torch.backends.cudnn.benchmark = True

    console.print(f"[green]Random seed set to {seed}[/green]")


def get_world_info() -> dict[str, int]:
    """
    Get distributed world information.

    Returns:
        Dictionary with rank, world_size, local_rank
    """
    if dist.is_initialized():
        return {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        }
    else:
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
        }


def compute_throughput(
    tokens_processed: int,
    time_elapsed: float,
    world_size: int = 1,
) -> dict[str, float]:
    """
    Compute training throughput metrics.

    Args:
        tokens_processed: Number of tokens processed
        time_elapsed: Time elapsed in seconds
        world_size: Number of processes

    Returns:
        Dictionary with throughput metrics
    """
    tokens_per_second = tokens_processed / time_elapsed
    tokens_per_second_per_gpu = tokens_per_second / world_size

    return {
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
        "time_per_1k_tokens": 1000 / tokens_per_second,
    }


# Global distributed manager instance
_dist_manager = DistributedManager()


def get_dist_manager() -> DistributedManager:
    """Get global distributed manager instance."""
    return _dist_manager


# Export main functions and classes
__all__ = [
    "DistributedManager",
    "FSDPConfig",
    "set_seed",
    "get_world_info",
    "compute_throughput",
    "get_dist_manager",
]
