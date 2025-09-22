"""
HuggingFace utility functions for WMTP framework.

This module provides safe model/tokenizer loading with local-first policy,
S3 fallback, and HuggingFace Hub as last resort. No direct transformers
imports should exist outside this module.
"""

from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .s3 import S3Manager

console = Console()


class HFModelLoader:
    """
    Safe model loader with local → S3 → HuggingFace Hub fallback.

    Ensures models are loaded from the most efficient source
    while maintaining reproducibility.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/models",
        s3_manager: S3Manager | None = None,
    ):
        """
        Initialize HuggingFace model loader.

        Args:
            cache_dir: Local cache directory for models
            s3_manager: Optional S3 manager for remote storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.s3_manager = s3_manager

    def from_pretrained(
        self,
        model_id: str,
        model_type: str = "auto",
        local_path: str | Path | None = None,
        s3_path: str | None = None,
        device_map: str | dict | None = None,
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Load model with local-first policy.

        Priority order:
        1. Local path if exists
        2. S3 path if configured
        3. HuggingFace Hub

        Args:
            model_id: HuggingFace model ID or path
            model_type: Model type ('auto', 'causal_lm', 'base')
            local_path: Optional local path override
            s3_path: Optional S3 path
            device_map: Device mapping for model
            torch_dtype: Data type for model weights
            trust_remote_code: Trust remote code in configs
            **kwargs: Additional model loading arguments

        Returns:
            Loaded model
        """
        # Try local path first
        if local_path:
            local_path = Path(local_path)
            if local_path.exists():
                console.print(f"[green]Loading model from local: {local_path}[/green]")
                return self._load_from_path(
                    local_path,
                    model_type,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )

        # Try S3 if configured
        if s3_path and self.s3_manager:
            cache_path = self.cache_dir / model_id.replace("/", "_")

            try:
                # Download model files from S3
                self._download_model_from_s3(s3_path, cache_path)

                if cache_path.exists():
                    console.print(
                        f"[green]Loading model from S3 cache: {cache_path}[/green]"
                    )
                    return self._load_from_path(
                        cache_path,
                        model_type,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        trust_remote_code=trust_remote_code,
                        **kwargs,
                    )
            except Exception as e:
                console.print(
                    f"[yellow]S3 loading failed: {e}. Falling back to HF Hub[/yellow]"
                )

        # Fall back to HuggingFace Hub
        console.print(f"[cyan]Loading model from HuggingFace Hub: {model_id}[/cyan]")
        return self._load_from_hub(
            model_id,
            model_type,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            cache_dir=str(self.cache_dir),
            **kwargs,
        )

    def _load_from_path(
        self,
        path: Path,
        model_type: str,
        **kwargs,
    ) -> PreTrainedModel:
        """Load model from local path."""
        if model_type == "causal_lm":
            return AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        elif model_type == "base":
            return AutoModel.from_pretrained(str(path), **kwargs)
        else:  # auto
            try:
                return AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
            except Exception:
                return AutoModel.from_pretrained(str(path), **kwargs)

    def _load_from_hub(
        self,
        model_id: str,
        model_type: str,
        **kwargs,
    ) -> PreTrainedModel:
        """Load model from HuggingFace Hub."""
        if model_type == "causal_lm":
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        elif model_type == "base":
            return AutoModel.from_pretrained(model_id, **kwargs)
        else:  # auto
            try:
                return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            except Exception:
                return AutoModel.from_pretrained(model_id, **kwargs)

    def _download_model_from_s3(self, s3_path: str, local_path: Path) -> None:
        """Download model files from S3."""
        if not self.s3_manager:
            raise ValueError("S3 manager not configured")

        # Essential model files to download
        model_files = [
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ]

        local_path.mkdir(parents=True, exist_ok=True)

        for file_name in model_files:
            s3_key = f"{s3_path}/{file_name}"
            local_file = local_path / file_name

            try:
                self.s3_manager.download_if_missing(s3_key, local_file)
            except FileNotFoundError:
                # Some files may not exist for all models
                continue

    def load_tokenizer(
        self,
        model_id: str,
        local_path: str | Path | None = None,
        s3_path: str | None = None,
        padding_side: str = "right",
        trust_remote_code: bool = False,
        **kwargs,
    ) -> PreTrainedTokenizer:
        """
        Load tokenizer with local-first policy.

        Args:
            model_id: HuggingFace model ID or path
            local_path: Optional local path override
            s3_path: Optional S3 path
            padding_side: Padding side ('left' or 'right')
            trust_remote_code: Trust remote code
            **kwargs: Additional tokenizer arguments

        Returns:
            Loaded tokenizer
        """
        tokenizer = None

        # Try local path first
        if local_path:
            local_path = Path(local_path)
            if local_path.exists():
                console.print(
                    f"[green]Loading tokenizer from local: {local_path}[/green]"
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    str(local_path),
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )

        # Try S3 if configured
        if tokenizer is None and s3_path and self.s3_manager:
            cache_path = self.cache_dir / f"{model_id.replace('/', '_')}_tokenizer"

            try:
                self._download_model_from_s3(s3_path, cache_path)

                if cache_path.exists():
                    console.print(
                        f"[green]Loading tokenizer from S3 cache: {cache_path}[/green]"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(cache_path),
                        trust_remote_code=trust_remote_code,
                        **kwargs,
                    )
            except Exception as e:
                console.print(f"[yellow]S3 tokenizer loading failed: {e}[/yellow]")

        # Fall back to HuggingFace Hub
        if tokenizer is None:
            console.print(
                f"[cyan]Loading tokenizer from HuggingFace Hub: {model_id}[/cyan]"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                cache_dir=str(self.cache_dir),
                **kwargs,
            )

        # Configure tokenizer
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_config(
        self,
        model_id: str,
        local_path: str | Path | None = None,
        s3_path: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> AutoConfig:
        """
        Load model configuration.

        Args:
            model_id: HuggingFace model ID
            local_path: Optional local path
            s3_path: Optional S3 path
            trust_remote_code: Trust remote code
            **kwargs: Additional config arguments

        Returns:
            Model configuration
        """
        # Try local path first
        if local_path:
            local_path = Path(local_path)
            if (local_path / "config.json").exists():
                return AutoConfig.from_pretrained(
                    str(local_path),
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )

        # Try S3 if configured
        if s3_path and self.s3_manager:
            cache_path = self.cache_dir / f"{model_id.replace('/', '_')}_config"
            cache_path.mkdir(parents=True, exist_ok=True)

            config_file = cache_path / "config.json"
            try:
                self.s3_manager.download_if_missing(
                    f"{s3_path}/config.json",
                    config_file,
                )

                if config_file.exists():
                    return AutoConfig.from_pretrained(
                        str(cache_path),
                        trust_remote_code=trust_remote_code,
                        **kwargs,
                    )
            except Exception as e:
                console.print(f"[yellow]S3 config loading failed: {e}[/yellow]")

        # Fall back to HuggingFace Hub
        return AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=str(self.cache_dir),
            **kwargs,
        )


def create_model_loader(
    config: dict[str, Any],
    s3_manager: S3Manager | None = None,
) -> HFModelLoader:
    """
    Create HuggingFace model loader from configuration.

    Args:
        config: Configuration dictionary
        s3_manager: Optional S3 manager

    Returns:
        HFModelLoader instance
    """
    cache_dir = config.get("paths", {}).get("cache", ".cache")
    cache_dir = Path(cache_dir) / "models"

    return HFModelLoader(cache_dir=cache_dir, s3_manager=s3_manager)


def resize_token_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    pad_to_multiple_of: int = 8,
) -> None:
    """
    Resize model token embeddings to match tokenizer.

    Args:
        model: Model to resize
        tokenizer: Tokenizer with vocabulary
        pad_to_multiple_of: Pad vocabulary size to multiple
    """
    vocab_size = len(tokenizer)

    # Pad to multiple for efficiency
    if pad_to_multiple_of > 0:
        vocab_size = (
            (vocab_size + pad_to_multiple_of - 1) // pad_to_multiple_of
        ) * pad_to_multiple_of

    model.resize_token_embeddings(vocab_size)
    console.print(f"[green]Resized token embeddings to {vocab_size}[/green]")


def get_model_size(model: PreTrainedModel) -> tuple[int, float]:
    """
    Get model size in parameters and GB.

    Args:
        model: Model to measure

    Returns:
        Tuple of (num_parameters, size_gb)
    """
    num_params = sum(p.numel() for p in model.parameters())

    # Estimate size based on parameter dtype
    param = next(model.parameters())
    if param.dtype == torch.float16 or param.dtype == torch.bfloat16:
        bytes_per_param = 2
    elif param.dtype == torch.float32:
        bytes_per_param = 4
    else:
        bytes_per_param = 4  # Default

    size_gb = (num_params * bytes_per_param) / (1024**3)

    return num_params, size_gb


def get_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.

    Args:
        dtype_str: String representation ('fp32', 'fp16', 'bf16')

    Returns:
        torch.dtype
    """
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. Use one of {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


# Export main functions and classes
__all__ = [
    "HFModelLoader",
    "create_model_loader",
    "resize_token_embeddings",
    "get_model_size",
    "get_dtype",
]
