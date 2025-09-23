"""
HuggingFace model loader with local-first S3 fallback support.

Implements the WMTP model loading strategy:
1. Check local path first
2. Fall back to S3 with caching if not found locally
3. Support for base, RM, and reference models
"""

from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...components.registry import loader_registry
from ...utils.hf import safe_from_pretrained
from .base_loader import ModelLoader

console = Console()


@loader_registry.register("hf-local-s3-loader", category="loader", version="v1")
class HFLocalS3Loader(ModelLoader):
    """
    HuggingFace model loader with local-first S3 fallback.

    Supports loading:
    - Base MTP model (7B)
    - Reward model (RM)
    - Reference model (for Rho-1)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize HF loader.

        Args:
            config: Configuration with model paths and S3 settings
        """
        super().__init__(config)

        # Extract model-specific paths
        self.model_paths = config.get("model_paths", {})
        self.s3_config = config.get("s3_config", {})

        # Default model IDs (fallback to HuggingFace Hub)
        # Note: Recipe/local paths always take precedence over these defaults.
        self.default_model_ids = {
            "base": "facebook/multi-token-prediction-7b",
            "rm": "sfair/Llama-3-8B-RM-Reward-Model",
            "ref": "princeton-nlp/Sheared-LLaMA-1.3B",
        }

    def load_model(
        self,
        path: str,
        model_type: str = "base",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Any:
        """
        Load model with local-first S3 fallback.

        Args:
            path: Model path or identifier
            model_type: Type of model ('base', 'rm', 'ref')
            device_map: Device mapping strategy
            torch_dtype: Model precision
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional model loading arguments

        Returns:
            Loaded model
        """
        # 1. Try local path from configuration
        local_path = self.model_paths.get(model_type)
        if local_path:
            local_path = Path(local_path)
            if local_path.exists():
                console.print(
                    f"[green]Loading {model_type} model from local: {local_path}[/green]"
                )
                return self._load_from_path(
                    local_path,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )

        # 2. Try direct path if provided
        direct_path = Path(path)
        if direct_path.exists():
            console.print(f"[green]Loading model from: {direct_path}[/green]")
            return self._load_from_path(
                direct_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        # 3. Try S3 with caching
        if self.s3_manager and self.s3_manager.connected:
            # Construct S3 key for model
            s3_key = f"models/{model_type}/{path}"
            cache_key = self.compute_cache_key(
                data_id=f"model_{model_type}",
                version=path,
            )

            try:
                cached_path = self.sync_directory_with_cache(
                    local_dir=None,
                    s3_prefix=s3_key,
                    cache_key=cache_key,
                )

                if cached_path and cached_path.exists():
                    console.print(
                        f"[green]Loading {model_type} model from S3 cache: {cached_path}[/green]"
                    )
                    return self._load_from_path(
                        cached_path,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        trust_remote_code=trust_remote_code,
                        **kwargs,
                    )
            except Exception as e:
                console.print(f"[yellow]Could not load from S3: {e}[/yellow]")

        # 4. Fall back to HuggingFace Hub
        model_id = self.default_model_ids.get(model_type, path)
        console.print(
            f"[cyan]Loading {model_type} model from HuggingFace Hub: {model_id}[/cyan]"
        )

        # Use safe loading utility from utils/hf.py
        return safe_from_pretrained(
            model_id,
            model_class=AutoModelForCausalLM,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            cache_dir=str(self.cache_dir / "huggingface"),
            **kwargs,
        )

    def load_tokenizer(
        self,
        path: str,
        model_type: str = "base",
        padding_side: str = "left",
        **kwargs,
    ) -> Any:
        """
        Load tokenizer with local-first S3 fallback.

        Args:
            path: Tokenizer path or identifier
            model_type: Type of model ('base', 'rm', 'ref')
            padding_side: Padding side for tokenizer
            **kwargs: Additional tokenizer arguments

        Returns:
            Loaded tokenizer
        """
        # 1. Try local path from configuration
        local_path = self.model_paths.get(model_type)
        if local_path:
            local_path = Path(local_path)
            if local_path.exists():
                console.print(
                    f"[green]Loading tokenizer from local: {local_path}[/green]"
                )
                return self._load_tokenizer_from_path(
                    local_path,
                    padding_side=padding_side,
                    **kwargs,
                )

        # 2. Try direct path if provided
        direct_path = Path(path)
        if direct_path.exists():
            console.print(f"[green]Loading tokenizer from: {direct_path}[/green]")
            return self._load_tokenizer_from_path(
                direct_path,
                padding_side=padding_side,
                **kwargs,
            )

        # 3. Try S3 with caching
        if self.s3_manager and self.s3_manager.connected:
            # Construct S3 key for tokenizer
            s3_key = f"models/{model_type}/{path}"
            cache_key = self.compute_cache_key(
                data_id=f"tokenizer_{model_type}",
                version=path,
            )

            try:
                cached_path = self.sync_directory_with_cache(
                    local_dir=None,
                    s3_prefix=s3_key,
                    cache_key=cache_key,
                )

                if cached_path and cached_path.exists():
                    console.print(
                        f"[green]Loading tokenizer from S3 cache: {cached_path}[/green]"
                    )
                    return self._load_tokenizer_from_path(
                        cached_path,
                        padding_side=padding_side,
                        **kwargs,
                    )
            except Exception as e:
                console.print(f"[yellow]Could not load tokenizer from S3: {e}[/yellow]")

        # 4. Fall back to HuggingFace Hub
        model_id = self.default_model_ids.get(model_type, path)
        console.print(
            f"[cyan]Loading tokenizer from HuggingFace Hub: {model_id}[/cyan]"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side=padding_side,
            cache_dir=str(self.cache_dir / "huggingface"),
            **kwargs,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load(
        self,
        path: str,
        model_type: str = "base",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Load both model and tokenizer.

        Args:
            path: Model path or identifier
            model_type: Type of model ('base', 'rm', 'ref')
            **kwargs: Additional arguments

        Returns:
            Dictionary with model and tokenizer
        """
        # Load model
        model = self.load_model(path, model_type=model_type, **kwargs)

        # Load tokenizer
        tokenizer = self.load_tokenizer(path, model_type=model_type, **kwargs)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_type": model_type,
            "path": path,
        }

    def load_all_models(
        self,
        base_path: str | None = None,
        rm_path: str | None = None,
        ref_path: str | None = None,
        **kwargs,
    ) -> dict[str, dict[str, Any]]:
        """
        Load all models needed for WMTP training.

        Args:
            base_path: Path to base MTP model
            rm_path: Path to reward model
            ref_path: Path to reference model
            **kwargs: Additional arguments

        Returns:
            Dictionary with all loaded models
        """
        models = {}

        # Load base model (required)
        base_path = (
            base_path or self.model_paths.get("base") or self.default_model_ids["base"]
        )
        models["base"] = self.load(base_path, model_type="base", **kwargs)

        # Load reward model if needed (for critic-wmtp)
        if rm_path or "rm" in self.model_paths:
            rm_path = (
                rm_path or self.model_paths.get("rm") or self.default_model_ids["rm"]
            )
            models["rm"] = self.load(rm_path, model_type="rm", **kwargs)

        # Load reference model if needed (for rho1-wmtp)
        if ref_path or "ref" in self.model_paths:
            ref_path = (
                ref_path or self.model_paths.get("ref") or self.default_model_ids["ref"]
            )
            models["ref"] = self.load(ref_path, model_type="ref", **kwargs)

        return models

    def _load_from_path(
        self,
        path: Path,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Any:
        """
        Load model from a local path.

        Args:
            path: Local path to model
            device_map: Device mapping strategy
            torch_dtype: Model precision
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments

        Returns:
            Loaded model
        """
        return AutoModelForCausalLM.from_pretrained(
            str(path),
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            **kwargs,
        )

    def _load_tokenizer_from_path(
        self,
        path: Path,
        padding_side: str = "left",
        **kwargs,
    ) -> Any:
        """
        Load tokenizer from a local path.

        Args:
            path: Local path to tokenizer
            padding_side: Padding side
            **kwargs: Additional arguments

        Returns:
            Loaded tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            padding_side=padding_side,
            local_files_only=True,
            **kwargs,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Execute loading operation within component framework.

        Args:
            ctx: Context dictionary

        Returns:
            Dictionary with loaded models
        """
        self.validate_initialized()

        # Determine what to load based on context
        load_all = ctx.get("load_all_models", False)

        if load_all:
            # Load all models for pipeline
            models = self.load_all_models(**ctx)
            return {
                "models": models,
                "loader": self.__class__.__name__,
            }
        else:
            # Load single model
            path = ctx.get("model_path") or ctx.get("path")
            model_type = ctx.get("model_type", "base")

            if not path:
                raise ValueError("No model path specified in context")

            result = self.load(path, model_type=model_type, **ctx)
            return {
                "model": result["model"],
                "tokenizer": result["tokenizer"],
                "model_type": model_type,
                "path": path,
                "loader": self.__class__.__name__,
            }
