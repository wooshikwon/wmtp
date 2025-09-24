"""HuggingFace model loader with local-first S3 fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.components.loader.base_loader import ModelLoader
from src.components.registry import loader_registry
from src.utils.s3 import S3Utils


@loader_registry.register(
    "hf-model", version="1.0.0", description="HuggingFace model loader"
)
class HFModelLoader(ModelLoader):
    """Load HuggingFace models from local or S3 storage."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.s3_utils = S3Utils()
        self.use_4bit = config.get("use_4bit", False) if config else False
        self.use_8bit = config.get("use_8bit", False) if config else False
        self.device = (
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            if config
            else "cuda"
        )

    def _get_quantization_config(self) -> dict[str, Any]:
        """Get quantization configuration if enabled."""
        if self.use_4bit:
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            }
        elif self.use_8bit:
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            }
        return {}

    def load_model(self, path: str, **kwargs) -> Any:
        """
        Load a HuggingFace model from local path or S3.

        Args:
            path: Local path or S3 URL to model
            **kwargs: Additional arguments for model loading

        Returns:
            Loaded model
        """
        local_path = Path(path)

        # Check if it's an S3 path
        if path.startswith("s3://"):
            # Download from S3 to local cache
            local_path = self.s3_utils.download_model(path)
            if not local_path.exists():
                raise FileNotFoundError(f"Failed to download model from {path}")

        # Load from local path
        if not local_path.exists():
            raise FileNotFoundError(f"Model not found at {local_path}")

        # Merge quantization config with kwargs
        load_kwargs = {**self._get_quantization_config(), **kwargs}

        # Handle device placement
        if "device_map" not in load_kwargs and not (self.use_4bit or self.use_8bit):
            load_kwargs["device_map"] = {"": self.device}

        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(local_path), torch_dtype=torch.bfloat16, **load_kwargs
            )
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace model from {local_path}: {e}"
            )

    def load_tokenizer(self, path: str, **kwargs) -> Any:
        """
        Load a HuggingFace tokenizer from local path or S3.

        Args:
            path: Local path or S3 URL to tokenizer
            **kwargs: Additional arguments for tokenizer loading

        Returns:
            Loaded tokenizer
        """
        local_path = Path(path)

        # Check if it's an S3 path
        if path.startswith("s3://"):
            # Download from S3 to local cache
            local_path = self.s3_utils.download_model(path)
            if not local_path.exists():
                raise FileNotFoundError(f"Failed to download tokenizer from {path}")

        # Load from local path
        if not local_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {local_path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(str(local_path), **kwargs)
            # Ensure padding token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {local_path}: {e}")

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Run the loader with the given context.

        Args:
            ctx: Context containing model_path and tokenizer_path

        Returns:
            Dictionary with model and tokenizer
        """
        model_path = ctx.get("model_path")
        tokenizer_path = ctx.get(
            "tokenizer_path", model_path
        )  # Use model path if tokenizer path not specified

        if not model_path:
            raise ValueError("model_path is required in context")

        result = {}

        # Load model
        result["model"] = self.load_model(model_path)

        # Load tokenizer if available
        if tokenizer_path:
            result["tokenizer"] = self.load_tokenizer(tokenizer_path)

        return result
