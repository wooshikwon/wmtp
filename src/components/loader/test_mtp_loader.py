"""Test MTP Model Loader for development and testing.

This loader creates a lightweight MTP model by wrapping Sheared-LLaMA-2.7B
with MTP heads for testing the WMTP pipeline on MacBook M3.
"""

from typing import Any, Dict
import torch
from pathlib import Path

from src.components.base import BaseComponent
from src.components.registry import loader_registry
from src.components.model.mtp_wrapper import MTPModelWrapper


@loader_registry.register("test-mtp-loader", category="loader", version="1.0.0")
class TestMTPLoader(BaseComponent):
    """Test MTP model loader for lightweight testing.
    
    This loader creates a small MTP model suitable for testing on
    consumer hardware (MacBook M3 with 64GB RAM).
    
    Features:
        - Uses Sheared-LLaMA-2.7B as base (10x smaller than 7B MTP)
        - Adds MTP heads dynamically
        - Optimized for MPS (Metal Performance Shaders)
        - Memory-efficient loading
    """
    
    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize test MTP loader.
        
        Args:
            config: Configuration dictionary containing:
                - base_model: Base model name (default: "princeton-nlp/Sheared-LLaMA-2.7B")
                - n_future_tokens: Number of MTP heads (default: 4)
                - device: Device to use (default: "auto")
                - use_cached: Whether to use cached model if available
        """
        super().__init__(config)
        
        # Model configuration
        self.base_model = self.config.get(
            "base_model", 
            "princeton-nlp/Sheared-LLaMA-2.7B"
        )
        self.n_future_tokens = self.config.get("n_future_tokens", 4)
        
        # Device configuration
        devices_config = self.config.get("devices", {})
        self.device = self._resolve_device(devices_config)
        
        # Cache configuration
        self.use_cached = self.config.get("use_cached", True)
        self.cache_dir = Path.home() / ".cache" / "wmtp" / "test_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _resolve_device(self, devices_config: dict) -> str:
        """Resolve device based on configuration and availability.
        
        Args:
            devices_config: Device configuration
            
        Returns:
            Device string ("cuda", "mps", or "cpu")
        """
        compute_backend = devices_config.get("compute_backend", "auto")
        
        if compute_backend == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        
        return compute_backend
    
    def setup(self, ctx: Dict[str, Any]) -> None:
        """Setup loader (no-op for this loader).
        
        Args:
            ctx: Setup context
        """
        super().setup(ctx)
        print(f"[TestMTPLoader] Setup complete. Device: {self.device}")
        
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load or create test MTP model.
        
        Args:
            inputs: Input dictionary containing:
                - model_path: (optional) Override base model path
                - force_reload: (optional) Force reload even if cached
                
        Returns:
            Dictionary containing:
                - model: MTPModelWrapper instance
                - config: Model configuration
                - device: Device being used
                - memory_footprint: Memory usage statistics
        """
        self.validate_initialized()
        
        # Check for override model path
        model_path = inputs.get("model_path", self.base_model)
        force_reload = inputs.get("force_reload", False)
        
        # Check cache
        cache_path = self.cache_dir / f"mtp_wrapper_{Path(model_path).name}.pt"
        
        if self.use_cached and cache_path.exists() and not force_reload:
            print(f"[TestMTPLoader] Loading cached model from {cache_path}")
            try:
                model = torch.load(cache_path, map_location=self.device)
                print("[TestMTPLoader] Successfully loaded cached model")
            except Exception as e:
                print(f"[TestMTPLoader] Cache load failed: {e}. Creating new model...")
                model = self._create_model(model_path)
                self._save_cache(model, cache_path)
        else:
            print(f"[TestMTPLoader] Creating new MTP wrapper for {model_path}")
            model = self._create_model(model_path)
            if self.use_cached:
                self._save_cache(model, cache_path)
        
        # Get memory footprint
        memory_info = model.get_memory_footprint()
        
        # Prepare for training if needed
        if inputs.get("prepare_training", False):
            model.prepare_for_training()
            
        # Optionally freeze base model for memory efficiency
        if inputs.get("freeze_base", False):
            model.freeze_base_model()
            memory_info = model.get_memory_footprint()  # Recalculate
            
        print(f"[TestMTPLoader] Model loaded successfully")
        print(f"[TestMTPLoader] Total params: {memory_info['total_params']:,}")
        print(f"[TestMTPLoader] Trainable params: {memory_info['trainable_params']:,}")
        print(f"[TestMTPLoader] Estimated memory: {memory_info['total_memory_gb']:.2f} GB")
        
        return {
            "model": model,
            "config": {
                "base_model": model_path,
                "n_future_tokens": self.n_future_tokens,
                "hidden_size": model.hidden_size,
                "vocab_size": model.vocab_size,
            },
            "device": self.device,
            "memory_footprint": memory_info
        }
    
    def _create_model(self, model_path: str) -> MTPModelWrapper:
        """Create new MTP wrapper model.
        
        Args:
            model_path: Base model path
            
        Returns:
            MTPModelWrapper instance
        """
        return MTPModelWrapper(
            base_model_name_or_path=model_path,
            n_future_tokens=self.n_future_tokens,
            device=self.device
        )
    
    def _save_cache(self, model: MTPModelWrapper, cache_path: Path):
        """Save model to cache.
        
        Args:
            model: Model to save
            cache_path: Path to save to
        """
        try:
            print(f"[TestMTPLoader] Saving model to cache: {cache_path}")
            torch.save(model, cache_path)
            print("[TestMTPLoader] Model cached successfully")
        except Exception as e:
            print(f"[TestMTPLoader] Failed to cache model: {e}")


# Alternative lightweight loader for even smaller models
@loader_registry.register("tiny-mtp-loader", category="loader", version="1.0.0")
class TinyMTPLoader(TestMTPLoader):
    """Ultra-lightweight MTP loader for minimal testing.
    
    Uses even smaller models like GPT2 or DistilGPT2 for rapid testing.
    """
    
    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize tiny MTP loader."""
        super().__init__(config)
        
        # Override with tiny model by default
        self.base_model = self.config.get(
            "base_model",
            "distilgpt2"  # Only 82M parameters!
        )