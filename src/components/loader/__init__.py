"""
Data and model loaders for WMTP framework - Phase 2 unified.

This module now provides unified loaders that handle all types
of datasets and models through a single interface.
"""

# Import base classes
from .base_loader import BaseLoader, DatasetLoader, ModelLoader

# Import specialized loaders
from .checkpoint_loader import CheckpointLoader

# Import unified loaders (Phase 2)
from .standardized_model_loader import StandardizedModelLoader
from .unified_data_loader import UnifiedDataLoader

# Export loaders
__all__ = [
    # Base classes
    "BaseLoader",
    "DatasetLoader",
    "ModelLoader",
    # Unified loaders (Phase 2)
    "StandardizedModelLoader",
    "UnifiedDataLoader",
    # Specialized loaders
    "CheckpointLoader",
]

# Registry keys for unified loaders
MODEL_LOADER_KEYS = {
    "standardized": "standardized-model-loader",
    "checkpoint": "checkpoint-loader",
}

DATASET_LOADER_KEYS = {
    "unified": "unified-data-loader",
}
