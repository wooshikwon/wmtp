"""
Data and model loaders for WMTP framework - Phase 2 unified.

This module now provides unified loaders that handle all types
of datasets and models through a single interface.
"""

# Import base classes
from .base_loader import BaseLoader, DatasetLoader, ModelLoader
from .unified_data_loader import UnifiedDataLoader

# Import unified loaders (Phase 2)
from .unified_model_loader import UnifiedModelLoader

# Export loaders
__all__ = [
    # Base classes
    "BaseLoader",
    "DatasetLoader",
    "ModelLoader",
    # Unified loaders (Phase 2)
    "UnifiedModelLoader",
    "UnifiedDataLoader",
]

# Registry keys for unified loaders
MODEL_LOADER_KEYS = {
    "unified": "unified-model-loader",
}

DATASET_LOADER_KEYS = {
    "unified": "unified-data-loader",
}
