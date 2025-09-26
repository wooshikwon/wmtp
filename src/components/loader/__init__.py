"""
Data and model loaders for WMTP framework - Phase 2 unified.

This module now provides unified loaders that handle all types
of datasets and models through a single interface.
"""

# Import base classes
from .base_loader import BaseLoader, DatasetLoader
from .base_loader import ModelLoader as BaseModelLoader

# Import specialized loaders
from .checkpoint_loader import CheckpointLoader

# Import unified loaders (Phase 2)
from .model_loader import ModelLoader  # Phase 2 리팩토링: 단순화된 모델 로더
from .data_loader import DataLoader  # Phase 2 리팩토링: 단순화된 데이터 로더

# Export loaders
__all__ = [
    # Base classes
    "BaseLoader",
    "DatasetLoader",
    "BaseModelLoader",
    # Unified loaders (Phase 2)
    "ModelLoader",  # Phase 2: 단순화된 버전
    "DataLoader",  # Phase 2: 단순화된 버전
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
