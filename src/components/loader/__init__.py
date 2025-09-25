"""
Data and model loaders for WMTP framework.

This module provides individual loaders for datasets and models,
each registered as separate components in the registry.
"""

# Import base classes
from .base_loader import BaseLoader, DatasetLoader, ModelLoader

# Import all dataset loaders (auto-registers them)
from .dataset import (
    CodeContestsDatasetLoader,
    CustomDatasetLoader,
    HumanEvalDatasetLoader,
    MBPPDatasetLoader,
)

# Import all model loaders (auto-registers them)
from .model import (
    CheckpointLoader,
    HFModelLoader,
    MTPNativeCPULoader,
    MTPNativeLoader,
    ShearedLLaMALoader,
    StarlingRMLoader,
)

# Export all loaders
__all__ = [
    # Base classes
    "BaseLoader",
    "DatasetLoader",
    "ModelLoader",
    # Model loaders
    "HFModelLoader",
    "MTPNativeLoader",
    "MTPNativeCPULoader",
    "StarlingRMLoader",
    "ShearedLLaMALoader",
    "CheckpointLoader",
    # Dataset loaders
    "MBPPDatasetLoader",
    "CodeContestsDatasetLoader",
    "HumanEvalDatasetLoader",
    "CustomDatasetLoader",
]

# Registry keys for loaders
MODEL_LOADER_KEYS = {
    "huggingface": "hf-model",
    "mtp-native": "mtp-native",
    "checkpoint": "checkpoint",
}

DATASET_LOADER_KEYS = {
    "mbpp": "mbpp-dataset",
    "codecontests": "codecontests-dataset",
    "humaneval": "humaneval-dataset",
    "custom": "custom-dataset",
}
