"""
Data and model loaders for WMTP framework.

This module provides loaders for datasets and models with
local-first S3 fallback support.
"""

# Import base classes
from .base_loader import BaseLoader, DatasetLoader, ModelLoader

# Import concrete implementations
from .dataset_contest_loader import CodeContestsDatasetLoader
from .dataset_mbpp_loader import MBPPDatasetLoader
from .hf_local_s3_loader import HFLocalS3Loader

# Export all loaders
__all__ = [
    # Base classes
    "BaseLoader",
    "DatasetLoader",
    "ModelLoader",
    # Concrete loaders
    "HFLocalS3Loader",
    "MBPPDatasetLoader",
    "CodeContestsDatasetLoader",
]

# Loader registry keys for reference
LOADER_REGISTRY_KEYS = {
    "model": "hf-local-s3-loader",
    "mbpp": "dataset-mbpp-loader",
    "contest": "dataset-contest-loader",
}


def get_loader_key(source: str) -> str:
    """
    Get the registry key for a data source.

    Args:
        source: Data source name

    Returns:
        Registry key for the loader
    """
    return LOADER_REGISTRY_KEYS.get(source, source)
