"""Common type definitions for WMTP project."""

from typing import TypeAlias, Any
import torch
from pathlib import Path

# Model output types
ModelOutput: TypeAlias = dict[str, Any] | torch.Tensor
LogitsOutput: TypeAlias = torch.Tensor  # [B, S, H, V] shape
HiddenStates: TypeAlias = torch.Tensor  # [B, S, D] shape
HeadWeights: TypeAlias = torch.Tensor  # [B, S, H] shape

# Data types
BatchData: TypeAlias = dict[str, torch.Tensor | list[str]]
DatasetDict: TypeAlias = dict[str, Any]  # From HuggingFace datasets
TokenizerOutput: TypeAlias = dict[str, list[int] | list[list[int]]]

# Config types
ConfigDict: TypeAlias = dict[str, Any]
RecipeDict: TypeAlias = dict[str, Any]

# Path types
PathLike: TypeAlias = str | Path

# Metrics types
MetricsDict: TypeAlias = dict[str, float | int | str]
EvalResults: TypeAlias = dict[str, Any]

__all__ = [
    "ModelOutput",
    "LogitsOutput",
    "HiddenStates",
    "HeadWeights",
    "BatchData",
    "DatasetDict",
    "TokenizerOutput",
    "ConfigDict",
    "RecipeDict",
    "PathLike",
    "MetricsDict",
    "EvalResults",
]