"""Core components for WMTP framework using registry pattern."""

# Import registry first
# Phase 2: Unified loaders (기존 개별 로더는 삭제됨)
from .loader import (
    UnifiedDataLoader,
    UnifiedModelLoader,
)

# Import all optimizers to register them
from .optimizer import AdamWBF16FusedOptimizer
from .registry import (
    loader_registry,
    optimizer_registry,
    scorer_registry,
    tokenizer_registry,
    trainer_registry,
)

# Import all scorers to register them
from .scorer import CriticDeltaScorer, Rho1ExcessScorer

# Import all tokenizers to register them
from .tokenizer import SentencePieceTokenizer

# Import all trainers to register them
from .trainer import MTPWeightedCETrainer

__all__ = [
    # Registries
    "loader_registry",
    "scorer_registry",
    "tokenizer_registry",
    "trainer_registry",
    "optimizer_registry",
    # Model Loaders
    "HFModelLoader",
    "MTPNativeLoader",
    "CheckpointLoader",
    # Dataset Loaders
    "MBPPDatasetLoader",
    "CodeContestsDatasetLoader",
    "HumanEvalDatasetLoader",
    "CustomDatasetLoader",
    # Scorers
    "CriticDeltaScorer",
    "Rho1ExcessScorer",
    # Trainers
    "MTPWeightedCETrainer",
    # Optimizers
    "AdamWBF16FusedOptimizer",
]
