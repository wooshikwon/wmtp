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

# Note: As of v2.1.0, CriticDeltaScorer has been integrated into CriticWmtpTrainer

# Import all tokenizers to register them
from .tokenizer import SentencePieceTokenizer

# Import all trainers to register them
# Phase 2: Individual trainers replace monolithic MTPWeightedCETrainer
from .trainer import (
    BaseWmtpTrainer,
    BaselineMtpTrainer,
    CriticWmtpTrainer,
    Rho1WmtpTrainer,
    CriticHeadPretrainer,
)

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
    # Scorers (CriticDeltaScorer integrated into CriticWmtpTrainer v2.1.0+)
    # Trainers (Phase 2: separated by algorithm)
    "BaseWmtpTrainer",
    "BaselineMtpTrainer",
    "CriticWmtpTrainer",
    "Rho1WmtpTrainer",
    "CriticHeadPretrainer",
    # Optimizers
    "AdamWBF16FusedOptimizer",
]
