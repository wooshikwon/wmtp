"""Core components for WMTP framework using registry pattern."""

# Import registry first
# Phase 2: Unified loaders (기존 개별 로더는 삭제됨)
from .loader import (
    DataLoader,  # Phase 2 리팩토링: 단순화된 데이터 로더
    ModelLoader,  # Phase 2 리팩토링: 단순화된 모델 로더
)

# Import all optimizers to register them
from .optimizer import AdamWFusedOptimizer
from .registry import (
    loader_registry,
    optimizer_registry,
    # scorer_registry,  # Removed in v2.1.0
    tokenizer_registry,
    trainer_registry,
)

# Note: As of v2.1.0, CriticDeltaScorer has been integrated into CriticWmtpTrainer
# Import all tokenizers to register them
from .tokenizer import SentencePieceTokenizer

# Import all trainers to register them
# Phase 2: Individual trainers replace monolithic MTPWeightedCETrainer
from .trainer import (
    BaselineMtpTrainer,
    BaseWmtpTrainer,
    CriticHeadPretrainer,
    CriticWmtpTrainer,
    Rho1WmtpTrainer,
)

__all__ = [
    # Registries
    "loader_registry",
    # "scorer_registry",  # Removed in v2.1.0 - scorer logic integrated into trainers
    "tokenizer_registry",
    "trainer_registry",
    "optimizer_registry",
    # Unified Loaders (Phase 2)
    "DataLoader",  # Phase 2: 단순화된 데이터 로더
    "ModelLoader",  # Phase 2: 단순화된 모델 로더
    # Tokenizers
    "SentencePieceTokenizer",
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
    "AdamWFusedOptimizer",
]
