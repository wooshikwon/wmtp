"""Core components for WMTP framework using registry pattern."""

# Import registry first
# Import all dataset loaders to register them
from .loader.dataset import (
    CodeContestsDatasetLoader,
    CustomDatasetLoader,
    HumanEvalDatasetLoader,
    MBPPDatasetLoader,
)

# Import all model loaders to register them
from .loader.model import (
    CheckpointLoader,
    HFModelLoader,
    MTPNativeLoader,
)

# Import all optimizers to register them
from .optimizer import AdamWBF16FusedOptimizer
from .registry import (
    loader_registry,
    optimizer_registry,
    scorer_registry,
    trainer_registry,
)

# Import all scorers to register them
from .scorer import CriticDeltaScorer, Rho1ExcessScorer

# Import all trainers to register them
from .trainer import MTPWeightedCETrainer

__all__ = [
    # Registries
    "loader_registry",
    "scorer_registry",
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
