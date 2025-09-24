"""Training and evaluation pipelines for WMTP framework."""

from .evaluation_pipeline import EvaluationPipeline  # noqa: F401
from .training_pipeline import run_training_pipeline  # noqa: F401

# Use run_training_pipeline for all algorithms (mtp-baseline, critic-wmtp, rho1-wmtp)
# The pipeline handles all three cases internally
run_training = run_training_pipeline

__all__ = [
    "run_training",
    "run_training_pipeline",
    "EvaluationPipeline",
]
