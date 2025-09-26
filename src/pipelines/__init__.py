"""Training and evaluation pipelines for WMTP framework."""

from .evaluation_pipeline import (  # noqa: F401
    EvaluationOutputs,
    run_evaluation_pipeline,
)
from .training_pipeline import run_training_pipeline  # noqa: F401

# Use run_training_pipeline for all algorithms (mtp-baseline, critic-wmtp, rho1-wmtp)
# The pipeline handles all three cases internally
run_training = run_training_pipeline
run_evaluation = run_evaluation_pipeline

__all__ = [
    "run_training",
    "run_training_pipeline",
    "run_evaluation",
    "run_evaluation_pipeline",
    "EvaluationOutputs",
]
