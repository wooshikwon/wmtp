"""
Reward computation components for WMTP Critic approach.

This module contains utilities for computing rewards at different granularities:
- Sequence-level rewards for Stage 1 value head training
"""

from .sequence_reward import compute_sequence_rewards  # noqa: F401

__all__ = ["compute_sequence_rewards"]
