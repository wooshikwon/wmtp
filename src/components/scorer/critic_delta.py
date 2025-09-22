"""
Critic-based token importance scorer implementation.

This module implements the critic-weighted scoring approach using
a reward model to compute token-level importance weights following
the WMTP Critic-Delta algorithm from BLUEPRINT.md.

Mathematical foundation:
1. RM sequence reward R → token-level rewards r_t via GAE
2. Value function V_θ(h_t) regression: min Σ_t (V_θ(h_t) - V̂_t)²
3. Delta computation: δ_t = V_t - V_{t-1} (V_{-1} = 0)
4. Weight normalization: w_t = softmax(δ_t / T) with mean=1.0 enforcement
"""

from typing import Any

import numpy as np
import torch.nn as nn

from src.components.base import BaseComponent
from src.components.registry import scorer_registry


@scorer_registry.register("critic-delta-v1", category="scorer", version="1.0.0")
class CriticDeltaScorer(BaseComponent):
    """
    Critic-based token importance scorer.

    Implements the two-stage approach:
    1. Stage 1: Train value head V_θ to predict cumulative rewards
    2. Stage 2: Use V_θ to compute token deltas for importance weighting
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize critic scorer.

        Args:
            config: Scorer configuration including:
                - target: Target for critic (default: "rm_sequence")
                - token_spread: Method for spreading rewards (default: "gae")
                - delta_mode: Delta computation mode (default: "td")
                - normalize: Normalization method (default: "zscore")
                - temperature: Softmax temperature (default: 0.7)
                - gamma: Discount factor for GAE (default: 0.99)
                - gae_lambda: GAE lambda parameter (default: 0.95)
        """
        super().__init__(config)
        self.target = self.config.get("target", "rm_sequence")
        self.token_spread = self.config.get("token_spread", "gae")
        self.delta_mode = self.config.get("delta_mode", "td")
        self.normalize = self.config.get("normalize", "zscore")
        self.temperature = self.config.get("temperature", 0.7)
        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)

        # Value head will be initialized when needed
        self.value_head: nn.Module | None = None
        self.value_head_trained = False

    def setup(self, ctx: dict[str, Any]) -> None:
        """Initialize value head if needed."""
        super().setup(ctx)

        # Initialize value head based on hidden size
        hidden_size = ctx.get("hidden_size", 4096)  # Default for 7B model
        if self.value_head is None:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )

    def spread_reward_to_tokens(
        self,
        sequence_reward: float,
        seq_length: int,
        attention_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Spread sequence-level reward to token-level rewards.

        Args:
            sequence_reward: Sequence-level reward from RM
            seq_length: Length of sequence
            attention_weights: Optional attention weights for spread

        Returns:
            Token-level reward array
        """
        if self.token_spread == "uniform":
            # Uniform distribution
            return np.full(seq_length, sequence_reward / seq_length)

        elif self.token_spread == "length":
            # Inversely proportional to length (shorter gets more)
            weights = np.arange(seq_length, 0, -1)
            weights = weights / weights.sum()
            return sequence_reward * weights

        elif self.token_spread == "attention":
            # Use attention weights if available
            if attention_weights is not None:
                weights = attention_weights / attention_weights.sum()
                return sequence_reward * weights
            else:
                # Fallback to uniform
                return np.full(seq_length, sequence_reward / seq_length)

        elif self.token_spread == "gae":
            # GAE-style backward attribution
            # Later tokens get discounted attribution
            discounts = np.power(self.gamma, np.arange(seq_length))
            discounts = discounts[::-1]  # Reverse for backward attribution
            weights = discounts / discounts.sum()
            return sequence_reward * weights

        else:
            # Default to uniform
            return np.full(seq_length, sequence_reward / seq_length)

    def compute_gae_returns(
        self, rewards: np.ndarray, values: np.ndarray, next_value: float = 0.0
    ) -> np.ndarray:
        """
        Compute GAE returns for value function targets.

        GAE formula:
        Â_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            rewards: Token-level rewards
            values: Current value estimates
            next_value: Bootstrap value for last state

        Returns:
            GAE advantage estimates (used as value targets)
        """
        seq_length = len(rewards)
        advantages = np.zeros(seq_length)
        last_gae = 0

        # Append next_value for TD computation
        values_with_next = np.append(values, next_value)

        # Backward pass to compute GAE
        for t in reversed(range(seq_length)):
            # TD error: δ_t = r_t + γV_{t+1} - V_t
            delta = rewards[t] + self.gamma * values_with_next[t + 1] - values[t]

            # GAE: Â_t = δ_t + (γλ)Â_{t+1}
            advantages[t] = delta + self.gamma * self.gae_lambda * last_gae
            last_gae = advantages[t]

        # Value targets: V̂_t = Â_t + V_t
        value_targets = advantages + values

        return value_targets

    def compute_deltas(self, values: np.ndarray) -> np.ndarray:
        """
        Compute temporal difference deltas.

        Following BLUEPRINT: δ_t = V_t - V_{t-1} where V_{-1} = 0

        Args:
            values: Value function predictions

        Returns:
            Delta values for each token
        """
        if self.delta_mode == "td":
            # Temporal difference: δ_t = V_t - V_{t-1}
            # For first token: δ_0 = V_0 - 0 = V_0
            deltas = np.diff(values, prepend=0)
        else:  # diff mode
            # Alternative: difference from mean
            deltas = values - np.mean(values)

        return deltas

    def normalize_and_weight(
        self, deltas: np.ndarray, epsilon: float = 0.05, max_weight: float = 3.0
    ) -> np.ndarray:
        """
        Apply full normalization pipeline as per BLUEPRINT.

        Pipeline:
        1. Z-score normalization
        2. Softmax with temperature
        3. Scale to mean 1.0
        4. Clip to [ε, W_max]
        5. Re-normalize to mean 1.0

        Args:
            deltas: Raw delta values
            epsilon: Minimum weight value
            max_weight: Maximum weight value

        Returns:
            Normalized weights with mean 1.0
        """
        # Step 1: Z-score normalization
        if self.normalize == "zscore":
            mean = np.mean(deltas)
            std = np.std(deltas) + 1e-8
            normalized = (deltas - mean) / std
        elif self.normalize == "minmax":
            min_val = np.min(deltas)
            max_val = np.max(deltas)
            range_val = max_val - min_val + 1e-8
            normalized = (deltas - min_val) / range_val
        else:
            normalized = deltas

        # Step 2: Softmax with temperature
        # w_t = softmax(δ_t / T)
        exp_values = np.exp(normalized / self.temperature)
        weights = exp_values / np.sum(exp_values)

        # Step 3: Scale to have mean 1.0
        # Since softmax sums to 1, multiply by length for mean 1
        weights = weights * len(weights)

        # Step 4: Clip to valid range [ε, W_max]
        weights = np.clip(weights, epsilon, max_weight)

        # Step 5: Re-normalize to ensure mean is exactly 1.0
        # But ensure we don't exceed max_weight after renormalization
        current_mean = np.mean(weights)
        if current_mean != 1.0:
            # Scale towards mean 1.0 but respect bounds
            scale_factor = 1.0 / current_mean
            scaled_weights = weights * scale_factor

            # If scaling would exceed max_weight, use alternative approach
            if np.any(scaled_weights > max_weight):
                # Use iterative approach to get as close to mean=1.0 as possible
                # while respecting bounds
                weights = weights - (current_mean - 1.0)
                weights = np.clip(weights, epsilon, max_weight)
            else:
                weights = scaled_weights

        # Final safety check for NaN/Inf
        if not np.all(np.isfinite(weights)):
            # Fallback to uniform weights
            weights = np.ones_like(weights)

        return weights

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Compute token importance scores using critic approach.

        Args:
            ctx: Context containing:
                - hidden_states: Hidden states from model [batch, seq_len, hidden]
                - rewards: Sequence-level rewards from RM [batch]
                - attention_weights: Optional attention weights [batch, seq_len]
                - seq_lengths: Actual sequence lengths [batch]

        Returns:
            Dictionary containing:
                - weights: Token-level importance weights [batch, seq_len]
                - deltas: Raw delta values [batch, seq_len]
                - values: Value function predictions [batch, seq_len]
                - statistics: Weight statistics
        """
        self.validate_initialized()

        # Extract inputs
        hidden_states = ctx.get("hidden_states")
        rewards = ctx.get("rewards", [1.0])  # Default reward if not provided
        seq_lengths = ctx.get("seq_lengths", [100])
        attention_weights = ctx.get("attention_weights")

        # For demonstration without actual model inputs
        if hidden_states is None:
            # Mock implementation
            batch_size = len(rewards)
            max_seq_len = max(seq_lengths)

            all_weights = []
            all_deltas = []
            all_values = []

            for b in range(batch_size):
                seq_len = seq_lengths[b]
                reward = rewards[b]

                # Step 1: Spread sequence reward to tokens
                token_rewards = self.spread_reward_to_tokens(
                    reward, seq_len, attention_weights[b] if attention_weights else None
                )

                # Step 2: Simulate value predictions (in real case, use value_head)
                # V_t should approximate cumulative future rewards
                cumulative = np.cumsum(token_rewards[::-1])[::-1]
                values = cumulative + np.random.randn(seq_len) * 0.1  # Add noise

                # Step 3: Compute GAE returns as targets
                value_targets = self.compute_gae_returns(
                    token_rewards, values, next_value=0.0
                )

                # In training mode, we would minimize (V_θ(h_t) - V̂_t)²
                # Here we use the targets as our value predictions
                values = value_targets

                # Step 4: Compute deltas δ_t = V_t - V_{t-1}
                deltas = self.compute_deltas(values)

                # Step 5: Normalize and compute weights
                weights = self.normalize_and_weight(
                    deltas,
                    epsilon=self.config.get("epsilon", 0.05),
                    max_weight=self.config.get("max_weight", 3.0),
                )

                all_weights.append(weights)
                all_deltas.append(deltas)
                all_values.append(values)

            # Stack results
            weights = np.array(all_weights)
            deltas = np.array(all_deltas)
            values = np.array(all_values)

        else:
            # Real implementation with model hidden states
            # This would use the actual value_head neural network
            raise NotImplementedError(
                "Full implementation requires actual model hidden states"
            )

        # Compute statistics
        statistics = {
            "mean_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "mean_value": float(np.mean(values)),
            "nan_count": int(np.sum(~np.isfinite(weights))),
        }

        # Verify statistical invariants
        assert (
            0.95 <= statistics["mean_weight"] <= 1.05
        ), f"Mean weight {statistics['mean_weight']} not in [0.95, 1.05]"
        assert statistics["nan_count"] == 0, "NaN values in weights"

        return {
            "weights": weights.tolist(),
            "deltas": deltas.tolist(),
            "values": values.tolist(),
            "statistics": statistics,
        }
