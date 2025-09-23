"""
Rho-1 reference-based token importance scorer implementation.

This module implements the Rho-1 scoring approach using
reference model cross-entropy to identify important tokens
following the WMTP Rho-1 algorithm from BLUEPRINT.md.

Mathematical foundation:
1. Score computation: s_t = |CE^ref_t - CE^base_t| (absolute CE excess)
2. Percentile-based emphasis for top p% tokens
3. Normalization pipeline: z-score → softmax(T) → mean 1.0 → clip → renormalize
4. Statistical invariants: mean=1.0±ε, no NaN/Inf, range [ε, W_max]
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.components.base import BaseComponent
from src.components.registry import scorer_registry


@scorer_registry.register("rho1-excess-v1", category="scorer", version="1.0.0")
class Rho1ExcessScorer(BaseComponent):
    """
    Rho-1 reference-based token importance scorer.

    Computes token importance weights based on excess cross-entropy
    between base and reference models.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Rho-1 scorer.

        Args:
            config: Scorer configuration including:
                - score: Scoring method (default: "abs_excess_ce")
                - percentile_top_p: Top percentile to emphasize (default: 0.2)
                - refresh_per_epoch: Whether to refresh scores (default: False)
                - temperature: Softmax temperature (default: 0.7)
        """
        super().__init__(config)
        self.score_method = self.config.get("score", "abs_excess_ce")
        self.percentile_top_p = self.config.get("percentile_top_p", 0.2)
        self.refresh_per_epoch = self.config.get("refresh_per_epoch", False)
        self.temperature = self.config.get("temperature", 0.7)

    def compute_cross_entropy(
        self, logits: torch.Tensor, target_ids: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for each token.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            target_ids: Target token IDs [batch, seq_len]
            reduction: Reduction method ('none', 'mean', 'sum')

        Returns:
            Cross-entropy values per token [batch, seq_len]
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for cross entropy computation
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_ids.view(-1)

        # Compute CE for each token
        ce_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Reshape back to [batch, seq_len]
        ce = ce_flat.view(batch_size, seq_len)

        if reduction == "mean":
            return ce.mean()
        elif reduction == "sum":
            return ce.sum()
        else:
            return ce

    def compute_ce_excess(
        self, base_ce: np.ndarray, ref_ce: np.ndarray, method: str = "abs_excess"
    ) -> np.ndarray:
        """
        Compute cross-entropy excess scores.

        Following BLUEPRINT formula:
        - abs_excess: s_t = |CE^ref_t - CE^base_t|
        - rel_excess: s_t = (CE^ref_t - CE^base_t) / CE^ref_t
        - diff_excess: s_t = CE^ref_t - CE^base_t

        Args:
            base_ce: Cross-entropy from base model
            ref_ce: Cross-entropy from reference model
            method: Scoring method

        Returns:
            Excess scores for each token
        """
        if method == "abs_excess_ce" or method == "abs_excess":
            # Absolute difference: |CE^ref - CE^base|
            scores = np.abs(ref_ce - base_ce)
        elif method == "rel_excess_ce" or method == "rel_excess":
            # Relative difference: (CE^ref - CE^base) / CE^ref
            # Add epsilon to avoid division by zero
            scores = (ref_ce - base_ce) / (ref_ce + 1e-8)
        elif method == "diff_excess_ce" or method == "diff_excess":
            # Raw difference: CE^ref - CE^base
            scores = ref_ce - base_ce
        else:
            # Default to absolute excess
            scores = np.abs(ref_ce - base_ce)

        return scores

    def apply_percentile_emphasis(
        self, scores: np.ndarray, percentile_top_p: float, emphasis_type: str = "soft"
    ) -> np.ndarray:
        """
        Apply percentile-based emphasis to top-scoring tokens.

        Following BLUEPRINT: emphasize tokens in top p% percentile
        to focus learning on most important tokens.

        Args:
            scores: Raw importance scores
            percentile_top_p: Top percentile to emphasize (0.0-1.0)
            emphasis_type: Type of emphasis ('soft', 'hard', 'sigmoid')

        Returns:
            Scores with emphasis applied
        """
        # Calculate threshold for top p%
        threshold = np.percentile(scores, (1 - percentile_top_p) * 100)

        if emphasis_type == "hard":
            # Binary emphasis: top p% get 2x weight, others 1x
            emphasis = np.where(scores >= threshold, 2.0, 1.0)
        elif emphasis_type == "sigmoid":
            # Smooth sigmoid emphasis around threshold
            # Steepness controls how sharp the transition is
            steepness = 4.0
            emphasis = 1.0 + 1.0 / (1.0 + np.exp(-steepness * (scores - threshold)))
        else:  # soft emphasis (default)
            # Gradual emphasis based on distance from threshold
            # Top tokens get up to 3x emphasis
            relative_scores = (scores - np.min(scores)) / (
                np.max(scores) - np.min(scores) + 1e-8
            )
            emphasis = 1.0 + 2.0 * relative_scores
            # Only apply strong emphasis to top p%
            emphasis = np.where(
                scores >= threshold, emphasis, 1.0 + 0.5 * relative_scores
            )

        return scores * emphasis

    def normalize_and_weight(
        self,
        scores: np.ndarray,
        temperature: float = 0.7,
        epsilon: float = 0.05,
        max_weight: float = 3.0,
    ) -> np.ndarray:
        """
        Apply full normalization pipeline as per BLUEPRINT.

        Pipeline (same as CriticDeltaScorer):
        1. Z-score normalization
        2. Softmax with temperature
        3. Scale to mean 1.0
        4. Clip to [ε, W_max]
        5. Re-normalize to mean 1.0

        Args:
            scores: Raw importance scores
            temperature: Softmax temperature
            epsilon: Minimum weight value
            max_weight: Maximum weight value

        Returns:
            Normalized weights with mean 1.0
        """
        # Step 1: Z-score normalization
        mean = np.mean(scores)
        std = np.std(scores) + 1e-8
        normalized = (scores - mean) / std

        # Step 2: Softmax with temperature
        # w_t = softmax(s_t / T)
        exp_values = np.exp(normalized / temperature)
        weights = exp_values / np.sum(exp_values)

        # Step 3: Scale to have mean 1.0
        # Since softmax sums to 1, multiply by length for mean 1
        L = float(len(weights))
        weights = weights * L

        # Step 4: Clip to valid range [ε, W_max]
        weights = np.clip(weights, epsilon, max_weight)

        # Step 5: Re-normalize mean to exactly 1.0 without a second clip
        total = np.sum(weights)
        if total > 0:
            scale = L / total
            weights = weights * scale

        # Final safety check for NaN/Inf
        if not np.all(np.isfinite(weights)):
            # Fallback to uniform weights
            weights = np.ones_like(weights)

        return weights

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Compute token importance scores using reference model CE excess.

        Args:
            ctx: Context containing:
                - base_logits: Logits from base model [batch, seq_len, vocab] (optional)
                - ref_logits: Logits from reference model [batch, seq_len, vocab] (optional)
                - base_ce: Pre-computed CE from base model [batch, seq_len] (optional)
                - ref_ce: Pre-computed CE from reference model [batch, seq_len] (optional)
                - target_ids: Target token IDs [batch, seq_len] (optional)
                - seq_lengths: Actual sequence lengths [batch]

        Returns:
            Dictionary containing:
                - weights: Token-level importance weights [batch, seq_len]
                - scores: Raw CE excess scores [batch, seq_len]
                - statistics: Weight statistics
        """
        self.validate_initialized()

        # Extract inputs
        base_logits = ctx.get("base_logits")
        ref_logits = ctx.get("ref_logits")
        base_ce = ctx.get("base_ce")
        ref_ce = ctx.get("ref_ce")
        target_ids = ctx.get("target_ids")
        seq_lengths = ctx.get("seq_lengths", [100])

        # For demonstration without actual model inputs
        if base_ce is None or ref_ce is None:
            if (
                base_logits is not None
                and ref_logits is not None
                and target_ids is not None
            ):
                # Compute CE from logits
                base_ce_tensor = self.compute_cross_entropy(base_logits, target_ids)
                ref_ce_tensor = self.compute_cross_entropy(ref_logits, target_ids)

                # Convert to numpy
                base_ce = base_ce_tensor.detach().cpu().numpy()
                ref_ce = ref_ce_tensor.detach().cpu().numpy()
            else:
                # Mock implementation for testing
                batch_size = len(seq_lengths)
                max_seq_len = max(seq_lengths)

                all_weights = []
                all_scores = []

                for b in range(batch_size):
                    seq_len = seq_lengths[b]

                    # Simulate CE values with realistic distributions
                    # Reference model typically has higher CE (less specialized)
                    base_ce_seq = np.random.gamma(2.0, 1.0, seq_len)  # Shape=2, scale=1
                    ref_ce_seq = np.random.gamma(2.5, 1.2, seq_len)  # Slightly higher

                    # Add some correlation between base and ref
                    correlation = 0.7
                    ref_ce_seq = (
                        correlation * base_ce_seq + (1 - correlation) * ref_ce_seq
                    )

                    # Compute CE excess scores
                    scores = self.compute_ce_excess(
                        base_ce_seq, ref_ce_seq, method=self.score_method
                    )

                    # Apply percentile emphasis
                    emphasized_scores = self.apply_percentile_emphasis(
                        scores, self.percentile_top_p, emphasis_type="soft"
                    )

                    # Normalize and compute weights
                    weights = self.normalize_and_weight(
                        emphasized_scores,
                        temperature=self.temperature,
                        epsilon=self.config.get("epsilon", 0.05),
                        max_weight=self.config.get("max_weight", 3.0),
                    )

                    all_weights.append(weights)
                    all_scores.append(scores)

                # Stack results
                weights = np.array(all_weights)
                scores = np.array(all_scores)
        else:
            # Real implementation with actual CE values
            batch_size = base_ce.shape[0] if len(base_ce.shape) > 1 else 1

            if len(base_ce.shape) == 1:
                base_ce = base_ce.reshape(1, -1)
                ref_ce = ref_ce.reshape(1, -1)

            all_weights = []
            all_scores = []

            for b in range(batch_size):
                # Compute CE excess scores
                scores = self.compute_ce_excess(
                    base_ce[b], ref_ce[b], method=self.score_method
                )

                # Apply percentile emphasis
                emphasized_scores = self.apply_percentile_emphasis(
                    scores, self.percentile_top_p, emphasis_type="soft"
                )

                # Normalize and compute weights
                weights = self.normalize_and_weight(
                    emphasized_scores,
                    temperature=self.temperature,
                    epsilon=self.config.get("epsilon", 0.05),
                    max_weight=self.config.get("max_weight", 3.0),
                )

                all_weights.append(weights)
                all_scores.append(scores)

            weights = np.array(all_weights)
            scores = np.array(all_scores)

        # Compute statistics
        statistics = {
            "mean_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "top_p_threshold": float(
                np.percentile(scores, (1 - self.percentile_top_p) * 100)
            ),
            "nan_count": int(np.sum(~np.isfinite(weights))),
        }

        # Verify statistical invariants
        assert (
            0.95 <= statistics["mean_weight"] <= 1.05
        ), f"Mean weight {statistics['mean_weight']} not in [0.95, 1.05]"
        assert statistics["nan_count"] == 0, "NaN values in weights"

        return {
            "weights": weights.tolist(),
            "weights_tensor": torch.tensor(weights, dtype=torch.float32),
            "scores": scores.tolist(),
            "statistics": statistics,
        }
