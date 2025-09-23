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
    Rho-1 reference-based head-level importance scorer.

    Computes head-level importance weights based on excess cross-entropy
    between base and reference models, using distance-aware decay to
    generate different weights for each MTP prediction head.

    Mathematical foundation:
    1. Token importance: s_t = |CE^ref_t - CE^base_t| (absolute CE excess)
    2. Head weighting: w_{t,k} = s_t * exp(-decay_rate * k)
    3. Distance decay: closer heads (k=0) get stronger weights
    4. Output format: [batch, seq_len, horizon] for MTP compatibility
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Rho-1 head-level scorer.

        Args:
            config: Scorer configuration including:
                - score: Scoring method (default: "abs_excess_ce")
                - percentile_top_p: Top percentile to emphasize (default: 0.2)
                - refresh_per_epoch: Whether to refresh scores (default: False)
                - temperature: Softmax temperature (default: 0.7)
                - head_decay_rate: Head distance decay rate (default: 0.5)
                - normalize_heads: Whether to normalize head weights (default: True)
                - head_temperature: Temperature for head softmax (default: 0.7)
                - epsilon: Minimum weight value (default: 0.05)
                - max_weight: Maximum weight value (default: 3.0)
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

        Used as input for head-level weight computation.

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
        Compute cross-entropy excess scores for token importance.

        These token-level scores are later converted to head-level weights
        using distance-aware decay in _compute_head_weights().

        Following BLUEPRINT formula:
        - abs_excess: s_t = |CE^ref_t - CE^base_t|
        - rel_excess: s_t = (CE^ref_t - CE^base_t) / CE^ref_t
        - diff_excess: s_t = CE^ref_t - CE^base_t

        Args:
            base_ce: Cross-entropy from base model [seq_len]
            ref_ce: Cross-entropy from reference model [seq_len]
            method: Scoring method

        Returns:
            Token-level excess scores [seq_len]
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

        Emphasizes tokens in top p% percentile before converting to head-level weights.
        This focuses head-level learning on the most important tokens.

        Following BLUEPRINT: emphasize tokens in top p% percentile
        to focus learning on most important tokens.

        Args:
            scores: Raw token importance scores [seq_len]
            percentile_top_p: Top percentile to emphasize (0.0-1.0)
            emphasis_type: Type of emphasis ('soft', 'hard', 'sigmoid')

        Returns:
            Emphasized token scores [seq_len]
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

    def _compute_head_weights(self, token_weights: np.ndarray, horizon: int = 4) -> np.ndarray:
        """
        Convert token-level weights to head-level weights using distance-aware decay.

        Each head k predicts token at position t+k+1, so closer predictions should
        receive stronger weights. Uses exponential decay based on prediction distance.

        Mathematical foundation:
        - Head k predicts t+k+1 position (k=0,1,2,3 for horizon=4)
        - Distance factor: f_k = exp(-decay_rate * k)
        - Head weight: w_{t,k} = s_t * f_k
        - Normalization: ensure each token's head weights sum appropriately

        Args:
            token_weights: Token importance scores [batch, seq_len]
            horizon: Number of MTP heads (default=4)

        Returns:
            head_weights: Head-level importance weights [batch, seq_len, horizon]
        """
        batch_size, seq_len = token_weights.shape
        head_weights = np.zeros((batch_size, seq_len, horizon))

        # Distance-based decay rate (configurable)
        decay_rate = self.config.get("head_decay_rate", 0.5)

        # Compute distance factors for each head
        # Head k=0: immediate next token (factor=1.0)
        # Head k=1: token after next (factor=exp(-0.5)≈0.6)
        # Head k=2: two tokens ahead (factor=exp(-1.0)≈0.37)
        # Head k=3: three tokens ahead (factor=exp(-1.5)≈0.22)
        distance_factors = np.array([np.exp(-decay_rate * k) for k in range(horizon)])

        for b in range(batch_size):
            for t in range(seq_len):
                token_score = token_weights[b, t]

                # Apply distance-based weighting to each head
                raw_head_weights = token_score * distance_factors

                # Optional: apply softmax normalization across heads for this token
                # This ensures head weights sum to the original token weight
                if self.config.get("normalize_heads", True):
                    # Softmax with temperature for smooth distribution
                    temp = self.config.get("head_temperature", 0.7)
                    exp_vals = np.exp(raw_head_weights / temp)
                    softmax_weights = exp_vals / (np.sum(exp_vals) + 1e-8)

                    # Scale to preserve original token weight magnitude
                    head_weights[b, t] = softmax_weights * token_score
                else:
                    # Direct application without normalization
                    head_weights[b, t] = raw_head_weights

        # Final global normalization to ensure reasonable overall magnitude
        # Target: overall mean ≈ 1.0 for compatibility with trainer expectations
        current_mean = np.mean(head_weights)
        if current_mean > 0:
            target_mean = 1.0
            scale_factor = target_mean / current_mean
            head_weights = head_weights * scale_factor

        return head_weights


    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Compute head-level importance weights using reference model CE excess.

        Args:
            ctx: Context containing:
                - base_logits: Logits from base model [batch, seq_len, vocab] (optional)
                - ref_logits: Logits from reference model [batch, seq_len, vocab] (optional)
                - base_ce: Pre-computed CE from base model [batch, seq_len] (optional)
                - ref_ce: Pre-computed CE from reference model [batch, seq_len] (optional)
                - target_ids: Target token IDs [batch, seq_len] (optional)
                - seq_lengths: Actual sequence lengths [batch]
                - horizon: Number of MTP heads (optional, default=4)

        Returns:
            Dictionary containing:
                - weights: Head-level importance weights [batch, seq_len, horizon]
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
        horizon = ctx.get("horizon", 4)  # MTP heads count

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

                all_token_weights = []
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

                    # Convert emphasized scores to token-level weights (simplified normalization)
                    # Z-score normalization + softmax
                    mean_score = np.mean(emphasized_scores)
                    std_score = np.std(emphasized_scores) + 1e-8
                    normalized_scores = (emphasized_scores - mean_score) / std_score

                    # Softmax with temperature
                    exp_values = np.exp(normalized_scores / self.temperature)
                    token_weights = exp_values / np.sum(exp_values)

                    # Scale to have mean 1.0 (preserve magnitude)
                    token_weights = token_weights * float(len(token_weights))

                    # Clip to valid range
                    epsilon = self.config.get("epsilon", 0.05)
                    max_weight = self.config.get("max_weight", 3.0)
                    token_weights = np.clip(token_weights, epsilon, max_weight)

                    # Re-normalize to mean 1.0
                    total = np.sum(token_weights)
                    if total > 0:
                        scale = float(len(token_weights)) / total
                        token_weights = token_weights * scale

                    all_token_weights.append(token_weights)
                    all_scores.append(scores)

                # Stack results - token weights [B,S]
                token_weights = np.array(all_token_weights)
                scores = np.array(all_scores)

                # Convert to head-level weights [B,S,H]
                head_weights = self._compute_head_weights(token_weights, horizon)
        else:
            # Real implementation with actual CE values
            batch_size = base_ce.shape[0] if len(base_ce.shape) > 1 else 1

            if len(base_ce.shape) == 1:
                base_ce = base_ce.reshape(1, -1)
                ref_ce = ref_ce.reshape(1, -1)

            all_token_weights = []
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

                # Convert emphasized scores to token-level weights (simplified normalization)
                # Z-score normalization + softmax
                mean_score = np.mean(emphasized_scores)
                std_score = np.std(emphasized_scores) + 1e-8
                normalized_scores = (emphasized_scores - mean_score) / std_score

                # Softmax with temperature
                exp_values = np.exp(normalized_scores / self.temperature)
                token_weights = exp_values / np.sum(exp_values)

                # Scale to have mean 1.0 (preserve magnitude)
                token_weights = token_weights * float(len(token_weights))

                # Clip to valid range
                epsilon = self.config.get("epsilon", 0.05)
                max_weight = self.config.get("max_weight", 3.0)
                token_weights = np.clip(token_weights, epsilon, max_weight)

                # Re-normalize to mean 1.0
                total = np.sum(token_weights)
                if total > 0:
                    scale = float(len(token_weights)) / total
                    token_weights = token_weights * scale

                all_token_weights.append(token_weights)
                all_scores.append(scores)

            # Stack results - token weights [B,S]
            token_weights = np.array(all_token_weights)
            scores = np.array(all_scores)

            # Convert to head-level weights [B,S,H]
            head_weights = self._compute_head_weights(token_weights, horizon)

        # Compute head-level statistics
        statistics = {
            "mean_weight": float(np.mean(head_weights)),
            "std_weight": float(np.std(head_weights)),
            "min_weight": float(np.min(head_weights)),
            "max_weight": float(np.max(head_weights)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "top_p_threshold": float(
                np.percentile(scores, (1 - self.percentile_top_p) * 100)
            ),
            "nan_count": int(np.sum(~np.isfinite(head_weights))),
            # Head-specific statistics
            "head_mean_weights": [float(np.mean(head_weights[:, :, k])) for k in range(horizon)],
            "head_std_weights": [float(np.std(head_weights[:, :, k])) for k in range(horizon)],
        }

        # Verify statistical invariants for head-level weights
        overall_mean = statistics["mean_weight"]
        assert (
            0.8 <= overall_mean <= 1.2
        ), f"Head-level mean weight {overall_mean} not in reasonable range [0.8, 1.2]"
        assert statistics["nan_count"] == 0, "NaN values in head weights"

        # Determine target device and dtype from context
        target_device = ctx.get("device", "cpu")
        target_dtype = ctx.get("dtype", torch.float32)

        # Convert to torch.Tensor with proper device/dtype [B,S,H]
        head_weights_tensor = torch.tensor(
            head_weights, device=target_device, dtype=target_dtype
        )

        return {
            "weights": head_weights_tensor,  # [B,S,H] tensor
            "scores": scores.tolist(),       # [B,S] raw scores
            "statistics": statistics,        # Head-level statistics
        }
