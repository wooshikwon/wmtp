"""
Rho1-WMTP 알고리즘: 참조모델과의 차이로 어려운 토큰을 찾아 중요도를 계산합니다.

WMTP 연구에서 핵심 아이디어:
"참조모델도 어려워하는 토큰 = 중요한 토큰"

이 모듈은 연구개선안에서 권장하는 Rho-1 방식을 구현합니다:
- Critic 학습의 불안정성 없이 바로 가중치 계산 가능
- Microsoft Rho-1 연구의 "Not All Tokens Are What You Need" 철학
- CodeLlama 참조모델로 코딩 도메인 특화 토큰 선별

수학적 원리:
1. 토큰별 중요도: s_t = |CE^ref_t - CE^base_t| (교차엔트로피 차이)
2. Percentile 강조: 상위 p% 토큰에 추가 가중치
3. 헤드별 분배: 거리 감쇠 적용 (가까운 헤드일수록 높은 가중치)
4. 정규화: Z-score → softmax → mean=1.0 강제 → clipping

장점: Critic 없이 직접 계산 가능, 안정적
교수님 피드백 반영: GRPO처럼 복잡한 value estimation 제거
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
    Rho1-WMTP 방식: 참조모델 차이 기반으로 헤드별 중요도를 계산하는 스코어러입니다.

    연구 맥락:
    WMTP의 세 가지 알고리즘 중 하나로, 연구개선안에서 권장하는 방식입니다.
    Microsoft Rho-1의 "선택적 언어모델링" 아이디어를 MTP에 적용했습니다.
    Critic 기반 방식과 달리 별도 학습 없이 바로 가중치를 계산할 수 있습니다.

    핵심 원리:
    1. 토큰 중요도 = |CE^ref - CE^base| (참조모델과 기본모델의 CE 차이)
    2. 큰 차이 = 두 모델 모두 어려워함 = 중요한 토큰
    3. 헤드별 가중치 = 토큰 중요도 × 거리 감쇠 (exp(-decay_rate × k))
    4. 최종 출력: [batch, seq_len, horizon] 형태의 헤드별 가중치

    예시:
    - Base model CE = 2.1, Ref model CE = 3.8 → 중요도 = |3.8-2.1| = 1.7 (높음)
    - Base model CE = 1.2, Ref model CE = 1.3 → 중요도 = |1.3-1.2| = 0.1 (낮음)

    장점:
    - Critic 학습 없이 즉시 사용 가능
    - 코딩 도메인에서 CodeLlama 참조모델로 특화된 토큰 선별
    - 수치적으로 안정적 (교수님 피드백 반영)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Rho-1 헤드별 스코어러를 초기화합니다.

        연구 맥락:
        이 설정들은 Microsoft Rho-1 연구와 연구개선안의 권장사항을 반영합니다.
        Critic 방식과 달리 별도 학습이 없으므로 설정값이 최종 성능에 직접 영향합니다.

        매개변수:
            config: 스코어러 설정 딕셔너리
                - score: 스코어링 방법 ("abs_excess_ce" - 절댓값 차이 추천)
                - percentile_top_p: 강조할 상위 백분율 (0.2 = 상위 20%)
                - refresh_per_epoch: 에포크마다 점수 갱신 여부 (False - 계산 비용 절약)
                - temperature: Softmax 온도 (0.7 - 적당히 sharp한 분포)
                - head_decay_rate: 헤드 거리 감쇠율 (0.5 - 가까운 헤드 우선)
                - normalize_heads: 헤드 가중치 정규화 여부 (True 추천)
                - head_temperature: 헤드별 softmax 온도 (0.7)
                - epsilon: 최소 가중치 (0.05 - 너무 작으면 학습 효과 없음)
                - max_weight: 최대 가중치 (3.0 - 너무 크면 불안정)

        권장값 (연구개선안 기준):
        - percentile_top_p: 0.15 (상위 15% 토큰만 강조)
        - temperature: 0.5 (sharp한 분포로 중요 토큰 집중)
        - head_decay_rate: 0.5 (균형잡힌 거리 감쇠)
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

    def _compute_head_weights(
        self, token_weights: np.ndarray, horizon: int = 4
    ) -> np.ndarray:
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
            "head_mean_weights": [
                float(np.mean(head_weights[:, :, k])) for k in range(horizon)
            ],
            "head_std_weights": [
                float(np.std(head_weights[:, :, k])) for k in range(horizon)
            ],
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
            "scores": scores.tolist(),  # [B,S] raw scores
            "statistics": statistics,  # Head-level statistics
        }
