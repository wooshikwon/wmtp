"""
Critic-WMTP 알고리즘: 강화학습의 가치함수로 토큰 중요도를 계산합니다.

WMTP 연구에서 핵심 아이디어:
"더 큰 가치 증가 = 더 중요한 토큰"

이 모듈은 연구제안서의 Critic-Weighted 방식을 구현합니다:
- Stage 1: RM(Reward Model) 보상으로 가치헤드 사전 학습
- Stage 2: 학습된 가치헤드로 토큰별 중요도 δ_t = V(s_t) - V(s_{t-1}) 계산

수학적 원리 (연구제안서 공식):
1. 시퀀스 보상 R → GAE로 토큰별 가치 목표값 V̂_t 생성
2. 가치함수 학습: min Σ_t (V_θ(h_t) - V̂_t)²
3. Delta 계산: δ_t = V_t - λV_{t-1} (TD error 관점, 여기서 V_{-1} = 0)
4. 헤드별 가중치: w_{t+k} = softmax([δ_{t+1}, δ_{t+2}, δ_{t+3}, δ_{t+4}])

최종 출력: [batch, seq_len, horizon] 형태의 헤드별 가중치
각 헤드 k는 t+(k+1) 위치의 토큰을 예측하므로 해당 위치의 delta를 가중치로 사용
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.components.base import BaseComponent
from src.components.registry import scorer_registry


@scorer_registry.register("critic-delta-v1", category="scorer", version="1.0.0")
class CriticDeltaScorer(BaseComponent):
    """
    Critic-WMTP 방식: 강화학습 가치함수로 토큰 중요도를 계산하는 스코어러입니다.

    연구 맥락:
    WMTP의 세 가지 알고리즘 중 하나로, 강화학습의 가치함수 개념을 활용합니다.
    "미래에 더 큰 보상을 가져다주는 토큰 = 더 중요한 토큰"이라는 직관을 구현합니다.

    2단계 학습 프로세스:
    Stage 1: 가치헤드 사전학습
    - RM이 제공한 시퀀스 보상 R을 토큰별로 분배 (GAE 사용)
    - Value Head V_θ(h_t)가 토큰별 누적 보상을 예측하도록 학습
    - 손실함수: L = Σ_t (V_θ(h_t) - V̂_t)²

    Stage 2: Delta 기반 가중치 생성
    - 학습된 가치함수로 각 토큰의 가치 V_t 계산
    - Delta 계산: δ_t = V_t - λV_{t-1} (TD error 관점의 가치 증가량)
    - 헤드별 가중치: softmax([δ_{t+1}, δ_{t+2}, δ_{t+3}, δ_{t+4}])

    장점: 이론적으로 탄탄한 강화학습 기반
    단점: 가치함수 학습의 불안정성 (교수님 피드백)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Critic 스코어러를 초기화합니다.

        연구 맥락:
        이 설정들은 연구제안서의 Critic-WMTP 알고리즘 구현에 필수적입니다.
        각 파라미터는 강화학습 이론과 실제 구현 안정성을 고려하여 설계되었습니다.

        매개변수:
            config: 스코어러 설정 딕셔너리
                - target: Critic 목표 ("rm_sequence" - RM 시퀀스 보상 사용)
                - token_spread: 보상 분배 방법 ("gae" - GAE 방식 추천)
                - delta_mode: Delta 계산 모드 ("td" - Temporal Difference)
                - normalize: 정규화 방법 ("zscore" - Z점수 정규화)
                - temperature: Softmax 온도 (0.7 - 적당히 sharp한 분포)
                - gamma: GAE 할인계수 (0.99 - 미래 보상 중시)
                - gae_lambda: GAE 람다값 (0.95 - bias-variance 균형)
                - discount_lambda: TD error 계산 시 V_{t-1}에 적용할 할인율 (0.95)

        주의사항:
        - gamma는 너무 낮으면 단기적 보상만 고려, 너무 높으면 분산 증가
        - temperature는 낮을수록 sharp, 높을수록 uniform한 분포
        - discount_lambda는 시간 차이를 보정하여 공정한 delta 계산
        """
        super().__init__(config)
        self.target = self.config.get("target", "rm_sequence")
        self.token_spread = self.config.get("token_spread", "gae")
        self.delta_mode = self.config.get("delta_mode", "td")
        self.normalize = self.config.get("normalize", "zscore")
        self.temperature = self.config.get("temperature", 0.7)
        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.discount_lambda = self.config.get(
            "discount_lambda", 0.95
        )  # TD error discount factor

        # 가치헤드는 필요할 때 초기화됩니다 (Stage 1에서 학습, Stage 2에서 사용)
        self.value_head: nn.Module | None = None
        self.value_head_trained = False  # Stage 1 완료 여부 표시

    def setup(self, ctx: dict[str, Any]) -> None:
        """
        가치헤드를 초기화하고 필요시 체크포인트에서 로드합니다.

        연구 맥락:
        Critic-WMTP는 2단계 학습이므로 가치헤드의 초기화가 중요합니다.
        Stage 1에서 학습한 가치헤드를 Stage 2에서 재사용해야 합니다.
        """
        super().setup(ctx)

        # 모델 크기에 따른 가치헤드 초기화 (7B 모델 기본값: 4096)
        hidden_size = ctx.get("hidden_size", 4096)
        if self.value_head is None:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),  # 은닉상태 → 중간층 (2048)
                nn.ReLU(),  # 비선형 활성화
                nn.Linear(hidden_size // 2, 1),  # 중간층 → 스칼라 가치 예측
            )

        # Stage 1에서 학습한 가치헤드 가중치 로드 (선택적)
        value_head_path = ctx.get("value_head_path")
        if value_head_path:
            try:
                state = torch.load(value_head_path, map_location="cpu")
                self.value_head.load_state_dict(state)
                self.value_head_trained = True  # Stage 1 완료됨
            except Exception:
                # 치명적 오류 아님: 랜덤 초기화된 헤드로 계속 진행
                self.value_head_trained = False

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

    def _compute_head_weights(self, values: np.ndarray, horizon: int = 4) -> np.ndarray:
        """
        연구제안서 정확 구현: 각 위치에서 헤드별 가중치 계산

        Position t에서 4개 헤드에 대응하는 가중치 (TD error 적용):
        - Head 0 (예측 t+1): δ_{t+1} = V_{t+1} - λV_t
        - Head 1 (예측 t+2): δ_{t+2} = V_{t+2} - λV_{t+1}
        - Head 2 (예측 t+3): δ_{t+3} = V_{t+3} - λV_{t+2}
        - Head 3 (예측 t+4): δ_{t+4} = V_{t+4} - λV_{t+3}

        그 다음 softmax([δ_{t+1}, δ_{t+2}, δ_{t+3}, δ_{t+4}])를 적용

        Args:
            values: [S] 형태의 value function 예측값
            horizon: MTP 헤드 수 (기본값 4)

        Returns:
            head_weights: [S, H] 형태의 헤드별 가중치 매트릭스
        """
        seq_len = len(values)
        head_weights = np.zeros((seq_len, horizon), dtype=np.float32)

        # 각 위치 t에서 헤드별 delta 계산
        for t in range(seq_len):
            deltas_t = []

            for k in range(horizon):
                future_pos = t + k + 1  # k번째 헤드는 t+(k+1) 예측

                if future_pos < seq_len:
                    # TD error: δ_{t+k+1} = V_{t+k+1} - λV_{t+k}
                    if t + k >= 0:
                        delta_k = (
                            values[future_pos] - self.discount_lambda * values[t + k]
                        )
                    else:
                        # 경계 처리: V_{-1} = 0으로 가정
                        delta_k = values[future_pos] - 0.0
                else:
                    # 시퀀스 끝을 넘어가는 경우 작은 값 사용
                    delta_k = -1.0  # 낮은 중요도

                deltas_t.append(delta_k)

            # 헤드별 softmax with temperature 적용
            deltas_array = np.array(deltas_t, dtype=np.float32)

            # Z-score normalization (선택적)
            if self.normalize == "zscore" and len(deltas_array) > 1:
                mean_delta = np.mean(deltas_array)
                std_delta = np.std(deltas_array) + 1e-8
                deltas_normalized = (deltas_array - mean_delta) / std_delta
            else:
                deltas_normalized = deltas_array

            # Softmax with temperature
            exp_values = np.exp(deltas_normalized / self.temperature)
            head_weights_t = exp_values / (np.sum(exp_values) + 1e-8)

            # 유효한 헤드만 남기고 나머지는 0
            for k in range(horizon):
                if t + k + 1 >= seq_len:
                    head_weights_t[k] = 0.0

            # 재정규화 (유효한 헤드들만)
            total_valid = np.sum(head_weights_t) + 1e-8
            head_weights_t = head_weights_t / total_valid

            head_weights[t] = head_weights_t

        return head_weights

    def compute_deltas(self, values: np.ndarray) -> np.ndarray:
        """
        Compute temporal difference deltas.

        Following TD error approach: δ_t = V_t - λV_{t-1} where V_{-1} = 0

        Args:
            values: Value function predictions

        Returns:
            Delta values for each token
        """
        if self.delta_mode == "td":
            # Temporal difference with discount: δ_t = V_t - λV_{t-1}
            # For first token: δ_0 = V_0 - 0 = V_0
            prev_values = np.concatenate(([0], values[:-1]))  # V_{-1} = 0
            deltas = values - self.discount_lambda * prev_values
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

        # TODO: 토큰별 softmax 삭제됨 - 헤드별 가중치 생성으로 교체 예정
        # 임시 uniform weights (후에 _compute_head_weights()로 교체)
        weights = np.ones_like(normalized)

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
                - weights: Head-level importance weights [batch, seq_len, horizon] ← 새로운!
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

                # Step 4: Compute deltas δ_t = V_t - λV_{t-1} (TD error)
                deltas = self.compute_deltas(values)

                # Step 5: 새로운 헤드별 가중치 계산 (연구제안서 구현)
                head_weights = self._compute_head_weights(
                    values, horizon=self.config.get("horizon", 4)
                )  # [S, H] 형태

                all_weights.append(head_weights)
                all_deltas.append(deltas)
                all_values.append(values)

            # Stack results - 이제 [B, S, H] 형태!
            weights = np.array(all_weights)  # [B, S, H]
            deltas = np.array(all_deltas)  # [B, S]
            values = np.array(all_values)  # [B, S]

        else:
            # Real implementation with model hidden states
            # Use the value head to predict values and compute deltas/weights
            if self.value_head is None:
                # Initialize a default head assuming last dim is hidden size
                hidden_size = int(hidden_states.shape[-1])
                self.value_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1),
                )

            hs = hidden_states
            if isinstance(hs, torch.Tensor):
                hs_tensor = hs
            else:
                hs_tensor = torch.tensor(hs, dtype=torch.float32)

            B, S, H = hs_tensor.shape
            # Ensure value_head on same device/dtype
            try:
                self.value_head = self.value_head.to(
                    device=hs_tensor.device, dtype=hs_tensor.dtype
                )
            except Exception:
                self.value_head = self.value_head.to(device=hs_tensor.device)
            with torch.set_grad_enabled(False):
                v_pred = self.value_head(hs_tensor.view(B * S, H)).view(B, S)

            values = v_pred.detach().cpu().numpy()

            # Compute deltas and head weights per sequence
            all_deltas = []
            all_weights = []
            horizon = self.config.get("horizon", 4)

            for b in range(B):
                seq_len = seq_lengths[b] if b < len(seq_lengths) else S
                vals = values[b, :seq_len]
                deltas = self.compute_deltas(vals)
                # 새로운 헤드별 가중치 계산 (연구제안서 구현)
                head_weights = self._compute_head_weights(
                    vals, horizon=horizon
                )  # [seq_len, H]
                all_deltas.append(deltas)
                all_weights.append(head_weights)

            # Pad back to [B, S] for deltas, [B, S, H] for weights
            def pad_to_len(arr, L):
                if len(arr) >= L:
                    return arr[:L]
                pad_shape = (
                    (L - len(arr),) + arr.shape[1:] if arr.ndim > 1 else (L - len(arr),)
                )
                pad = np.ones(pad_shape, dtype=arr.dtype)
                return np.concatenate([arr, pad], axis=0)

            deltas = np.stack([pad_to_len(d, S) for d in all_deltas], axis=0)  # [B, S]
            weights = np.stack(
                [pad_to_len(w, S) for w in all_weights], axis=0
            )  # [B, S, H]

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

        # Determine target device and dtype from context
        target_device = ctx.get("device", "cpu")
        target_dtype = ctx.get("dtype", torch.float32)

        # Convert to torch.Tensor with proper device/dtype
        weights_tensor = torch.tensor(weights, device=target_device, dtype=target_dtype)

        return {
            "weights": weights_tensor,
            "deltas": deltas.tolist(),
            "values": values.tolist(),
            "statistics": statistics,
        }
