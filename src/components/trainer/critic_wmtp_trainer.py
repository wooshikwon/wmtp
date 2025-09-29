"""
Critic WMTP Trainer - 강화학습 가치함수 기반 WMTP 알고리즘 (Scorer 통합 버전)

가치함수의 델타(δ_t = V_t - V_{t-1})를 사용하여 토큰 중요도를 계산합니다.
"미래에 더 큰 보상을 가져다주는 토큰 = 더 중요한 토큰"이라는 직관을 구현합니다.

[리팩토링 v2.1.0]
- CriticDeltaScorer 완전 통합: Value Head 및 Delta 계산 로직을 직접 구현
- 성능 향상: scorer.run() 호출 오버헤드 제거
- 코드 명확성: Critic 로직이 한 파일에 집중

특징:
- Value Head 직접 관리: nn.Sequential(Linear, ReLU, Linear) 구조
- TD Error 계산: δ_t = V_t - λV_{t-1} 직접 구현
- 헤드별 가중치: softmax([δ_{t+1}, δ_{t+2}, δ_{t+3}, δ_{t+4}]) 직접 계산
- Stage 1 체크포인트 지원: 사전학습된 Value Head 로드 가능

수학적 공식:
    δ_t = V_t - λV_{t-1} (TD error 관점의 가치 증가량)
    w_{t+k} = softmax([δ_{t+1}, δ_{t+2}, δ_{t+3}, δ_{t+4}])_k
    L_WMTP = Σ(k=0 to H-1) w_{t+k} × CE_k
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from src.components.registry import trainer_registry
from src.components.trainer.base_wmtp_trainer import (
    BaseWmtpTrainer,
    compute_weighted_mtp_loss,
)
from src.utils.reward_utils import compute_sequence_rewards

console = Console()


@trainer_registry.register("critic-wmtp", category="trainer", version="2.1.0")
class CriticWmtpTrainer(BaseWmtpTrainer):
    """Critic WMTP 트레이너 - 가치함수 델타 기반 WMTP 알고리즘.

    연구 철학 "Not All Tokens Are What You Need"의 강화학습 구현:
        강화학습의 가치함수를 활용하여 각 토큰의 미래 보상 기여도를 측정하고,
        이를 기반으로 동적 가중치를 계산하여 WMTP 학습을 수행합니다.

    🔬 핵심 동작:
        1. CriticDeltaScorer를 통해 가치함수 델타 계산: δ_t = V_t - λV_{t-1}
        2. 델타를 헤드별 가중치로 변환: softmax([δ_{t+1}, δ_{t+2}, δ_{t+3}, δ_{t+4}])
        3. 가중된 MTP 손실 계산: L = Σ w_{t+k} × CE_k

    2단계 학습 프로세스:
        Stage 1: 가치헤드 사전학습 (CriticStage1Pretrainer 사용)
        - RM 보상으로부터 토큰별 가치 목표값 생성
        - Value Head가 토큰별 누적 보상을 정확히 예측하도록 학습

        Stage 2: Critic-WMTP 학습 (이 클래스)
        - 사전학습된 가치헤드로 토큰별 중요도 계산
        - 동적 가중치를 적용한 WMTP 손실로 모델 훈련

    장점:
        - 동적 적응: 각 시퀀스와 위치에 맞는 최적 가중치 계산
        - 높은 표현력: 복잡한 토큰 패턴의 중요도 학습 가능
        - 이론적 근거: 강화학습의 가치함수 이론에 기반

    사용 사례:
        - 복잡한 추론이 필요한 태스크 (수학, 코딩 등)
        - 토큰 간 장기 의존성이 중요한 시퀀스
        - 높은 성능이 필요한 프로덕션 환경
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Critic WMTP 트레이너 초기화.

        Value Head와 관련 하이퍼파라미터를 직접 관리합니다.
        """
        super().__init__(config)

        # Recipe 기반 설정 로드 (Factory에서 전달)
        self.critic_cfg = self.config.get("critic_config", {})

        # Phase 2.1: discount_lambda 파라미터 (Recipe에서)
        self.discount_lambda = float(
            self.critic_cfg.get("discount_lambda", 0.95)
        )  # TD error 할인율
        # self.temperature는 setup()에서 설정

        # Value Head는 setup()에서 초기화
        self.value_head: nn.Module | None = None
        self.rm_model: Any = None  # Reward Model 저장

    def setup(self, ctx: dict[str, Any]) -> None:
        """Value Head를 초기화하고 필요시 체크포인트에서 로드.

        Stage 1에서 사전학습된 Value Head를 Stage 2에서도 계속 학습합니다.
        """
        super().setup(ctx)

        # Weight temperature 파라미터 설정 (Phase 1 통합: recipe.loss.weight_temperature에서)
        # Backward compatibility: temperature → weight_temperature
        self.temperature = float(
            self.loss_cfg.get("weight_temperature")
            or self.loss_cfg.get("temperature", 0.7)
        )

        # RM model 저장 (Stage 2에서도 사용)
        self.rm_model = ctx.get("rm_model")  # Value loss 계산용

        # 모델 hidden size 가져오기 (모델에서 직접 추출)
        hidden_size = None
        if hasattr(self.model, "config"):
            # HuggingFace 스타일 모델
            hidden_size = getattr(
                self.model.config,
                "hidden_size",
                getattr(self.model.config, "n_embd", None),
            )

        if hidden_size is None:
            # ctx에서 시도
            hidden_size = ctx.get("hidden_size")

        if hidden_size is None:
            raise ValueError(
                f"Failed to extract hidden_size from model. "
                f"Model config attributes: {dir(self.model.config) if hasattr(self.model, 'config') else 'No config'}"
            )

        # Value Head 초기화
        if self.value_head is None:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),  # 은닉상태 → 중간층
                nn.ReLU(),  # 비선형 활성화
                nn.Linear(hidden_size // 2, 1),  # 중간층 → 스칼라 가치
            )

        # Stage 1에서 학습한 Value Head 가중치 로드
        # Pipeline이 Stage 1 결과를 value_head_path로 전달
        value_head_path = ctx.get("value_head_path")

        if value_head_path:
            try:
                # GPU 환경 일관성: 현재 device에 맞게 로드
                map_location = self.device if self.device else "cpu"
                state = torch.load(value_head_path, map_location=map_location)
                self.value_head.load_state_dict(state)
                console.print(
                    f"[green]✓ Loaded Stage 1 Value Head from {value_head_path} to {map_location}[/green]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]⚠ Failed to load Stage 1 Value Head: {e}[/yellow]"
                )
                console.print("[yellow]  Using random initialization instead[/yellow]")
        else:
            console.print(
                "[yellow]ℹ No Stage 1 Value Head provided, using random initialization[/yellow]"
            )

        # Value Head를 모델과 같은 device로 이동
        if self.device:
            self.value_head = self.value_head.to(self.device)

        # Value Head를 optimizer에 포함 (Stage 2 continuous learning)
        if self.optimizer is not None:
            # Critic 설정 가져오기 (value_lr 등)
            critic_config = (
                ctx.get("recipe", {}).critic
                if hasattr(ctx.get("recipe", {}), "critic")
                else {}
            )
            value_lr = (
                float(critic_config.get("value_lr", 5e-5))
                if isinstance(critic_config, dict)
                else 5e-5
            )

            # Value Head parameters를 별도 param group으로 추가
            self.optimizer.add_param_group(
                {
                    "params": self.value_head.parameters(),
                    "lr": value_lr,  # Higher LR for value head
                }
            )
            console.print(
                f"[green]✓ Value Head added to optimizer with lr={value_lr}[/green]"
            )

    def _compute_deltas(self, values: torch.Tensor) -> torch.Tensor:
        """TD error 계산: δ_t = V_t - λV_{t-1}.

        Args:
            values: [B, S] 형태의 value predictions

        Returns:
            deltas: [B, S] 형태의 TD errors
        """
        B, S = values.shape

        # V_{-1} = 0으로 가정하여 이전 값 준비
        zeros = torch.zeros((B, 1), device=values.device, dtype=values.dtype)
        prev_values = torch.cat([zeros, values[:, :-1]], dim=1)  # [B, S]

        # TD error with discount
        deltas = values - self.discount_lambda * prev_values
        return deltas

    def _compute_head_weights_from_values(
        self, values: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """가치함수로부터 헤드별 가중치 계산.

        Position t에서 4개 헤드에 대응하는 가중치:
        - Head 0 (예측 t+1): δ_{t+1} = V_{t+1} - λV_t
        - Head 1 (예측 t+2): δ_{t+2} = V_{t+2} - λV_{t+1}
        - Head 2 (예측 t+3): δ_{t+3} = V_{t+3} - λV_{t+2}
        - Head 3 (예측 t+4): δ_{t+4} = V_{t+4} - λV_{t+3}

        Args:
            values: [B, S] 형태의 value predictions
            valid_mask: [B, S] 형태의 유효 토큰 마스크

        Returns:
            head_weights: [B, S, H] 형태의 헤드별 가중치
        """
        B, S = values.shape
        H = self.horizon

        # Delta 계산
        deltas = self._compute_deltas(values)  # [B, S]

        # 헤드별 가중치 계산
        head_weights = torch.zeros((B, S, H), device=values.device, dtype=values.dtype)

        for t in range(S):
            delta_list = []
            valid_heads = []

            for k in range(H):
                future_pos = t + k + 1  # k번째 헤드는 t+(k+1) 예측

                if future_pos < S:
                    # 유효한 위치의 delta 사용
                    delta_k = deltas[:, future_pos]  # [B]
                    delta_list.append(delta_k)
                    valid_heads.append(k)
                else:
                    # 시퀀스 끝을 넘어가는 경우 매우 작은 값
                    delta_k = torch.full(
                        (B,), -10.0, device=values.device, dtype=values.dtype
                    )
                    delta_list.append(delta_k)

            if delta_list:
                # Stack deltas for all heads: [B, H]
                delta_tensor = torch.stack(delta_list, dim=1)

                # 유효하지 않은 헤드에 대한 마스킹 (Softmax 전!)
                # 시퀀스 경계를 넘는 예측은 매우 작은 값으로 설정하여
                # Softmax 후 자연스럽게 0에 가까워지도록 함
                for k in range(H):
                    if t + k + 1 >= S:
                        delta_tensor[:, k] = -1e10

                # Softmax with temperature (이제 gradient-safe)
                weights_t = F.softmax(delta_tensor / self.temperature, dim=1)  # [B, H]

                # 수치 안정성을 위한 클리핑만 적용 (inplace 수정 없음)
                weights_t = torch.clamp(weights_t, min=1e-8, max=1.0)

                head_weights[:, t, :] = weights_t

        # Valid mask 적용
        head_weights = head_weights * valid_mask.unsqueeze(-1)

        return head_weights

    @torch.no_grad()
    def _compute_sequence_rewards(
        self,
        rm_model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Stage 1과 동일한 reward 계산 (공통 유틸리티 사용)"""

        if rm_model is None:
            # Fallback: use negative CE as pseudo reward
            return self._compute_pseudo_rewards(input_ids, attention_mask)

        # 공통 유틸리티 함수 사용
        return compute_sequence_rewards(rm_model, input_ids, attention_mask, amp_dtype)

    @torch.no_grad()
    def _compute_pseudo_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pseudo rewards using negative CE from base model"""

        # Use common utility with base model as fallback RM
        return compute_sequence_rewards(
            self.model, input_ids, attention_mask, amp_dtype=self._amp_dtype
        )

    def _compute_gae_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> torch.Tensor:
        """토큰별 value target 계산 (Stage 1과 동일)"""
        B, S = values.shape
        returns = torch.zeros_like(values)

        for b in range(B):
            gae = 0
            for t in reversed(range(S)):
                next_val = 0.0 if t == S - 1 else values[b, t + 1]

                # TD error
                delta = rewards[b, t] + gamma * next_val - values[b, t]

                # GAE
                gae = delta + gamma * gae_lambda * gae
                returns[b, t] = values[b, t] + gae

        return returns

    def compute_head_weights(
        self,
        logits: torch.Tensor,  # noqa: ARG002
        target_labels: torch.Tensor,
        **kwargs  # noqa: ARG002
    ) -> torch.Tensor:
        """Value Head를 사용한 직접 헤드 가중치 계산.

        Hidden states로부터 가치함수를 예측하고 TD error를 계산하여
        MTP 헤드별 가중치를 생성합니다.

        Args:
            logits: MTP 모델 출력 [B, S, H, V]
            target_labels: 3D 타겟 라벨 [B, S, H] - MTPDataCollator 생성
            **kwargs: hidden_states 등 추가 정보

        Returns:
            head_weights: Critic 기반 가중치 [B, S, H]

        Raises:
            RuntimeError: Value Head가 초기화되지 않은 경우
            ValueError: hidden_states가 없는 경우
        """
        if self.value_head is None:
            raise RuntimeError("Value Head not initialized. Call setup() first.")

        # Hidden states 추출
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError(
                "CriticWmtpTrainer requires 'hidden_states' from model outputs. "
                "Ensure the model returns hidden states."
            )

        B, S, H = target_labels.shape

        # Hidden states shape 검증
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden_states shape [B, S, D], got {hidden_states.shape}"
            )

        # Value Head를 같은 device로 이동
        if self.value_head.training != self.model.training:
            self.value_head.train(self.model.training)

        # Value prediction
        with torch.set_grad_enabled(self.model.training):
            # [B, S, D] -> [B*S, D] -> [B*S, 1] -> [B, S]
            B_hs, S_hs, D = hidden_states.shape
            values = (
                self.value_head(hidden_states.view(B_hs * S_hs, D))
                .view(B_hs, S_hs)
                .squeeze(-1)
            )

            # Shape alignment if needed
            if values.shape[0] != B or values.shape[1] != S:
                values = values[:B, :S]

        # Valid mask 계산 (ignore_index=-100인 토큰 제외)
        # 3D 라벨에서 2D 마스크 생성 (모든 헤드가 동일한 유효성 가정)
        valid_mask = (target_labels[:, :, 0] != -100).float()  # [B, S]

        # 헤드별 가중치 계산
        head_weights = self._compute_head_weights_from_values(values, valid_mask)

        # 메트릭을 위해 deltas 저장
        with torch.no_grad():
            deltas = self._compute_deltas(values)
            self._last_deltas = deltas
            self._last_values = values

        return head_weights

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Critic WMTP 훈련 스텝 - 가치함수 델타 기반 동적 가중치 WMTP 손실 계산.

        Args:
            batch: 훈련 배치 데이터 (input_ids, labels, attention_mask 등)

        Returns:
            메트릭 딕셔너리 (loss, lr, critic 특화 메트릭 포함)
        """
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        self.model.train()
        self.value_head.train()  # Value Head도 학습 모드로

        # 배치를 디바이스로 이동
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()
        }

        input_ids = batch["input_ids"]
        target_labels: torch.Tensor = batch[
            "labels"
        ]  # [B, S, H] - MTPDataCollator 생성
        attention_mask = batch.get("attention_mask")

        # autocast 디바이스 타입 결정
        if torch.cuda.is_available():
            autocast_device = "cuda"
        elif torch.backends.mps.is_available() and str(self.device).startswith("mps"):
            autocast_device = "cpu"  # MPS는 아직 autocast 미지원
        else:
            autocast_device = "cpu"

        with torch.autocast(
            device_type=autocast_device,
            dtype=self._amp_dtype,
        ):
            # 모델 forward pass (hidden_states 포함 반환 필요)
            outputs: dict[str, Any] | torch.Tensor = self.model(**batch)

            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]  # [B, S, H, V] 예상
            else:
                logits = outputs  # tensor라고 가정

            # Shape 검증
            if logits.ndim != 4:
                raise ValueError(
                    f"Expected logits shape [B,S,H,V], got {tuple(logits.shape)}"
                )

            # gradient 활성화
            if not logits.requires_grad:
                logits = logits.detach().requires_grad_(True)

            # hidden_states 추출 (CriticScorer에 필요)
            from src.utils.model_utils import extract_hidden_states

            try:
                hidden_states = extract_hidden_states(outputs)
            except ValueError as e:
                raise RuntimeError(
                    f"CriticWmtpTrainer requires valid hidden_states [B,S,D] from model outputs. "
                    f"Error: {e}. Ensure your model is configured to return hidden states."
                ) from e

            # 🎯 Critic WMTP: 가치함수 델타 기반 동적 가중치 계산
            head_weights = self.compute_head_weights(
                logits, target_labels, hidden_states=hidden_states
            )

            # Value prediction (gradient enabled for training)
            B, S, D = hidden_states.shape
            values = (
                self.value_head(hidden_states.view(B * S, D)).view(B, S).squeeze(-1)
            )  # [B, S]

            # WMTP 손실 계산 (간소화된 3D 라벨 기반)
            weighted_loss, valid_mask, ce_per_head = compute_weighted_mtp_loss(
                logits=logits,  # [B, S, H, V]
                target_labels=target_labels,  # [B, S, H] - MTPDataCollator 생성
                head_weights=head_weights,  # [B, S, H] - 동적 가중치
                ignore_index=-100,
                config=self.config,  # MPS 경로 판단용 설정 전달
            )

            # Value Loss 계산 (auxiliary loss for continuous learning)
            value_loss = torch.tensor(0.0, device=self.device)
            critic_config = self.config.get("critic", {})
            if self.rm_model is not None or critic_config.get(
                "use_pseudo_rewards", True
            ):
                # Compute rewards
                rewards = self._compute_sequence_rewards(
                    self.rm_model, input_ids, attention_mask, amp_dtype=self._amp_dtype
                )

                # Spread rewards to tokens (uniform for simplicity)
                B, S = values.shape
                token_rewards = torch.zeros_like(values)
                for b in range(B):
                    seq_reward = rewards[b]
                    token_rewards[b, :] = seq_reward / S  # Uniform distribution

                # Compute returns using GAE
                with torch.no_grad():
                    returns = self._compute_gae_returns(
                        token_rewards,
                        values.detach(),
                        gamma=critic_config.get("gamma", 0.99),
                        gae_lambda=critic_config.get("gae_lambda", 0.95),
                    )

                # MSE loss for value prediction
                value_loss = F.mse_loss(
                    values[valid_mask.bool()], returns[valid_mask.bool()]
                )

            # Phase 2.2: Clean loss structure - main loss fixed at 1.0
            auxiliary_coef = float(critic_config.get("auxiliary_loss_coef", 0.1))

            # Main WMTP loss (1.0) + auxiliary value loss
            loss = weighted_loss + auxiliary_coef * value_loss

        # 역전파 및 최적화
        loss.backward()

        # 연구 취지: Critic TD의 자연스러운 신호를 보존 (no gradient clipping)

        # 대신 gradient 안정성 체크 (연구 취지 유지하면서 안정성 확보)
        total_norm = 0.0
        for p in list(self.model.parameters()) + list(self.value_head.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        # 극단적인 경우만 체크 (자연스러운 학습은 허용)
        if total_norm > 100.0:  # Very high threshold for research integrity
            console.print(
                f"[yellow]⚠ Large gradient norm detected: {total_norm:.2f} at step {self.global_step}[/yellow]"
            )

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        # MLflow 로깅 (100 step마다 + 핵심 메트릭만)
        if self.mlflow is not None and self.global_step % 100 == 0:
            try:
                # 핵심 메트릭만 로깅
                metrics = {
                    "train/loss": float(loss.detach().item()),
                    "train/wmtp_loss": float(weighted_loss.item()),
                    "train/value_loss": float(value_loss.item()),
                }

                # Critic 특화 메트릭 (중요한 것만)
                if hasattr(self, "_last_values") and self._last_values is not None:
                    metrics["train/value_mean"] = float(self._last_values.mean().item())

                # 가중치 통계 (평균만)
                with torch.no_grad():
                    w_eff = head_weights[
                        valid_mask.unsqueeze(-1).expand(-1, -1, self.horizon)
                    ]
                    if w_eff.numel() > 0:
                        metrics["train/weight_mean"] = float(w_eff.mean().item())

                self.mlflow.log_metrics(metrics, step=self.global_step)
            except Exception:
                pass

        # 실패 감지 (NaN/Inf 체크)
        if not torch.isfinite(loss) or not torch.isfinite(head_weights).all():
            raise RuntimeError(
                f"NaN/Inf detected at step {self.global_step}; aborting training."
            )

        return {
            "loss": float(loss.detach().item()),
            "wmtp_loss": float(weighted_loss.item()),
            "value_loss": float(value_loss.item()),
            "lr": float(getattr(self.optimizer, "_last_lr", 0.0)),
        }
