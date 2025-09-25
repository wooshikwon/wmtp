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

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from src.components.trainer.base_wmtp_trainer import BaseWmtpTrainer, compute_weighted_mtp_loss
from src.components.registry import trainer_registry

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

        # Critic 특화 하이퍼파라미터
        self.discount_lambda = self.config.get("discount_lambda", 0.95)  # TD error 할인율
        self.temperature = self.config.get("temperature", 0.7)  # Softmax 온도

        # Value Head는 setup()에서 초기화
        self.value_head: nn.Module | None = None

    def setup(self, ctx: dict[str, Any]) -> None:
        """Value Head를 초기화하고 필요시 체크포인트에서 로드.

        Stage 1에서 사전학습된 Value Head를 Stage 2에서 재사용합니다.
        """
        super().setup(ctx)

        # 모델 hidden size 가져오기 (7B 모델 기본값: 4096)
        hidden_size = ctx.get("hidden_size", 4096)

        # Value Head 초기화
        if self.value_head is None:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),  # 은닉상태 → 중간층
                nn.ReLU(),  # 비선형 활성화
                nn.Linear(hidden_size // 2, 1),  # 중간층 → 스칼라 가치
            )

        # Stage 1에서 학습한 Value Head 가중치 로드 (선택적)
        value_head_path = ctx.get("value_head_path")
        if value_head_path:
            try:
                import os
                if os.path.exists(value_head_path):
                    state = torch.load(value_head_path, map_location="cpu")
                    self.value_head.load_state_dict(state)
                    console.print(f"[green]✓ Loaded Value Head from {value_head_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Failed to load Value Head: {e}[/yellow]")
                # 치명적 오류 아님: 랜덤 초기화된 헤드로 계속 진행

        # Value Head를 모델과 같은 device로 이동
        if self.device:
            self.value_head = self.value_head.to(self.device)

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
                    delta_k = torch.full((B,), -10.0, device=values.device, dtype=values.dtype)
                    delta_list.append(delta_k)

            if delta_list:
                # Stack deltas for all heads: [B, H]
                delta_tensor = torch.stack(delta_list, dim=1)

                # Softmax with temperature
                weights_t = F.softmax(delta_tensor / self.temperature, dim=1)  # [B, H]

                # 유효하지 않은 헤드는 0으로 마스킹
                for k in range(H):
                    if t + k + 1 >= S:
                        weights_t[:, k] = 0.0

                # 재정규화 (유효한 헤드만)
                weights_sum = weights_t.sum(dim=1, keepdim=True).clamp(min=1e-8)
                weights_t = weights_t / weights_sum

                head_weights[:, t, :] = weights_t

        # Valid mask 적용
        head_weights = head_weights * valid_mask.unsqueeze(-1)

        return head_weights

    def compute_head_weights(self, logits: torch.Tensor, target_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Value Head를 사용한 직접 헤드 가중치 계산.

        Hidden states로부터 가치함수를 예측하고 TD error를 계산하여
        MTP 헤드별 가중치를 생성합니다.

        Args:
            logits: MTP 모델 출력 [B, S, H, V]
            target_ids: 타겟 토큰 ID [B, S]
            **kwargs: hidden_states 등 추가 정보

        Returns:
            head_weights: Critic 기반 가중치 [B, S, H]

        Raises:
            RuntimeError: Value Head가 초기화되지 않은 경우
            ValueError: hidden_states가 없는 경우
        """
        if self.value_head is None:
            raise RuntimeError(
                "Value Head not initialized. Call setup() first."
            )

        # Hidden states 추출
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError(
                "CriticWmtpTrainer requires 'hidden_states' from model outputs. "
                "Ensure the model returns hidden states."
            )

        B, S = target_ids.shape

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
            values = self.value_head(
                hidden_states.view(B_hs * S_hs, D)
            ).view(B_hs, S_hs).squeeze(-1)

            # Shape alignment if needed
            if values.shape[0] != B or values.shape[1] != S:
                values = values[:B, :S]

        # Valid mask 계산 (ignore_index=-100인 토큰 제외)
        valid_mask = (target_ids != -100).float()

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

        # 배치를 디바이스로 이동
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        target_ids: torch.Tensor = batch["labels"]  # [B, S]

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
            hidden_states = None
            try:
                if isinstance(outputs, dict) and "hidden_states" in outputs:
                    hs = outputs["hidden_states"]
                    hidden_states = hs[-1] if isinstance(hs, (list, tuple)) else hs
                elif hasattr(outputs, "hidden_states"):
                    hs = outputs.hidden_states
                    hidden_states = hs[-1] if isinstance(hs, (list, tuple)) else hs
            except Exception:
                pass

            if hidden_states is None or hidden_states.ndim != 3:
                raise RuntimeError(
                    "CriticWmtpTrainer requires valid hidden_states [B,S,D] from model outputs. "
                    "Ensure your model is configured to return hidden states."
                )

            # 🎯 Critic WMTP: 가치함수 델타 기반 동적 가중치 계산
            head_weights = self.compute_head_weights(
                logits, target_ids, hidden_states=hidden_states
            )

            # WMTP 손실 계산 (BaseWmtpTrainer의 공통 함수 사용)
            weighted_loss, valid_mask, ce_per_head = compute_weighted_mtp_loss(
                logits=logits,  # [B, S, H, V]
                target_ids=target_ids,  # [B, S]
                head_weights=head_weights,  # [B, S, H] - 동적 가중치
                horizon=self.horizon,
                ignore_index=-100,
            )

            # Lambda scaling
            lambda_w = float(self.loss_cfg.get("lambda", 0.3))
            loss = lambda_w * weighted_loss  # 최종 스칼라 손실

        # 역전파 및 최적화
        loss.backward()

        # 그래디언트 클리핑
        grad_clip = float(getattr(self.optimizer, "grad_clip", 1.0))
        if math.isfinite(grad_clip) and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        # MLflow 로깅 (선택적)
        if self.mlflow is not None:
            try:
                # 헤드별 CE 평균 (진단용)
                with torch.no_grad():
                    B, S, H, V = logits.shape
                    ce_head_means = []
                    for k in range(H):
                        shift = k + 1
                        valid_len = S - shift
                        if valid_len <= 0:
                            ce_head_means.append(torch.tensor(0.0, device=logits.device))
                            continue
                        logits_k = logits[:, :valid_len, k, :]
                        labels_k = target_ids[:, shift : shift + valid_len]
                        ce_k = F.cross_entropy(
                            logits_k.transpose(1, 2),
                            labels_k,
                            ignore_index=-100,
                            reduction="none",
                        )
                        ce_head_means.append(ce_k.mean())
                    ce_head_means = torch.stack(ce_head_means)

                    # 기본 메트릭
                    metrics = {
                        f"train/ce_head_{i}": float(x)
                        for i, x in enumerate(ce_head_means)
                    }
                    metrics.update({
                        "train/loss": float(loss.detach().item()),
                        "train/ce_mean": float(
                            (ce_per_head[valid_mask.unsqueeze(-1).expand(-1, -1, H)]).mean().item()
                        ) if valid_mask.any() else 0.0,
                    })

                    # 가중치 통계 (동적 가중치 분석용)
                    w_eff = head_weights[valid_mask.unsqueeze(-1).expand(-1, -1, H)]
                    if w_eff.numel() > 0:
                        weight_stats = {
                            "train/weight_mean": float(w_eff.mean().item()),
                            "train/weight_min": float(w_eff.min().item()),
                            "train/weight_max": float(w_eff.max().item()),
                            "train/weight_std": float(w_eff.std().item()),
                        }

                        # 가중치 분포 백분위수 (계획서 요구사항)
                        try:
                            weight_stats.update({
                                "train/weight_p25": float(torch.quantile(w_eff, 0.25).item()),
                                "train/weight_p75": float(torch.quantile(w_eff, 0.75).item()),
                                "train/weight_p95": float(torch.quantile(w_eff, 0.95).item()),
                            })
                        except Exception:
                            # 폴백 (이전 PyTorch 버전용)
                            sorted_w = torch.sort(w_eff)[0]
                            n = sorted_w.numel()
                            weight_stats.update({
                                "train/weight_p25": float(sorted_w[int(n * 0.25)].item()),
                                "train/weight_p75": float(sorted_w[int(n * 0.75)].item()),
                                "train/weight_p95": float(sorted_w[int(n * 0.95)].item()),
                            })

                        # 실패 감지 (NaN/극값 체크)
                        weight_stats.update({
                            "train/nan_weights": int((~torch.isfinite(head_weights)).sum().item()),
                            "train/extreme_weights": int((head_weights > 5.0).sum().item()),
                        })

                        metrics.update(weight_stats)

                    # Critic 특화 메트릭 (직접 계산된 값 기반)
                    if hasattr(self, "_last_deltas") and self._last_deltas is not None:
                        metrics["train/critic_delta_mean"] = float(self._last_deltas.mean().item())
                        metrics["train/critic_delta_std"] = float(self._last_deltas.std().item())
                        metrics["train/critic_algorithm"] = 1  # Critic 플래그

                    if hasattr(self, "_last_values") and self._last_values is not None:
                        metrics["train/value_mean"] = float(self._last_values.mean().item())
                        metrics["train/value_std"] = float(self._last_values.std().item())

                    # 유효 토큰 비율
                    total_tokens = float(valid_mask.numel())
                    valid_tokens = float(valid_mask.sum().item())
                    metrics["train/valid_token_ratio"] = (
                        valid_tokens / total_tokens if total_tokens > 0 else 0.0
                    )

                self.mlflow.log_metrics(metrics, step=self.global_step)
            except Exception:
                # 로깅 오류로 훈련 중단 방지
                pass

        # 실패 감지 (NaN/Inf 체크)
        if (
            not torch.isfinite(loss)
            or not torch.isfinite(ce_per_head).all()
            or not torch.isfinite(head_weights).all()
        ):
            if self.mlflow is not None:
                try:
                    self.mlflow.log_metrics(
                        {"train/failure": 1.0}, step=self.global_step
                    )
                except Exception:
                    pass
            raise RuntimeError(
                "Detected NaN/Inf in loss or inputs; aborting training step."
            )

        return {
            "loss": float(loss.detach().item()),
            "lr": float(getattr(self.optimizer, "_last_lr", 0.0)),
        }