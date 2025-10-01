"""
Baseline MTP Trainer - 가장 단순한 WMTP 알고리즘

균등 가중치를 사용하는 baseline 구현으로 WMTP의 기본 동작을 제공합니다.
모든 토큰과 모든 헤드에 동일한 가중치 1.0을 적용합니다.

특징:
- Scorer 없음: 외부 토큰 중요도 계산 불필요
- 균등 가중치: w_{t+k} = 1.0 (모든 k에 대해)
- 최고 성능: 복잡한 가중치 계산 없이 순수 MTP 성능 측정 가능
- 기준선 역할: 다른 WMTP 알고리즘의 성능 비교 기준

수학적 공식:
    L_WMTP = Σ(k=0 to H-1) 1.0 × CE_k
    = CE_0 + CE_1 + CE_2 + CE_3 (H=4인 경우)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from rich.console import Console

from src.components.registry import trainer_registry
from src.components.trainer.base_wmtp_trainer import (
    BaseWmtpTrainer,
    compute_weighted_mtp_loss,
)

console = Console()


@trainer_registry.register("baseline-mtp", category="trainer", version="2.0.0")
class BaselineMtpTrainer(BaseWmtpTrainer):
    """Baseline MTP 트레이너 - 균등 가중치 WMTP 알고리즘.

    연구 철학 "Not All Tokens Are What You Need"의 기준선 구현:
        모든 토큰에 동일한 중요도를 부여하여 표준 MTP와 유사하게 동작합니다.
        복잡한 가중치 계산 없이 WMTP 프레임워크의 기본 성능을 측정할 수 있습니다.

    🔬 핵심 동작:
        1. 모든 헤드에 균등 가중치 적용: w_{t+k} = 1.0
        2. MTP 손실 계산: L = Σ CE_k (일반적인 MTP와 동일)
        3. 체크포인트 관리 및 MLflow 로깅

    장점:
        - 가장 빠른 학습 속도 (가중치 계산 오버헤드 없음)
        - 안정적 성능 (복잡한 scorer 로직 없음)
        - 다른 알고리즘의 성능 비교 기준선

    사용 사례:
        - WMTP 시스템의 기본 성능 측정
        - 복잡한 알고리즘의 효과 검증을 위한 baseline
        - 빠른 실험 및 디버깅
    """

    def compute_head_weights(
        self,
        logits: torch.Tensor,
        target_labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """균등 헤드 가중치 계산 - 모든 헤드에 1.0 가중치.

        가장 단순한 가중치 계산으로 모든 MTP 헤드에 동일한 중요도를 부여합니다.

        Args:
            logits: MTP 모델 출력 [B, S, H, V]
            target_labels: 3D 타겟 라벨 [B, S, H] - MTPDataCollator 생성
            **kwargs: 사용되지 않음 (기준선 알고리즘)

        Returns:
            head_weights: 균등 가중치 [B, S, H] - 모든 값이 1.0
        """
        B, S, H, V = logits.shape
        return torch.ones((B, S, H), device=logits.device, dtype=logits.dtype)

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """MTP Baseline 훈련 스텝 - 균등 가중치로 간단한 WMTP 손실 계산.

        Args:
            batch: 훈련 배치 데이터 (input_ids, labels, attention_mask 등)

        Returns:
            메트릭 딕셔너리 (loss, lr 등)
        """
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        self.model.train()

        # 배치를 디바이스로 이동
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()
        }

        target_labels: torch.Tensor = batch[
            "labels"
        ]  # [B, S, H] - MTPDataCollator 생성

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
            # 모델 forward pass
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

            # gradient 활성화 (일부 모델이 detached tensor 반환하는 경우)
            if not logits.requires_grad:
                logits = logits.detach().requires_grad_(True)

            # 🎯 MTP Baseline: 균등 가중치로 단순한 손실 계산
            head_weights = self.compute_head_weights(logits, target_labels)

            # WMTP 손실 계산 (간소화된 3D 라벨 기반)
            weighted_loss, valid_mask, ce_per_head = compute_weighted_mtp_loss(
                logits=logits,  # [B, S, H, V]
                target_labels=target_labels,  # [B, S, H] - MTPDataCollator 생성
                head_weights=head_weights,  # [B, S, H] - 모두 1.0
                ignore_index=-100,
                config=self.config,  # MPS 경로 판단용 설정 전달
            )

            # Lambda scaling (설정에서 가져오기)
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
                            ce_head_means.append(
                                torch.tensor(0.0, device=logits.device)
                            )
                            continue
                        logits_k = logits[:, :valid_len, k, :]
                        labels_k = target_labels[:, shift : shift + valid_len]
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
                    metrics.update(
                        {
                            "train/loss": float(loss.detach().item()),
                            "train/ce_mean": float(
                                (
                                    ce_per_head[
                                        valid_mask.unsqueeze(-1).expand(-1, -1, H)
                                    ]
                                )
                                .mean()
                                .item()
                            )
                            if valid_mask.any()
                            else 0.0,
                        }
                    )

                    # Baseline 특화 메트릭
                    metrics.update(
                        {
                            "train/baseline_uniform_weight": 1.0,  # 항상 1.0
                            "train/baseline_algorithm": 1,  # Baseline 플래그
                        }
                    )

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
