"""
Rho-1 WMTP Trainer - Reference 모델 비교 기반 WMTP 알고리즘

Microsoft Rho-1 연구의 "Not All Tokens Are What You Need" 철학을 MTP에 적용한 핵심 알고리즘입니다.
Reference 모델과 Base 모델의 Cross-Entropy 차이로 토큰 중요도를 계산합니다.

특징:
- Reference CE 비교: |CE^ref - CE^base| 기반 토큰 중요도 측정
- 효율적 구현: 한 번의 forward pass로 reference CE 계산
- 정확한 정렬: MTP 헤드별로 적절한 time step과 매칭
- 최고 성능: 연구개선안에서 권장하는 가장 효과적인 방법

수학적 공식:
    CE^ref_t = CrossEntropy(ref_model(input[:t]), target[t])
    CE^base_{t+k} = CrossEntropy(base_model_head_k(input[:t]), target[t+k])
    excess_loss = |CE^ref_t - CE^base_{t+k}| (적절한 time step 정렬 후)
    w_{t+k} = softmax(excess_loss / temperature)_k
    L_WMTP = Σ(k=0 to H-1) w_{t+k} × CE_k
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from rich.console import Console

from src.components.trainer.base_wmtp_trainer import BaseWmtpTrainer, compute_weighted_mtp_loss
from src.components.registry import trainer_registry

console = Console()


@trainer_registry.register("rho1-wmtp", category="trainer", version="2.0.0")
class Rho1WmtpTrainer(BaseWmtpTrainer):
    """Rho-1 WMTP 트레이너 - Reference 모델 비교 기반 WMTP 알고리즘.

    연구 철학 "Not All Tokens Are What You Need"의 핵심 구현:
        Microsoft Rho-1 연구의 선택적 언어모델링 아이디어를 MTP에 적용하여,
        Reference 모델과 Base 모델이 모두 어려워하는 토큰을 중요 토큰으로 식별하고
        해당 토큰에 높은 가중치를 부여합니다.

    🔬 핵심 동작:
        1. Reference 모델로 각 위치의 next token CE 계산
        2. MTP 모델의 헤드별 CE와 정확한 time step 매칭
        3. Excess loss 계산: |CE^ref - CE^base|
        4. Softmax 가중치 변환 및 WMTP 손실 적용

    ⭐ 정렬 방식 (핵심):
        - Reference: t 시점에서 t+1 토큰 예측
        - MTP Head k: t 시점에서 t+k+1 토큰 예측
        - 정렬: head_k vs ref(t+k → t+k+1)

    장점:
        - 이론적 근거: Microsoft Rho-1 연구에서 입증된 효과
        - Pretrainer 불필요: 별도 학습 없이 바로 가중치 계산
        - 정확한 비교: 동일한 예측 태스크끼리 CE 비교
        - 높은 성능: 연구개선안에서 권장하는 최고 성능 알고리즘

    필수 요구사항:
        - ref_model: Reference 모델 (ctx에서 제공)
        - temperature: Softmax 온도 파라미터 (config)
    """

    def setup(self, ctx: dict[str, Any]) -> None:
        """Rho-1 트레이너 초기화 - Reference 모델 필수 확인.

        Args:
            ctx: 컨텍스트 딕셔너리 (ref_model 포함 필요)

        Raises:
            ValueError: Reference 모델이 제공되지 않은 경우
        """
        super().setup(ctx)

        self.ref_model: torch.nn.Module | None = ctx.get("ref_model")
        if self.ref_model is None:
            raise ValueError(
                "Rho1WmtpTrainer requires 'ref_model' in context. "
                "Please provide a reference model for CE comparison."
            )

        # Recipe 기반 설정 로드 (Factory에서 전달)
        self.rho1_cfg = self.config.get("rho1_config", {})
        
        # Dual mode 파라미터 로드 
        self.selection_mode = self.rho1_cfg.get("selection_mode", "weighted")
        self.skip_threshold_pct = float(self.rho1_cfg.get("skip_threshold_percentile", 0.3))
        
        # Weight softmax temperature (weighted mode에서 사용)
        # Backward compatibility: temperature → weight_temperature
        self.temperature = float(
            self.loss_cfg.get("weight_temperature") or
            self.loss_cfg.get("temperature", 0.7)
        )
        if self.temperature <= 0:
            raise ValueError(f"Weight temperature must be positive, got {self.temperature}")
            
        # Phase 1.2: CE Difference Threshold 파라미터 (노이즈 필터링)
        self.min_ce_diff = float(self.rho1_cfg.get("min_ce_diff", 0.01))
        if self.min_ce_diff < 0:
            raise ValueError(f"min_ce_diff must be non-negative, got {self.min_ce_diff}")

        console.print(f"[green]Rho-1 WMTP initialized:[/green]")
        console.print(f"  Mode: {self.selection_mode}")
        if self.selection_mode == "token_skip":
            console.print(f"  Skip threshold: {self.skip_threshold_pct:.1%} (bottom)")
        else:
            console.print(f"  Weight temperature: {self.temperature}")
        console.print(f"  Min CE diff threshold: {self.min_ce_diff}")

    def compute_reference_ce(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """효율적 Reference CE 계산 (한 번의 forward pass).

        Reference 모델로 각 위치에서 다음 토큰의 Cross-Entropy를 계산합니다.
        MTP 헤드와의 정확한 비교를 위해 위치별로 CE를 분리합니다.

        Args:
            input_ids: 입력 토큰 시퀀스 [B, S]
            target_ids: 타겟 토큰 시퀀스 [B, S]

        Returns:
            ref_ce_all: 위치별 Reference CE [B, S-1] - t위치에서 t+1토큰 예측 CE
        """
        with torch.no_grad():
            # Reference 모델 forward pass
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),  # 전체 시퀀스 사용
            )

            # logits 추출
            if isinstance(ref_outputs, dict) and "logits" in ref_outputs:
                ref_logits = ref_outputs["logits"]  # [B, S, V]
            else:
                ref_logits = ref_outputs

            if ref_logits.ndim != 3:
                raise ValueError(f"Reference logits should be 3D [B,S,V], got {ref_logits.shape}")

            # 각 위치에서 다음 토큰의 CE 계산
            # ref_logits[:, :-1] = t 위치의 logits (t+1 토큰 예측)
            # target_ids[:, 1:] = t+1 위치의 실제 토큰
            ref_ce_all = F.cross_entropy(
                ref_logits[:, :-1].transpose(1, 2),  # [B, V, S-1]
                target_ids[:, 1:],                   # [B, S-1]
                reduction='none'
            )  # [B, S-1] - 각 위치 t에서 t+1 토큰에 대한 CE

        return ref_ce_all

    def align_ref_ce_to_mtp(self, ref_ce_all: torch.Tensor, mtp_ce_heads: torch.Tensor) -> torch.Tensor:
        """Reference CE를 MTP 헤드와 정렬.

        Reference 모델의 위치별 CE를 MTP 헤드별 CE와 올바른 time step으로 매칭합니다.

        정렬 원리:
        - MTP Head k: t 시점에서 t+k+1 토큰 예측
        - Reference: t 시점에서 t+1 토큰 예측
        - 매칭: head_k[t] ↔ reference[t+k] (둘 다 t+k+1 토큰 예측)

        Args:
            ref_ce_all: Reference CE [B, S-1] - 위치 t에서 t+1 토큰 예측 CE
            mtp_ce_heads: MTP 헤드별 CE [B, S, H]

        Returns:
            aligned_ref_ce: MTP 헤드와 정렬된 Reference CE [B, S, H]
        """
        B, S, H = mtp_ce_heads.shape
        aligned_ref_ce = torch.zeros_like(mtp_ce_heads)

        # 각 헤드별로 적절한 reference CE 매칭
        for k in range(H):
            # Head k는 t+k+1 토큰을 예측
            # Reference에서 t+k 위치의 CE는 t+k+1 토큰에 대한 예측
            if k < ref_ce_all.size(1):  # reference CE 범위 내인지 확인
                valid_len = min(S - k - 1, ref_ce_all.size(1) - k)
                if valid_len > 0:
                    # 헤드 k의 처음 valid_len 위치에 reference[k:k+valid_len] 매칭
                    aligned_ref_ce[:, :valid_len, k] = ref_ce_all[:, k:k+valid_len]

        return aligned_ref_ce

    def compute_head_weights(self, logits: torch.Tensor, target_ids: torch.Tensor, ce_per_head: torch.Tensor, **kwargs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rho-1 방식: |CE^ref - CE^base| 기반 가중치 계산.

        Reference 모델과 Base 모델의 CE 차이를 계산하여 토큰 중요도를 측정하고,
        이를 MTP 헤드별 가중치로 변환합니다.

        Args:
            logits: MTP 모델 출력 [B, S, H, V] (사용되지 않음, ce_per_head 사용)
            target_ids: 타겟 토큰 ID [B, S]
            ce_per_head: MTP 헤드별 CE [B, S, H] - compute_weighted_mtp_loss에서 계산됨
            **kwargs: input_ids 등 추가 정보

        Returns:
            - Weighted mode: head_weights만 반환 [B, S, H]
            - Token skip mode: (head_weights, selection_mask) 튜플 반환

        Raises:
            ValueError: input_ids가 제공되지 않은 경우
        """
        # Reference CE 계산을 위한 input_ids 필요
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError(
                "Rho1WmtpTrainer requires 'input_ids' for reference CE calculation. "
                "Ensure input_ids are passed in kwargs."
            )

        # 1. Reference 모델로 CE 계산
        ref_ce_all = self.compute_reference_ce(input_ids, target_ids)

        # 2. MTP 헤드와 Reference CE 정렬
        aligned_ref_ce = self.align_ref_ce_to_mtp(ref_ce_all, ce_per_head)

        # 3. Excess loss 계산: |CE^ref - CE^base|
        # 큰 차이 = 두 모델 모두 어려워함 = 중요한 토큰
        excess_loss = torch.abs(ce_per_head - aligned_ref_ce)  # [B, S, H]
        
        # Phase 1.2: CE Difference Threshold 적용 (노이즈 필터링)
        excess_loss = self._apply_ce_threshold(excess_loss)

        # 4. Selection mode에 따라 분기
        if self.selection_mode == "token_skip":
            return self._compute_token_skip_weights(excess_loss)
        else:
            return self._compute_weighted_weights(excess_loss)

    def _compute_token_skip_weights(
        self, 
        excess_loss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rho-1 Original: 상위 토큰만 선택, 나머지 제외.
        
        Args:
            excess_loss: [B, S, H] - 각 토큰-헤드의 excess loss
            
        Returns:
            head_weights: [B, S, H] - 선택된 토큰은 1.0, 제외는 0.0
            selection_mask: [B, S, H] - 동일 (바이너리 마스크)
        """
        B, S, H = excess_loss.shape
        
        # 배치별로 threshold 계산
        flat_loss = excess_loss.view(B, -1)  # [B, S*H]
        
        # 하위 k% percentile 값 구하기
        k_threshold = torch.quantile(
            flat_loss, 
            self.skip_threshold_pct,  # 하위 30% 기본값
            dim=1, 
            keepdim=True
        ).view(B, 1, 1)  # [B, 1, 1]
        
        # 임계값 이상인 토큰만 선택 (binary mask)
        selection_mask = (excess_loss >= k_threshold).float()  # [B, S, H]
        
        # 선택된 토큰에만 균등 가중치 부여
        head_weights = selection_mask.clone()
        
        # 통계 로깅
        selected_ratio = selection_mask.mean()
        console.print(f"[cyan]Token Skip: {selected_ratio:.1%} tokens selected[/cyan]")
        
        return head_weights, selection_mask
    
    def _compute_weighted_weights(
        self, 
        excess_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        WMTP: 모든 토큰에 연속적 가중치 적용 (기존 방식).
        
        Args:
            excess_loss: [B, S, H] - 각 토큰-헤드의 excess loss
            
        Returns:
            weights: [B, S, H] - Softmax 가중치
        """
        # Softmax로 연속적 가중치 계산
        weights = F.softmax(excess_loss / self.temperature, dim=-1)  # [B, S, H]
        
        # 통계 로깅
        weight_std = weights.std()
        console.print(f"[cyan]Weighted: std={weight_std:.3f}[/cyan]")
        
        return weights  # selection_mask 없이 weights만 반환
    
    def _apply_ce_threshold(self, excess_loss: torch.Tensor) -> torch.Tensor:
        """
        Phase 1.2: CE Difference Threshold 적용 - 노이즈 필터링.
        
        너무 작은 CE 차이는 노이즈로 간주하고 0으로 처리하여
        가중치 계산에서 제외합니다.
        
        Args:
            excess_loss: [B, S, H] - 원본 excess loss 값
            
        Returns:
            filtered_excess_loss: [B, S, H] - threshold 적용된 excess loss
        """
        if self.min_ce_diff <= 0:
            return excess_loss  # threshold 비활성화 시 원본 값 반환
            
        # Threshold 적용: min_ce_diff 미만은 0으로 처리
        filtered_loss = torch.where(
            excess_loss >= self.min_ce_diff,
            excess_loss,
            torch.zeros_like(excess_loss)
        )
        
        # Edge case 처리: 모든 값이 threshold 미만인 경우
        B, S, H = filtered_loss.shape
        
        # 배치별로 처리
        for b in range(B):
            batch_loss = filtered_loss[b]  # [S, H]
            
            # 유효한 값이 하나라도 있는지 확인
            if torch.all(batch_loss == 0):
                # 모든 값이 0이면 uniform weight fallback
                console.print(
                    f"[yellow]⚠️ Batch {b}: All excess_loss < {self.min_ce_diff}, using uniform weights[/yellow]"
                )
                # 균등 가중치로 대체 (1/H 대신 1.0 사용 - softmax에서 정규화됨)
                filtered_loss[b] = torch.ones_like(batch_loss)
        
        return filtered_loss

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Rho-1 WMTP 훈련 스텝 - Reference CE 비교 기반 동적 가중치 WMTP 손실 계산.

        Args:
            batch: 훈련 배치 데이터 (input_ids, labels, attention_mask 등)

        Returns:
            메트릭 딕셔너리 (loss, lr, rho1 특화 메트릭 포함)
        """
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        self.model.train()

        # 배치를 디바이스로 이동
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        target_ids: torch.Tensor = batch["labels"]  # [B, S]
        input_ids: torch.Tensor = batch.get("input_ids")

        if input_ids is None:
            raise ValueError("Rho1WmtpTrainer requires 'input_ids' in batch")

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

            # gradient 활성화
            if not logits.requires_grad:
                logits = logits.detach().requires_grad_(True)

            # 🎯 단계 1: MTP 헤드별 CE 계산 (임시 균등 가중치로)
            B, S, H, V = logits.shape
            temp_weights = torch.ones((B, S, H), device=logits.device, dtype=logits.dtype)

            _, valid_mask, ce_per_head = compute_weighted_mtp_loss(
                logits=logits,
                target_ids=target_ids,
                head_weights=temp_weights,
                horizon=self.horizon,
                ignore_index=-100,
            )

            # 🎯 단계 2: Rho-1 가중치 계산 (Reference CE 비교)
            result = self.compute_head_weights(
                logits, target_ids, ce_per_head, input_ids=input_ids
            )
            
            # 반환값 타입에 따라 처리
            if isinstance(result, tuple):
                head_weights, selection_mask = result
            else:
                head_weights = result
                selection_mask = None  # Weighted mode

            # 🎯 단계 3: 최종 가중 WMTP 손실 계산
            weighted_loss, valid_mask, ce_per_head = compute_weighted_mtp_loss(
                logits=logits,  # [B, S, H, V]
                target_ids=target_ids,  # [B, S]
                head_weights=head_weights,  # [B, S, H] - Rho-1 가중치
                selection_mask=selection_mask,  # [B, S, H] - Token skip mask (새로 추가)
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

                    # 가중치 통계 (Rho-1 가중치 분석용)
                    w_eff = head_weights[valid_mask.unsqueeze(-1).expand(-1, -1, H)]
                    if w_eff.numel() > 0:
                        weight_stats = {
                            "train/weight_mean": float(w_eff.mean().item()),
                            "train/weight_min": float(w_eff.min().item()),
                            "train/weight_max": float(w_eff.max().item()),
                            "train/weight_std": float(w_eff.std().item()),
                        }

                        # 가중치 분포 백분위수
                        try:
                            weight_stats.update({
                                "train/weight_p25": float(torch.quantile(w_eff, 0.25).item()),
                                "train/weight_p75": float(torch.quantile(w_eff, 0.75).item()),
                                "train/weight_p95": float(torch.quantile(w_eff, 0.95).item()),
                            })
                        except Exception:
                            sorted_w = torch.sort(w_eff)[0]
                            n = sorted_w.numel()
                            weight_stats.update({
                                "train/weight_p25": float(sorted_w[int(n * 0.25)].item()),
                                "train/weight_p75": float(sorted_w[int(n * 0.75)].item()),
                                "train/weight_p95": float(sorted_w[int(n * 0.95)].item()),
                            })

                        weight_stats.update({
                            "train/nan_weights": int((~torch.isfinite(head_weights)).sum().item()),
                            "train/extreme_weights": int((head_weights > 5.0).sum().item()),
                        })

                        metrics.update(weight_stats)

                    # Rho-1 특화 메트릭 (excess loss 분석)
                    try:
                        # Reference CE 재계산 (로깅용)
                        ref_ce_all = self.compute_reference_ce(input_ids, target_ids)
                        aligned_ref_ce = self.align_ref_ce_to_mtp(ref_ce_all, ce_per_head)
                        excess_loss = torch.abs(ce_per_head - aligned_ref_ce)

                        excess_eff = excess_loss[valid_mask.unsqueeze(-1).expand(-1, -1, H)]
                        if excess_eff.numel() > 0:
                            # Excess loss 통계
                            metrics.update({
                                "train/rho1_excess_mean": float(excess_eff.mean().item()),
                                "train/rho1_excess_std": float(excess_eff.std().item()),
                                "train/rho1_excess_max": float(excess_eff.max().item()),
                            })

                            # 높은 excess loss 토큰 비율 (중요 토큰 비율)
                            threshold = excess_eff.mean() + excess_eff.std()
                            important_tokens = float((excess_eff > threshold).sum().item())
                            total_tokens = float(excess_eff.numel())
                            metrics["train/rho1_important_ratio"] = (
                                important_tokens / total_tokens if total_tokens > 0 else 0.0
                            )

                            metrics["train/rho1_algorithm"] = 1  # Rho-1 플래그
                            metrics["train/rho1_temperature"] = self.temperature
                    except Exception:
                        # Reference CE 계산 실패시 무시
                        pass

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