"""WMTP 핵심 구현체 - MTP Weighted Cross-Entropy Trainer.

연구 철학의 실현: "Not All Tokens Are What You Need"
=================================================

이 트레이너는 WMTP 연구의 핵심 아이디어를 실제로 구현합니다:
기존 MTP의 균등한 토큰 가중치 대신, 토큰별 중요도를 동적으로 계산하여
가중치를 적용한 새로운 손실 함수를 사용합니다.

🔬 WMTP 손실 공식:
    L_WMTP = Σ(k=1 to H) w_{t+k} × CE_k

    여기서:
    - w_{t+k}: k번째 헤드의 토큰별 중요도 가중치
    - CE_k: k번째 예측 헤드의 Cross-Entropy 손실
    - H: 예측 헤드 수 (일반적으로 4개: t+1, t+2, t+3, t+4)

알고리즘별 가중치 계산 방식:
    - mtp-baseline: w_{t+k} = 1.0 (균등 가중치, Scorer=None)
    - critic-wmtp: w_{t+k} = f(δ_t) where δ_t = V_t - V_{t-1}
    - rho1-wmtp: w_{t+k} = |CE^ref_t - CE^base_t|

기술적 특징:
    - Mixed Precision 지원: BF16/FP16/FP32 자동 선택
    - FSDP (Fully Sharded Data Parallel) 분산 훈련
    - 그래디언트 클리핑으로 안정성 보장
    - MLflow 자동 로깅으로 실험 추적
    - 동적 메모리 최적화 및 배치 처리
"""

from __future__ import annotations  # Python 3.10+ 타입 힌트 호환성

import math  # 수학 연산 (가중치 정규화 등)
from typing import Any  # 범용 타입 힌트

import numpy as np  # 수치 연산
import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 함수형 API (cross_entropy 등)
from rich.console import Console  # 컬러풀한 콘솔 출력

from src.components.base import BaseComponent  # WMTP 컴포넌트 베이스 클래스
from src.components.registry import trainer_registry  # 트레이너 레지스트리
from src.utils import get_dist_manager  # 분산 훈련 매니저

console = Console()  # 전역 콘솔 객체


def _compute_weighted_mtp_loss(
    logits: torch.Tensor,  # [B, S, H, V]
    target_ids: torch.Tensor,  # [B, S]
    head_weights: torch.Tensor,  # [B, S, H] - 새로운 헤드별 가중치!
    horizon: int,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    연구제안서 정확 구현: L_WMTP = Σ(k=0 to H-1) w_{t+k} × CE_k

    각 헤드별 CE에 해당 헤드의 가중치를 직접 적용하여
    토큰 중요도가 손실에 정확히 반영되도록 함.

    Args:
        logits: [batch, seq_len, horizon, vocab] - MTP 모델 출력
        target_ids: [batch, seq_len] - 타겟 라벨
        head_weights: [batch, seq_len, horizon] - 헤드별 가중치 매트릭스
        horizon: 예측 헤드 수 (4)
        ignore_index: 무시할 라벨 값

    Returns:
        weighted_loss: 가중 평균 손실 (scalar)
        valid_mask: 유효한 위치 마스크 [batch, seq_len]
    """
    # Input validation
    if not isinstance(logits, torch.Tensor) or not isinstance(target_ids, torch.Tensor):
        raise TypeError("logits and target_ids must be torch.Tensor")

    if not isinstance(head_weights, torch.Tensor):
        raise TypeError("head_weights must be torch.Tensor")

    if logits.ndim != 4:
        raise ValueError(f"logits must be 4D [B,S,H,V], got shape {logits.shape}")

    if target_ids.ndim != 2:
        raise ValueError(f"target_ids must be 2D [B,S], got shape {target_ids.shape}")

    if head_weights.ndim != 3:
        raise ValueError(
            f"head_weights must be 3D [B,S,H], got shape {head_weights.shape}"
        )

    bsz, seqlen, H, vocab = logits.shape
    if target_ids.shape != (bsz, seqlen):
        raise ValueError(
            f"Shape mismatch: logits {logits.shape} vs target_ids {target_ids.shape}"
        )

    if head_weights.shape != (bsz, seqlen, H):
        raise ValueError(
            f"Shape mismatch: head_weights {head_weights.shape} vs expected {(bsz, seqlen, H)}"
        )

    if horizon != H:
        raise ValueError(
            f"Mismatch between logits heads ({H}) and configured horizon ({horizon})"
        )

    device = logits.device
    dtype = logits.dtype

    # 헤드별 가중 CE 누적용
    weighted_ce_sum = torch.zeros((bsz, seqlen), device=device, dtype=dtype)
    total_weights = torch.zeros((bsz, seqlen), device=device, dtype=dtype)

    # 각 헤드별로 CE 계산 및 가중치 적용
    for k in range(H):
        shift = k + 1  # k번째 헤드는 t+(k+1) 위치 예측
        valid_len = seqlen - shift
        if valid_len <= 0:
            continue

        # 유효 영역 슬라이싱
        logits_k = logits[:, :valid_len, k, :]  # [B, valid_len, V]
        labels_k = target_ids[:, shift : shift + valid_len]  # [B, valid_len]
        weights_k = head_weights[:, :valid_len, k]  # [B, valid_len]

        # 헤드별 CE 계산
        ce_k = F.cross_entropy(
            logits_k.transpose(1, 2),  # [B, V, valid_len]
            labels_k,
            ignore_index=ignore_index,
            reduction="none",
        )  # [B, valid_len]

        # 유효 위치 마스킹 (ignore_index 제외)
        valid_k_mask = (labels_k != ignore_index).to(dtype)  # [B, valid_len]

        # 가중 CE: w_{t+k} × CE_k (연구제안서 공식!)
        weighted_ce_k = weights_k * ce_k * valid_k_mask  # [B, valid_len]
        effective_weights_k = weights_k * valid_k_mask  # [B, valid_len]

        # 전체 시퀀스에 누적 ([B, S] 형태로 맞춤)
        weighted_ce_sum[:, :valid_len] += weighted_ce_k
        total_weights[:, :valid_len] += effective_weights_k

    # 가중 평균 계산 (분모 0 방지)
    total_weights_clamped = torch.clamp(total_weights, min=1e-8)
    weighted_loss_per_token = weighted_ce_sum / total_weights_clamped

    # 유효 마스크: 최소 하나의 헤드에서 유효한 위치
    valid_mask = total_weights > 1e-8

    # 최종 스칼라 손실: 유효 토큰들의 평균
    if valid_mask.any():
        final_loss = (
            weighted_loss_per_token * valid_mask.to(dtype)
        ).sum() / valid_mask.sum().to(dtype)
    else:
        final_loss = torch.tensor(0.0, device=device, dtype=dtype)

    return final_loss, valid_mask


@trainer_registry.register(
    "mtp-weighted-ce-trainer", category="trainer", version="1.0.0"
)
class MTPWeightedCETrainer(BaseComponent):
    """WMTP 통합 트레이너 - 모든 알고리즘의 핵심 실행기.

    연구 철학 "Not All Tokens Are What You Need"의 실제 구현:
        이 클래스는 세 가지 WMTP 알고리즘(mtp-baseline, critic-wmtp, rho1-wmtp)을
        모두 지원하는 통합 트레이너입니다. 알고리즘 간 차이는 오직 Scorer에 의한
        토큰 가중치 계산 방식뿐이며, 나머지 훈련 로직은 완전히 공유됩니다.

    🔬 핵심 동작 원리:
        1. Scorer에서 토큰별 중요도 가중치 w_{t+k} 계산
        2. 각 MTP 헤드별로 Cross-Entropy 손실 CE_k 계산
        3. WMTP 공식 적용: L_WMTP = Σ w_{t+k} × CE_k
        4. 혼합 정밀도와 분산 훈련으로 안정적 최적화

    알고리즘별 동작 차이:
        - mtp-baseline: scorer=None → 모든 w_{t+k} = 1.0
        - critic-wmtp: CriticScorer → δ_t = V_t - V_{t-1} 기반 가중치
        - rho1-wmtp: Rho1Scorer → |CE^ref_t - CE^base_t| 기반 가중치

    필수 설정 키:
        - n_heads: MTP 헤드 수 (일반적으로 4)
        - horizon: 예측 범위 (n_heads와 동일)
        - loss_config: 손실 함수 설정 (정규화, 온도 등)
        - mixed_precision: 혼합 정밀도 ("bf16"/"fp16"/"fp32")
        - fsdp_config: FSDP 분산 훈련 설정 (dict 또는 None)
        - scorer: 토큰 가중치 계산기 (None이면 baseline)

    선택적 설정:
        - full_finetune: 전체 파인튜닝 여부
        - lora_config: LoRA 설정 (메모리 효율적 파인튜닝)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """WMTP 트레이너 초기화.

        Args:
            config: 트레이너 설정 딕셔너리 (horizon, 손실 설정, Scorer 등)
        """
        super().__init__(config)

        # 주요 컴포넌트들 (setup에서 초기화됨)
        self.model: nn.Module | None = None  # Facebook MTP 모델
        self.optimizer = None  # AdamW 등 최적화기

        # 훈련 상태 추적
        self.global_step: int = 0  # 전역 훈련 스텝 카운터

        # MTP 설정
        self.horizon: int = int(self.config.get("horizon", 4))  # 예측 헤드 수

        # Scorer 출력 캐싱 (성능 최적화용)
        self._last_score_out: dict[str, Any] | None = None

    def setup(self, ctx: dict[str, Any]) -> None:
        """트레이너 초기화 - 모델, 분산 훈련, Scorer 등 모든 컴포넌트 설정.

        이 메서드는 파이프라인에서 제공받은 컴포넌트들을 연결하고
        WMTP 훈련에 필요한 모든 설정을 완료합니다.

        Args:
            ctx: 컨텍스트 딕셔너리 (model, optimizer, scorers, tokenizers 등)
        """
        super().setup(ctx)
        dm = get_dist_manager()  # 분산 훈련 매니저

        # 필수 컴포넌트 검증 및 설정
        model: nn.Module | None = ctx.get("model")  # Facebook MTP 모델
        optimizer = ctx.get("optimizer")  # AdamW 등 최적화기
        if model is None:
            raise ValueError("트레이너에 'model'이 필요합니다 (ctx에서 누락)")
        if optimizer is None:
            raise ValueError("트레이너에 'optimizer'가 필요합니다 (ctx에서 누락)")

        # 분산 훈련: FSDP 래핑 (선택적)
        fsdp_cfg = self.config.get("fsdp_config")
        if fsdp_cfg:
            # Fully Sharded Data Parallel로 모델 래핑
            model = dm.setup_fsdp(model, fsdp_cfg)

        self.model = model
        self.optimizer = optimizer

        # 혼합 정밀도 설정: 메모리와 속도 최적화
        mp = str(self.config.get("mixed_precision", "bf16")).lower()
        if mp not in {"bf16", "fp16", "fp32"}:
            mp = "bf16"  # 기본값: BFloat16 (권장)

        self._amp_dtype = (
            torch.bfloat16  # BF16: 안정성과 성능의 균형
            if mp == "bf16"
            else (torch.float16 if mp == "fp16" else torch.float32)  # FP16 또는 FP32
        )

        # 🎯 핵심: 알고리즘별 토큰 가중치 계산 Scorer 연결
        self.scorer = self.config.get(
            "scorer"
        )  # None(baseline), CriticScorer, Rho1Scorer

        # WMTP 손실 함수 설정
        self.loss_cfg = self.config.get("loss_config", {})

        # MLflow 실험 추적 (선택적)
        self.mlflow = ctx.get("mlflow_manager")

        # 알고리즘별 보조 모델들 (선택적 - 해당 알고리즘에서만 사용)
        self.ref_model: nn.Module | None = ctx.get("ref_model")  # Rho-1용 참조 모델
        self.rm_model: nn.Module | None = ctx.get("rm_model")  # Critic용 보상 모델
        self.base_tokenizer = ctx.get("base_tokenizer")  # 기본 토크나이저
        self.ref_tokenizer = ctx.get("ref_tokenizer")  # 참조 모델용 토크나이저

        # 체크포인트 관리 설정
        self.dist_manager = dm  # 분산 훈련 매니저 (체크포인트 저장/로드용)

        # Recipe에서 체크포인트 설정 파싱
        recipe = ctx.get("recipe")  # Recipe 객체 가져오기 (없으면 None)
        if (
            recipe
            and hasattr(recipe, "train")
            and hasattr(recipe.train, "checkpointing")
        ):
            checkpointing = recipe.train.checkpointing
            self.save_interval = getattr(checkpointing, "save_interval", 100)
            self.keep_last = getattr(checkpointing, "keep_last", 3)
            self.save_final = getattr(checkpointing, "save_final", True)
        else:
            # 기본값 설정
            self.save_interval = 100
            self.keep_last = 3
            self.save_final = True

        # 알고리즘 정보 저장 (recipe에서 추출)
        self.algorithm = (
            getattr(recipe.train, "algo", "wmtp")
            if recipe and hasattr(recipe, "train")
            else "wmtp"
        )

        # 체크포인트 디렉토리 설정
        from pathlib import Path

        run_name = (
            getattr(recipe, "run", {}).get("name", "default") if recipe else "default"
        )
        self.checkpoint_dir = Path("./checkpoints") / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 저장된 체크포인트 목록 관리
        self.saved_checkpoints = []

        # 재개 처리 로직
        self.start_step = 0
        self.resume_metrics = {}

        resume_checkpoint = ctx.get("resume_checkpoint")
        if resume_checkpoint:
            checkpoint_data = self.dist_manager.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                checkpoint_path=str(resume_checkpoint),
            )

            self.start_step = checkpoint_data.get("step", 0)
            self.resume_metrics = checkpoint_data.get("metrics", {})

            console.print(
                f"[green]Model and optimizer states restored from step {self.start_step}[/green]"
            )

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        self.model.train()

        # Note: input_ids may be unused by this method
        target_ids: torch.Tensor = batch["labels"]  # [B, S]

        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=self._amp_dtype,
        ):
            # Model is expected to output logits for each horizon head
            outputs: dict[str, Any] | torch.Tensor = self.model(**batch)

            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]  # [B, S, H, V] expected
            else:
                logits = outputs  # assume tensor

            # Shape validation
            if logits.ndim != 4:
                raise ValueError(
                    f"Expected logits shape [B,S,H,V], got {tuple(logits.shape)}"
                )

            # Ensure logits require grad for tests/models that return detached tensors
            if not logits.requires_grad:
                logits = logits.detach().requires_grad_(True)

            # 새로운 헤드별 가중치 기반 손실 계산 시작

            # Build scorer context to get token weights
            if self.scorer is not None:
                scorer_ctx = {
                    "base_logits": logits[:, :, 0, :],  # provide one head if needed
                    "target_ids": target_ids,
                    "seq_lengths": [int(target_ids.shape[1])]
                    * int(target_ids.shape[0]),
                }

                # Provide hidden_states to critic scorer if available
                try:
                    hidden_states = None
                    if isinstance(outputs, dict) and "hidden_states" in outputs:
                        hs = outputs["hidden_states"]
                        hidden_states = hs[-1] if isinstance(hs, (list | tuple)) else hs
                    elif hasattr(outputs, "hidden_states"):
                        hs = outputs.hidden_states
                        hidden_states = hs[-1] if isinstance(hs, (list | tuple)) else hs
                    if hidden_states is not None and hidden_states.ndim == 3:
                        scorer_ctx["hidden_states"] = hidden_states
                except Exception:
                    # Hidden states are optional; ignore failures
                    pass
                # Optionally compute reference logits for Rho-1 scorer
                if self.ref_model is not None:
                    try:
                        with (
                            torch.no_grad(),
                            torch.autocast(
                                device_type="cuda"
                                if torch.cuda.is_available()
                                else "cpu",
                                dtype=self._amp_dtype,
                            ),
                        ):
                            ref_outputs = self.ref_model(
                                input_ids=batch.get("input_ids"),
                                attention_mask=batch.get("attention_mask"),
                            )
                            ref_logits = (
                                ref_outputs["logits"]
                                if isinstance(ref_outputs, dict)
                                and "logits" in ref_outputs
                                else ref_outputs
                            )
                        if ref_logits is not None and ref_logits.ndim == 3:
                            ref_vocab = ref_logits.shape[-1]
                            # ensure no negative labels included for max
                            valid_tids = target_ids[target_ids >= 0]
                            max_tid = (
                                int(valid_tids.max().item())
                                if valid_tids.numel() > 0
                                else 0
                            )
                            if max_tid < ref_vocab:
                                scorer_ctx["ref_logits"] = ref_logits
                    except Exception:
                        from rich.console import Console as _C

                        _C().print(
                            "[yellow]Reference forward failed; fallback to base-only scoring this step.[/yellow]"
                        )

                # 새로운 헤드별 가중치 기반 접근 (연구제안서 구현)
                score_out = self.scorer.run(scorer_ctx)
                head_weights_out = score_out.get("weights")  # [B, S, H] 형태

                # Store score_out for extended metrics
                self._last_score_out = score_out

                # Convert head weights to tensor
                if isinstance(head_weights_out, torch.Tensor):
                    head_weights = head_weights_out.to(
                        device=logits.device, dtype=logits.dtype
                    )
                else:
                    head_weights_np = np.asarray(head_weights_out)
                    head_weights = torch.tensor(
                        head_weights_np, device=logits.device, dtype=logits.dtype
                    )

                # 새로운 가중 MTP 손실 계산 (연구제안서 정확 구현)
                weighted_loss, valid_mask = _compute_weighted_mtp_loss(
                    logits=logits,  # [B, S, H, V]
                    target_ids=target_ids,  # [B, S]
                    head_weights=head_weights,  # [B, S, H]
                    horizon=self.horizon,
                    ignore_index=-100,
                )

                # Lambda scaling
                lambda_w = float(self.loss_cfg.get("lambda", 0.3))
                loss = lambda_w * weighted_loss  # 최종 스칼라 손실

            else:
                # Scorer가 없는 경우: uniform weights 사용
                B, S, H, V = logits.shape
                uniform_weights = torch.ones(
                    (B, S, H), device=logits.device, dtype=logits.dtype
                )

                weighted_loss, valid_mask = _compute_weighted_mtp_loss(
                    logits=logits,
                    target_ids=target_ids,
                    head_weights=uniform_weights,
                    horizon=self.horizon,
                    ignore_index=-100,
                )

                lambda_w = float(self.loss_cfg.get("lambda", 0.3))
                loss = lambda_w * weighted_loss
                self._last_score_out = None

            # 새로운 헤드별 가중치 기반 손실 계산 완료
            # loss는 위에서 이미 계산됨

        # Backward and optimize
        loss.backward()

        # Grad clip (from optimizer component if available)
        grad_clip = float(getattr(self.optimizer, "grad_clip", 1.0))
        if math.isfinite(grad_clip) and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        # Optional MLflow logging
        if self.mlflow is not None:
            try:
                # Per-head CE means (diagnostics)
                with torch.no_grad():
                    bsz, seqlen, H, vocab = logits.shape
                    # Approximate head CE means using valid regions only
                    ce_head_means = []
                    for k in range(H):
                        shift = k + 1
                        valid_len = seqlen - shift
                        if valid_len <= 0:
                            ce_head_means.append(
                                torch.tensor(0.0, device=logits.device)
                            )
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
                    metrics = {
                        f"train/ce_head_{i}": float(x)
                        for i, x in enumerate(ce_head_means)
                    }
                    metrics.update(
                        {
                            "train/loss": float(loss.detach().item()),
                            "train/ce_mean": float(
                                (ce_per_token[valid_mask]).mean().item()
                            )
                            if valid_mask.any()
                            else 0.0,
                        }
                    )
                    # Extended weight statistics on aligned region
                    w_eff = weights[valid_mask]
                    if w_eff.numel() > 0:
                        # Basic weight statistics
                        weight_stats = {
                            "train/weight_mean": float(w_eff.mean().item()),
                            "train/weight_min": float(w_eff.min().item()),
                            "train/weight_max": float(w_eff.max().item()),
                            "train/weight_std": float(w_eff.std().item()),
                        }

                        # Weight distribution percentiles (계획서 요구사항)
                        try:
                            weight_stats.update(
                                {
                                    "train/weight_p25": float(
                                        torch.quantile(w_eff, 0.25).item()
                                    ),
                                    "train/weight_p75": float(
                                        torch.quantile(w_eff, 0.75).item()
                                    ),
                                    "train/weight_p95": float(
                                        torch.quantile(w_eff, 0.95).item()
                                    ),
                                }
                            )
                        except Exception:
                            # Fallback if quantile fails (e.g., older PyTorch versions)
                            sorted_w = torch.sort(w_eff)[0]
                            n = sorted_w.numel()
                            weight_stats.update(
                                {
                                    "train/weight_p25": float(
                                        sorted_w[int(n * 0.25)].item()
                                    ),
                                    "train/weight_p75": float(
                                        sorted_w[int(n * 0.75)].item()
                                    ),
                                    "train/weight_p95": float(
                                        sorted_w[int(n * 0.95)].item()
                                    ),
                                }
                            )

                        # Failure gates strengthening (계획서 요구사항)
                        weight_stats.update(
                            {
                                "train/nan_weights": int(
                                    (~torch.isfinite(weights)).sum().item()
                                ),
                                "train/extreme_weights": int(
                                    (weights > 5.0).sum().item()
                                ),
                            }
                        )

                        metrics.update(weight_stats)

                    # Scorer-specific metrics (계획서 요구사항: 방식별 특화 지표)
                    if hasattr(self, "_last_score_out") and self._last_score_out:
                        # Detect scorer type
                        scorer_type = (
                            self.scorer.__class__.__name__.lower()
                            if self.scorer
                            else "unknown"
                        )

                        if "rho1" in scorer_type:
                            # Rho-1 specific metrics
                            scores = self._last_score_out.get("scores")
                            if scores:
                                scores_tensor = (
                                    torch.tensor(scores)
                                    if not isinstance(scores, torch.Tensor)
                                    else scores
                                )
                                total_tokens = float(scores_tensor.numel())
                                # 임계값 이상의 토큰들의 비율 (usage ratio)
                                threshold = 0.5  # 임계값 설정
                                high_score_tokens = float(
                                    (scores_tensor > threshold).sum().item()
                                )
                                metrics["train/rho1_usage_ratio"] = (
                                    high_score_tokens / total_tokens
                                    if total_tokens > 0
                                    else 0.0
                                )

                        elif "critic" in scorer_type:
                            # Critic specific metrics
                            deltas = self._last_score_out.get("deltas")
                            if deltas:
                                deltas_tensor = (
                                    torch.tensor(deltas)
                                    if not isinstance(deltas, torch.Tensor)
                                    else deltas
                                )
                                metrics["train/critic_delta_mean"] = float(
                                    deltas_tensor.mean().item()
                                )
                                metrics["train/critic_delta_std"] = float(
                                    deltas_tensor.std().item()
                                )
                    # Valid token ratio
                    total_tokens = float(valid_mask.numel())
                    valid_tokens = float(valid_mask.sum().item())
                    metrics["train/valid_token_ratio"] = (
                        valid_tokens / total_tokens if total_tokens > 0 else 0.0
                    )
                self.mlflow.log_metrics(metrics, step=self.global_step)
            except Exception:
                # Never fail training on logging errors
                pass

        # Failure gates
        if (
            not torch.isfinite(loss)
            or not torch.isfinite(ce_per_token).all()
            or not torch.isfinite(weights).all()
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

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        체크포인트 저장 기능이 포함된 확장된 훈련 루프.
        주기적 체크포인트 저장과 최종 모델 저장을 지원합니다.

        Args:
            ctx: 'train_dataloader'와 'max_steps' 포함

        Returns:
            훈련 메트릭 딕셔너리
        """
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        dataloader = ctx.get("train_dataloader")
        if dataloader is None:
            raise ValueError("Trainer.run expects 'train_dataloader' in ctx")
        max_steps: int | None = ctx.get("max_steps")

        epoch = 0  # 단순화를 위해 epoch=0으로 설정
        metrics = {}

        console.print(
            f"[green]체크포인트 저장 활성화: 매 {self.save_interval}스텝마다 저장[/green]"
        )
        console.print(f"[green]체크포인트 디렉토리: {self.checkpoint_dir}[/green]")

        for step, batch in enumerate(dataloader):
            current_step = step + 1

            # 재개시 이미 완료된 스텝 건너뛰기
            if current_step <= self.start_step:
                continue

            # 기존 훈련 스텝 실행 (변경 없음)
            out = self.train_step(batch)
            metrics = out

            # 주기적 체크포인트 저장
            if current_step % self.save_interval == 0:
                try:
                    checkpoint_path = self._save_checkpoint(
                        epoch, current_step, metrics
                    )
                    self.saved_checkpoints = self._manage_checkpoints(
                        self.saved_checkpoints, checkpoint_path
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]체크포인트 저장 실패 (스텝 {current_step}): {e}[/yellow]"
                    )

            # 최대 스텝 도달 시 종료
            if max_steps is not None and current_step >= max_steps:
                break

        # 최종 체크포인트 저장
        if self.save_final:
            try:
                final_step = step + 1 if "step" in locals() else 1
                final_path = self._save_final_checkpoint(epoch, final_step, metrics)
                console.print(f"[green]최종 모델 저장 완료: {final_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]최종 모델 저장 실패: {e}[/yellow]")

        return metrics

    def _save_checkpoint(self, epoch: int, step: int, metrics: dict) -> Path:
        """
        단일 체크포인트 저장.

        Args:
            epoch: 현재 에폭
            step: 현재 스텝
            metrics: 훈련 메트릭

        Returns:
            저장된 체크포인트 경로
        """

        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"

        # FSDP 호환 체크포인트 저장
        self.dist_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=str(checkpoint_path),
            epoch=epoch,
            step=step,
            metrics=metrics,
            algorithm=getattr(self, "algorithm", "wmtp"),
            mlflow_run_id=self.mlflow.get_run_id() if self.mlflow else None,
        )

        # MLflow에 아티팩트 업로드 (있는 경우)
        if self.mlflow is not None:
            try:
                self.mlflow.log_artifact(
                    local_path=checkpoint_path, artifact_path="checkpoints"
                )
                console.print(
                    f"[green]Checkpoint uploaded to MLflow: {checkpoint_path.name}[/green]"
                )
            except Exception as e:
                console.print(f"[yellow]MLflow upload warning: {e}[/yellow]")

        console.print(f"[green]체크포인트 저장 완료: {checkpoint_path}[/green]")
        return checkpoint_path

    def _manage_checkpoints(
        self, saved_checkpoints: list, new_checkpoint: Path
    ) -> list:
        """
        체크포인트 파일 개수 관리 (keep_last 개만 유지).

        Args:
            saved_checkpoints: 기존 체크포인트 목록
            new_checkpoint: 새로 저장된 체크포인트

        Returns:
            업데이트된 체크포인트 목록
        """
        saved_checkpoints.append(new_checkpoint)

        # keep_last 개수 초과 시 오래된 파일 삭제
        while len(saved_checkpoints) > self.keep_last:
            old_checkpoint = saved_checkpoints.pop(0)
            try:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    console.print(
                        f"[blue]이전 체크포인트 삭제: {old_checkpoint.name}[/blue]"
                    )
            except Exception as e:
                console.print(f"[yellow]체크포인트 삭제 실패: {e}[/yellow]")

        return saved_checkpoints

    def _save_final_checkpoint(self, epoch: int, step: int, metrics: dict) -> Path:
        """
        최종 모델 저장.

        Args:
            epoch: 최종 에폭
            step: 최종 스텝
            metrics: 최종 메트릭

        Returns:
            저장된 최종 모델 경로
        """

        final_path = self.checkpoint_dir / "final_model.pt"

        # 최종 체크포인트 저장
        self.dist_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=str(final_path),
            epoch=epoch,
            step=step,
            metrics=metrics,
            algorithm=getattr(self, "algorithm", "wmtp"),
            final_model=True,
            mlflow_run_id=self.mlflow.get_run_id() if self.mlflow else None,
        )

        # MLflow 모델 레지스트리 등록 및 아티팩트 업로드
        if self.mlflow is not None:
            try:
                # 모델 이름 생성 (recipe에서 알고리즘 정보 사용)
                model_name = f"wmtp-{self.algorithm}"

                # 모델 레지스트리 등록
                self.mlflow.log_model(
                    model=self.model,
                    artifact_path="final_model",
                    registered_model_name=model_name,
                )

                # 체크포인트 파일 업로드
                self.mlflow.log_artifact(
                    local_path=final_path, artifact_path="final_checkpoint"
                )

                console.print(f"[green]MLflow 모델 등록 완료: {model_name}[/green]")
            except Exception as e:
                console.print(
                    f"[yellow]MLflow model registration warning: {e}[/yellow]"
                )

        return final_path
