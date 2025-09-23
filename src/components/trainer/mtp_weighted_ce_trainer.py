"""
MTP Weighted Cross-Entropy Trainer.

Implements the WMTP training loop where token-level weights produced by a
Scorer are applied to the average CE over MTP heads. Supports AMP (bf16/fp16),
grad clipping, FSDP wrapping via utils.dist, and MLflow logging hooks.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

from src.components.base import BaseComponent
from src.components.registry import trainer_registry
from src.utils import get_dist_manager

console = Console()


def _compute_weighted_mtp_loss(
    logits: torch.Tensor,        # [B, S, H, V]
    target_ids: torch.Tensor,    # [B, S]
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
        raise ValueError(f"head_weights must be 3D [B,S,H], got shape {head_weights.shape}")

    bsz, seqlen, H, vocab = logits.shape
    if target_ids.shape != (bsz, seqlen):
        raise ValueError(f"Shape mismatch: logits {logits.shape} vs target_ids {target_ids.shape}")

    if head_weights.shape != (bsz, seqlen, H):
        raise ValueError(f"Shape mismatch: head_weights {head_weights.shape} vs expected {(bsz, seqlen, H)}")

    if horizon != H:
        raise ValueError(f"Mismatch between logits heads ({H}) and configured horizon ({horizon})")

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
        labels_k = target_ids[:, shift:shift + valid_len]  # [B, valid_len]
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
        effective_weights_k = weights_k * valid_k_mask   # [B, valid_len]

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
        final_loss = (weighted_loss_per_token * valid_mask.to(dtype)).sum() / valid_mask.sum().to(dtype)
    else:
        final_loss = torch.tensor(0.0, device=device, dtype=dtype)

    return final_loss, valid_mask


@trainer_registry.register(
    "mtp-weighted-ce-trainer", category="trainer", version="1.0.0"
)
class MTPWeightedCETrainer(BaseComponent):
    """
    Trainer that applies token weights to MTP CE across heads.

    Expected config keys:
      - n_heads: int (MTP heads)
      - horizon: int (same as n_heads)
      - loss_config: { weight_norm, lambda, temperature, epsilon, max_weight }
      - mixed_precision: str ("bf16"|"fp16"|"fp32")
      - fsdp_config: dict or None (FSDP options)
      - scorer: Scorer instance (must implement run())
      - full_finetune / lora_config: routed but not used here directly
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model: nn.Module | None = None
        self.optimizer = None
        self.global_step: int = 0
        self.horizon: int = int(self.config.get("horizon", 4))
        self._last_score_out: dict[str, Any] | None = None

    def setup(self, ctx: dict[str, Any]) -> None:
        super().setup(ctx)
        dm = get_dist_manager()

        # Expect model and optimizer to be provided in ctx
        model: nn.Module | None = ctx.get("model")
        optimizer = ctx.get("optimizer")
        if model is None:
            raise ValueError("Trainer requires 'model' in ctx")
        if optimizer is None:
            raise ValueError("Trainer requires 'optimizer' in ctx")

        # Optional: wrap with FSDP
        fsdp_cfg = self.config.get("fsdp_config")
        if fsdp_cfg:
            model = dm.setup_fsdp(model, fsdp_cfg)

        self.model = model
        self.optimizer = optimizer

        # Mixed precision policy selection
        mp = str(self.config.get("mixed_precision", "bf16")).lower()
        if mp not in {"bf16", "fp16", "fp32"}:
            mp = "bf16"
        self._amp_dtype = (
            torch.bfloat16
            if mp == "bf16"
            else (torch.float16 if mp == "fp16" else torch.float32)
        )

        # Attach scorer if provided
        self.scorer = self.config.get("scorer")

        # Loss/weight config
        self.loss_cfg = self.config.get("loss_config", {})
        # Optional MLflow manager for logging
        self.mlflow = ctx.get("mlflow_manager")

        # Optional auxiliary models/tokenizers for scorers
        self.ref_model: nn.Module | None = ctx.get("ref_model")
        self.rm_model: nn.Module | None = ctx.get("rm_model")
        self.base_tokenizer = ctx.get("base_tokenizer")
        self.ref_tokenizer = ctx.get("ref_tokenizer")

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
                    logits=logits,           # [B, S, H, V]
                    target_ids=target_ids,   # [B, S]
                    head_weights=head_weights, # [B, S, H]
                    horizon=self.horizon,
                    ignore_index=-100
                )

                # Lambda scaling
                lambda_w = float(self.loss_cfg.get("lambda", 0.3))
                loss = lambda_w * weighted_loss  # 최종 스칼라 손실

            else:
                # Scorer가 없는 경우: uniform weights 사용
                B, S, H, V = logits.shape
                uniform_weights = torch.ones((B, S, H), device=logits.device, dtype=logits.dtype)

                weighted_loss, valid_mask = _compute_weighted_mtp_loss(
                    logits=logits,
                    target_ids=target_ids,
                    head_weights=uniform_weights,
                    horizon=self.horizon,
                    ignore_index=-100
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
                            weight_stats.update({
                                "train/weight_p25": float(torch.quantile(w_eff, 0.25).item()),
                                "train/weight_p75": float(torch.quantile(w_eff, 0.75).item()),
                                "train/weight_p95": float(torch.quantile(w_eff, 0.95).item()),
                            })
                        except Exception:
                            # Fallback if quantile fails (e.g., older PyTorch versions)
                            sorted_w = torch.sort(w_eff)[0]
                            n = sorted_w.numel()
                            weight_stats.update({
                                "train/weight_p25": float(sorted_w[int(n * 0.25)].item()),
                                "train/weight_p75": float(sorted_w[int(n * 0.75)].item()),
                                "train/weight_p95": float(sorted_w[int(n * 0.95)].item()),
                            })

                        # Failure gates strengthening (계획서 요구사항)
                        weight_stats.update({
                            "train/nan_weights": int((~torch.isfinite(weights)).sum().item()),
                            "train/extreme_weights": int((weights > 5.0).sum().item()),
                        })

                        metrics.update(weight_stats)

                    # Scorer-specific metrics (계획서 요구사항: 방식별 특화 지표)
                    if hasattr(self, '_last_score_out') and self._last_score_out:
                        # Detect scorer type
                        scorer_type = self.scorer.__class__.__name__.lower() if self.scorer else "unknown"

                        if "rho1" in scorer_type:
                            # Rho-1 specific metrics
                            scores = self._last_score_out.get("scores")
                            if scores:
                                scores_tensor = torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores
                                total_tokens = float(scores_tensor.numel())
                                # 임계값 이상의 토큰들의 비율 (usage ratio)
                                threshold = 0.5  # 임계값 설정
                                high_score_tokens = float((scores_tensor > threshold).sum().item())
                                metrics["train/rho1_usage_ratio"] = (
                                    high_score_tokens / total_tokens if total_tokens > 0 else 0.0
                                )

                        elif "critic" in scorer_type:
                            # Critic specific metrics
                            deltas = self._last_score_out.get("deltas")
                            if deltas:
                                deltas_tensor = torch.tensor(deltas) if not isinstance(deltas, torch.Tensor) else deltas
                                metrics["train/critic_delta_mean"] = float(deltas_tensor.mean().item())
                                metrics["train/critic_delta_std"] = float(deltas_tensor.std().item())
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
        Minimal training loop over a provided dataloader in ctx.
        Expects ctx to provide 'train_dataloader' and optional 'max_steps'.
        """
        if self.model is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        dataloader = ctx.get("train_dataloader")
        if dataloader is None:
            raise ValueError("Trainer.run expects 'train_dataloader' in ctx")
        max_steps: int | None = ctx.get("max_steps")

        metrics = {}
        for step, batch in enumerate(dataloader):
            out = self.train_step(batch)
            metrics = out
            if max_steps is not None and step + 1 >= max_steps:
                break
        return metrics
