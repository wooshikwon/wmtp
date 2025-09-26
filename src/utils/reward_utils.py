"""
Reward computation utilities for WMTP training.

Common reward calculation functions shared between CriticHeadPretrainer and CriticWmtpTrainer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_sequence_rewards(
    rm_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    amp_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """공통 시퀀스별 보상 계산 함수.

    RM 모델을 사용하여 시퀀스별 보상을 계산하거나,
    RM이 명시적인 보상을 제공하지 않으면 negative CE를 fallback으로 사용합니다.

    Args:
        rm_model: Reward Model 또는 일반 LLM
        input_ids: 토큰 ID 텐서 [batch_size, seq_len]
        attention_mask: 선택적 어텐션 마스크
        amp_dtype: 혼합 정밀도를 위한 데이터 타입

    Returns:
        torch.Tensor: 각 시퀀스에 대한 보상 점수 [batch_size]
    """
    # 혼합 정밀도 설정
    autocast_kwargs = {}
    if amp_dtype is not None:
        autocast_kwargs = {
            "device_type": ("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": amp_dtype,
        }

    # RM 모델 추론 실행
    with (
        torch.autocast(**autocast_kwargs)
        if autocast_kwargs
        else torch.autocast(enabled=False, device_type="cpu")
    ):
        outputs = rm_model(input_ids=input_ids, attention_mask=attention_mask)

    # 1단계: 명시적인 보상 출력이 있는지 확인
    if isinstance(outputs, dict):
        for key in ("reward", "rewards", "score", "scores", "value", "values"):
            if key in outputs:
                vals = outputs[key]
                if isinstance(vals, torch.Tensor):
                    return vals.detach().float().view(-1)  # [batch_size] 형태로 변환
                try:
                    return torch.tensor(
                        [float(v) for v in vals], device=input_ids.device
                    )
                except Exception:
                    pass

    # 2단계: 명시적 보상이 없으면 logits 해석
    logits = (
        outputs.get("logits")
        if isinstance(outputs, dict)
        else getattr(outputs, "logits", None)
    )
    if isinstance(logits, torch.Tensor):
        # SequenceClassification/Regression: [B] or [B,1] or [B,C]
        if logits.ndim == 1:
            return logits.detach().float().view(-1)
        if logits.ndim == 2:
            if logits.size(1) == 1:
                return logits.squeeze(1).detach().float().view(-1)
            # multi-class → max-logit as reward proxy
            return logits.max(dim=1).values.detach().float().view(-1)

        # CausalLM: [B, S, V] → negative CE fallback
        if logits.ndim == 3:
            B, S, V = logits.shape
            if S <= 1:
                return torch.zeros(B, device=input_ids.device, dtype=torch.float32)

            logits_shifted = logits[:, :-1, :].transpose(1, 2).contiguous()
            labels_shifted = input_ids[:, 1:].contiguous()

            ce = F.cross_entropy(logits_shifted, labels_shifted, reduction="none")

            if attention_mask is not None:
                mask = attention_mask[:, 1:].to(dtype=ce.dtype)
                token_counts = torch.clamp(mask.sum(dim=1), min=1.0)
                ce_mean = (ce * mask).sum(dim=1) / token_counts
            else:
                ce_mean = ce.mean(dim=1)

            return -ce_mean.detach().float()

    # 출력 형태를 해석할 수 없는 경우 0 보상
    return torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.float32)
