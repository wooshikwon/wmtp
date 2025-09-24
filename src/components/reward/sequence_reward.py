"""
WMTP Critic Stage 1: 시퀀스 보상 계산 모듈

연구 맥락:
이 모듈은 Critic-WMTP의 Stage 1에서만 사용되는 시퀀스 품질 평가기입니다.
연구제안서의 2단계 학습 과정 중 첫 번째 단계를 담당합니다.

WMTP에서의 역할:
1. Critic-WMTP Stage 1 전용 (Rho-1이나 baseline에서는 사용 안함)
2. RM(Reward Model)으로 전체 문장의 품질을 점수로 계산
3. 하나의 스칼라 보상값을 반환 (예: 3.7점)
4. CriticDeltaScorer가 이 보상을 토큰별로 분배 (GAE 사용)
5. 가치헤드가 토큰별 누적 보상을 학습하는 기초 데이터 제공

학습 흐름:
시퀀스 → RM → 시퀀스 보상 R → GAE 분배 → 토큰별 가치 목표 → 가치헤드 학습

RM이 없을 때 fallback: 평균 교차엔트로피의 음수 사용 (-mean_CE)
"더 자연스러운 문장 = 더 높은 점수" 원리
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
) -> list[float]:
    """
    Critic Stage 1을 위한 시퀀스별 보상을 계산합니다.

    연구 맥락:
    이 함수는 연구제안서 Critic-WMTP의 Stage 1에서 핵심 역할을 담당합니다.
    전체 문장을 평가하여 하나의 보상 점수를 반환하는 것이 목적입니다.

    WMTP Critic 학습 과정에서의 역할:
    1. 이 함수가 시퀀스별 보상 R_t를 제공
    2. CriticDeltaScorer가 R_t를 토큰별 가치로 분배 (GAE 사용)
    3. 가치헤드가 이 토큰별 가치를 학습 목표로 사용
    4. Stage 2에서 학습된 가치헤드로 토큰 중요도 계산

    매개변수:
        rm_model: 보상모델 (예: Llama-3-8B-RM) 또는 일반 LLM
        input_ids: 토큰 ID 텐서 [batch_size, seq_len]
        attention_mask: 선택적 어텐션 마스크
        amp_dtype: 혼합 정밀도를 위한 데이터 타입

    반환값:
        배치의 각 시퀀스에 대한 보상 점수 리스트

    예시:
        >>> rewards = compute_sequence_rewards(rm, tokens)
        >>> print(rewards)  # [2.3, -0.7, 1.8]  시퀀스별 품질 점수

    주의사항:
        RM이 명시적인 보상 출력이 없으면 음의 평균 교차엔트로피로 fallback
        "더 자연스러운 문장 = 더 높은 점수" 원리 사용
    """
    # 혼합 정밀도 설정 (GPU 메모리 절약 및 속도 향상)
    autocast_kwargs = {}
    if amp_dtype is not None:
        autocast_kwargs = {
            "device_type": ("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": amp_dtype,
        }

    # RM 모델 추론 실행 (그래디언트 계산 비활성화)
    with (
        torch.autocast(**autocast_kwargs)
        if autocast_kwargs
        else torch.autocast(enabled=False, device_type="cpu")
    ):
        outputs = rm_model(input_ids=input_ids, attention_mask=attention_mask)

    # 1단계: 명시적인 보상 출력이 있는지 확인
    # 전용 RM 모델들은 보통 이런 키들로 보상값을 반환
    if isinstance(outputs, dict):
        for key in ("reward", "rewards", "score", "scores", "value", "values"):
            if key in outputs:
                vals = outputs[key]
                if isinstance(vals, torch.Tensor):
                    vals = vals.detach().float().view(-1)  # [batch_size] 형태로 변환
                    return vals.tolist()
                try:
                    return [float(v) for v in vals]
                except Exception:
                    pass

    # 2단계: 명시적 보상이 없으면 logits에서 계산 (fallback)
    logits = (
        outputs["logits"]
        if isinstance(outputs, dict) and "logits" in outputs
        else outputs
    )
    # 유효하지 않은 출력인 경우 기본값 반환
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        return [0.0 for _ in range(int(input_ids.shape[0]))]

    B, S, V = logits.shape  # [배치, 시퀀스길이, 어휘크기]
    if S <= 1:  # 시퀀스가 너무 짧으면 의미있는 보상 계산 불가
        return [0.0 for _ in range(B)]

    # Next-token 예측을 위한 shift (t시점 입력 → t+1시점 예측)
    logits_shifted = logits[:, :-1, :].transpose(1, 2).contiguous()  # [B, V, S-1]
    labels_shifted = input_ids[:, 1:].contiguous()  # [B, S-1]

    # 토큰별 교차엔트로피 계산
    ce = F.cross_entropy(logits_shifted, labels_shifted, reduction="none")  # [B, S-1]

    # 어텐션 마스크 적용 (패딩 토큰 제외)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].to(dtype=ce.dtype)  # [B, S-1]
        token_counts = torch.clamp(mask.sum(dim=1), min=1.0)  # 유효 토큰 수
        ce_mean = (ce * mask).sum(dim=1) / token_counts  # 마스킹된 평균 CE
    else:
        ce_mean = ce.mean(dim=1)  # 단순 평균 CE

    # 음의 교차엔트로피를 보상으로 사용 (낮은 CE = 높은 보상)
    # 연구 맥락: "더 자연스러운 문장 = 더 높은 점수"
    rewards = (-ce_mean).detach().float().tolist()
    return rewards
