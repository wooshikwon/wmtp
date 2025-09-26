"""WMTP MTP Data Collator - 단순화된 Utils 버전

사용자 제안에 따른 실용적 단순화:
- BaseComponent 패턴 제거
- Factory 패턴 복잡성 제거
- 직접적이고 명확한 인터페이스 제공
- WMTP 알고리즘 전용 최적화

핵심 기능:
- 2D 라벨 [B, S] → 3D MTP 라벨 [B, S, H] 변환
- DataCollatorForLanguageModeling 기반 패딩
- 단순한 함수형 인터페이스
"""

from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling


class MTPDataCollator:
    """MTP(Multi-Token Prediction)용 단순화된 Data Collator

    모든 WMTP 알고리즘(baseline-mtp, critic-wmtp, rho1-wmtp)에서 사용하는
    통합 collator. BaseComponent 복잡성을 제거하고 직접적인 사용을 위해 설계.

    주요 기능:
    1. DataCollatorForLanguageModeling 기반 기본 패딩
    2. MTP용 multi-horizon 라벨 생성 [B, S] → [B, S, H]
    3. GPU 효율성을 위한 pad_to_multiple_of 지원
    """

    def __init__(self, tokenizer, horizon: int = 4, pad_to_multiple_of: int = 8):
        """MTP Data Collator 초기화

        Args:
            tokenizer: HuggingFace tokenizer 인스턴스
            horizon: MTP 예측 범위 (기본값: 4, t+1,t+2,t+3,t+4)
            pad_to_multiple_of: GPU 효율성을 위한 패딩 단위 (기본값: 8)
        """
        self.tokenizer = tokenizer
        self.horizon = horizon

        # DataCollatorForLanguageModeling 초기화 - Composition 패턴
        self.base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 인과적 언어 모델링
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """MTP 배치 생성 - 기본 패딩 + MTP 라벨 생성

        Args:
            features: 배치 내 개별 샘플들의 리스트

        Returns:
            MTP 훈련을 위한 배치 딕셔너리:
            - input_ids: [B, S] 입력 토큰
            - attention_mask: [B, S] 어텐션 마스크
            - labels: [B, S, H] MTP 라벨 (3D)
        """
        # 1. 기본 패딩 처리 (DataCollatorForLanguageModeling 사용)
        batch = self.base_collator(features)

        # 2. MTP 라벨 생성: [B, S] → [B, S, H]
        if "labels" in batch:
            batch["labels"] = self._create_mtp_labels(batch["labels"])

        return batch

    def _create_mtp_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """[B, S] 라벨을 [B, S, H] MTP 라벨로 변환

        각 위치 t에서 t+1, t+2, t+3, t+4 토큰을 예측하는 MTP 라벨 생성.
        Meta 2024 논문의 multi-token prediction 구조를 정확히 구현.

        Args:
            labels: 2D 라벨 텐서 [B, S]

        Returns:
            3D MTP 라벨 텐서 [B, S, H]
        """
        B, S = labels.shape
        device = labels.device
        dtype = labels.dtype

        # MTP 라벨 텐서 초기화 (-100은 무시할 라벨)
        mtp_labels = torch.full((B, S, self.horizon), -100, dtype=dtype, device=device)

        # 각 horizon에 대해 라벨 생성
        for h in range(self.horizon):
            shift = h + 1  # t+1, t+2, t+3, t+4
            if shift < S:
                # 각 위치에서 shift만큼 앞의 토큰을 라벨로 사용
                mtp_labels[:, : S - shift, h] = labels[:, shift:]

        return mtp_labels


def create_mtp_collator(
    tokenizer, horizon: int = 4, pad_to_multiple_of: int = 8
) -> MTPDataCollator:
    """MTP Data Collator 생성 함수 - 간단한 팩토리 함수

    ComponentFactory.create_collator()를 대체하는 단순하고 직접적인 함수.
    WMTP 알고리즘에서 필요한 MTP collator를 즉시 생성.

    Args:
        tokenizer: HuggingFace tokenizer 인스턴스
        horizon: MTP 예측 범위 (기본값: 4)
        pad_to_multiple_of: GPU 효율성을 위한 패딩 단위 (기본값: 8)

    Returns:
        설정된 MTPDataCollator 인스턴스

    Example:
        ```python
        from src.utils.mtp_collator import create_mtp_collator

        collator = create_mtp_collator(tokenizer, horizon=4)
        train_dl = DataLoader(dataset, collate_fn=collator, ...)
        ```
    """
    return MTPDataCollator(
        tokenizer=tokenizer, horizon=horizon, pad_to_multiple_of=pad_to_multiple_of
    )
