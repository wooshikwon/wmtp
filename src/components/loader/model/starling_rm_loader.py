"""
Berkeley Starling-RM-7B-alpha Reward Model 로더

WMTP Critic 알고리즘의 핵심 컴포넌트인 Reward Model 로더.
Starling-RM은 Llama-2-7b-chat 기반으로 동일한 SentencePiece tokenizer.model 사용.

주요 특징:
- 7B 파라미터 Reward Model
- Sequence classification head로 보상 점수 예측
- WMTP의 토큰별 가중치 계산에 활용
- Facebook MTP와 동일한 토크나이저로 완벽 호환
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.components.loader.base_loader import ModelLoader
from src.components.registry import loader_registry


@loader_registry.register(
    "starling-rm",
    category="loader",
    version="1.0.0",
    description="Berkeley Starling-RM-7B-alpha reward model loader",
)
class StarlingRMLoader(ModelLoader):
    """
    Starling Reward Model 로더

    WMTP Critic-WMTP 알고리즘의 핵심 RM 모델:
    - Value Function 기반 토큰 가중치 계산
    - δ_t = V_t - V_{t-1} 차분값으로 중요도 측정
    - Facebook MTP와 동일한 SentencePiece 토크나이저 사용

    모델 구조:
    - Base: Llama-2-7b-chat
    - Head: Reward modeling head (sequence classification)
    - Tokenizer: SentencePiece (SHA256: 9e556afd...)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        초기화

        Args:
            config: 설정 딕셔너리
                - device: 실행 디바이스 (cuda/cpu/mps)
                - use_4bit: 4비트 양자화 여부
                - use_8bit: 8비트 양자화 여부
                - cache_dir: 모델 캐시 디렉토리
        """
        super().__init__(config or {})

        # 디바이스 설정
        self.device = config.get("device", "auto") if config else "auto"

        # 양자화 옵션
        self.use_4bit = config.get("use_4bit", False) if config else False
        self.use_8bit = config.get("use_8bit", False) if config else False

        # 모델 ID
        self.model_id = "berkeley-nest/Starling-RM-7B-alpha"

    def load_model(self, path: str, **kwargs) -> Any:
        """
        Starling-RM 모델 로드

        Reward Model은 시퀀스 분류 태스크용 헤드를 포함:
        - 입력: 프롬프트 + 응답
        - 출력: 스칼라 보상 점수

        Args:
            path: 모델 경로 또는 캐시 디렉토리
            **kwargs: 추가 옵션
                - force_download: 강제 다운로드 여부
                - local_files_only: 로컬 파일만 사용

        Returns:
            로드된 Starling-RM 모델

        Raises:
            ImportError: transformers 패키지 미설치
            RuntimeError: 모델 로드 실패
        """
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "transformers 패키지가 필요합니다. " "설치: uv pip install transformers"
            )

        # 로컬 경로 확인
        local_path = Path(path)
        cache_dir = str(local_path) if local_path.exists() else None

        # 양자화 설정
        model_kwargs = {}
        if self.use_4bit or self.use_8bit:
            from transformers import BitsAndBytesConfig

            if self.use_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("📦 4비트 양자화 활성화 (NF4)")
            elif self.use_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                print("📦 8비트 양자화 활성화")

        # 디바이스 설정
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        elif self.device != "cpu":
            model_kwargs["device_map"] = {"": self.device}

        # 모델 로드
        try:
            print(f"🚀 Starling-RM 모델 로드 중: {self.model_id}")

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16
                if not (self.use_4bit or self.use_8bit)
                else None,
                trust_remote_code=True,  # 커스텀 코드 허용
                **model_kwargs,
                **kwargs,
            )

            # 평가 모드 설정
            model.eval()

            # 모델 정보 출력
            param_count = sum(p.numel() for p in model.parameters())
            print("✅ Starling-RM 로드 완료")
            print(f"   - 파라미터: {param_count:,} ({param_count / 1e9:.1f}B)")
            print(f"   - 디바이스: {next(model.parameters()).device}")
            print("   - Reward head 출력: 스칼라 보상 점수")

            return model

        except Exception as e:
            raise RuntimeError(
                f"Starling-RM 모델 로드 실패: {self.model_id}\n" f"에러: {e}"
            )

    # load_tokenizer는 부모 클래스의 통합 SentencePiece 메서드 상속
    # Starling-RM은 Llama-2 기반으로 동일한 tokenizer.model 사용
