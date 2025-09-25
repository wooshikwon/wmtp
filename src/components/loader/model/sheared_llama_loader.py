"""
Princeton Sheared-LLaMA-2.7B 참조 모델 로더

구조적 가지치기(Structured Pruning)로 경량화된 LLaMA 모델 로더.
Sheared-LLaMA는 원본 LLaMA-2-7B의 2.7B 파라미터 버전으로,
동일한 SentencePiece tokenizer.model 사용.

주요 특징:
- 2.7B 파라미터로 빠른 실험 가능
- 원본 대비 60% 크기로 메모리 효율적
- WMTP Rho1 알고리즘의 참조 모델로 활용
- Facebook MTP와 동일한 토크나이저로 완벽 호환
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.components.loader.base_loader import ModelLoader
from src.components.registry import loader_registry


@loader_registry.register(
    "sheared-llama",
    category="loader",
    version="1.0.0",
    description="Princeton Sheared-LLaMA-2.7B reference model loader",
)
class ShearedLLaMALoader(ModelLoader):
    """
    Sheared-LLaMA 참조 모델 로더

    WMTP Rho1-WMTP 알고리즘의 참조 모델:
    - Reference model과 base model의 CE 차이 계산
    - |CE^ref_t - CE^base_t|로 어려운 토큰 식별
    - Facebook MTP와 동일한 SentencePiece 토크나이저 사용

    모델 구조:
    - Base: LLaMA-2-7B를 2.7B로 가지치기
    - Method: Structured pruning (구조적 가지치기)
    - Context: 4096 토큰 (원본과 동일)
    - Tokenizer: SentencePiece (SHA256: 9e556afd...)

    백업 옵션:
    더 강한 참조 모델이 필요한 경우 사용
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

        # 양자화 옵션 (작은 모델이므로 기본적으로 비활성화)
        self.use_4bit = config.get("use_4bit", False) if config else False
        self.use_8bit = config.get("use_8bit", False) if config else False

        # 모델 ID
        self.model_id = "princeton-nlp/Sheared-LLaMA-2.7B"

    def load_model(self, path: str, **kwargs) -> Any:
        """
        Sheared-LLaMA 모델 로드

        2.7B 경량 모델로 빠른 추론과 실험 지원:
        - 메모리 사용량: ~11GB (FP32) / ~5.5GB (BF16)
        - 추론 속도: 7B 모델 대비 2.5x 빠름
        - 성능: 많은 벤치마크에서 원본 7B의 95% 성능 유지

        Args:
            path: 모델 경로 또는 캐시 디렉토리
            **kwargs: 추가 옵션
                - force_download: 강제 다운로드 여부
                - local_files_only: 로컬 파일만 사용
                - revision: 모델 버전/브랜치

        Returns:
            로드된 Sheared-LLaMA 모델

        Raises:
            ImportError: transformers 패키지 미설치
            RuntimeError: 모델 로드 실패
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers 패키지가 필요합니다. " "설치: uv pip install transformers"
            )

        # 로컬 경로 확인
        local_path = Path(path)
        cache_dir = str(local_path) if local_path.exists() else None

        # 양자화 설정 (작은 모델이므로 선택적)
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
                print("📦 4비트 양자화 활성화 (메모리 ~1.4GB)")
            elif self.use_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                print("📦 8비트 양자화 활성화 (메모리 ~2.7GB)")

        # 디바이스 설정
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        elif self.device != "cpu":
            model_kwargs["device_map"] = {"": self.device}

        # 모델 로드
        try:
            print(f"🚀 Sheared-LLaMA 모델 로드 중: {self.model_id}")

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16
                if not (self.use_4bit or self.use_8bit)
                else None,
                trust_remote_code=False,  # 공식 transformers 코드만 사용
                **model_kwargs,
                **kwargs,
            )

            # 평가 모드 설정
            model.eval()

            # 모델 정보 출력
            param_count = sum(p.numel() for p in model.parameters())
            print("✅ Sheared-LLaMA 로드 완료")
            print(f"   - 파라미터: {param_count:,} ({param_count / 1e9:.1f}B)")
            print(f"   - 디바이스: {next(model.parameters()).device}")
            print("   - 컨텍스트 길이: 4096 토큰")
            print("   - 용도: WMTP Rho1 참조 모델")

            return model

        except Exception as e:
            raise RuntimeError(
                f"Sheared-LLaMA 모델 로드 실패: {self.model_id}\n" f"에러: {e}"
            )

    # load_tokenizer는 부모 클래스의 통합 SentencePiece 메서드 상속
    # Sheared-LLaMA는 원본 LLaMA-2와 동일한 tokenizer.model 사용
