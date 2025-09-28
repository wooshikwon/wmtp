"""
HuggingFace Transformers 기반 통합 토크나이저 컴포넌트

모델별 내장 토크나이저를 HuggingFace transformers 라이브러리에서 직접 로드하여
SentencePiece 의존성 없이 다양한 모델을 지원합니다.

핵심 특징:
- AutoTokenizer를 통한 자동 토크나이저 감지
- 모델별 최적화된 토크나이저 사용 (GPT-2, LLAMA, BERT 등)
- 싱글톤 패턴으로 메모리 효율성 극대화
- Registry 시스템과 완벽한 통합
- HuggingFace Hub 캐싱 지원
"""

import logging
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer

from datasets import Dataset

from ..base import BaseComponent
from ..registry import tokenizer_registry

logger = logging.getLogger(__name__)


@tokenizer_registry.register("hf-transformers", category="tokenizer", version="1.0.0")
class HfTransformersTokenizer(BaseComponent):
    """
    HuggingFace Transformers 기반 통합 토크나이저.

    모델별 내장 토크나이저를 사용하여:
    - 완벽한 모델-토크나이저 호환성 보장
    - 다양한 토크나이저 타입 지원 (BPE, WordPiece, SentencePiece)
    - HuggingFace 생태계와 완전한 통합
    """

    # 클래스 레벨 싱글톤 변수 (모델별 캐싱)
    _tokenizers: dict[str, PreTrainedTokenizer] = {}
    _instances: dict[str, "HfTransformersTokenizer"] = {}

    def __new__(cls, config: dict[str, Any], *args, **kwargs):
        """모델별 싱글톤 패턴 구현"""
        # 모델 ID 추출
        model_id = cls._extract_model_id(config)

        # 모델별 인스턴스 재사용
        if model_id not in cls._instances:
            cls._instances[model_id] = super().__new__(cls)
        return cls._instances[model_id]

    def __init__(self, config: dict[str, Any]):
        """
        초기화.

        Args:
            config: 모델 ID, 패딩 설정 등 구성
        """
        # 이미 초기화되었다면 skip
        if hasattr(self, "_initialized"):
            return

        super().__init__(config)

        # 모델 ID 추출 및 저장
        self.model_id = self._extract_model_id(config)

        # 토크나이저 설정
        self.pad_side = config.get("tokenizer_pad_side", "left")  # 생성 모델 기본값
        self.padding = config.get("padding", False)
        self.truncation = config.get("truncation", True)
        self.max_length = config.get("max_length", 512)

        # HuggingFace Hub 캐시 디렉토리 (선택적)
        self.cache_dir = os.getenv("HF_HOME", None)

        self._initialized = True
        logger.debug(f"HfTransformersTokenizer 인스턴스 생성: {self.model_id}")

    @classmethod
    def _extract_model_id(cls, config: dict[str, Any]) -> str:
        """설정에서 모델 ID 추출"""
        # 직접 지정된 model_id
        if "model_id" in config:
            return config["model_id"]

        # base_id에서 추출
        if "base_id" in config:
            return config["base_id"]

        # 경로에서 추출
        if "model_path" in config:
            path = Path(config["model_path"])
            # 경로의 마지막 부분이 모델 이름인 경우
            if path.name in ["distilgpt2", "gpt2", "llama", "bert"]:
                return path.name
            return str(path)

        # 기본값
        return "gpt2"

    def setup(self, ctx: dict[str, Any]) -> None:
        """
        토크나이저 초기화 (실제 로딩).

        Args:
            ctx: 실행 컨텍스트 (config, recipe 포함)
        """
        # BaseComponent의 기본 setup 호출
        super().setup(ctx)

        # 컨텍스트에서 추가 설정 추출
        self._update_from_context(ctx)

        # 토크나이저 로드 보장
        self._ensure_tokenizer_loaded()

    def _update_from_context(self, ctx: dict[str, Any]) -> None:
        """실행 컨텍스트에서 설정 업데이트"""
        config = ctx.get("config", {})
        # Phase 3: recipe.model 제거됨 - config에서만 모델 정보 추출

        # Config에서 모델 경로 추출
        if config and hasattr(config, "paths"):
            if hasattr(config.paths, "models"):
                model_path = Path(config.paths.models.base)
                # 경로가 실제 모델 디렉토리인 경우
                if (model_path / "tokenizer_config.json").exists():
                    self.model_id = str(model_path)
                # 경로에서 모델 이름 추출
                elif model_path.name:
                    self.model_id = model_path.name

        logger.debug(
            f"토크나이저 설정 업데이트: model_id={self.model_id}, pad_side={self.pad_side}"
        )

    def _ensure_tokenizer_loaded(self) -> None:
        """
        HuggingFace 토크나이저 로드 보장 (싱글톤).

        모델별로 한 번만 로드되며, 이후 호출에서는 캐시된 인스턴스 재사용.
        """
        # 이미 로드되었으면 skip
        if self.model_id in HfTransformersTokenizer._tokenizers:
            logger.debug(f"토크나이저 재사용 (이미 로드됨): {self.model_id}")
            return

        # 로컬 모델 디렉토리 체크
        local_path = Path(self.model_id)
        if local_path.exists() and (local_path / "tokenizer_config.json").exists():
            try:
                logger.info(f"로컬 모델 디렉토리에서 토크나이저 로드: {local_path}")
                tokenizer = AutoTokenizer.from_pretrained(
                    str(local_path),
                    use_fast=True,  # Rust 기반 빠른 토크나이저 사용
                    trust_remote_code=False,  # 보안: 원격 코드 실행 방지
                    cache_dir=self.cache_dir,
                    local_files_only=True,
                )
            except Exception as e:
                logger.warning(f"로컬 로드 실패, HuggingFace Hub 시도: {e}")
                tokenizer = None
        else:
            tokenizer = None

        # HuggingFace Hub에서 로드
        if tokenizer is None:
            try:
                logger.info(f"HuggingFace Hub에서 토크나이저 다운로드: {self.model_id}")
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    use_fast=True,
                    trust_remote_code=False,
                    cache_dir=self.cache_dir,
                )
            except Exception as e:
                # 최종 폴백: 기본 GPT-2 토크나이저
                if self.model_id != "gpt2":
                    logger.warning(f"{self.model_id} 로드 실패, GPT-2 폴백: {e}")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            "gpt2",
                            use_fast=True,
                            trust_remote_code=False,
                            cache_dir=self.cache_dir,
                        )
                        logger.warning("⚠️ GPT-2 토크나이저로 폴백 (호환성 주의)")
                    except Exception as e2:
                        raise RuntimeError(
                            f"토크나이저 로드 완전 실패: {self.model_id}\n"
                            f"원인: {e}\n"
                            f"GPT-2 폴백도 실패: {e2}\n"
                            "해결 방법:\n"
                            "1. 인터넷 연결 확인 (HuggingFace Hub 접근)\n"
                            "2. 모델 ID 확인 (예: 'distilgpt2', 'gpt2')\n"
                            "3. transformers 업데이트: pip install -U transformers"
                        )
                else:
                    raise RuntimeError(
                        f"토크나이저 로드 실패: {self.model_id}\n"
                        f"원인: {e}\n"
                        "해결 방법:\n"
                        "1. 인터넷 연결 확인\n"
                        "2. HuggingFace Hub 접근 확인\n"
                        "3. 캐시 디렉토리 권한 확인"
                    )

        # 토크나이저 설정
        self._configure_tokenizer(tokenizer)

        # 캐시에 저장
        HfTransformersTokenizer._tokenizers[self.model_id] = tokenizer

        logger.info(
            f"✅ HuggingFace 토크나이저 로드 성공: {self.model_id}\n"
            f"   어휘 크기: {tokenizer.vocab_size}\n"
            f"   패딩 방향: {tokenizer.padding_side}\n"
            f"   특수 토큰: bos={tokenizer.bos_token}, eos={tokenizer.eos_token}, pad={tokenizer.pad_token}"
        )

    def _configure_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """토크나이저 설정 적용"""
        # 패딩 방향 설정
        tokenizer.padding_side = self.pad_side

        # 패드 토큰 설정 (GPT 모델들은 기본적으로 pad_token이 없음)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.debug(f"패드 토큰을 EOS 토큰으로 설정: {tokenizer.eos_token}")
            else:
                # 새로운 패드 토큰 추가
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                logger.debug("새로운 [PAD] 토큰 추가")

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        토크나이저 및 관련 정보 반환.

        Args:
            ctx: 실행 컨텍스트

        Returns:
            토크나이저, 어휘 크기, 특수 토큰 정보
        """
        self.validate_initialized()

        if self.model_id not in HfTransformersTokenizer._tokenizers:
            self._ensure_tokenizer_loaded()

        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        if tokenizer is None:
            raise RuntimeError(f"토크나이저가 초기화되지 않았습니다: {self.model_id}")

        return {
            "tokenizer": self,  # HuggingFace 호환 인터페이스 제공
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": {
                "bos_id": tokenizer.bos_token_id,
                "eos_id": tokenizer.eos_token_id,
                "pad_id": tokenizer.pad_token_id,
                "unk_id": tokenizer.unk_token_id,
            },
            "source": "local" if Path(self.model_id).exists() else "huggingface_hub",
            "model_id": self.model_id,
        }

    def __call__(
        self,
        text: str | list[str],
        truncation: bool | None = None,
        max_length: int | None = None,
        padding: bool | str = None,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        HuggingFace 호환 토크나이징 인터페이스.

        Args:
            text: 입력 텍스트 또는 텍스트 리스트
            truncation: max_length로 자르기 여부
            max_length: 최대 시퀀스 길이
            padding: 패딩 전략 ('max_length', True, False)
            return_attention_mask: attention mask 반환 여부
            return_tensors: 반환 타입 ('pt' for PyTorch, 'np' for numpy, None for list)
            **kwargs: 추가 토크나이저 인자

        Returns:
            토크나이징 결과 딕셔너리
        """
        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        if tokenizer is None:
            raise RuntimeError(f"토크나이저가 초기화되지 않았습니다: {self.model_id}")

        # 제공된 값 또는 기본값 사용
        truncation = truncation if truncation is not None else self.truncation
        max_length = max_length if max_length is not None else self.max_length
        padding = padding if padding is not None else self.padding

        # padding=True를 'max_length'로 변환 (일관성)
        if padding is True:
            padding = "max_length"

        return tokenizer(
            text,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
            **kwargs,
        )

    def tokenize_dataset(
        self,
        dataset: Dataset,
        max_length: int,
        text_column: str | None = None,
        remove_columns: list[str] | None = None,
        load_from_cache_file: bool = True,
        num_proc: int | None = None,
        **kwargs,
    ) -> Dataset:
        """
        HuggingFace Dataset 토크나이징.

        Args:
            dataset: 입력 데이터셋
            max_length: 최대 시퀀스 길이
            text_column: 토크나이징할 텍스트 컬럼명
            remove_columns: 토크나이징 후 제거할 컬럼들
            load_from_cache_file: 캐시된 토크나이징 사용 여부
            num_proc: 병렬 처리 프로세스 수
            **kwargs: 추가 인자

        Returns:
            토크나이징된 데이터셋
        """
        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        if tokenizer is None:
            raise RuntimeError(f"토크나이저가 초기화되지 않았습니다: {self.model_id}")

        def tokenize_function(examples):
            # 텍스트 컬럼 결정
            if text_column:
                texts = examples[text_column]
            else:
                # 일반적인 컬럼명들 시도
                for col in ["text", "content", "prompt", "full_text", "input"]:
                    if col in examples:
                        texts = examples[col]
                        break
                else:
                    raise ValueError(
                        f"텍스트 컬럼을 찾을 수 없습니다. 가능한 컬럼: {examples.keys()}"
                    )

            # 토크나이징
            tokenized = self(
                texts,
                truncation=True,
                max_length=max_length,
                padding=False,  # 데이터셋 토크나이징 시에는 패딩 안함
                return_attention_mask=True,
            )

            # 언어 모델링용 레이블 추가 (input_ids의 복사본)
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # 제거할 컬럼 결정
        if remove_columns is None:
            remove_columns = dataset.column_names

        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_columns,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
            desc=f"HuggingFace 토크나이징 ({self.model_id})",
        )

    # 일반적인 토크나이저 속성들 위임
    @property
    def vocab_size(self) -> int:
        """어휘 크기"""
        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        return tokenizer.vocab_size if tokenizer else 0

    @property
    def pad_token_id(self) -> int:
        """패드 토큰 ID"""
        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        return tokenizer.pad_token_id if tokenizer else 0

    @property
    def eos_token_id(self) -> int:
        """EOS 토큰 ID"""
        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        return tokenizer.eos_token_id if tokenizer else 0

    @property
    def bos_token_id(self) -> int:
        """BOS 토큰 ID"""
        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        return tokenizer.bos_token_id if tokenizer else 0

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """토큰 ID를 텍스트로 디코딩"""
        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        if tokenizer is None:
            raise RuntimeError(f"토크나이저가 초기화되지 않았습니다: {self.model_id}")
        return tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, sequences: list[list[int]], **kwargs) -> list[str]:
        """토큰 ID 시퀀스 배치를 텍스트로 디코딩"""
        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        if tokenizer is None:
            raise RuntimeError(f"토크나이저가 초기화되지 않았습니다: {self.model_id}")
        return tokenizer.batch_decode(sequences, **kwargs)

    def get_hf_tokenizer(self) -> PreTrainedTokenizer:
        """DataCollator에서 사용할 순수 HuggingFace tokenizer 반환.

        Pipeline의 복잡한 추출 로직을 대체하는 명확한 인터페이스.
        DataCollatorForLanguageModeling이 기대하는 PreTrainedTokenizer 인스턴스를 직접 제공.

        Returns:
            PreTrainedTokenizer: 순수 HuggingFace tokenizer 인스턴스

        Raises:
            RuntimeError: tokenizer가 초기화되지 않은 경우
        """
        if self.model_id not in HfTransformersTokenizer._tokenizers:
            self._ensure_tokenizer_loaded()

        tokenizer = HfTransformersTokenizer._tokenizers.get(self.model_id)
        if tokenizer is None:
            raise RuntimeError(f"토크나이저가 초기화되지 않았습니다: {self.model_id}")

        return tokenizer

    @classmethod
    def reset(cls, model_id: str | None = None):
        """
        토크나이저 인스턴스 초기화 (주로 테스트용).

        Args:
            model_id: 특정 모델만 초기화. None이면 전체 초기화.
        """
        if model_id:
            cls._tokenizers.pop(model_id, None)
            cls._instances.pop(model_id, None)
            logger.debug(f"토크나이저 초기화: {model_id}")
        else:
            cls._tokenizers.clear()
            cls._instances.clear()
            logger.debug("모든 토크나이저 싱글톤 초기화됨")
