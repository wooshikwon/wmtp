"""
HuggingFace 호환 SentencePiece 토크나이저

SentencePieceTokenizer를 내부적으로 사용하면서 HuggingFace 인터페이스를 제공하는
ComponentFactory 패턴 호환 토크나이저입니다.

주요 기능:
- ComponentFactory Registry 등록 지원
- HuggingFace 스타일 __call__ 메서드 제공
- Dataset 토크나이징 유틸리티 메서드
- 완전한 토큰 정보 반환 (input_ids, attention_mask, labels)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

from datasets import Dataset

from src.components.base import BaseComponent
from src.components.registry import tokenizer_registry

logger = logging.getLogger(__name__)


@tokenizer_registry.register("hf-sentencepiece", category="tokenizer", version="1.0.0")
@tokenizer_registry.register("hf", category="tokenizer", version="1.0.0")
@tokenizer_registry.register("huggingface", category="tokenizer", version="1.0.0")
class HfSentencePieceTokenizer(BaseComponent):
    """ComponentFactory 패턴 호환 HuggingFace 스타일 SentencePiece 토크나이저

    SentencePieceTokenizer를 내부적으로 사용하면서 HuggingFace 호환 인터페이스를
    제공하는 Registry 등록 가능한 컴포넌트입니다.

    특징:
    - BaseComponent 상속으로 ComponentFactory 패턴 완전 지원
    - tokenizer(text, truncation=True, max_length=512) 호출 지원
    - {"input_ids": [...], "attention_mask": [...]} 형태 반환
    - Dataset.map()과 호환되는 tokenize_dataset() 메서드 제공
    - Registry에서 직접 생성 가능
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """ComponentFactory 패턴 호환 초기화

        Args:
            config: 컴포넌트 설정 (tokenizer_path 등)
        """
        super().__init__(config)
        self.sentence_piece_tokenizer = None
        self.sp = None
        self.vocab_size = None
        self.bos_id = None
        self.eos_id = None
        self.pad_id = None
        self.unk_id = None

    def setup(self, ctx: dict[str, Any]) -> None:
        """컴포넌트 초기화 - SentencePieceTokenizer 생성 및 설정

        Args:
            ctx: 실행 컨텍스트 (tokenizer_path 등)
        """
        super().setup(ctx)

        # 내부적으로 SentencePieceTokenizer 사용
        from .sentencepiece_tokenizer import SentencePieceTokenizer

        self.sentence_piece_tokenizer = SentencePieceTokenizer(self.config)
        self.sentence_piece_tokenizer.setup(ctx)

        # SentencePiece 토크나이저 실행하여 processor 얻기
        result = self.sentence_piece_tokenizer.run({})
        self.sp = result["tokenizer"]
        self.vocab_size = self.sp.get_piece_size()

        # 특수 토큰 ID 저장
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() != -1 else self.sp.unk_id()
        self.unk_id = self.sp.unk_id()

        logger.info(
            f"HfSentencePieceTokenizer 초기화됨:\n"
            f"  - 어휘 크기: {self.vocab_size}\n"
            f"  - BOS ID: {self.bos_id}\n"
            f"  - EOS ID: {self.eos_id}\n"
            f"  - PAD ID: {self.pad_id}\n"
            f"  - UNK ID: {self.unk_id}"
        )

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """ComponentFactory 패턴 실행 메서드

        Args:
            ctx: 실행 컨텍스트

        Returns:
            토크나이저 정보가 포함된 딕셔너리
        """
        self.validate_initialized()

        return {
            "tokenizer": self,  # HF 호환 인터페이스를 가진 자기 자신 반환
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "bos_id": self.bos_id,
                "eos_id": self.eos_id,
                "pad_id": self.pad_id,
                "unk_id": self.unk_id,
            },
            "source": "Registry/HfSentencePieceTokenizer",
        }

    def __call__(
        self,
        text: Union[str, List[str]],
        truncation: bool = True,
        max_length: int = 512,
        padding: Union[bool, str] = False,
        return_attention_mask: bool = True,
        return_tensors: str = None,
        **kwargs
    ) -> Dict[str, List[int]]:
        """HuggingFace 스타일 토크나이징 메서드

        기존 파이프라인에서 사용하던 tokenizer() 호출과 완전 호환되도록
        동일한 인터페이스와 반환값을 제공합니다.

        Args:
            text: 토크나이징할 텍스트 (단일 또는 리스트)
            truncation: 최대 길이 초과시 자를지 여부
            max_length: 최대 토큰 길이
            padding: 패딩 방식 (현재는 False만 지원)
            return_attention_mask: attention_mask 반환 여부
            return_tensors: 텐서 형태 반환 (현재는 None만 지원)
            **kwargs: 추가 파라미터

        Returns:
            HuggingFace 형태 딕셔너리:
            {
                "input_ids": [int, ...],
                "attention_mask": [int, ...] (옵션)
            }
        """
        # 단일 텍스트 처리
        if isinstance(text, str):
            return self._encode_single(
                text, truncation, max_length, return_attention_mask
            )

        # 리스트 텍스트 처리 (배치)
        results = []
        for single_text in text:
            result = self._encode_single(
                single_text, truncation, max_length, return_attention_mask
            )
            results.append(result)

        # 배치 결과 병합
        return self._merge_batch_results(results)

    def _encode_single(
        self,
        text: str,
        truncation: bool,
        max_length: int,
        return_attention_mask: bool
    ) -> Dict[str, List[int]]:
        """단일 텍스트 인코딩"""
        # SentencePiece로 토큰화
        tokens = self.sp.encode(text, out_type=int)

        # Truncation 처리
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]

        # 결과 구성
        result = {"input_ids": tokens}

        # Attention mask 생성 (모든 토큰이 실제 토큰이므로 1)
        if return_attention_mask:
            result["attention_mask"] = [1] * len(tokens)

        return result

    def _merge_batch_results(self, results: List[Dict[str, List[int]]]) -> Dict[str, List[List[int]]]:
        """배치 결과 병합"""
        merged = {}
        for key in results[0].keys():
            merged[key] = [result[key] for result in results]
        return merged

    def tokenize_dataset(
        self,
        dataset: Dataset,
        max_length: int,
        text_column: str = None,
        remove_columns: List[str] = None,
        **kwargs
    ) -> Dataset:
        """Dataset 전체를 토크나이징하는 유틸리티 메서드

        기존 파이프라인의 중첩 함수를 대체하는 깔끔한 메서드입니다.
        Dataset.map()과 함께 사용되어 전체 데이터셋을 효율적으로 처리합니다.

        Args:
            dataset: 토크나이징할 Dataset
            max_length: 최대 토큰 길이
            text_column: 텍스트가 들어있는 컬럼명 (자동 감지)
            remove_columns: 제거할 컬럼들 (자동으로 원본 텍스트 컬럼 제거)
            **kwargs: Dataset.map() 추가 파라미터

        Returns:
            토크나이징된 Dataset
        """
        def tokenize_function(example: Dict[str, Any]) -> Dict[str, Any]:
            """개별 샘플 토크나이징 함수"""
            # 텍스트 추출 - 다양한 데이터셋 형식 지원
            if text_column:
                text = example.get(text_column, "")
            else:
                # 자동 감지: 일반적인 텍스트 컬럼명들 시도
                text = (
                    example.get("full_text") or
                    example.get("prompt") or
                    example.get("text") or
                    example.get("content") or
                    ""
                )

            # HuggingFace 스타일로 토크나이징
            tokenized = self(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True
            )

            # 언어모델용 labels 생성 (input_ids와 동일)
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Dataset 토크나이징 실행
        if remove_columns is None:
            remove_columns = dataset.column_names

        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=remove_columns,
            desc="HF호환 토크나이저로 데이터셋 토크나이징",
            load_from_cache_file=kwargs.get("load_from_cache_file", True),
            **{k: v for k, v in kwargs.items() if k != "load_from_cache_file"}
        )

        logger.info(
            f"데이터셋 토크나이징 완료:\n"
            f"  - 원본 샘플 수: {len(dataset)}\n"
            f"  - 토크나이징된 샘플 수: {len(tokenized_dataset)}\n"
            f"  - 최대 길이: {max_length}\n"
            f"  - 컬럼: {tokenized_dataset.column_names}"
        )

        return tokenized_dataset

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """토큰 ID를 텍스트로 디코딩"""
        return self.sp.decode(token_ids)

    def batch_decode(self, batch_token_ids: List[List[int]], **kwargs) -> List[str]:
        """배치 토큰 ID들을 텍스트 리스트로 디코딩"""
        return [self.decode(token_ids, **kwargs) for token_ids in batch_token_ids]

    # HuggingFace 토크나이저 호환 속성들
    @property
    def vocab(self):
        """어휘 사전 (완전한 구현은 필요시 추가)"""
        return None

    @property
    def model_max_length(self) -> int:
        """모델 최대 길이 (기본값)"""
        return 512

    def __repr__(self) -> str:
        return (
            f"HfSentencePieceTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"model_max_length={self.model_max_length})"
        )