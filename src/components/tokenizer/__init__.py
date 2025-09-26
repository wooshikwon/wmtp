"""
WMTP 듀얼 토크나이저 모듈

두 가지 토크나이저를 Registry에 등록하여 용도에 따라 선택적 사용:
1. SentencePieceTokenizer: Raw SentencePiece 인터페이스 (내부 사용)
2. HfSentencePieceTokenizer: HuggingFace 호환 인터페이스 (파이프라인 사용)

모든 WMTP 모델이 동일한 SentencePiece tokenizer.model을 사용하도록 통합 관리
S3에서 직접 메모리로 로드하여 VESSL 환경 완벽 지원
"""

from .sentencepiece_tokenizer import SentencePieceTokenizer  # Raw SentencePiece 인터페이스
from .hf_sentencepiece_tokenizer import HfSentencePieceTokenizer  # HuggingFace 호환 SentencePiece wrapper
from .hf_transformers_tokenizer import HfTransformersTokenizer  # Pure HuggingFace transformers tokenizer

__all__ = ["SentencePieceTokenizer", "HfSentencePieceTokenizer", "HfTransformersTokenizer"]
