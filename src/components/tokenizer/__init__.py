"""
WMTP S3 기반 통합 토크나이저 모듈

모든 WMTP 모델이 동일한 SentencePiece tokenizer.model을 사용하도록 통합 관리
S3에서 직접 메모리로 로드하여 VESSL 환경 완벽 지원
"""

from .sentence_piece import SentencePieceTokenizer  # S3 기반 통합 토크나이저

__all__ = ["SentencePieceTokenizer"]
