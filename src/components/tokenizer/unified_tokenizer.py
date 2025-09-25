"""
WMTP 통합 SentencePiece 토크나이저

모든 WMTP 모델(Facebook MTP, Starling-RM, Sheared-LLaMA)이
동일한 Llama-2 기반 SentencePiece tokenizer.model 사용
SHA256: 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347

싱글톤 패턴으로 메모리 효율성 극대화
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class UnifiedSentencePieceTokenizer:
    """
    단일 SentencePiece 토크나이저 관리

    모든 WMTP 모델이 동일한 토크나이저를 공유하여:
    - 메모리 사용량 최소화
    - 토큰화 일관성 보장
    - 로딩 속도 향상
    """

    _instance: Optional["UnifiedSentencePieceTokenizer"] = None
    _tokenizer: Any | None = None
    _tokenizer_path: Path | None = None

    def __new__(cls):
        """싱글톤 인스턴스 보장"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_tokenizer(cls, tokenizer_path: Path | None = None) -> Any:
        """
        통합 토크나이저 인스턴스 반환

        Args:
            tokenizer_path: tokenizer.model 파일 경로
                           None이면 기본 경로들을 순차 탐색

        Returns:
            SentencePieceProcessor 인스턴스

        Raises:
            ImportError: sentencepiece 패키지 미설치
            FileNotFoundError: tokenizer.model 파일 없음
        """
        # 이미 로드된 토크나이저가 있으면 재사용
        if cls._tokenizer is not None:
            logger.debug(f"토크나이저 재사용: {cls._tokenizer_path}")
            return cls._tokenizer

        # SentencePiece import
        try:
            from sentencepiece import SentencePieceProcessor
        except ImportError:
            raise ImportError(
                "sentencepiece 패키지가 필요합니다. "
                "설치: uv pip install sentencepiece"
            )

        # 토크나이저 파일 경로 탐색
        if tokenizer_path is None:
            # 기본 경로들 순차 확인
            default_paths = [
                Path("models/tokenizer.model"),  # 프로젝트 루트 기준
                Path("models/7b_1t_4/tokenizer.model"),  # Facebook MTP
                Path(".cache/tokenizer.model"),  # 캐시 디렉토리
            ]

            for path in default_paths:
                if path.exists():
                    tokenizer_path = path
                    logger.info(f"기본 경로에서 토크나이저 발견: {path}")
                    break

            if tokenizer_path is None:
                raise FileNotFoundError(
                    f"tokenizer.model 파일을 찾을 수 없습니다. "
                    f"확인한 경로: {[str(p) for p in default_paths]}"
                )

        # 경로 유효성 확인
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"토크나이저 파일이 존재하지 않습니다: {tokenizer_path}"
            )

        # SentencePiece 토크나이저 로드
        try:
            tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
            cls._tokenizer = tokenizer
            cls._tokenizer_path = tokenizer_path

            logger.info(
                f"✅ SentencePiece 토크나이저 로드 성공: {tokenizer_path}\n"
                f"   어휘 크기: {tokenizer.get_piece_size()}\n"
                f"   BOS ID: {tokenizer.bos_id()}, EOS ID: {tokenizer.eos_id()}"
            )

            return tokenizer

        except Exception as e:
            raise RuntimeError(f"토크나이저 로드 실패: {tokenizer_path}\n" f"에러: {e}")

    @classmethod
    def reset(cls):
        """
        토크나이저 인스턴스 초기화 (주로 테스트용)
        """
        cls._tokenizer = None
        cls._tokenizer_path = None
        logger.debug("토크나이저 인스턴스 초기화됨")

    @classmethod
    def is_loaded(cls) -> bool:
        """토크나이저 로드 여부 확인"""
        return cls._tokenizer is not None

    @classmethod
    def get_vocab_size(cls) -> int:
        """어휘 크기 반환"""
        if cls._tokenizer is None:
            raise RuntimeError("토크나이저가 로드되지 않았습니다")
        return cls._tokenizer.get_piece_size()

    @classmethod
    def get_special_tokens(cls) -> dict:
        """특수 토큰 ID 반환"""
        if cls._tokenizer is None:
            raise RuntimeError("토크나이저가 로드되지 않았습니다")

        return {
            "bos_id": cls._tokenizer.bos_id(),
            "eos_id": cls._tokenizer.eos_id(),
            "pad_id": cls._tokenizer.pad_id(),
            "unk_id": cls._tokenizer.unk_id(),
        }


# 편의를 위한 전역 함수
def get_unified_tokenizer(tokenizer_path: Path | None = None) -> Any:
    """
    통합 토크나이저 인스턴스 반환 (간편 함수)

    Args:
        tokenizer_path: tokenizer.model 파일 경로

    Returns:
        SentencePieceProcessor 인스턴스
    """
    return UnifiedSentencePieceTokenizer.get_tokenizer(tokenizer_path)
