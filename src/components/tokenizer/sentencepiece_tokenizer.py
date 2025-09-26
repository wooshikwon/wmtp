"""
S3 기반 통합 SentencePiece 토크나이저 컴포넌트

이 모듈은 S3에서 직접 tokenizer.model을 메모리로 로드하여
파일시스템 의존성을 완전히 제거합니다. VESSL 환경에 최적화되었으며,
로컬 개발을 위한 폴백 옵션도 제공합니다.

핵심 특징:
- S3에서 메모리로 직접 로드 (파일시스템 미사용)
- 싱글톤 패턴으로 메모리 효율성 극대화
- Registry 시스템과 완벽한 통합
- 환경변수 기반 로컬 개발 지원
"""

import logging
import os
from pathlib import Path
from typing import Any

from ...utils.s3 import create_s3_manager
from ..base import BaseComponent
from ..registry import tokenizer_registry

logger = logging.getLogger(__name__)


@tokenizer_registry.register("sentencepiece", category="tokenizer", version="2.0.0")
class SentencePieceTokenizer(BaseComponent):
    """
    S3 기반 통합 SentencePiece 토크나이저.

    모든 WMTP 모델이 동일한 tokenizer.model을 공유하여:
    - 메모리 사용량 최소화 (싱글톤 패턴)
    - 토큰화 일관성 보장
    - S3에서 직접 로드로 VESSL 호환성 확보
    """

    # 클래스 레벨 싱글톤 변수
    _processor = None
    _instance = None

    # S3 경로 상수
    S3_TOKENIZER_PATH = "models/shared/tokenizer.model"

    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: dict[str, Any]):
        """
        초기화.

        Args:
            config: S3 설정 및 기타 구성
        """
        # 이미 초기화되었다면 skip
        if hasattr(self, "_initialized"):
            return

        super().__init__(config)

        # S3 Manager 생성 또는 전달받기
        self.s3_manager = config.get("s3_manager") or create_s3_manager(config)

        # 로컬 개발 환경 경로 (선택적)
        self.local_path = os.getenv("TOKENIZER_MODEL_PATH")

        self._initialized = True
        logger.debug("SentencePieceTokenizer 인스턴스 생성")

    def setup(self, ctx: dict[str, Any]) -> None:
        """
        토크나이저 초기화 (실제 로딩).

        Args:
            ctx: 실행 컨텍스트
        """
        # BaseComponent의 기본 setup 호출하여 initialized = True 설정
        super().setup(ctx)
        self._ensure_processor_loaded()

    def _load_from_s3(self) -> bytes:
        """
        S3에서 tokenizer.model을 바이트로 다운로드.

        Returns:
            모델 파일의 바이트 데이터

        Raises:
            RuntimeError: S3 연결 실패 또는 다운로드 실패
        """
        if not self.s3_manager or not self.s3_manager.connected:
            raise RuntimeError("S3 연결이 필요합니다. AWS 자격증명을 확인하세요.")

        try:
            logger.info(f"S3에서 토크나이저 다운로드 시작: {self.S3_TOKENIZER_PATH}")
            model_bytes = self.s3_manager.download_to_bytes(self.S3_TOKENIZER_PATH)
            logger.info(f"✅ S3 다운로드 완료: {len(model_bytes)} bytes")
            return model_bytes

        except FileNotFoundError:
            raise FileNotFoundError(
                f"S3에서 토크나이저 모델을 찾을 수 없습니다: {self.S3_TOKENIZER_PATH}\n"
                f"S3 버킷: {self.s3_manager.bucket}\n"
                f"먼저 다음 명령으로 업로드하세요:\n"
                f"  aws s3 cp models/7b_1t_4/tokenizer.model "
                f"s3://{self.s3_manager.bucket}/{self.S3_TOKENIZER_PATH}"
            )
        except Exception as e:
            raise RuntimeError(f"S3 다운로드 실패: {e}")

    def _ensure_processor_loaded(self) -> None:
        """
        SentencePieceProcessor 로드 보장 (싱글톤).

        한 번만 로드되며, 이후 호출에서는 캐시된 인스턴스 재사용.
        """
        # 이미 로드되었으면 skip
        if SentencePieceTokenizer._processor is not None:
            logger.debug("토크나이저 재사용 (이미 로드됨)")
            return

        # SentencePiece import
        try:
            from sentencepiece import SentencePieceProcessor
        except ImportError:
            raise ImportError(
                "sentencepiece 패키지가 필요합니다.\n"
                "설치: uv pip install sentencepiece"
            )

        # S3에서 로드 시도
        if self.s3_manager and self.s3_manager.connected:
            try:
                # S3에서 바이트로 다운로드
                model_bytes = self._load_from_s3()

                # SentencePiece 초기화 (메모리에서 직접)
                sp = SentencePieceProcessor()
                sp.LoadFromSerializedProto(model_bytes)

                SentencePieceTokenizer._processor = sp
                logger.info(
                    f"✅ S3에서 토크나이저 로드 성공\n"
                    f"   어휘 크기: {sp.get_piece_size()}\n"
                    f"   BOS ID: {sp.bos_id()}, EOS ID: {sp.eos_id()}"
                )
                return

            except Exception as e:
                logger.warning(f"S3 로드 실패, 로컬 폴백 시도: {e}")

        # 로컬 개발 환경 폴백
        if self.local_path and Path(self.local_path).exists():
            try:
                sp = SentencePieceProcessor(model_file=self.local_path)
                SentencePieceTokenizer._processor = sp
                logger.info(
                    f"✅ 로컬 경로에서 토크나이저 로드: {self.local_path}\n"
                    f"   어휘 크기: {sp.get_piece_size()}"
                )
                return

            except Exception as e:
                logger.error(f"로컬 로드 실패: {e}")

        # 기본 경로들 탐색 (캐시 제거)
        default_paths = [
            Path("models/7b_1t_4/tokenizer.model"),
            Path("models/tokenizer.model"),
        ]

        for path in default_paths:
            if path.exists():
                try:
                    sp = SentencePieceProcessor(model_file=str(path))
                    SentencePieceTokenizer._processor = sp
                    logger.info(
                        f"✅ 기본 경로에서 토크나이저 로드: {path}\n"
                        f"   어휘 크기: {sp.get_piece_size()}"
                    )
                    return
                except Exception as e:
                    logger.warning(f"경로 {path} 로드 실패: {e}")

        # 모든 시도 실패
        raise RuntimeError(
            "토크나이저 로드 실패. 다음 중 하나를 확인하세요:\n"
            "1. S3 연결 및 자격증명 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
            f"2. S3에 파일 업로드: s3://{getattr(self.s3_manager, 'bucket', 'your-bucket')}/{self.S3_TOKENIZER_PATH}\n"
            "3. 로컬 개발용 환경변수: export TOKENIZER_MODEL_PATH=/path/to/tokenizer.model\n"
            f"4. 기본 경로에 파일 복사: {default_paths[0]}"
        )

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        토크나이저 및 관련 정보 반환.

        Args:
            ctx: 실행 컨텍스트

        Returns:
            토크나이저, 어휘 크기, 특수 토큰 정보
        """
        self.validate_initialized()

        if SentencePieceTokenizer._processor is None:
            self._ensure_processor_loaded()

        if SentencePieceTokenizer._processor is None:
            raise RuntimeError("토크나이저가 초기화되지 않았습니다")

        sp = SentencePieceTokenizer._processor

        return {
            "tokenizer": sp,
            "vocab_size": sp.get_piece_size(),
            "special_tokens": {
                "bos_id": sp.bos_id(),
                "eos_id": sp.eos_id(),
                "pad_id": sp.pad_id(),
                "unk_id": sp.unk_id(),
            },
            "source": "S3"
            if self.s3_manager and self.s3_manager.connected
            else "local",
        }

    @classmethod
    def reset(cls):
        """
        토크나이저 인스턴스 초기화 (주로 테스트용).
        """
        cls._processor = None
        cls._instance = None
        logger.debug("토크나이저 싱글톤 초기화됨")
