"""
WMTP 통합 경로 시스템: 로컬/S3 경로 자동 판별 및 해석

핵심 목표:
- 로컬 파일 경로와 S3 URI를 자동으로 구분
- 모델/데이터셋 타입 자동 감지
- 설정 복잡도 감소 및 일관된 경로 처리

사용 예시:
>>> resolver = PathResolver()
>>> path_type, resolved = resolver.resolve("s3://wmtp/models/7b_mtp")
>>> print(f"타입: {path_type}, 경로: {resolved}")
타입: s3, 경로: s3://wmtp/models/7b_mtp

>>> path_type, resolved = resolver.resolve("./models/local_model.pth")
>>> print(f"타입: {path_type}, 경로: {resolved}")
타입: local, 경로: /absolute/path/to/models/local_model.pth
"""

from pathlib import Path
from typing import Literal


class PathResolver:
    """경로 타입 자동 판별 및 해석

    WMTP 실험에서 로컬과 클라우드 환경 간의 투명한 전환을 지원합니다.
    경로만 보고 자동으로 로컬/S3를 판별하여, 설정 파일 단순화를 달성합니다.
    """

    # 모델 파일 확장자
    MODEL_EXTENSIONS = [".pth", ".pt", ".safetensors", ".bin", ".ckpt"]

    # 데이터셋 마커
    DATASET_MARKERS = ["dataset", "data", "benchmark", "eval"]
    DATASET_EXTENSIONS = [".json", ".jsonl", ".txt", ".csv", ".parquet"]

    def resolve(self, path: str) -> tuple[Literal["local", "s3"], str]:
        """경로를 해석하여 타입과 정규화된 경로를 반환

        Args:
            path: 입력 경로 (로컬 경로 또는 S3 URI)

        Returns:
            (type, resolved_path) 튜플
            - type: "local" 또는 "s3"
            - resolved_path: 정규화된 경로

        Examples:
            >>> resolver.resolve("s3://bucket/key")
            ("s3", "s3://bucket/key")

            >>> resolver.resolve("./models/local.pth")
            ("local", "/absolute/path/to/models/local.pth")
        """
        if not path:
            raise ValueError("경로가 비어있습니다")

        # S3 경로 처리
        if path.startswith("s3://"):
            # S3 URI는 그대로 반환
            return "s3", path

        # 로컬 경로 처리
        # 상대 경로를 절대 경로로 변환
        path_obj = Path(path)
        absolute_path = path_obj.absolute()

        return "local", str(absolute_path)

    def is_model_path(self, path: str) -> bool:
        """모델 파일 경로인지 확인

        Args:
            path: 확인할 경로

        Returns:
            모델 파일이면 True, 아니면 False

        Examples:
            >>> resolver.is_model_path("model.pth")
            True
            >>> resolver.is_model_path("data.json")
            False
        """
        path_lower = path.lower()

        # 확장자로 판별
        for ext in self.MODEL_EXTENSIONS:
            if path_lower.endswith(ext):
                return True

        # 모델 관련 키워드로 추가 판별
        model_keywords = ["model", "checkpoint", "weights", "ckpt"]
        for keyword in model_keywords:
            if keyword in path_lower:
                return True

        return False

    def is_dataset_path(self, path: str) -> bool:
        """데이터셋 경로인지 확인

        Args:
            path: 확인할 경로

        Returns:
            데이터셋 경로면 True, 아니면 False

        Examples:
            >>> resolver.is_dataset_path("dataset/mbpp/test.json")
            True
            >>> resolver.is_dataset_path("model.pth")
            False
        """
        path_lower = path.lower()

        # 데이터셋 마커로 판별
        for marker in self.DATASET_MARKERS:
            if marker in path_lower:
                return True

        # 확장자로 판별
        for ext in self.DATASET_EXTENSIONS:
            if path_lower.endswith(ext):
                return True

        return False

    def get_path_category(self, path: str) -> Literal["model", "dataset", "unknown"]:
        """경로의 카테고리를 판별

        Args:
            path: 분류할 경로

        Returns:
            "model", "dataset", 또는 "unknown"

        Examples:
            >>> resolver.get_path_category("s3://wmtp/models/7b.pth")
            "model"
            >>> resolver.get_path_category("dataset/mbpp/")
            "dataset"
        """
        if self.is_model_path(path):
            return "model"
        elif self.is_dataset_path(path):
            return "dataset"
        else:
            return "unknown"

    def extract_bucket_and_key(self, s3_path: str) -> tuple[str, str]:
        """S3 경로에서 버킷과 키를 추출

        Args:
            s3_path: S3 URI (s3://bucket/key 형식)

        Returns:
            (bucket, key) 튜플

        Raises:
            ValueError: S3 URI 형식이 아닌 경우

        Examples:
            >>> resolver.extract_bucket_and_key("s3://wmtp/models/7b.pth")
            ("wmtp", "models/7b.pth")
        """
        if not s3_path.startswith("s3://"):
            raise ValueError(f"S3 URI 형식이 아닙니다: {s3_path}")

        # s3:// 제거
        path_without_scheme = s3_path[5:]

        # 버킷과 키 분리
        parts = path_without_scheme.split("/", 1)
        bucket = parts[0]

        if len(parts) > 1:
            key = parts[1]
        else:
            key = ""

        if not bucket:
            raise ValueError(f"버킷 이름이 비어있습니다: {s3_path}")

        return bucket, key

    def normalize_s3_path(self, path: str, default_bucket: str = "wmtp") -> str:
        """S3 경로를 정규화

        간단한 경로를 완전한 S3 URI로 변환합니다.

        Args:
            path: 입력 경로
            default_bucket: 기본 버킷 이름

        Returns:
            정규화된 S3 URI

        Examples:
            >>> resolver.normalize_s3_path("models/7b.pth")
            "s3://wmtp/models/7b.pth"

            >>> resolver.normalize_s3_path("s3://custom/path")
            "s3://custom/path"
        """
        # 이미 S3 URI면 그대로 반환
        if path.startswith("s3://"):
            return path

        # 상대 경로면 기본 버킷 사용
        return f"s3://{default_bucket}/{path}"


def create_path_resolver() -> PathResolver:
    """PathResolver 인스턴스 생성 헬퍼 함수

    Returns:
        새로운 PathResolver 인스턴스
    """
    return PathResolver()
