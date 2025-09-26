"""
WMTP 통합 경로 시스템: 프로토콜 기반 경로 해석

핵심 목표:
- S3 URI, File 프로토콜, 로컬 경로를 명확하게 구분
- 경로에 프로토콜을 직접 포함하는 직관적 시스템
- 설정 복잡도 감소 및 일관된 경로 처리

지원 프로토콜:
- s3://bucket/key: S3 경로
- file:///absolute/path: 절대 경로 명시
- file://./relative/path: 상대 경로 명시
- ./path 또는 /path: 로컬 경로 (자동으로 절대 경로로 변환)

사용 예시:
>>> resolver = PathResolver()
>>> path_type, resolved = resolver.resolve("s3://wmtp/models/7b_mtp")
>>> print(f"타입: {path_type}, 경로: {resolved}")
타입: s3, 경로: s3://wmtp/models/7b_mtp

>>> path_type, resolved = resolver.resolve("file:///home/user/models/test.pth")
>>> print(f"타입: {path_type}, 경로: {resolved}")
타입: file, 경로: /home/user/models/test.pth
"""

from pathlib import Path
from typing import Literal


class PathResolver:
    """경로 프로토콜 해석 및 정규화

    WMTP 실험에서 S3와 로컬 리소스를 투명하게 처리합니다.
    경로에 포함된 프로토콜을 해석하여 적절한 로더를 선택할 수 있게 합니다.
    """

    def resolve(self, path: str) -> tuple[Literal["local", "s3", "file"], str]:
        """경로를 해석하여 타입과 정규화된 경로를 반환

        Args:
            path: 입력 경로 (로컬 경로, S3 URI, 또는 File 프로토콜)

        Returns:
            (type, resolved_path) 튜플
            - type: "local", "s3", 또는 "file"
            - resolved_path: 정규화된 경로

        Examples:
            >>> resolver.resolve("s3://bucket/key")
            ("s3", "s3://bucket/key")

            >>> resolver.resolve("file:///home/user/model.pth")
            ("file", "/home/user/model.pth")

            >>> resolver.resolve("./models/local.pth")
            ("local", "/absolute/path/to/models/local.pth")
        """
        if not path:
            raise ValueError("경로가 비어있습니다")

        # S3 경로 처리
        if path.startswith("s3://"):
            return "s3", path

        # File 프로토콜 처리
        if path.startswith("file://"):
            file_path = path[7:]  # "file://" 제거

            if file_path.startswith("/"):
                # file:/// (절대 경로)
                return "file", file_path
            elif file_path.startswith("./"):
                # file://./ (상대 경로)
                relative_path = file_path[2:]
                absolute_path = Path.cwd() / relative_path
                return "file", str(absolute_path.resolve())
            else:
                # file://path (상대 경로, ./ 없음)
                absolute_path = Path.cwd() / file_path
                return "file", str(absolute_path.resolve())

        # 로컬 경로 처리 (프로토콜 없는 경우)
        path_obj = Path(path)
        absolute_path = path_obj.absolute()
        return "local", str(absolute_path)

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
        key = parts[1] if len(parts) > 1 else ""

        if not bucket:
            raise ValueError(f"버킷 이름이 비어있습니다: {s3_path}")

        return bucket, key

    def normalize_path(self, path: str) -> str:
        """경로를 프로토콜 포함 형식으로 정규화

        모든 경로를 명시적 프로토콜이 포함된 형식으로 변환합니다.
        Phase 2 Config 간소화에서 활용될 핵심 메서드입니다.

        Args:
            path: 입력 경로

        Returns:
            프로토콜이 포함된 정규화된 경로

        Examples:
            >>> resolver.normalize_path("./models/test.pth")
            "file:///absolute/path/to/models/test.pth"

            >>> resolver.normalize_path("s3://bucket/key")
            "s3://bucket/key"
        """
        path_type, resolved = self.resolve(path)

        if path_type == "s3":
            return resolved
        elif path_type in ["local", "file"]:
            # 로컬과 file 타입 모두 file:// 프로토콜로 통일
            return f"file://{resolved}"
        else:
            return resolved


def create_path_resolver() -> PathResolver:
    """PathResolver 인스턴스 생성

    Returns:
        새로운 PathResolver 인스턴스
    """
    return PathResolver()


def resolve_checkpoint_path(base_path: str, run_identifier: str) -> tuple[str, bool]:
    """
    체크포인트 경로 해석

    WMTP Phase 3 기능:
    CheckpointConfig의 base_path와 run_identifier를 결합하여
    실제 체크포인트 저장 경로를 생성합니다.

    Args:
        base_path: 체크포인트 기본 경로 (프로토콜 포함)
        run_identifier: 실행 식별자 (MLflow run_id 또는 run_name)

    Returns:
        (resolved_path, is_s3): 해석된 경로와 S3 여부

    Examples:
        >>> resolve_checkpoint_path("file://./checkpoints", "a1b2c3d4e5f6")
        ("./checkpoints/a1b2c3d4e5f6", False)

        >>> resolve_checkpoint_path("s3://wmtp/checkpoints", "a1b2c3d4e5f6")
        ("s3://wmtp/checkpoints/a1b2c3d4e5f6", True)

        >>> resolve_checkpoint_path("./checkpoints", "experiment_1")
        ("./checkpoints/experiment_1", False)
    """
    resolver = PathResolver()
    path_type, resolved_base = resolver.resolve(base_path)

    if path_type == "s3":
        # S3 경로: 슬래시 정규화 후 run_identifier 추가
        resolved_path = f"{resolved_base.rstrip('/')}/{run_identifier}"
        return resolved_path, True
    elif path_type in ["file", "local"]:
        # 로컬 경로: resolved_base는 이미 정규화됨
        resolved_path = f"{resolved_base.rstrip('/')}/{run_identifier}"
        return resolved_path, False
    else:
        # 기본값 처리 (하위 호환성)
        resolved_path = f"{base_path.rstrip('/')}/{run_identifier}"
        return resolved_path, False
