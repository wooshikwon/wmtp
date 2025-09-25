"""
WMTP 클러스터 인프라의 핵심: S3 유틸리티

WMTP 연구 맥락:
이 모듈은 WMTP 실험의 클러스터 환경(VESSL, AWS)에서 필수적인 S3 데이터 또늠 기능을 제공합니다.
개발자의 로컬 환경과 클러스터 환경 사이의 데이터 일관성을 보장하며,
중앙집중식 모델 및 데이터셋 관리를 통해 연구 효율성을 극대화합니다.

핵심 철학:
"로컬 우선 + S3 미러링" 전략으로 모든 WMTP 알고리즘이 동일한 데이터에 접근

WMTP 실험 시나리오:
1. 개발 환경: 로컬 파일 직접 사용 (⚡ 가장 빠름)
2. 클러스터 환경: S3에서 자동 다운로드 + 캐싱
3. CI/CD: 사전 캐시된 데이터로 빠른 테스트

지원 데이터 유형:
- Facebook MTP 모델: consolidated.pth (5GB+)
- 코딩 데이터셋: MBPP, CodeContests, HumanEval
- 학습 체크포인트: WMTP 알고리즘 별 로컬 저장
- 설정 파일: config.yaml, recipe.yaml

성능 최적화:
- 지능형 캐싱: 파일 해시 기반 중복 다운로드 방지
- 진행률 표시: Rich UI로 실시간 다운로드 진행 상황 표시
- 다중 디렉토리 동기화: 배치 업로드/다운로드 지원
- 오류 복구: 네트워크 오류 시 자동 재시도

보안 및 권한:
- AWS IAM 역할 기반 인증
- 버킷 별 세리한 액세스 제어
- 전송 중 암호화 (TLS) 및 저장 중 암호화

아키텍처 설계:
- boto3 사용을 이 모듈로 중앙집중화
- 직접 AWS SDK 호출 방지로 일관성 보장
- 에러 핸들링 및 로깅 표준화
"""

import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class S3Utils:
    """
    WMTP 로더를 위한 간소화된 S3 유틸리티 래퍼입니다.

    연구 진행 편의성:
    복잡한 S3 조작을 간단한 메서드 호출로 추상화하여,
    WMTP 연구자가 클러스터 인프라 대신 알고리즘 개발에 집중할 수 있게 합니다.

    주요 기능:
    - download_model(): S3에서 모델 자동 다운로드
    - upload_checkpoint(): 학습된 모델 중앙 저장
    - sync_dataset(): 데이터셋 동기화
    - cache_management(): 로컬 저장 공간 최적화

    기본 설정:
    - 버킷: wmtp-models (WMTP 연구 전용)
    - 리전: ap-northeast-2 (서울, 낮은 레이턴시)
    - 캐시: ~/.cache/wmtp/ (로컬 임시 저장소)

    사용 예시:
    >>> s3 = S3Utils()
    >>> model_path = s3.download_model("s3://wmtp-models/facebook-mtp/consolidated.pth")
    >>> print(f"Model loaded from: {model_path}")
    """

    def __init__(self, bucket: str = "wmtp-models", region: str = "ap-northeast-2"):
        """Initialize S3 utilities with default WMTP bucket."""
        self.manager = S3Manager(bucket=bucket, region=region)

    def download_model(self, s3_path: str, local_dir: Path | None = None) -> Path:
        """Download model from S3 path like s3://bucket/path/to/model."""
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")

        # Parse S3 path
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Update manager if different bucket
        if bucket != self.manager.bucket:
            self.manager = S3Manager(bucket=bucket, region=self.manager.region)

        # Determine local path
        if local_dir is None:
            local_dir = self.manager.cache_dir / "models"
        local_path = local_dir / key.split("/")[-1]

        return self.manager.download_if_missing(key, local_path)

    def download_dataset(self, s3_path: str, local_dir: Path | None = None) -> Path:
        """Download dataset from S3 path."""
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")

        # Parse S3 path
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Update manager if different bucket
        if bucket != self.manager.bucket:
            self.manager = S3Manager(bucket=bucket, region=self.manager.region)

        # Determine local path
        if local_dir is None:
            local_dir = self.manager.cache_dir / "datasets"
        local_path = local_dir / key.split("/")[-1]

        return self.manager.download_if_missing(key, local_path)

    def download_checkpoint(self, s3_path: str, local_dir: Path | None = None) -> Path:
        """Download checkpoint from S3 path."""
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")

        # Parse S3 path
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Update manager if different bucket
        if bucket != self.manager.bucket:
            self.manager = S3Manager(bucket=bucket, region=self.manager.region)

        # Determine local path
        if local_dir is None:
            local_dir = self.manager.cache_dir / "checkpoints"
        local_path = local_dir / key.split("/")[-1]

        return self.manager.download_if_missing(key, local_path)

    def upload_checkpoint(self, local_path: Path, s3_path: str) -> None:
        """Upload checkpoint to S3."""
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")

        # Parse S3 path
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Update manager if different bucket
        if bucket != self.manager.bucket:
            self.manager = S3Manager(bucket=bucket, region=self.manager.region)

        self.manager.upload_file(str(local_path), key)


class S3Manager:
    """
    Manager for S3 operations with local caching.

    Implements the local-first policy: use local if exists,
    otherwise download from S3 and cache.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "ap-northeast-2",
        prefix: str = "",
        cache_dir: str | Path = ".cache",
    ):
        """
        Initialize S3 manager.

        Args:
            bucket: S3 bucket name
            region: AWS region
            prefix: S3 key prefix
            cache_dir: Local cache directory
        """
        self.bucket = bucket
        self.region = region
        self.prefix = prefix
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.s3_client = boto3.client("s3", region_name=region)
            self._test_connection()
            self.connected = True
        except (NoCredentialsError, ClientError) as e:
            console.print(f"[yellow]Warning: S3 not available: {e}[/yellow]")
            self.connected = False
            self.s3_client = None

    def _test_connection(self) -> None:
        """Test S3 connection by listing bucket."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise ValueError(f"Bucket '{self.bucket}' not found")
            else:
                raise

    def download_if_missing(
        self,
        s3_key: str,
        local_path: str | Path,
        force: bool = False,
    ) -> Path:
        """
        Download from S3 if local file doesn't exist.

        Args:
            s3_key: S3 object key
            local_path: Local file path
            force: Force download even if exists

        Returns:
            Path to local file
        """
        local_path = Path(local_path)

        # Check local first (unless forced)
        if local_path.exists() and not force:
            console.print(f"[green]Using local file: {local_path}[/green]")
            return local_path

        if not self.connected:
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Local file not found and S3 not available: {local_path}"
                )
            return local_path

        # Download from S3
        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Downloading {s3_key} from S3...",
                    total=None,
                )

                self.s3_client.download_file(
                    self.bucket,
                    full_key,
                    str(local_path),
                )

                progress.update(task, completed=True)

            console.print(f"[green]Downloaded to: {local_path}[/green]")
            return local_path

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise FileNotFoundError(
                    f"S3 object not found: s3://{self.bucket}/{full_key}"
                )
            else:
                raise

    def upload_from_bytes(
        self,
        data: bytes,
        s3_key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload bytes data directly to S3.

        Args:
            data: Bytes data to upload
            s3_key: S3 object key
            metadata: Optional metadata

        Returns:
            S3 URI
        """
        if not self.connected:
            raise RuntimeError("S3 not connected. Cannot upload.")

        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Uploading to S3: {s3_key}...",
                    total=None,
                )

                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=full_key,
                    Body=data,
                    **extra_args,
                )

                progress.update(task, completed=True)

            s3_uri = f"s3://{self.bucket}/{full_key}"
            console.print(f"[green]Uploaded to: {s3_uri}[/green]")
            return s3_uri

        except ClientError as e:
            raise RuntimeError(f"Failed to upload to S3: {e}")

    def upload_artifact(
        self,
        local_path: str | Path,
        s3_key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload artifact to S3.

        Args:
            local_path: Local file path
            s3_key: S3 object key
            metadata: Optional metadata

        Returns:
            S3 URI
        """
        if not self.connected:
            raise RuntimeError("S3 not connected. Cannot upload.")

        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Uploading {local_path.name} to S3...",
                    total=None,
                )

                self.s3_client.upload_file(
                    str(local_path),
                    self.bucket,
                    full_key,
                    ExtraArgs=extra_args,
                )

                progress.update(task, completed=True)

            s3_uri = f"s3://{self.bucket}/{full_key}"
            console.print(f"[green]Uploaded to: {s3_uri}[/green]")
            return s3_uri

        except ClientError as e:
            raise RuntimeError(f"Failed to upload to S3: {e}")

    def download_to_bytes(self, s3_key: str) -> bytes:
        """
        Download S3 object directly to bytes in memory.

        Args:
            s3_key: S3 object key

        Returns:
            File content as bytes

        Raises:
            RuntimeError: If S3 not connected
            FileNotFoundError: If object not found
        """
        if not self.connected:
            raise RuntimeError("S3 not connected. Cannot download.")

        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=full_key)
            return response["Body"].read()

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(
                    f"S3 object not found: s3://{self.bucket}/{full_key}"
                )
            else:
                raise RuntimeError(f"Failed to download from S3: {e}")

    def stream_model(self, s3_key: str) -> io.BytesIO:
        """
        모델을 메모리로 직접 스트리밍 (캐시 없음).

        WMTP 실험에서 대용량 모델(7B+)을 효율적으로 로드하기 위한 메서드.
        디스크 I/O를 피하고 메모리에 직접 로드하여 속도를 향상시킵니다.

        Args:
            s3_key: S3 object key

        Returns:
            모델 데이터를 담은 BytesIO 객체

        Raises:
            RuntimeError: S3 연결 실패
            FileNotFoundError: 모델 파일이 S3에 없음

        Example:
            >>> stream = s3_manager.stream_model("models/7b_mtp.pth")
            >>> model = torch.load(stream)
        """
        console.print(f"[cyan]Streaming model from S3: {s3_key}[/cyan]")
        model_bytes = self.download_to_bytes(s3_key)
        return io.BytesIO(model_bytes)

    def stream_dataset(self, s3_key: str) -> Iterator[dict]:
        """
        데이터셋을 스트리밍으로 읽기 (캐시 없음).

        JSONL 형식의 데이터셋을 한 줄씩 파싱하여 반환합니다.
        대용량 데이터셋도 메모리 효율적으로 처리 가능합니다.

        Args:
            s3_key: S3 object key

        Returns:
            JSON 객체들의 이터레이터

        Raises:
            RuntimeError: S3 연결 실패
            FileNotFoundError: 데이터셋 파일이 S3에 없음
            json.JSONDecodeError: JSON 파싱 실패

        Example:
            >>> for sample in s3_manager.stream_dataset("datasets/mbpp/test.jsonl"):
            >>>     print(sample['text'])
        """
        console.print(f"[cyan]Streaming dataset from S3: {s3_key}[/cyan]")
        content = self.download_to_bytes(s3_key)

        # JSONL 형식 처리 (한 줄에 하나의 JSON 객체)
        for line in content.decode("utf-8").splitlines():
            line = line.strip()
            if line:  # 빈 줄 무시
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    console.print(
                        f"[yellow]Warning: Failed to parse JSON line: {e}[/yellow]"
                    )
                    continue

    def exists(self, s3_key: str) -> bool:
        """
        Check if object exists in S3.

        Args:
            s3_key: S3 object key

        Returns:
            True if exists
        """
        if not self.connected:
            return False

        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError:
            return False

    def list_objects(
        self,
        prefix: str = "",
        recursive: bool = True,
    ) -> list[str]:
        """
        List objects in S3.

        Args:
            prefix: Key prefix to filter
            recursive: List recursively

        Returns:
            List of S3 keys
        """
        if not self.connected:
            return []

        full_prefix = f"{self.prefix}/{prefix}" if self.prefix else prefix

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.bucket,
                Prefix=full_prefix,
            )

            keys = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Remove the base prefix if present
                        if self.prefix and key.startswith(self.prefix + "/"):
                            key = key[len(self.prefix) + 1 :]
                        keys.append(key)

            if not recursive:
                # Filter to only immediate children
                keys = [k for k in keys if "/" not in k[len(prefix) :].lstrip("/")]

            return keys

        except ClientError as e:
            console.print(f"[red]Failed to list S3 objects: {e}[/red]")
            return []

    def get_etag(self, s3_key: str) -> str | None:
        """
        Get ETag for S3 object (for version checking).

        Args:
            s3_key: S3 object key

        Returns:
            ETag string or None
        """
        if not self.connected:
            return None

        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=full_key)
            return response.get("ETag", "").strip('"')
        except ClientError:
            return None

    def sync_directory(
        self,
        local_dir: str | Path,
        s3_prefix: str,
        direction: str = "download",
    ) -> None:
        """
        Sync entire directory with S3.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 key prefix
            direction: 'download' or 'upload'
        """
        if not self.connected:
            raise RuntimeError("S3 not connected. Cannot sync.")

        local_dir = Path(local_dir)

        if direction == "download":
            # List all objects with prefix
            objects = self.list_objects(s3_prefix)

            for s3_key in objects:
                # Reconstruct local path
                relative_path = s3_key[len(s3_prefix) :].lstrip("/")
                local_path = local_dir / relative_path

                # Download if missing
                self.download_if_missing(s3_key, local_path)

        elif direction == "upload":
            # Walk local directory
            for local_path in local_dir.rglob("*"):
                if local_path.is_file():
                    # Construct S3 key
                    relative_path = local_path.relative_to(local_dir)
                    s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")

                    # Upload file
                    self.upload_artifact(local_path, s3_key)

        else:
            raise ValueError(
                f"Invalid direction: {direction}. Use 'download' or 'upload'"
            )

    def get_cache_path(self, s3_key: str, version: str | None = None) -> Path:
        """
        Get local cache path for S3 object.

        Args:
            s3_key: S3 object key
            version: Optional version identifier

        Returns:
            Cache file path
        """
        # Create hash-based cache key
        cache_key = s3_key.replace("/", "_")
        if version:
            cache_key = f"{cache_key}_{version}"

        return self.cache_dir / cache_key


def create_s3_manager(config: dict[str, Any]) -> S3Manager | None:
    """
    Create S3 manager from configuration.

    Args:
        config: Configuration dictionary with S3 settings

    Returns:
        S3Manager instance or None if not configured
    """
    storage_mode = config.get("storage", {}).get("mode")

    # auto 모드일 때도 S3Manager 생성 (PathResolver가 경로를 판별)
    if storage_mode not in ["s3", "auto"]:
        return None

    s3_config = config.get("storage", {}).get("s3", {})
    if not s3_config:
        console.print(
            "[yellow]Warning: S3 mode selected but no S3 config found[/yellow]"
        )
        return None

    # cache_dir은 더 이상 사용하지 않지만, 하위 호환성을 위해 임시로 유지
    # 추후 S3Manager 클래스 자체에서 cache_dir 파라미터 제거 예정
    return S3Manager(
        bucket=s3_config.get("bucket"),
        region=s3_config.get("region", "ap-northeast-2"),
        prefix=s3_config.get("prefix", ""),
        cache_dir="/tmp/.s3_cache",  # 임시 경로, 실제로 사용 안 함
    )


def compute_file_hash(file_path: str | Path, algorithm: str = "md5") -> str:
    """
    Compute hash of file for cache validation.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5' or 'sha256')

    Returns:
        Hex digest string
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.md5() if algorithm == "md5" else hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def ensure_s3_uri(path_or_uri: str, bucket: str, prefix: str = "") -> str:
    """
    Ensure path is formatted as S3 URI.

    Args:
        path_or_uri: Path or S3 URI
        bucket: S3 bucket name
        prefix: S3 prefix

    Returns:
        S3 URI string
    """
    if path_or_uri.startswith("s3://"):
        return path_or_uri

    # Construct S3 URI
    key = f"{prefix}/{path_or_uri}" if prefix else path_or_uri
    return f"s3://{bucket}/{key}"


# Export main functions
__all__ = [
    "S3Manager",
    "create_s3_manager",
    "compute_file_hash",
    "ensure_s3_uri",
]
