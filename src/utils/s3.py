"""
WMTP 클러스터 인프라의 핵심: S3 스트리밍 유틸리티 (Cache-Free)

WMTP 연구 맥락:
이 모듈은 WMTP 실험의 클러스터 환경(VESSL, AWS)에서 필수적인 S3 데이터 스트리밍 기능을 제공합니다.
개발자의 로컬 환경과 클러스터 환경 사이의 데이터 일관성을 보장하며,
메모리 기반 직접 스트리밍을 통해 연구 효율성을 극대화합니다.

핵심 철학:
"로컬 우선 → S3 직접 스트리밍" 전략으로 모든 WMTP 알고리즘이 동일한 데이터에 접근 (캐시 제거)

WMTP 실험 시나리오:
1. 개발 환경: 로컬 파일 직접 사용 (⚡ 가장 빠름)
2. 클러스터 환경: S3에서 메모리로 직접 스트리밍
3. CI/CD: 로컬 파일 사용

지원 데이터 유형:
- Facebook MTP 모델: consolidated.pth (5GB+) → 메모리 스트리밍
- 코딩 데이터셋: MBPP, CodeContests, HumanEval → 라인별 스트리밍
- 학습 체크포인트: WMTP 알고리즘 별 메모리 로드
- 설정 파일: config.yaml, recipe.yaml

성능 최적화:
- 메모리 스트리밍: 디스크 I/O 완전 제거
- 진행률 표시: Rich UI로 실시간 다운로드 진행 상황 표시
- 라인별 처리: 대용량 데이터셋도 메모리 효율적 처리
- 오류 복구: 네트워크 오류 시 자동 재시도

보안 및 권한:
- AWS IAM 역할 기반 인증
- 버킷 별 세밀한 액세스 제어
- 전송 중 암호화 (TLS) 및 저장 중 암호화

아키텍처 설계:
- boto3 사용을 이 모듈로 중앙집중화
- 직접 AWS SDK 호출 방지로 일관성 보장
- 에러 핸들링 및 로깅 표준화
"""

# hashlib import 제거됨 - compute_file_hash 함수 삭제로 불필요
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


# S3Utils 클래스 제거됨 - 실제로 사용되지 않음
# 모든 로더는 S3Manager를 직접 사용하여 중간 레이어 불필요


class S3Manager:
    """
    Manager for S3 operations with local caching.

    Implements S3 streaming with no local cache dependency.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "ap-northeast-2",
        prefix: str = "",
    ):
        """
        Initialize S3 manager.

        Args:
            bucket: S3 bucket name
            region: AWS region
            prefix: S3 key prefix
        """
        self.bucket = bucket
        self.region = region
        self.prefix = prefix

        try:
            self.s3_client = boto3.client("s3", region_name=region)
            # 간단한 연결 테스트
            self.s3_client.head_bucket(Bucket=self.bucket)
            self.connected = True
        except (NoCredentialsError, ClientError) as e:
            console.print(f"[yellow]Warning: S3 not available: {e}[/yellow]")
            self.connected = False
            self.s3_client = None

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


def create_s3_manager(config: dict[str, Any]) -> S3Manager | None:
    """
    Phase 2 리팩토링: S3 매니저 생성 (storage 의존성 제거)

    기존 storage 설정 방식:
        storage:
          mode: s3  # 또는 auto
          s3:
            bucket: wmtp
            region: ap-northeast-2

    새로운 방식:
        s3_auth:
          default_bucket: wmtp
          region: ap-northeast-2

    Args:
        config: Configuration dictionary with S3 settings

    Returns:
        S3Manager instance or None if not configured
    """
    # Phase 2: 새로운 s3_auth 방식 우선 사용
    if "s3_auth" in config:
        s3_auth = config["s3_auth"]
        return S3Manager(
            bucket=s3_auth.get("default_bucket"),
            region=s3_auth.get("region", "ap-northeast-2"),
            prefix="",  # 새 방식에서는 prefix 사용 안 함
        )

    # 하위 호환성: 기존 storage 방식 지원
    storage = config.get("storage", {})
    if isinstance(storage, dict):
        storage_mode = storage.get("mode")

        # auto 모드일 때도 S3Manager 생성 (PathResolver가 경로를 판별)
        if storage_mode not in ["s3", "auto"]:
            return None

        s3_config = storage.get("s3", {})
        if not s3_config:
            console.print(
                "[yellow]Warning: S3 mode selected but no S3 config found[/yellow]"
            )
            return None

        return S3Manager(
            bucket=s3_config.get("bucket"),
            region=s3_config.get("region", "ap-northeast-2"),
            prefix=s3_config.get("prefix", ""),
        )

    # S3 설정이 없으면 None 반환
    return None


# Export main functions
__all__ = [
    "S3Manager",
    "create_s3_manager",
]
