"""
대용량 모델을 위한 S3 최적화 전송 유틸리티

WMTP의 10GB+ 모델 파일들을 효율적으로 다운로드하기 위한 최적화:
- boto3 TransferConfig로 멀티파트 병렬 다운로드
- 대용량/소용량 파일 지능적 처리
- 진행률 표시 및 네트워크 재시도 로직
- 메모리 효율적 처리
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from boto3.s3.transfer import TransferConfig
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()


class OptimizedS3Transfer:
    """S3 대용량 모델 최적화 전송 클래스"""

    def __init__(self, s3_client, bucket: str):
        """
        Args:
            s3_client: boto3 S3 client
            bucket: S3 버킷 이름
        """
        self.s3_client = s3_client
        self.bucket = bucket

        # WMTP 대용량 모델용 최적화 설정
        self.transfer_config = TransferConfig(
            multipart_threshold=1024 * 25,  # 25MB 이상부터 멀티파트
            max_concurrency=10,  # 최대 10개 병렬 연결
            multipart_chunksize=1024 * 25,  # 25MB 청크 크기
            use_threads=True,  # 스레드 풀 활성화
            num_download_attempts=3,  # 네트워크 실패시 3회 재시도
            max_io_queue=100,  # I/O 큐 크기 (동시 요청 수)
            io_chunksize=1024 * 256,  # 256KB I/O 청크 (메모리 효율)
        )

        # 대용량/소용량 파일 구분 임계값 (100MB)
        self.large_file_threshold = 100 * 1024 * 1024

    def get_file_info(self, key_prefix: str) -> list[tuple[str, int]]:
        """S3에서 파일 정보 조회 (키, 크기)"""
        files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=key_prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                size = obj["Size"]

                # 디렉토리는 제외
                if not key.endswith("/"):
                    files.append((key, size))

        return files

    def download_file_optimized(
        self,
        s3_key: str,
        local_path: Path,
        file_size: int,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> tuple[bool, str | None]:
        """최적화된 단일 파일 다운로드

        Args:
            s3_key: S3 객체 키
            local_path: 로컬 저장 경로
            file_size: 파일 크기 (진행률용)
            progress: Rich Progress 인스턴스
            task_id: Progress 태스크 ID

        Returns:
            (성공 여부, 에러 메시지)
        """
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # 진행률 콜백 함수
            def progress_callback(bytes_transferred):
                if progress and task_id:
                    progress.update(task_id, advance=bytes_transferred)

            # boto3의 최적화된 다운로드 (자동 멀티파트 적용)
            self.s3_client.download_file(
                self.bucket,
                s3_key,
                str(local_path),
                Config=self.transfer_config,
                Callback=progress_callback,
            )

            return True, None

        except Exception as e:
            error_msg = f"Download failed for {s3_key}: {e}"
            console.print(f"[red]❌ {error_msg}[/red]")
            return False, error_msg

    def download_model_directory(
        self, s3_key_prefix: str, temp_dir: Path, show_progress: bool = True
    ) -> tuple[bool, list[str]]:
        """모델 디렉토리 전체를 최적화된 방식으로 다운로드

        Args:
            s3_key_prefix: S3 키 프리픽스 (예: "models/llama-7b/")
            temp_dir: 임시 디렉토리 경로
            show_progress: 진행률 표시 여부

        Returns:
            (성공 여부, 다운로드된 파일 목록)
        """
        # 1. 파일 정보 조회
        files = self.get_file_info(s3_key_prefix)
        if not files:
            console.print(f"[yellow]⚠️ No files found at {s3_key_prefix}[/yellow]")
            return False, []

        # 2. 대용량/소용량 파일 분류
        large_files = [
            (key, size) for key, size in files if size > self.large_file_threshold
        ]
        small_files = [
            (key, size) for key, size in files if size <= self.large_file_threshold
        ]

        total_size = sum(size for _, size in files)
        console.print(
            f"[cyan]📦 Downloading {len(files)} files ({total_size / (1024**3):.2f}GB)[/cyan]"
        )
        console.print(f"  Large files (>100MB): {len(large_files)}")
        console.print(f"  Small files (≤100MB): {len(small_files)}")

        downloaded_files = []
        success_count = 0

        if show_progress:
            # 3. 진행률 표시와 함께 다운로드
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2,
            ) as progress:
                # 대용량 파일 순차 처리 (멀티파트 자동 적용)
                for s3_key, file_size in large_files:
                    file_name = s3_key.split("/")[-1]
                    local_path = temp_dir / file_name

                    task_id = progress.add_task(f"📥 {file_name}", total=file_size)

                    success, error = self.download_file_optimized(
                        s3_key, local_path, file_size, progress, task_id
                    )

                    if success:
                        success_count += 1
                        downloaded_files.append(str(local_path))
                        progress.update(task_id, completed=file_size)

                # 소용량 파일 병렬 처리 (최대 5개 동시)
                if small_files:
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_task = {}

                        for s3_key, file_size in small_files:
                            file_name = s3_key.split("/")[-1]
                            local_path = temp_dir / file_name

                            task_id = progress.add_task(
                                f"📥 {file_name}", total=file_size
                            )

                            future = executor.submit(
                                self.download_file_optimized,
                                s3_key,
                                local_path,
                                file_size,
                                progress,
                                task_id,
                            )
                            future_to_task[future] = (
                                s3_key,
                                local_path,
                                task_id,
                                file_size,
                            )

                        for future in as_completed(future_to_task):
                            s3_key, local_path, task_id, file_size = future_to_task[
                                future
                            ]
                            try:
                                success, error = future.result()
                                if success:
                                    success_count += 1
                                    downloaded_files.append(str(local_path))
                                    progress.update(task_id, completed=file_size)
                            except Exception as e:
                                console.print(
                                    f"[red]❌ Exception for {s3_key}: {e}[/red]"
                                )

        else:
            # 진행률 없이 다운로드 (조용한 모드)
            for s3_key, file_size in files:
                file_name = s3_key.split("/")[-1]
                local_path = temp_dir / file_name

                success, error = self.download_file_optimized(
                    s3_key, local_path, file_size
                )
                if success:
                    success_count += 1
                    downloaded_files.append(str(local_path))

        # 결과 요약
        total_files = len(files)
        if success_count == total_files:
            console.print(
                f"[bold green]✅ Download complete: {success_count}/{total_files} files[/bold green]"
            )
            return True, downloaded_files
        else:
            console.print(
                f"[yellow]⚠️ Partial success: {success_count}/{total_files} files[/yellow]"
            )
            return success_count > 0, downloaded_files

    def download_specific_files(
        self,
        s3_key_prefix: str,
        required_files: list[str],
        temp_dir: Path,
        show_progress: bool = True,
    ) -> tuple[bool, list[str]]:
        """특정 파일들만 선택적 다운로드 (HuggingFace 모델 전용)

        Args:
            s3_key_prefix: S3 키 프리픽스
            required_files: 다운로드할 파일명 목록
            temp_dir: 임시 디렉토리
            show_progress: 진행률 표시 여부

        Returns:
            (성공 여부, 다운로드된 파일 목록)
        """
        downloaded_files = []
        success_count = 0

        if show_progress:
            with Progress(console=console) as progress:
                for filename in required_files:
                    s3_key = f"{s3_key_prefix.rstrip('/')}/{filename}"
                    local_path = temp_dir / filename

                    # 파일 크기 조회
                    try:
                        response = self.s3_client.head_object(
                            Bucket=self.bucket, Key=s3_key
                        )
                        file_size = response["ContentLength"]

                        task_id = progress.add_task(f"📥 {filename}", total=file_size)

                        success, error = self.download_file_optimized(
                            s3_key, local_path, file_size, progress, task_id
                        )

                        if success:
                            success_count += 1
                            downloaded_files.append(str(local_path))
                            progress.update(task_id, completed=file_size)

                    except Exception as e:
                        console.print(f"[yellow]⚠️ Skipping {filename}: {e}[/yellow]")
                        continue
        else:
            # 조용한 모드
            for filename in required_files:
                s3_key = f"{s3_key_prefix.rstrip('/')}/{filename}"
                local_path = temp_dir / filename

                try:
                    response = self.s3_client.head_object(
                        Bucket=self.bucket, Key=s3_key
                    )
                    file_size = response["ContentLength"]

                    success, error = self.download_file_optimized(
                        s3_key, local_path, file_size
                    )
                    if success:
                        success_count += 1
                        downloaded_files.append(str(local_path))

                except Exception:
                    continue

        console.print(
            f"[green]📁 Downloaded {success_count}/{len(required_files)} files[/green]"
        )
        return success_count > 0, downloaded_files
