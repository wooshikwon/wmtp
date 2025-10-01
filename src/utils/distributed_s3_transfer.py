"""
분산 S3 다운로드 시스템 - 대용량 모델 고속 전송

핵심 최적화:
1. Range 요청 기반 청크 분할 다운로드
2. 멀티스레드/멀티프로세스 병렬 처리
3. 동적 워커 수 조정
4. 중단된 다운로드 재개 지원
5. 메모리 효율적 스트리밍
"""

import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import boto3
import psutil
from botocore.config import Config
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

console = Console()


class DistributedS3Transfer:
    """분산 S3 전송 클래스 - 최대 성능 추출"""

    def __init__(
        self,
        s3_client=None,
        bucket: str = None,
        max_workers: int | None = None,
        use_multiprocess: bool = False,
        chunk_size_mb: int = 50,
        enable_acceleration: bool = True,
    ):
        """
        Args:
            s3_client: boto3 S3 client (없으면 자동 생성)
            bucket: S3 버킷 이름
            max_workers: 최대 워커 수 (None이면 자동 결정)
            use_multiprocess: 멀티프로세스 사용 여부 (CPU 집약적 작업용)
            chunk_size_mb: 청크 크기 (MB)
            enable_acceleration: S3 Transfer Acceleration 사용 여부
        """
        self.bucket = bucket or os.getenv("S3_BUCKET_NAME", "wmtp")

        # S3 클라이언트 설정 (연결 풀 확대)
        if s3_client is None:
            config = Config(
                max_pool_connections=100,  # 연결 풀 크기 증가
                retries={
                    "max_attempts": 3,
                    "mode": "adaptive",  # 적응형 재시도
                },
            )

            # Transfer Acceleration 엔드포인트 사용
            if enable_acceleration:
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name=os.getenv("AWS_DEFAULT_REGION", "eu-north-1"),
                    config=config,
                    endpoint_url=f"https://{self.bucket}.s3-accelerate.amazonaws.com",
                )
            else:
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name=os.getenv("AWS_DEFAULT_REGION", "eu-north-1"),
                    config=config,
                )
        else:
            self.s3_client = s3_client

        # 워커 수 자동 결정 (메모리 안전 설정)
        if max_workers is None:
            cpu_count = mp.cpu_count()
            # 메모리 안전을 위해 보수적으로 설정
            self.max_workers = min(cpu_count, 8)  # 최대 8개 워커로 제한
        else:
            self.max_workers = min(max_workers, 8)  # 사용자 지정도 8개로 제한

        self.use_multiprocess = use_multiprocess
        self.chunk_size = chunk_size_mb * 1024 * 1024  # MB to bytes

        # 진행률 추적용 락
        self.progress_lock = Lock()
        self.download_stats = {}

        # 재개 가능한 다운로드를 위한 메타데이터 저장
        self.metadata_dir = Path(".wmtp_download_cache")
        self.metadata_dir.mkdir(exist_ok=True)

    def get_optimal_workers(self, file_size: int) -> int:
        """파일 크기에 따른 최적 워커 수 결정 (메모리 안전)"""
        size_gb = file_size / (1024**3)

        if size_gb < 1:
            return min(2, self.max_workers)  # 작은 파일: 2 워커
        elif size_gb < 5:
            return min(4, self.max_workers)  # 중간 파일: 4 워커
        elif size_gb < 10:
            return min(6, self.max_workers)  # 큰 파일: 6 워커
        else:
            return min(8, self.max_workers)  # 초대형 파일: 최대 8 워커

    def get_file_info(self, s3_key: str) -> dict[str, Any]:
        """S3 파일 정보 조회"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return {
                "size": response["ContentLength"],
                "etag": response.get("ETag", "").strip('"'),
                "last_modified": response.get("LastModified"),
                "content_type": response.get("ContentType"),
            }
        except Exception as e:
            console.print(f"[red]Error getting file info for {s3_key}: {e}[/red]")
            return None

    def download_chunk(
        self,
        s3_key: str,
        start_byte: int,
        end_byte: int,
        chunk_file: Path,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> tuple[bool, str | None]:
        """단일 청크 다운로드"""
        try:
            # Range 요청으로 특정 바이트 범위만 다운로드
            response = self.s3_client.get_object(
                Bucket=self.bucket, Key=s3_key, Range=f"bytes={start_byte}-{end_byte}"
            )

            # 스트리밍으로 메모리 효율적 쓰기 (작은 버퍼 사용)
            with open(chunk_file, "wb") as f:
                for chunk in response["Body"].iter_chunks(
                    chunk_size=256 * 1024
                ):  # 256KB씩 (메모리 절약)
                    f.write(chunk)
                    if progress and task_id:
                        with self.progress_lock:
                            progress.update(task_id, advance=len(chunk))

            return True, None

        except Exception as e:
            error_msg = f"Failed to download chunk {start_byte}-{end_byte}: {e}"
            return False, error_msg

    def merge_chunks(self, chunk_files: list[Path], output_file: Path) -> bool:
        """청크 파일들을 하나로 병합"""
        try:
            with open(output_file, "wb") as output:
                for chunk_file in sorted(chunk_files):
                    with open(chunk_file, "rb") as chunk:
                        # 효율적인 파일 복사
                        while True:
                            data = chunk.read(1024 * 1024 * 10)  # 10MB 버퍼
                            if not data:
                                break
                            output.write(data)
                    # 청크 파일 삭제
                    chunk_file.unlink()
            return True
        except Exception as e:
            console.print(f"[red]Error merging chunks: {e}[/red]")
            return False

    def download_file_distributed(
        self,
        s3_key: str,
        local_path: Path,
        progress: Progress | None = None,
        resume: bool = True,
    ) -> tuple[bool, str | None]:
        """분산 다운로드 메인 함수"""

        # 파일 정보 조회
        file_info = self.get_file_info(s3_key)
        if not file_info:
            return False, f"Failed to get file info for {s3_key}"

        file_size = file_info["size"]
        etag = file_info["etag"]

        # 메타데이터 파일 경로
        meta_file = (
            self.metadata_dir / f"{hashlib.md5(s3_key.encode()).hexdigest()}.json"
        )

        # 재개 가능한 다운로드 확인
        completed_chunks = set()
        if resume and meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                if meta.get("etag") == etag:
                    completed_chunks = set(meta.get("completed_chunks", []))
                    console.print(
                        f"[yellow]Resuming download: {len(completed_chunks)} chunks already done[/yellow]"
                    )

        # 최적 워커 수 결정
        num_workers = self.get_optimal_workers(file_size)

        # 청크 분할 계산
        num_chunks = max(1, (file_size + self.chunk_size - 1) // self.chunk_size)
        num_workers = min(num_workers, num_chunks)  # 청크 수보다 많은 워커는 불필요

        console.print(f"[cyan]📊 File: {s3_key}[/cyan]")
        console.print(f"[cyan]   Size: {file_size/(1024**3):.2f}GB[/cyan]")
        console.print(
            f"[cyan]   Chunks: {num_chunks} × {self.chunk_size/(1024**2):.0f}MB[/cyan]"
        )
        console.print(f"[cyan]   Workers: {num_workers} parallel downloads[/cyan]")

        # 청크 파일 준비
        temp_dir = local_path.parent / f".tmp_{local_path.name}"
        temp_dir.mkdir(exist_ok=True, parents=True)

        # 진행률 표시
        task_id = None
        if progress:
            task_id = progress.add_task(
                f"📥 {local_path.name}",
                total=file_size - len(completed_chunks) * self.chunk_size,
            )

        # 청크 다운로드 작업 생성
        chunk_tasks = []
        for i in range(num_chunks):
            if i in completed_chunks:
                continue  # 이미 완료된 청크 스킵

            start_byte = i * self.chunk_size
            end_byte = min(start_byte + self.chunk_size - 1, file_size - 1)
            chunk_file = temp_dir / f"chunk_{i:06d}"

            chunk_tasks.append((i, start_byte, end_byte, chunk_file))

        # 병렬 다운로드 실행
        failed_chunks = []
        completed_new_chunks = []

        # 실행자 선택 (프로세스 vs 스레드)
        Executor = ProcessPoolExecutor if self.use_multiprocess else ThreadPoolExecutor

        with Executor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self.download_chunk,
                    s3_key,
                    start,
                    end,
                    chunk_file,
                    progress,
                    task_id,
                ): (chunk_id, chunk_file)
                for chunk_id, start, end, chunk_file in chunk_tasks
            }

            for future in as_completed(futures):
                chunk_id, chunk_file = futures[future]
                success, error = future.result()

                if success:
                    completed_new_chunks.append(chunk_id)
                    completed_chunks.add(chunk_id)

                    # 메타데이터 업데이트 (재개 지원)
                    with open(meta_file, "w") as f:
                        json.dump(
                            {
                                "s3_key": s3_key,
                                "etag": etag,
                                "completed_chunks": list(completed_chunks),
                                "total_chunks": num_chunks,
                            },
                            f,
                        )
                else:
                    failed_chunks.append((chunk_id, error))

        # 다운로드 성공 여부 확인
        if failed_chunks:
            console.print(
                f"[red]❌ Failed to download {len(failed_chunks)} chunks[/red]"
            )
            for chunk_id, error in failed_chunks:
                console.print(f"[red]   Chunk {chunk_id}: {error}[/red]")
            return False, "Some chunks failed to download"

        # 모든 청크 파일 병합
        console.print(f"[yellow]🔀 Merging {num_chunks} chunks...[/yellow]")
        all_chunk_files = [temp_dir / f"chunk_{i:06d}" for i in range(num_chunks)]

        if self.merge_chunks(all_chunk_files, local_path):
            # 임시 디렉토리 및 메타데이터 정리
            temp_dir.rmdir()
            meta_file.unlink(missing_ok=True)

            console.print(f"[green]✅ Successfully downloaded: {local_path}[/green]")
            return True, None
        else:
            return False, "Failed to merge chunks"

    def download_directory_distributed(
        self,
        s3_prefix: str,
        local_dir: Path,
        file_pattern: str | None = None,
        show_progress: bool = True,
    ) -> tuple[bool, list[str]]:
        """디렉토리 전체를 분산 다운로드"""

        # S3 객체 목록 조회
        console.print(
            f"[cyan]📂 Listing objects in s3://{self.bucket}/{s3_prefix}[/cyan]"
        )

        files_to_download = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                size = obj["Size"]

                # 디렉토리 제외
                if key.endswith("/"):
                    continue

                # 패턴 필터링
                if file_pattern and not Path(key).match(file_pattern):
                    continue

                files_to_download.append((key, size))

        if not files_to_download:
            console.print(f"[yellow]No files found in {s3_prefix}[/yellow]")
            return False, []

        # 통계 표시
        total_size = sum(size for _, size in files_to_download)
        console.print(
            f"[cyan]📊 Found {len(files_to_download)} files, {total_size/(1024**3):.2f}GB total[/cyan]"
        )

        # 파일 크기별로 정렬 (큰 파일 먼저 - 로드 밸런싱)
        files_to_download.sort(key=lambda x: x[1], reverse=True)

        downloaded_files = []
        failed_files = []

        # 진행률 표시
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2,
            ) as progress:
                # 전체 진행률
                overall_task = progress.add_task(
                    "📦 Overall Progress", total=len(files_to_download)
                )

                # 각 파일 다운로드
                for s3_key, _file_size in files_to_download:
                    # 로컬 경로 계산
                    relative_path = s3_key[len(s3_prefix) :].lstrip("/")
                    local_path = local_dir / relative_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    # 분산 다운로드 실행
                    success, error = self.download_file_distributed(
                        s3_key, local_path, progress=progress, resume=True
                    )

                    if success:
                        downloaded_files.append(str(local_path))
                    else:
                        failed_files.append((s3_key, error))

                    progress.update(overall_task, advance=1)
        else:
            # 진행률 없이 다운로드
            for s3_key, _file_size in files_to_download:
                relative_path = s3_key[len(s3_prefix) :].lstrip("/")
                local_path = local_dir / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                success, error = self.download_file_distributed(
                    s3_key, local_path, resume=True
                )

                if success:
                    downloaded_files.append(str(local_path))
                else:
                    failed_files.append((s3_key, error))

        # 결과 요약
        if failed_files:
            console.print(
                f"[yellow]⚠️ {len(failed_files)} files failed to download[/yellow]"
            )
            for key, error in failed_files[:5]:  # 처음 5개만 표시
                console.print(f"[red]   {key}: {error}[/red]")

        console.print(
            f"[green]✅ Downloaded {len(downloaded_files)}/{len(files_to_download)} files[/green]"
        )

        return len(downloaded_files) > 0, downloaded_files

    def benchmark_download_speed(self, test_file_size_mb: int = 100) -> float:
        """다운로드 속도 벤치마크"""
        test_key = f"benchmark/test_{test_file_size_mb}mb.bin"
        test_file = Path(f"/tmp/benchmark_test_{test_file_size_mb}mb.bin")

        console.print("[cyan]🏃 Running download speed benchmark...[/cyan]")

        try:
            # 테스트 파일 생성 (없으면)
            try:
                self.get_file_info(test_key)
            except Exception:  # noqa: S110
                console.print("[yellow]Creating test file in S3...[/yellow]")
                test_data = os.urandom(test_file_size_mb * 1024 * 1024)
                self.s3_client.put_object(
                    Bucket=self.bucket, Key=test_key, Body=test_data
                )

            # 다운로드 속도 측정
            start_time = time.time()
            success, _ = self.download_file_distributed(
                test_key, test_file, resume=False
            )
            elapsed_time = time.time() - start_time

            if success:
                speed_mbps = (test_file_size_mb * 8) / elapsed_time
                console.print(
                    f"[green]✅ Download speed: {speed_mbps:.2f} Mbps[/green]"
                )

                # 정리
                test_file.unlink(missing_ok=True)
                return speed_mbps
            else:
                console.print("[red]❌ Benchmark failed[/red]")
                return 0.0

        except Exception as e:
            console.print(f"[red]❌ Benchmark error: {e}[/red]")
            return 0.0

    def get_system_stats(self) -> dict[str, Any]:
        """시스템 리소스 상태 확인"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "network_mbps": self.estimate_network_bandwidth(),
            "disk_free_gb": psutil.disk_usage("/").free / (1024**3),
        }

    def estimate_network_bandwidth(self) -> float:
        """네트워크 대역폭 추정"""
        net_stats = psutil.net_io_counters()
        time.sleep(1)
        net_stats_after = psutil.net_io_counters()

        bytes_recv = net_stats_after.bytes_recv - net_stats.bytes_recv
        mbps = (bytes_recv * 8) / (1024 * 1024)
        return mbps

    def auto_optimize_settings(self):
        """시스템 상태에 따라 설정 자동 최적화"""
        stats = self.get_system_stats()

        # CPU 사용률이 높으면 워커 수 감소
        if stats["cpu_percent"] > 80:
            self.max_workers = max(4, self.max_workers // 2)
            console.print(
                f"[yellow]High CPU usage, reducing workers to {self.max_workers}[/yellow]"
            )

        # 메모리 부족시 청크 크기 감소
        if stats["memory_percent"] > 85:
            self.chunk_size = self.chunk_size // 2
            console.print(
                f"[yellow]High memory usage, reducing chunk size to {self.chunk_size/(1024**2):.0f}MB[/yellow]"
            )

        # 디스크 공간 확인
        if stats["disk_free_gb"] < 10:
            console.print(
                f"[red]⚠️ Low disk space: {stats['disk_free_gb']:.2f}GB free[/red]"
            )

        return stats


def display_download_summary(stats: dict[str, Any]):
    """다운로드 요약 정보 표시"""
    table = Table(title="Download Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Files", str(stats.get("total_files", 0)))
    table.add_row("Downloaded", str(stats.get("downloaded_files", 0)))
    table.add_row("Failed", str(stats.get("failed_files", 0)))
    table.add_row("Total Size", f"{stats.get('total_size_gb', 0):.2f} GB")
    table.add_row("Elapsed Time", f"{stats.get('elapsed_time', 0):.1f} seconds")
    table.add_row("Average Speed", f"{stats.get('avg_speed_mbps', 0):.2f} Mbps")

    console.print(table)
