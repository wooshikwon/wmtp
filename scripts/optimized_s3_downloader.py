#!/usr/bin/env python3
"""
AWS S3 대용량 파일 최적화 다운로더

10GB+ 모델 파일들을 위한 멀티파트 병렬 다운로드 최적화
- TransferConfig로 병렬 다운로드 설정
- 진행률 표시
- 네트워크 재시도 로직
- 메모리 효율적 처리
"""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
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


class OptimizedS3Downloader:
    """S3 대용량 파일 최적화 다운로더"""

    def __init__(
        self, aws_access_key: str, aws_secret_key: str, region: str = "eu-north-1"
    ):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region,
        )
        self.bucket = "wmtp"

        # 대용량 파일용 최적화 설정
        self.transfer_config = TransferConfig(
            multipart_threshold=1024 * 25,  # 25MB 이상부터 멀티파트
            max_concurrency=10,  # 최대 10개 병렬 연결
            multipart_chunksize=1024 * 25,  # 25MB 청크
            use_threads=True,  # 스레드 풀 사용
            num_download_attempts=3,  # 재시도 3회
            max_io_queue=100,  # I/O 큐 크기
            io_chunksize=1024 * 256,  # 256KB I/O 청크 (메모리 효율)
        )

        console.print("[green]✅ S3 최적화 다운로더 초기화[/green]")
        console.print("  멀티파트 임계점: 25MB")
        console.print("  최대 병렬 연결: 10")
        console.print("  청크 크기: 25MB")

    def get_file_list(self, prefix: str) -> list[tuple[str, int]]:
        """S3 파일 리스트와 크기 가져오기"""
        console.print(f"[yellow]S3 파일 리스트 조회: {prefix}[/yellow]")

        files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                size = obj["Size"]
                files.append((key, size))

                # 파일 크기를 사람이 읽기 쉬운 형태로 표시
                if size > 1024**3:  # GB
                    size_str = f"{size / (1024**3):.1f}GB"
                elif size > 1024**2:  # MB
                    size_str = f"{size / (1024**2):.1f}MB"
                else:  # KB
                    size_str = f"{size / 1024:.1f}KB"

                console.print(f"  📁 {key.split('/')[-1]} ({size_str})")

        return files

    def download_file_optimized(
        self,
        s3_key: str,
        local_path: Path,
        file_size: int,
        progress: Progress,
        task_id: TaskID,
    ):
        """최적화된 단일 파일 다운로드"""
        try:
            # 파일 크기가 클 경우 멀티파트 다운로드 자동 적용
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # 진행률 콜백 함수
            def progress_callback(bytes_transferred):
                progress.update(task_id, advance=bytes_transferred)

            # S3 다운로드 (boto3가 자동으로 멀티파트 판단)
            self.s3_client.download_file(
                self.bucket,
                s3_key,
                str(local_path),
                Config=self.transfer_config,
                Callback=progress_callback,
            )

            return True, None

        except ClientError as e:
            error_msg = f"S3 Client Error: {e}"
            return False, error_msg
        except Exception as e:
            error_msg = f"Download Error: {e}"
            return False, error_msg

    def download_model_parallel(self, source_prefix: str, local_dir: Path) -> bool:
        """병렬 모델 다운로드"""
        console.print("[bold blue]🚀 최적화된 S3 다운로드 시작[/bold blue]")
        console.print(f"소스: s3://{self.bucket}/{source_prefix}")
        console.print(f"대상: {local_dir}")

        # 파일 리스트 가져오기
        files = self.get_file_list(source_prefix)
        if not files:
            console.print("[red]❌ 다운로드할 파일이 없습니다[/red]")
            return False

        # 총 파일 크기 계산
        total_size = sum(size for _, size in files)
        console.print(
            f"[cyan]총 다운로드 크기: {total_size / (1024**3):.2f}GB ({len(files)}개 파일)[/cyan]"
        )

        # 진행률 표시기 설정
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,  # 초당 4회 갱신
        ) as progress:
            # 각 파일별 다운로드 태스크 생성
            download_tasks = []
            for s3_key, file_size in files:
                file_name = s3_key.split("/")[-1]
                local_path = local_dir / s3_key.replace(source_prefix, "").lstrip("/")

                task_id = progress.add_task(f"📥 {file_name}", total=file_size)

                download_tasks.append((s3_key, local_path, file_size, task_id))

            # 대용량 파일은 순차 다운로드, 작은 파일들은 병렬 처리
            large_files = [
                (key, path, size, task_id)
                for key, path, size, task_id in download_tasks
                if size > 100 * 1024 * 1024
            ]  # 100MB 이상
            small_files = [
                (key, path, size, task_id)
                for key, path, size, task_id in download_tasks
                if size <= 100 * 1024 * 1024
            ]

            success_count = 0

            # 대용량 파일 순차 처리 (멀티파트 자동 적용)
            for s3_key, local_path, file_size, task_id in large_files:
                success, error = self.download_file_optimized(
                    s3_key, local_path, file_size, progress, task_id
                )
                if success:
                    success_count += 1
                    progress.update(task_id, completed=file_size)
                else:
                    console.print(f"[red]❌ 실패: {s3_key} - {error}[/red]")

            # 작은 파일들 병렬 처리 (최대 5개 동시)
            if small_files:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_task = {}
                    for s3_key, local_path, file_size, task_id in small_files:
                        future = executor.submit(
                            self.download_file_optimized,
                            s3_key,
                            local_path,
                            file_size,
                            progress,
                            task_id,
                        )
                        future_to_task[future] = (s3_key, task_id, file_size)

                    for future in as_completed(future_to_task):
                        s3_key, task_id, file_size = future_to_task[future]
                        try:
                            success, error = future.result()
                            if success:
                                success_count += 1
                                progress.update(task_id, completed=file_size)
                            else:
                                console.print(f"[red]❌ 실패: {s3_key} - {error}[/red]")
                        except Exception as e:
                            console.print(f"[red]❌ 예외: {s3_key} - {e}[/red]")

        # 결과 요약
        total_files = len(files)
        if success_count == total_files:
            console.print(
                f"[bold green]🎉 다운로드 완료: {success_count}/{total_files} 파일 성공[/bold green]"
            )
            return True
        else:
            console.print(
                f"[yellow]⚠️ 부분 성공: {success_count}/{total_files} 파일 완료[/yellow]"
            )
            return success_count > 0


def test_sheared_llama_download():
    """Sheared LLaMA 다운로드 테스트"""
    console.print("[bold blue]🧪 Sheared LLaMA 2.7B 최적화 다운로드 테스트[/bold blue]")

    # AWS 인증 정보 - 환경변수에서 로드
    downloader = OptimizedS3Downloader(
        os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        local_dir = temp_path / "sheared_llama"

        console.print(f"[cyan]임시 디렉토리: {local_dir}[/cyan]")

        # 최적화된 다운로드 실행
        success = downloader.download_model_parallel(
            "models/Sheared-LLaMA-2.7B/", local_dir
        )

        if success:
            # 다운로드된 파일 확인
            console.print("[green]📂 다운로드된 파일들:[/green]")
            for file_path in local_dir.rglob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    if size > 1024**3:
                        size_str = f"{size / (1024**3):.2f}GB"
                    elif size > 1024**2:
                        size_str = f"{size / (1024**2):.1f}MB"
                    else:
                        size_str = f"{size / 1024:.1f}KB"
                    console.print(f"  ✅ {file_path.name} ({size_str})")

            console.print(
                f"[bold green]✅ 다운로드 성공! 임시폴더: {local_dir}[/bold green]"
            )
            console.print("[yellow]💡 실제 변환 작업을 계속 진행하시겠습니까?[/yellow]")
        else:
            console.print("[red]❌ 다운로드 실패[/red]")


if __name__ == "__main__":
    # 환경변수 설정 - .env 파일에서 로드
    from dotenv import load_dotenv

    load_dotenv()

    test_sheared_llama_download()
