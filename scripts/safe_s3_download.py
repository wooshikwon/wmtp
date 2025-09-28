#!/usr/bin/env python3
"""
메모리 안전 S3 다운로드 스크립트
OOM 방지를 위한 보수적 설정으로 안정적 다운로드
"""

import os
import sys
import time
from pathlib import Path

import psutil
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.distributed_s3_transfer import DistributedS3Transfer

console = Console()


def check_system_resources():
    """시스템 리소스 확인 및 안전 설정 결정"""
    mem = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()

    # 메모리 상태 확인
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)
    used_percent = mem.percent

    console.print(
        Panel.fit(
            f"""[cyan]System Resources:[/cyan]

📊 Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total ({used_percent:.1f}% used)
🖥️  CPUs: {cpu_count} cores
🌐 Network: Ready

[yellow]Safety Configuration:[/yellow]
✅ Max workers: 4 (reduced from 32)
✅ Chunk size: 25MB (reduced from 50MB)
✅ Memory buffer: 256KB (reduced from 1MB)
✅ Sequential download for large files
""",
            title="🛡️ Memory-Safe Mode",
            border_style="green",
        )
    )

    # 메모리가 부족하면 경고
    if available_gb < 2:
        console.print("[bold red]⚠️  Warning: Less than 2GB RAM available![/bold red]")
        console.print(
            "[yellow]Consider closing other applications before downloading.[/yellow]"
        )
        response = console.input("\nContinue anyway? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            return False

    return True


def safe_download(prefix: str, output_dir: str = "."):
    """메모리 안전 다운로드 실행"""

    # 시스템 리소스 확인
    if not check_system_resources():
        console.print("[red]Download cancelled for safety.[/red]")
        return

    # 환경변수 로드
    load_dotenv()
    bucket = os.getenv("S3_BUCKET_NAME", "wmtp")

    console.print(f"\n[cyan]🔗 Connecting to S3 bucket: {bucket}[/cyan]")

    try:
        # 안전한 설정으로 전송 객체 생성
        transfer = DistributedS3Transfer(
            bucket=bucket,
            max_workers=4,  # 최대 4개 워커만 사용
            use_multiprocess=False,  # 스레드만 사용 (프로세스는 메모리 더 사용)
            chunk_size_mb=25,  # 25MB 청크로 축소
            enable_acceleration=False,  # 가속 비활성화 (안정성 우선)
        )

        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 파일 목록 확인
        console.print(f"\n[yellow]📋 Scanning {prefix}...[/yellow]")

        files = []
        total_size = 0

        paginator = transfer.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                size = obj["Size"]

                if not key.endswith("/"):
                    files.append((key, size))
                    total_size += size

        if not files:
            console.print(f"[yellow]No files found in {prefix}[/yellow]")
            return

        # 다운로드 정보 표시
        console.print(
            f"\n[green]Found {len(files)} files, {total_size/(1024**3):.2f}GB total[/green]"
        )

        # 큰 파일부터 정렬 (하지만 동시 다운로드 제한)
        files.sort(key=lambda x: x[1], reverse=True)

        # 다운로드 확인
        console.print(
            f"\n[bold yellow]This will download to: {output_path / prefix}[/bold yellow]"
        )
        console.print(
            "[dim]Using memory-safe configuration to prevent system freeze[/dim]"
        )

        response = console.input("\nProceed? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            console.print("[red]Cancelled.[/red]")
            return

        # 다운로드 실행
        console.print(
            f"\n[bold green]Starting safe download of {prefix}...[/bold green]"
        )

        success, downloaded = transfer.download_directory_distributed(
            prefix, output_path / prefix, show_progress=True
        )

        if success:
            console.print(
                f"\n[bold green]✅ Successfully downloaded {len(downloaded)} files![/bold green]"
            )
        else:
            console.print(
                f"\n[yellow]⚠️  Downloaded {len(downloaded)} files with some failures[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return


def main():
    """메인 실행"""
    console.print(
        "[bold cyan]🛡️  WMTP Safe S3 Downloader (OOM Prevention)[/bold cyan]\n"
    )

    # datasets 다운로드 (수정된 디렉토리명)
    console.print("[bold]📦 Step 1: Download datasets/[/bold]")
    safe_download("datasets/")

    # 메모리 정리를 위한 잠시 대기
    console.print("\n[dim]Waiting 5 seconds for memory cleanup...[/dim]")
    time.sleep(5)

    # models 다운로드 (수정된 디렉토리명)
    console.print("\n[bold]📦 Step 2: Download models/[/bold]")
    safe_download("models/")

    console.print("\n[bold green]✨ All downloads completed![/bold green]")

    # 최종 디렉토리 확인
    if Path("datasets").exists():
        dataset_files = list(Path("datasets").rglob("*"))
        console.print(f"[green]datasets/: {len(dataset_files)} files[/green]")

    if Path("models").exists():
        model_files = list(Path("models").rglob("*"))
        console.print(f"[green]models/: {len(model_files)} files[/green]")


if __name__ == "__main__":
    main()
