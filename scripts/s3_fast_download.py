#!/usr/bin/env python3
"""
S3 고속 다운로드 스크립트 - 최적화 버전
"""

import os
import subprocess
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv
from rich.console import Console

console = Console()


def download_with_aws_cli(source_s3_path: str, local_dir: str, region: str = "eu-north-1"):
    """AWS CLI를 사용한 고속 다운로드"""
    console.print("[bold blue]🚀 AWS CLI 고속 다운로드 시작[/bold blue]")

    # AWS CLI 명령어 구성
    cmd = [
        "aws", "s3", "sync",
        source_s3_path,
        local_dir,
        "--region", region,
        "--cli-read-timeout", "0",
        "--cli-connect-timeout", "60",
        "--max-concurrent-requests", "20",
        "--max-bandwidth", "1000MB/s"
    ]

    console.print(f"명령어: {' '.join(cmd)}")

    try:
        # 실행
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode == 0:
            console.print("[green]✅ 다운로드 완료![/green]")
            return True
        else:
            console.print(f"[red]❌ 다운로드 실패 (코드: {result.returncode})[/red]")
            return False

    except Exception as e:
        console.print(f"[red]❌ 에러: {e}[/red]")
        return False


def test_download_speed(bucket: str = "wmtp", region: str = "eu-north-1"):
    """다운로드 속도 테스트"""
    console.print("[yellow]📊 S3 연결 속도 테스트 중...[/yellow]")

    # 환경변수 로드
    load_dotenv()

    # S3 클라이언트 생성
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region
    )

    # 작은 파일로 테스트
    test_key = "models/Sheared-LLaMA-2.7B/config.json"

    import time
    start = time.time()

    try:
        # 메타데이터만 가져오기
        response = s3.head_object(Bucket=bucket, Key=test_key)
        latency = (time.time() - start) * 1000

        console.print(f"✅ S3 연결 성공!")
        console.print(f"  리전: {region}")
        console.print(f"  지연시간: {latency:.0f}ms")

        if latency > 500:
            console.print("[yellow]⚠️ 지연시간이 높습니다. 가까운 리전 사용을 권장합니다.[/yellow]")
            console.print("  한국: ap-northeast-2 (서울)")
            console.print("  일본: ap-northeast-1 (도쿄)")
            console.print("  미국: us-east-1 (버지니아)")

    except Exception as e:
        console.print(f"[red]❌ 연결 실패: {e}[/red]")


def download_sheared_llama_fast():
    """Sheared-LLaMA 고속 다운로드"""

    # 1. 속도 테스트
    test_download_speed()

    # 2. AWS CLI로 다운로드
    source = "s3://wmtp/models/Sheared-LLaMA-2.7B/"
    dest = "./sheared-llama-download/"

    console.print(f"\n[cyan]다운로드할 모델: Sheared-LLaMA-2.7B[/cyan]")
    console.print(f"[cyan]저장 위치: {dest}[/cyan]")

    # AWS CLI 사용 가능 확인
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        console.print("[green]✅ AWS CLI 사용 가능[/green]")

        # AWS CLI로 다운로드
        success = download_with_aws_cli(source, dest)

        if success:
            console.print(f"\n[bold green]✨ 모델이 {dest}에 다운로드되었습니다![/bold green]")

    except subprocess.CalledProcessError:
        console.print("[red]❌ AWS CLI가 설치되지 않았습니다.[/red]")
        console.print("설치: brew install awscli")


def main():
    """메인 실행"""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 속도 테스트만
        test_download_speed()
    else:
        # 전체 다운로드
        download_sheared_llama_fast()


if __name__ == "__main__":
    main()