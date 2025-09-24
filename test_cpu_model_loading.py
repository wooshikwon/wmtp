#!/usr/bin/env python3
"""
M3 MacBook Pro 메모리 최적화 모델 로딩 테스트

CPU 전용 Facebook MTP 로더의 메모리 사용량을 실시간 모니터링하면서
27GB 모델이 64GB RAM에서 안전하게 로드되는지 확인합니다.
"""

import os
import time

import psutil

# MLflow 환경변수 설정
os.environ["MLFLOW_TRACKING_URI"] = "./mlflow_runs"
os.environ["MLFLOW_REGISTRY_URI"] = "./mlflow_runs"

from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()


def get_memory_info():
    """현재 메모리 사용량 정보 반환."""
    memory = psutil.virtual_memory()
    process = psutil.Process()
    process_memory = process.memory_info()

    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent": memory.percent,
        "process_gb": process_memory.rss / (1024**3),
        "process_vms_gb": process_memory.vms / (1024**3),
    }


def create_memory_table(mem_info, status="Loading..."):
    """메모리 정보 테이블 생성."""
    table = Table(title=f"🖥️  M3 MacBook Pro 메모리 모니터링 - {status}")

    table.add_column("항목", style="cyan")
    table.add_column("값", style="white")
    table.add_column("비율", style="yellow")

    # 시스템 메모리
    table.add_row("시스템 전체 RAM", f"{mem_info['total_gb']:.1f} GB", "100%")
    table.add_row(
        "사용 중 메모리", f"{mem_info['used_gb']:.1f} GB", f"{mem_info['percent']:.1f}%"
    )
    table.add_row(
        "사용 가능 메모리",
        f"{mem_info['available_gb']:.1f} GB",
        f"{100 - mem_info['percent']:.1f}%",
    )
    table.add_row("", "", "")

    # 프로세스 메모리
    table.add_row(
        "현재 프로세스 (실제)",
        f"{mem_info['process_gb']:.1f} GB",
        f"{mem_info['process_gb'] / mem_info['total_gb'] * 100:.1f}%",
    )
    table.add_row(
        "현재 프로세스 (가상)",
        f"{mem_info['process_vms_gb']:.1f} GB",
        f"{mem_info['process_vms_gb'] / mem_info['total_gb'] * 100:.1f}%",
    )

    return table


def test_memory_optimized_loading():
    """메모리 최적화 모델 로딩 테스트."""
    console.print(
        "[bold green]🚀 M3 MacBook Pro CPU 최적화 모델 로딩 테스트[/bold green]"
    )
    console.print(
        "실시간 메모리 모니터링과 함께 27GB Facebook MTP 모델을 로드합니다.\n"
    )

    # 초기 메모리 상태 확인
    initial_mem = get_memory_info()
    console.print("🔍 초기 메모리 상태:")
    console.print(f"   - 사용 가능: {initial_mem['available_gb']:.1f} GB")
    console.print(f"   - 프로세스: {initial_mem['process_gb']:.1f} GB\n")

    if initial_mem["available_gb"] < 20:
        console.print(
            "[red]⚠️  사용 가능한 메모리가 20GB 미만입니다. 다른 애플리케이션을 종료하세요.[/red]"
        )
        return False

    # 모니터링과 함께 로딩 시작
    with Live(
        create_memory_table(initial_mem, "준비 중..."), refresh_per_second=2
    ) as live:
        try:
            # 설정 로딩
            live.update(create_memory_table(get_memory_info(), "설정 로딩 중..."))
            from src.settings import load_config, load_recipe

            config = load_config("configs/config.experiment.yaml")
            recipe = load_recipe("configs/recipe.baseline_quick.yaml")

            # ComponentFactory 생성
            live.update(create_memory_table(get_memory_info(), "Factory 생성 중..."))
            from src.factory.component_factory import ComponentFactory

            factory = ComponentFactory()

            # CPU 최적화 로더 생성
            live.update(create_memory_table(get_memory_info(), "로더 생성 중..."))
            loader = factory.create_model_loader(config, recipe)

            loader_type = type(loader).__name__
            live.update(
                create_memory_table(get_memory_info(), f"로더 생성 완료: {loader_type}")
            )
            time.sleep(2)

            if loader_type != "MTPNativeCPULoader":
                console.print(
                    f"[yellow]⚠️  예상한 CPU 로더가 아닙니다: {loader_type}[/yellow]"
                )

            # 모델 로딩 시작
            live.update(create_memory_table(get_memory_info(), "모델 로딩 시작..."))

            model_path = "models/7b_1t_4"

            # 단계별 메모리 모니터링
            start_time = time.time()
            model = loader.load_model(model_path)
            end_time = time.time()

            # 최종 메모리 상태
            final_mem = get_memory_info()
            live.update(
                create_memory_table(
                    final_mem, f"로딩 완료! ({end_time - start_time:.1f}초)"
                )
            )

        except Exception as e:
            final_mem = get_memory_info()
            live.update(create_memory_table(final_mem, f"오류 발생: {str(e)[:50]}..."))
            console.print(f"\n[red]❌ 모델 로딩 실패: {e}[/red]")
            return False

    # 결과 분석
    memory_used = final_mem["process_gb"] - initial_mem["process_gb"]

    console.print("\n[bold blue]📊 로딩 결과 분석[/bold blue]")
    console.print("✅ 모델 로딩 성공!")
    console.print(f"⏱️  소요 시간: {end_time - start_time:.1f}초")
    console.print(f"💾 메모리 사용량 증가: {memory_used:.1f} GB")
    console.print(f"📈 최종 시스템 메모리 사용률: {final_mem['percent']:.1f}%")

    # 간단한 추론 테스트
    console.print("\n[bold blue]🧪 간단한 추론 테스트[/bold blue]")
    try:
        # 더미 입력으로 모델 테스트
        import torch

        dummy_input = torch.randint(0, 1000, (1, 10))  # 배치=1, 시퀀스=10

        with torch.no_grad():
            output = model(dummy_input, start_pos=0)
            console.print(f"✅ 추론 성공: 출력 shape = {output.shape}")

    except Exception as e:
        console.print(f"⚠️  추론 테스트 실패: {e}")

    return True


if __name__ == "__main__":
    success = test_memory_optimized_loading()

    if success:
        console.print("\n[bold green]🎉 CPU 최적화 모델 로딩 성공![/bold green]")
        console.print("이제 전체 실험을 실행할 수 있습니다.")
    else:
        console.print("\n[bold red]💥 모델 로딩 실패[/bold red]")
        console.print("메모리 최적화가 더 필요합니다.")
