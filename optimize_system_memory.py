#!/usr/bin/env python3
"""
M3 MacBook Pro 시스템 메모리 최적화 도구

27GB Facebook MTP 모델 로딩을 위한 시스템 메모리를 최적화합니다.
"""

import subprocess

import psutil
from rich.console import Console
from rich.panel import Panel

console = Console()


def check_system_memory():
    """현재 시스템 메모리 상태 확인."""
    memory = psutil.virtual_memory()

    console.print(
        Panel.fit(
            f"🖥️  M3 MacBook Pro 메모리 상태\n\n"
            f"전체 RAM: {memory.total / (1024**3):.1f} GB\n"
            f"사용 중: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)\n"
            f"사용 가능: {memory.available / (1024**3):.1f} GB\n"
            f"여유 공간: {memory.free / (1024**3):.1f} GB",
            title="현재 메모리 상태",
        )
    )

    return memory.available / (1024**3)


def find_memory_hogs():
    """메모리를 많이 사용하는 프로세스 찾기."""
    console.print("\n[bold blue]🔍 메모리 사용량 상위 프로세스[/bold blue]")

    processes = []
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            memory_mb = proc.info["memory_info"].rss / (1024**2)
            if memory_mb > 100:  # 100MB 이상만 표시
                processes.append(
                    {
                        "pid": proc.info["pid"],
                        "name": proc.info["name"],
                        "memory_gb": memory_mb / 1024,
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # 메모리 사용량으로 정렬
    processes.sort(key=lambda x: x["memory_gb"], reverse=True)

    for i, proc in enumerate(processes[:10]):
        console.print(
            f"{i+1:2d}. {proc['name']:<20} {proc['memory_gb']:>6.1f} GB (PID: {proc['pid']})"
        )

    return processes


def suggest_optimizations(available_gb):
    """메모리 최적화 제안."""
    console.print("\n[bold yellow]💡 메모리 최적화 제안[/bold yellow]")

    if available_gb < 30:
        console.print("[red]⚠️  사용 가능한 메모리가 30GB 미만입니다![/red]")
        console.print("\n추천 조치:")
        console.print("1. 다른 애플리케이션 종료 (Chrome, Slack, IDE 등)")
        console.print("2. Docker Desktop 중지")
        console.print("3. 메모리 정리 실행")

    elif available_gb < 40:
        console.print("[yellow]⚠️  메모리 여유가 부족합니다.[/yellow]")
        console.print("\n권장 조치:")
        console.print("1. 불필요한 브라우저 탭 정리")
        console.print("2. 대용량 애플리케이션 일시 종료")

    else:
        console.print("[green]✅ 메모리 상태가 양호합니다![/green]")
        console.print("27GB 모델 로딩에 충분한 여유 공간이 있습니다.")


def clean_memory():
    """메모리 정리 시도."""
    console.print("\n[bold blue]🧹 메모리 정리 실행[/bold blue]")

    try:
        # Python 가비지 컬렉션
        import gc

        gc.collect()
        console.print("✅ Python 가비지 컬렉션 완료")

        # macOS 메모리 압축 해제 (메모리 확보)
        try:
            subprocess.run(["sudo", "purge"], check=True, capture_output=True)
            console.print("✅ macOS 메모리 캐시 정리 완료")
        except subprocess.CalledProcessError:
            console.print("⚠️  메모리 캐시 정리 실패 (sudo 권한 필요)")

    except Exception as e:
        console.print(f"❌ 메모리 정리 중 오류: {e}")


def main():
    """메인 함수."""
    console.print("[bold green]🔧 M3 MacBook Pro 메모리 최적화 도구[/bold green]")
    console.print("27GB Facebook MTP 모델 로딩을 위한 시스템 최적화\n")

    # 현재 메모리 상태 확인
    available_gb = check_system_memory()

    # 메모리 사용량 분석
    processes = find_memory_hogs()

    # 최적화 제안
    suggest_optimizations(available_gb)

    # 메모리 정리 옵션
    if available_gb < 35:
        console.print("\n[bold blue]메모리 정리를 실행하시겠습니까?[/bold blue]")
        response = input("y/N: ").lower().strip()
        if response == "y":
            clean_memory()

            # 정리 후 상태 확인
            console.print("\n[bold green]정리 후 메모리 상태:[/bold green]")
            check_system_memory()

    console.print("\n[bold cyan]📋 다음 단계:[/bold cyan]")
    console.print("1. python test_cpu_model_loading.py  # 최적화된 모델 로딩 테스트")
    console.print("2. python run_quick_experiment.py    # 전체 실험 실행")


if __name__ == "__main__":
    main()
