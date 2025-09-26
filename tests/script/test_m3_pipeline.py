#!/usr/bin/env python3
"""WMTP M3 Pipeline 종합 테스트 스크립트

이 스크립트는 MacBook M3 환경에서 WMTP의 모든 알고리즘 변형을 체계적으로 테스트합니다:
1. baseline-mtp: 균등 가중치 기본 MTP
2. critic-wmtp: Value Function 기반 동적 가중치
3. rho1-wmtp-tokenskip: Token Skip 모드 Rho1 WMTP
4. rho1-wmtp-weighted: Weighted 모드 Rho1 WMTP

각 알고리즘에 대해 dry-run과 실제 학습을 모두 테스트하여
전체 파이프라인의 안정성을 검증합니다.

Usage:
    # 특정 config와 recipe로 테스트
    python test_m3_pipeline.py --config tests/configs/config.local_test.yaml --recipe tests/configs/recipe.critic_wmtp.yaml

    # Dry-run으로 설정 검증만
    python test_m3_pipeline.py --config tests/configs/config.local_test.yaml --recipe tests/configs/recipe.mtp_baseline.yaml --dry-run

    # 상세 출력으로 디버깅
    python test_m3_pipeline.py --config tests/configs/config.local_test.yaml --recipe tests/configs/recipe.rho1_wmtp_tokenskip.yaml --verbose
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import warnings

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.text import Text
from rich import box

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

console = Console()

# 테스트할 레시피 목록과 설명
RECIPES = {
    "mtp_baseline": {
        "name": "MTP Baseline",
        "description": "균등 가중치 기본 Multi-Token Prediction",
        "config": "tests/configs/config.local_test.yaml",
        "recipe": "tests/configs/recipe.mtp_baseline.yaml",
        "expected_algo": "baseline-mtp"
    },
    "critic_wmtp": {
        "name": "Critic WMTP",
        "description": "Value Function 기반 동적 가중치",
        "config": "tests/configs/config.local_test.yaml",
        "recipe": "tests/configs/recipe.critic_wmtp.yaml",
        "expected_algo": "critic-wmtp"
    },
    "rho1_wmtp_tokenskip": {
        "name": "Rho1 WMTP (Token Skip)",
        "description": "Token Skip 모드 - 하위 30% 토큰 제거",
        "config": "tests/configs/config.local_test.yaml",
        "recipe": "tests/configs/recipe.rho1_wmtp_tokenskip.yaml",
        "expected_algo": "rho1-wmtp"
    },
    "rho1_wmtp_weighted": {
        "name": "Rho1 WMTP (Weighted)",
        "description": "Weighted 모드 - 연속적 토큰 가중치",
        "config": "tests/configs/config.local_test.yaml",
        "recipe": "tests/configs/recipe.rho1_wmtp_weighted.yaml",
        "expected_algo": "rho1-wmtp"
    }
}

class TestResult:
    """테스트 결과를 저장하는 클래스"""
    def __init__(self, recipe_name: str):
        self.recipe_name = recipe_name
        self.dry_run_success: Optional[bool] = None
        self.dry_run_time: Optional[float] = None
        self.dry_run_error: Optional[str] = None
        self.train_success: Optional[bool] = None
        self.train_time: Optional[float] = None
        self.train_error: Optional[str] = None

    @property
    def overall_success(self) -> bool:
        """전체 테스트 성공 여부"""
        return (
            (self.dry_run_success is None or self.dry_run_success) and
            (self.train_success is None or self.train_success)
        )

def check_environment() -> List[str]:
    """테스트 환경 검증"""
    issues = []

    # PyTorch 및 MPS 확인
    try:
        import torch
        console.print(f"✓ PyTorch {torch.__version__}")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            console.print("✓ MPS (Metal Performance Shaders) 사용 가능")
        else:
            console.print("⚠ MPS 없음, CPU 사용")
            issues.append("MPS unavailable")
    except ImportError:
        issues.append("PyTorch not installed")

    # 메모리 확인
    try:
        import psutil
        mem = psutil.virtual_memory()
        console.print(f"✓ Memory: {mem.total / (1024**3):.1f}GB total, {mem.available / (1024**3):.1f}GB available")
        if mem.available < 8 * (1024**3):  # 8GB
            issues.append("Low memory (<8GB available)")
    except ImportError:
        console.print("⚠ psutil not installed, cannot check memory")

    # 필수 디렉토리 확인
    required_dirs = [
        Path("tests/configs"),
        Path("tests/tiny_models"),
        Path("tests/test_dataset"),
        Path("src/cli"),
        Path("src/pipelines")
    ]

    for dir_path in required_dirs:
        if dir_path.exists():
            console.print(f"✓ {dir_path}")
        else:
            console.print(f"❌ {dir_path}")
            issues.append(f"Missing directory: {dir_path}")

    return issues

def validate_config_files() -> List[str]:
    """설정 파일 유효성 검사"""
    issues = []

    for recipe_key, recipe_info in RECIPES.items():
        config_path = Path(recipe_info["config"])
        recipe_path = Path(recipe_info["recipe"])

        # 파일 존재 확인
        if not config_path.exists():
            issues.append(f"Config file missing: {config_path}")
            continue
        if not recipe_path.exists():
            issues.append(f"Recipe file missing: {recipe_path}")
            continue

        # YAML 파싱 확인
        try:
            import yaml
            with open(config_path) as f:
                yaml.safe_load(f)
            with open(recipe_path) as f:
                recipe_data = yaml.safe_load(f)

            # 알고리즘 일치 확인
            actual_algo = recipe_data.get("train", {}).get("algo")
            expected_algo = recipe_info["expected_algo"]
            if actual_algo != expected_algo:
                issues.append(f"{recipe_key}: Algorithm mismatch - expected {expected_algo}, got {actual_algo}")

        except Exception as e:
            issues.append(f"{recipe_key}: YAML parse error - {e}")

    return issues

def run_single_test(config_path: str, recipe_path: str, test_name: str, dry_run: bool = False, verbose: bool = False) -> Tuple[bool, float, Optional[str]]:
    """단일 테스트 실행

    Args:
        config_path: 설정 파일 경로
        recipe_path: 레시피 파일 경로
        test_name: 테스트 이름
        dry_run: Dry-run 모드 여부
        verbose: 상세 출력 여부

    Returns:
        (성공여부, 실행시간, 에러메시지)
    """
    # CLI 명령 구성
    cmd = [
        sys.executable, "-m", "src.cli.train",
        "--config", config_path,
        "--recipe", recipe_path,
        "--run-name", f"test_{test_name}",
        "--tags", f"test,m3,{test_name}"
    ]

    if dry_run:
        cmd.append("--dry-run")

    if verbose:
        cmd.append("--verbose")

    # 실행
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,  # 프로젝트 루트
            capture_output=True,
            text=True,
            timeout=300 if dry_run else 1800  # dry-run: 5분, train: 30분
        )
        end_time = time.time()

        success = result.returncode == 0
        error = None if success else f"Exit code: {result.returncode}\nSTDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}"

        return success, end_time - start_time, error

    except subprocess.TimeoutExpired:
        return False, time.time() - start_time, "Test timeout"
    except Exception as e:
        return False, time.time() - start_time, f"Process error: {e}"

def run_test(config_path: str, recipe_path: str, dry_run: bool, verbose: bool) -> TestResult:
    """단일 설정으로 테스트 실행"""
    # 테스트 이름 생성
    recipe_name = Path(recipe_path).stem.replace('recipe.', '')
    result = TestResult(recipe_name)

    console.print(f"\n[bold blue]🧪 {recipe_name} 테스트 시작[/bold blue]")

    if dry_run:
        console.print(f"[yellow]Dry-run 검증 중...[/yellow]")
        success, duration, error = run_single_test(config_path, recipe_path, recipe_name, dry_run=True, verbose=verbose)
        result.dry_run_success = success
        result.dry_run_time = duration
        result.dry_run_error = error

        if success:
            console.print(f"[green]✅ Dry-run 성공 ({duration:.1f}초)[/green]")
        else:
            console.print(f"[red]❌ Dry-run 실패 ({duration:.1f}초)[/red]")
            if verbose and error:
                console.print(f"[dim]Error: {error[:200]}...[/dim]")
    else:
        # Dry-run 먼저 실행
        console.print(f"[yellow]Dry-run 검증 중...[/yellow]")
        success, duration, error = run_single_test(config_path, recipe_path, recipe_name, dry_run=True, verbose=verbose)
        result.dry_run_success = success
        result.dry_run_time = duration
        result.dry_run_error = error

        if success:
            console.print(f"[green]✅ Dry-run 성공 ({duration:.1f}초)[/green]")

            # 실제 학습 실행
            console.print(f"[green]실제 학습 테스트 중...[/green]")
            success, duration, error = run_single_test(config_path, recipe_path, recipe_name, dry_run=False, verbose=verbose)
            result.train_success = success
            result.train_time = duration
            result.train_error = error

            if success:
                console.print(f"[green]✅ 학습 성공 ({duration:.1f}초)[/green]")
            else:
                console.print(f"[red]❌ 학습 실패 ({duration:.1f}초)[/red]")
                if verbose and error:
                    console.print(f"[dim]Error: {error[:200]}...[/dim]")
        else:
            console.print(f"[red]❌ Dry-run 실패 ({duration:.1f}초)[/red]")
            console.print("[yellow]Dry-run 실패로 인해 학습 테스트 스킵[/yellow]")
            if verbose and error:
                console.print(f"[dim]Error: {error[:200]}...[/dim]")

    return result

def print_summary(result: TestResult, dry_run: bool):
    """테스트 결과 요약 출력"""
    # 결과 테이블 생성
    table = Table(title="WMTP M3 Pipeline 테스트 결과", box=box.ROUNDED)
    table.add_column("테스트", style="cyan")
    table.add_column("Dry-run", justify="center")

    if not dry_run:
        table.add_column("학습", justify="center")
        table.add_column("전체", justify="center")

    # Dry-run 상태
    if result.dry_run_success is True:
        dry_status = f"[green]✅ {result.dry_run_time:.1f}s[/green]"
    elif result.dry_run_success is False:
        dry_status = f"[red]❌ {result.dry_run_time:.1f}s[/red]"
    else:
        dry_status = "[dim]미실행[/dim]"

    if dry_run:
        success = result.dry_run_success
        table.add_row(
            result.recipe_name,
            dry_status
        )
    else:
        # 학습 상태
        if result.train_success is True:
            train_status = f"[green]✅ {result.train_time:.1f}s[/green]"
        elif result.train_success is False:
            train_status = f"[red]❌ {result.train_time:.1f}s[/red]"
        else:
            train_status = "[yellow]스킵[/yellow]"

        # 전체 상태
        if result.overall_success:
            overall_status = "[green]✅ 성공[/green]"
            success = True
        else:
            overall_status = "[red]❌ 실패[/red]"
            success = False

        table.add_row(
            result.recipe_name,
            dry_status,
            train_status,
            overall_status
        )

    console.print(table)

    # 최종 결과
    if success:
        console.print("\n[bold green]🎉 테스트 성공! WMTP 파이프라인이 정상 작동합니다.[/bold green]")
    else:
        console.print("\n[bold red]⚠️ 테스트 실패[/bold red]")
        # 실패 원인 상세 표시
        if result.dry_run_success is False and result.dry_run_error:
            console.print(f"  Dry-run 오류: {result.dry_run_error[:100]}...")
        if result.train_success is False and result.train_error:
            console.print(f"  학습 오류: {result.train_error[:100]}...")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="WMTP M3 Pipeline 종합 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--config", "-c",
        required=True,
        help="환경 설정 YAML 파일 경로"
    )
    parser.add_argument(
        "--recipe", "-r",
        required=True,
        help="훈련 레시피 YAML 파일 경로"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run만 실행 (실제 학습 스킵)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력"
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="환경 검사 스킵"
    )

    args = parser.parse_args()

    # 파일 존재 확인
    config_path = Path(args.config)
    recipe_path = Path(args.recipe)

    if not config_path.exists():
        parser.error(f"Config 파일을 찾을 수 없습니다: {config_path}")
    if not recipe_path.exists():
        parser.error(f"Recipe 파일을 찾을 수 없습니다: {recipe_path}")

    # 헤더 출력
    console.print(Panel.fit(
        "[bold cyan]WMTP M3 Pipeline 테스트[/bold cyan]\n"
        f"Config: {args.config}\n"
        f"Recipe: {args.recipe}",
        title="🧪 테스트 시작"
    ))

    # 환경 검사
    if not args.skip_env_check:
        console.print("\n[bold]환경 검사 중...[/bold]")
        env_issues = check_environment()
        if env_issues:
            console.print(f"\n[yellow]환경 이슈 발견: {len(env_issues)}개[/yellow]")
            for issue in env_issues:
                console.print(f"  • {issue}")

            if not console.input("\n계속 진행하시겠습니까? [y/N]: ").lower().startswith('y'):
                console.print("테스트 중단")
                return

    # 설정 파일 검증
    console.print("\n[bold]설정 파일 검증 중...[/bold]")
    try:
        import yaml
        with open(args.config) as f:
            config_data = yaml.safe_load(f)
        with open(args.recipe) as f:
            recipe_data = yaml.safe_load(f)

        algo = recipe_data.get("train", {}).get("algo")
        console.print(f"✅ 알고리즘: {algo}")
    except Exception as e:
        console.print(f"[red]설정 파일 검증 실패: {e}[/red]")
        return

    mode_desc = "Dry-run 검증만" if args.dry_run else "Dry-run + 실제 학습"
    console.print(f"\n[cyan]🔧 테스트 모드: {mode_desc}[/cyan]")

    # 사용자 확인
    if not args.dry_run:
        if not console.input("\n실제 학습을 포함한 테스트를 시작하시겠습니까? [y/N]: ").lower().startswith('y'):
            console.print("테스트 중단")
            return

    # 테스트 실행
    console.print("\n[bold green]🚀 테스트 시작![/bold green]")

    start_time = time.time()
    result = run_test(args.config, args.recipe, args.dry_run, args.verbose)
    total_time = time.time() - start_time

    # 결과 출력
    console.print(f"\n[bold]⏱️ 총 소요시간: {total_time:.1f}초[/bold]")
    print_summary(result, args.dry_run)

if __name__ == "__main__":
    # 경고 억제
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    main()