#!/usr/bin/env python3
"""
WMTP Framework 빠른 실험 스크립트

M3 MacBook Pro에서 세 가지 알고리즘을 비교하는 실험:
1. MTP Baseline (균등 가중치)
2. Critic WMTP (가치 함수 기반 가중치)
3. Rho-1 WMTP (참조 모델 기반 가중치)

각 실험은 50 스텝으로 제한하여 총 10분 내 완료 목표
"""

import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.pipelines import run_training
from src.settings import load_config, load_recipe

console = Console()


def run_experiment(
    algo_name: str, config_path: str, recipe_path: str, max_steps: int = 50
):
    """
    단일 알고리즘 실험 실행

    Args:
        algo_name: 알고리즘 이름 (표시용)
        config_path: 환경 설정 파일 경로
        recipe_path: 레시피 파일 경로
        max_steps: 최대 훈련 스텝 수

    Returns:
        dict: 실험 결과 (메트릭, 시간 등)
    """
    console.print(f"\n[bold blue]🔬 {algo_name} 실험 시작[/bold blue]")
    console.print(f"📝 레시피: {recipe_path}")

    start_time = time.time()

    try:
        # MLflow run 강제 종료 (이전 실험의 잔여 run 정리)
        try:
            import mlflow

            if mlflow.active_run():
                console.print("[dim]⚠️ 이전 MLflow run 종료 중...[/dim]")
                mlflow.end_run()
        except Exception:
            pass

        # 설정 로드
        config = load_config(config_path)
        recipe = load_recipe(recipe_path)

        console.print(f"[cyan]✅ 설정 로드 완료: {recipe.train.algo}[/cyan]")

        # 실험 실행 (max_steps로 제한)
        console.print(f"[yellow]⚡ 훈련 시작 (최대 {max_steps} 스텝)[/yellow]")

        # 고유한 실행명으로 MLflow run 충돌 방지
        import uuid

        unique_run_name = f"{algo_name.lower()}_quick_{uuid.uuid4().hex[:8]}"

        results = run_training(
            config,
            recipe,
            run_name=unique_run_name,
            tags=["quick_experiment", algo_name.lower()],
            dry_run=False,
            max_steps=max_steps,
        )

        end_time = time.time()
        duration = end_time - start_time

        console.print(f"[green]✅ {algo_name} 완료! ({duration:.1f}초)[/green]")

        return {
            "algorithm": algo_name,
            "duration": duration,
            "success": True,
            "metrics": results.trainer_metrics
            if hasattr(results, "trainer_metrics")
            else {},
            "final_loss": results.final_loss
            if hasattr(results, "final_loss")
            else None,
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        console.print(f"[red]❌ {algo_name} 실패: {e}[/red]")

        # 실패 시에도 MLflow run 정리
        try:
            import mlflow

            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
        except Exception:
            pass

        return {
            "algorithm": algo_name,
            "duration": duration,
            "success": False,
            "error": str(e),
            "metrics": {},
            "final_loss": None,
        }


def main():
    """메인 실험 함수"""

    console.print("[bold green]🚀 WMTP 빠른 비교 실험 시작[/bold green]")
    console.print("M3 MacBook Pro 64GB RAM 환경에서 세 알고리즘 비교")
    console.print("목표: 각 실험당 5분, 총 15분 내 완료\n")

    # 실험 설정
    config_path = "configs/config.experiment.yaml"
    experiments = [
        {
            "name": "MTP Baseline",
            "recipe": "configs/recipe.baseline_quick.yaml",
            "description": "기본 MTP (균등한 토큰 가중치)",
        },
        {
            "name": "Critic WMTP",
            "recipe": "configs/recipe.critic_quick.yaml",
            "description": "가치 함수 기반 토큰 가중치",
        },
        {
            "name": "Rho-1 WMTP",
            "recipe": "configs/recipe.rho1_quick.yaml",
            "description": "참조 모델 기반 토큰 가중치",
        },
    ]

    # 파일 존재 확인
    console.print("🔍 필수 파일 확인 중...")

    if not Path(config_path).exists():
        console.print(f"[red]❌ 설정 파일 없음: {config_path}[/red]")
        return

    for exp in experiments:
        if not Path(exp["recipe"]).exists():
            console.print(f"[red]❌ 레시피 파일 없음: {exp['recipe']}[/red]")
            return

    # 모델과 데이터셋 확인
    required_paths = [
        "models/7b_1t_4/consolidated.pth",
        "models/Llama_3_8B_RM",
        "models/codellama_7b_python",
        "dataset/mbpp",
    ]

    for path in required_paths:
        if not Path(path).exists():
            console.print(f"[red]❌ 필수 파일/디렉토리 없음: {path}[/red]")
            return

    console.print("[green]✅ 모든 필수 파일 확인 완료[/green]\n")

    # 실험 실행
    total_start_time = time.time()
    results = []

    for i, exp in enumerate(experiments, 1):
        console.print(f"[blue]━━━ 실험 {i}/3: {exp['name']} ━━━[/blue]")
        console.print(f"[dim]{exp['description']}[/dim]")

        result = run_experiment(
            exp["name"],
            config_path,
            exp["recipe"],
            max_steps=50,  # 빠른 실험을 위해 50 스텝만
        )

        results.append(result)

        # 짧은 휴식 (메모리 정리 시간)
        if i < len(experiments):
            console.print("[dim]잠시 대기 중... (메모리 정리)[/dim]")
            time.sleep(10)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # 결과 요약 테이블
    console.print(
        f"\n[bold green]🎉 모든 실험 완료! (총 {total_duration/60:.1f}분)[/bold green]"
    )

    table = Table(title="실험 결과 요약")
    table.add_column("알고리즘", style="cyan")
    table.add_column("상태", style="green")
    table.add_column("소요시간", style="yellow")
    table.add_column("최종 손실", style="magenta")
    table.add_column("설명", style="dim")

    algorithm_descriptions = {
        "MTP Baseline": "균등 가중치 (비교 기준)",
        "Critic WMTP": "RM 가치함수 가중치",
        "Rho-1 WMTP": "참조모델 비교 가중치",
    }

    for result in results:
        status = "✅ 성공" if result["success"] else "❌ 실패"
        duration = f"{result['duration']:.1f}초"
        final_loss = f"{result['final_loss']:.4f}" if result["final_loss"] else "N/A"
        description = algorithm_descriptions.get(result["algorithm"], "")

        table.add_row(result["algorithm"], status, duration, final_loss, description)

    console.print(table)

    # 실험 분석
    console.print("\n[bold blue]📊 실험 분석[/bold blue]")

    successful_results = [r for r in results if r["success"]]

    if len(successful_results) >= 2:
        console.print("\n[green]🔍 주요 관찰 사항:[/green]")

        # 손실 비교
        baseline_loss = None
        for r in successful_results:
            if "Baseline" in r["algorithm"] and r["final_loss"]:
                baseline_loss = r["final_loss"]
                break

        if baseline_loss:
            console.print(f"• 기준선 손실: {baseline_loss:.4f}")

            for r in successful_results:
                if "Baseline" not in r["algorithm"] and r["final_loss"]:
                    improvement = (
                        (baseline_loss - r["final_loss"]) / baseline_loss * 100
                    )
                    if improvement > 0:
                        console.print(
                            f"• {r['algorithm']}: {improvement:.1f}% 개선 (손실: {r['final_loss']:.4f})"
                        )
                    else:
                        console.print(
                            f"• {r['algorithm']}: {abs(improvement):.1f}% 악화 (손실: {r['final_loss']:.4f})"
                        )

        # 시간 비교
        avg_time = sum(r["duration"] for r in successful_results) / len(
            successful_results
        )
        console.print(f"• 평균 실험 시간: {avg_time:.1f}초")

        console.print("\n[cyan]💡 해석:[/cyan]")
        console.print("• 이 결과는 매우 짧은 훈련(50 스텝)으로 얻어진 것입니다")
        console.print("• 실제 성능 차이를 보려면 더 긴 훈련이 필요합니다")
        console.print("• 각 알고리즘의 수렴 패턴과 안정성이 중요합니다")

    else:
        console.print(
            "[yellow]⚠️  성공한 실험이 부족하여 비교 분석을 수행할 수 없습니다.[/yellow]"
        )

    # MLflow 안내
    console.print("\n[blue]📈 상세 결과 보기:[/blue]")
    console.print("MLflow 웹 UI로 자세한 메트릭을 확인하세요:")
    console.print("mlflow ui --backend-store-uri ./mlflow_runs")
    console.print("http://localhost:5000 에서 접속 가능")


if __name__ == "__main__":
    main()
