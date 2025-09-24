#!/usr/bin/env python3
"""
WMTP 알고리즘 성능 비교 분석 프레임워크

"Not All Tokens Are What You Need" 철학을 바탕으로 한
세 가지 WMTP 알고리즘의 성능을 종합적으로 비교 분석합니다.

분석 대상 알고리즘:
1. MTP Baseline: 균등한 토큰 가중치 (비교 기준)
2. Critic WMTP: Value Function 기반 동적 토큰 가중치
3. Rho-1 WMTP: Reference Model 차이 기반 토큰 가중치
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import mlflow
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@dataclass
class ExperimentResult:
    """단일 실험 결과를 담는 데이터 클래스."""

    algorithm: str
    run_id: str
    run_name: str
    start_time: datetime
    end_time: datetime | None
    duration_seconds: float
    status: str
    final_loss: float | None
    metrics: dict[str, float]
    parameters: dict[str, Any]
    tags: dict[str, str]


@dataclass
class ComparisonReport:
    """알고리즘 비교 분석 보고서."""

    baseline_result: ExperimentResult | None
    critic_result: ExperimentResult | None
    rho1_result: ExperimentResult | None
    comparison_metrics: dict[str, float]
    performance_ranking: list[str]
    convergence_analysis: dict[str, Any]
    token_weighting_effectiveness: dict[str, float]


class WMTPPerformanceAnalyzer:
    """WMTP 알고리즘 성능 분석기."""

    def __init__(self, mlflow_tracking_uri: str = "./mlflow_runs"):
        """
        성능 분석기 초기화.

        Args:
            mlflow_tracking_uri: MLflow 추적 URI
        """
        self.mlflow_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.experiment_name = "wmtp/quick_comparison"

        console.print("[bold blue]🔬 WMTP 성능 분석기 초기화[/bold blue]")
        console.print(f"MLflow URI: {mlflow_tracking_uri}")

    def discover_experiment_runs(self) -> list[ExperimentResult]:
        """MLflow에서 실험 결과들을 자동 발견."""
        console.print("\n[bold yellow]🔍 실험 결과 자동 발견 중...[/bold yellow]")

        try:
            # 실험 가져오기
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                console.print(
                    f"[red]❌ 실험을 찾을 수 없음: {self.experiment_name}[/red]"
                )
                return []

            # 모든 실행 가져오기
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id], output_format="list"
            )

            results = []
            for run in runs:
                # 알고리즘 타입 추론
                algorithm = self._infer_algorithm_type(run)
                if not algorithm:
                    continue

                # 실험 결과 객체 생성
                result = ExperimentResult(
                    algorithm=algorithm,
                    run_id=run.info.run_id,
                    run_name=run.info.run_name or f"{algorithm}_{run.info.run_id[:8]}",
                    start_time=datetime.fromtimestamp(run.info.start_time / 1000),
                    end_time=datetime.fromtimestamp(run.info.end_time / 1000)
                    if run.info.end_time
                    else None,
                    duration_seconds=(run.info.end_time - run.info.start_time) / 1000
                    if run.info.end_time
                    else 0,
                    status=run.info.status,
                    final_loss=run.data.metrics.get("final_loss"),
                    metrics=dict(run.data.metrics),
                    parameters=dict(run.data.params),
                    tags=dict(run.data.tags),
                )
                results.append(result)

            console.print(f"✅ 발견된 실험: {len(results)}개")
            return results

        except Exception as e:
            console.print(f"[red]❌ 실험 발견 실패: {e}[/red]")
            return []

    def _infer_algorithm_type(self, run) -> str | None:
        """실행 정보로부터 알고리즘 타입 추론."""
        run_name = (run.info.run_name or "").lower()
        tags = {k.lower(): v.lower() for k, v in run.data.tags.items()}

        # 태그 기반 추론
        if "baseline" in tags.values() or "mtp baseline" in run_name:
            return "MTP Baseline"
        elif "critic" in tags.values() or "critic" in run_name:
            return "Critic WMTP"
        elif "rho1" in tags.values() or "rho-1" in run_name or "rho1" in run_name:
            return "Rho-1 WMTP"

        # 파라미터 기반 추론
        algo_param = run.data.params.get("algorithm", "").lower()
        if "baseline" in algo_param:
            return "MTP Baseline"
        elif "critic" in algo_param:
            return "Critic WMTP"
        elif "rho1" in algo_param:
            return "Rho-1 WMTP"

        return None

    def analyze_convergence(self, results: list[ExperimentResult]) -> dict[str, Any]:
        """수렴 패턴 분석."""
        console.print("\n[bold blue]📈 수렴 패턴 분석[/bold blue]")

        convergence_data = {}

        for result in results:
            if result.status != "FINISHED" or not result.final_loss:
                continue

            # 기본 수렴 메트릭
            convergence_data[result.algorithm] = {
                "final_loss": result.final_loss,
                "training_time": result.duration_seconds,
                "convergence_speed": result.final_loss
                / max(result.duration_seconds, 1),  # loss/time
                "stability": self._calculate_stability(result),
            }

        return convergence_data

    def _calculate_stability(self, result: ExperimentResult) -> float:
        """훈련 안정성 계산 (더미 구현)."""
        # 실제로는 loss curve의 분산을 계산해야 함
        # 현재는 final_loss 기반 추정
        if result.final_loss and result.final_loss > 0:
            return 1.0 / (1.0 + result.final_loss)  # 낮은 loss = 높은 안정성
        return 0.0

    def compare_token_weighting_effectiveness(
        self, results: list[ExperimentResult]
    ) -> dict[str, float]:
        """토큰 가중치 효과성 비교."""
        console.print("\n[bold blue]⚖️  토큰 가중치 효과성 분석[/bold blue]")

        effectiveness = {}
        baseline_loss = None

        # 기준선 찾기
        for result in results:
            if result.algorithm == "MTP Baseline" and result.final_loss:
                baseline_loss = result.final_loss
                break

        if not baseline_loss:
            console.print("[yellow]⚠️  기준선 결과를 찾을 수 없음[/yellow]")
            return {}

        # 상대적 개선도 계산
        for result in results:
            if not result.final_loss:
                continue

            if result.algorithm == "MTP Baseline":
                effectiveness[result.algorithm] = 0.0  # 기준선
            else:
                improvement = (baseline_loss - result.final_loss) / baseline_loss * 100
                effectiveness[result.algorithm] = improvement

        return effectiveness

    def generate_performance_ranking(
        self, results: list[ExperimentResult]
    ) -> list[str]:
        """성능 순위 생성."""
        # final_loss 기준으로 정렬 (낮을수록 좋음)
        finished_results = [
            r for r in results if r.status == "FINISHED" and r.final_loss
        ]

        if not finished_results:
            return []

        sorted_results = sorted(finished_results, key=lambda x: x.final_loss)
        return [r.algorithm for r in sorted_results]

    def create_comparison_report(
        self, results: list[ExperimentResult]
    ) -> ComparisonReport:
        """종합 비교 보고서 생성."""
        console.print("\n[bold green]📊 종합 비교 보고서 생성 중...[/bold green]")

        # 알고리즘별 최신 결과 선택
        baseline_result = None
        critic_result = None
        rho1_result = None

        for result in results:
            if result.algorithm == "MTP Baseline":
                baseline_result = result
            elif result.algorithm == "Critic WMTP":
                critic_result = result
            elif result.algorithm == "Rho-1 WMTP":
                rho1_result = result

        # 분석 수행
        convergence_analysis = self.analyze_convergence(results)
        token_weighting_effectiveness = self.compare_token_weighting_effectiveness(
            results
        )
        performance_ranking = self.generate_performance_ranking(results)

        # 비교 메트릭 계산
        comparison_metrics = {}
        if baseline_result and baseline_result.final_loss:
            comparison_metrics["baseline_final_loss"] = baseline_result.final_loss

            for result in results:
                if result.algorithm != "MTP Baseline" and result.final_loss:
                    key = f"{result.algorithm.lower().replace(' ', '_')}_improvement"
                    improvement = (
                        baseline_result.final_loss - result.final_loss
                    ) / baseline_result.final_loss
                    comparison_metrics[key] = improvement

        return ComparisonReport(
            baseline_result=baseline_result,
            critic_result=critic_result,
            rho1_result=rho1_result,
            comparison_metrics=comparison_metrics,
            performance_ranking=performance_ranking,
            convergence_analysis=convergence_analysis,
            token_weighting_effectiveness=token_weighting_effectiveness,
        )

    def display_analysis_results(self, report: ComparisonReport):
        """분석 결과를 Rich 테이블로 표시."""
        console.print("\n[bold green]🎉 WMTP 알고리즘 성능 비교 분석 결과[/bold green]")

        # 1. 기본 정보 테이블
        basic_table = Table(title="📋 실험 기본 정보")
        basic_table.add_column("알고리즘", style="cyan")
        basic_table.add_column("상태", style="green")
        basic_table.add_column("최종 손실", style="yellow")
        basic_table.add_column("훈련 시간", style="magenta")
        basic_table.add_column("설명", style="dim")

        algorithms_info = {
            "MTP Baseline": ("기본 MTP (비교 기준)", report.baseline_result),
            "Critic WMTP": ("Value Function 기반", report.critic_result),
            "Rho-1 WMTP": ("Reference Model 기반", report.rho1_result),
        }

        for algo, (description, result) in algorithms_info.items():
            if result:
                status = "✅ 완료" if result.status == "FINISHED" else "❌ 실패"
                loss = f"{result.final_loss:.4f}" if result.final_loss else "N/A"
                time = (
                    f"{result.duration_seconds:.1f}s"
                    if result.duration_seconds
                    else "N/A"
                )
            else:
                status = "❓ 없음"
                loss = "N/A"
                time = "N/A"

            basic_table.add_row(algo, status, loss, time, description)

        console.print(basic_table)

        # 2. 성능 순위
        if report.performance_ranking:
            console.print("\n[bold blue]🏆 성능 순위 (최종 손실 기준)[/bold blue]")
            for i, algo in enumerate(report.performance_ranking, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                console.print(f"{medal} {i}위: {algo}")

        # 3. 토큰 가중치 효과성
        if report.token_weighting_effectiveness:
            console.print("\n[bold blue]⚖️  토큰 가중치 효과성[/bold blue]")
            for algo, effectiveness in report.token_weighting_effectiveness.items():
                if effectiveness > 0:
                    console.print(f"• {algo}: +{effectiveness:.2f}% 개선 🔥")
                elif effectiveness < 0:
                    console.print(f"• {algo}: {effectiveness:.2f}% 악화 📉")
                else:
                    console.print(f"• {algo}: 기준선 📊")

        # 4. 수렴 분석
        if report.convergence_analysis:
            conv_table = Table(title="📈 수렴 패턴 분석")
            conv_table.add_column("알고리즘", style="cyan")
            conv_table.add_column("수렴 속도", style="yellow")
            conv_table.add_column("안정성", style="green")
            conv_table.add_column("평가", style="magenta")

            for algo, data in report.convergence_analysis.items():
                speed = f"{data['convergence_speed']:.6f}"
                stability = f"{data['stability']:.3f}"

                # 평가 생성
                if data["stability"] > 0.8:
                    evaluation = "우수 ⭐⭐⭐"
                elif data["stability"] > 0.6:
                    evaluation = "양호 ⭐⭐"
                else:
                    evaluation = "개선필요 ⭐"

                conv_table.add_row(algo, speed, stability, evaluation)

            console.print(f"\n{conv_table}")

    def save_detailed_report(
        self, report: ComparisonReport, output_path: str = "wmtp_analysis_report.json"
    ):
        """상세 분석 보고서를 JSON 파일로 저장."""
        console.print("\n[bold blue]💾 상세 보고서 저장 중...[/bold blue]")

        # 직렬화 가능한 형태로 변환
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_experiments": sum(
                    1
                    for r in [
                        report.baseline_result,
                        report.critic_result,
                        report.rho1_result,
                    ]
                    if r
                ),
                "performance_ranking": report.performance_ranking,
                "comparison_metrics": report.comparison_metrics,
            },
            "algorithms": {},
            "convergence_analysis": report.convergence_analysis,
            "token_weighting_effectiveness": report.token_weighting_effectiveness,
        }

        # 각 알고리즘 결과 추가
        for name, result in [
            ("baseline", report.baseline_result),
            ("critic", report.critic_result),
            ("rho1", report.rho1_result),
        ]:
            if result:
                report_data["algorithms"][name] = {
                    "algorithm": result.algorithm,
                    "run_id": result.run_id,
                    "status": result.status,
                    "final_loss": result.final_loss,
                    "duration_seconds": result.duration_seconds,
                    "metrics": result.metrics,
                    "parameters": result.parameters,
                }

        # JSON 파일로 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        console.print(f"✅ 보고서 저장 완료: {output_path}")


def main():
    """메인 분석 함수."""
    console.print(
        Panel.fit(
            "[bold green]🧪 WMTP 알고리즘 성능 비교 분석[/bold green]\n\n"
            '"Not All Tokens Are What You Need" 철학을 바탕으로 한\n'
            "세 가지 가중치 전략의 종합적 성능 분석",
            title="WMTP Performance Analysis",
        )
    )

    # 분석기 초기화
    analyzer = WMTPPerformanceAnalyzer()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 실험 결과 발견
        task = progress.add_task("실험 결과 수집 중...", total=None)
        results = analyzer.discover_experiment_runs()
        progress.update(task, description="✅ 실험 결과 수집 완료")

        if not results:
            console.print("[red]❌ 분석할 실험 결과를 찾을 수 없습니다.[/red]")
            console.print("먼저 실험을 실행하세요: python run_quick_experiment.py")
            return

        # 비교 분석 수행
        progress.update(task, description="성능 분석 중...")
        report = analyzer.create_comparison_report(results)

        progress.update(task, description="✅ 분석 완료")

    # 결과 표시
    analyzer.display_analysis_results(report)

    # 상세 보고서 저장
    analyzer.save_detailed_report(report)

    # 다음 단계 안내
    console.print("\n[bold cyan]🚀 다음 단계:[/bold cyan]")
    console.print("1. 더 긴 훈련을 위해 configs에서 max_steps 증가")
    console.print("2. 다른 데이터셋으로 일반화 성능 확인")
    console.print("3. 하이퍼파라미터 튜닝으로 성능 최적화")


if __name__ == "__main__":
    main()
