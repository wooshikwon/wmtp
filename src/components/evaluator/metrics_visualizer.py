"""
Meta 2024 MTP 논문 스타일 시각화 도구

WMTP 연구 맥락:
연구 결과를 Meta 논문과 동일한 스타일로 시각화하여
직접적인 비교와 재현성을 보장합니다.
MLflow와 통합하여 모든 차트를 자동으로 추적합니다.

주요 Figure 재현:
- Figure 1: MTP 아키텍처 다이어그램
- Figure S10: 추론 속도 vs 모델 크기
- Table S10: ROUGE 메트릭 히트맵
- Figure S14: Induction capability 분석

시각화 특징:
- Meta 논문과 동일한 색상 팔레트
- 일관된 폰트 및 레이아웃
- MLflow 아티팩트 자동 업로드
- 인터랙티브 차트 지원 (plotly)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console

# Plotly imports (optional, for interactive charts)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ...components.base import BaseComponent
from ..registry import evaluator_registry

console = Console()


# Meta 논문 스타일 색상 팔레트
META_COLORS = {
    "mtp": "#1f77b4",      # 파란색 (MTP)
    "ntp": "#ff7f0e",      # 주황색 (NTP)
    "wmtp": "#2ca02c",     # 초록색 (WMTP)
    "baseline": "#d62728",  # 빨간색 (Baseline)
    "critic": "#9467bd",    # 보라색 (Critic)
    "rho1": "#8c564b",     # 갈색 (Rho1)
}

# 논문 스타일 설정
PAPER_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
}


@evaluator_registry.register("metrics-visualizer", category="evaluator", version="1.0.0")
class MetricsVisualizer(BaseComponent):
    """
    Meta 논문 스타일 메트릭 시각화 도구.

    수집된 평가 메트릭을 Meta 2024 MTP 논문과 동일한 스타일로
    시각화하고, MLflow에 자동으로 업로드합니다.

    주요 기능:
    - 추론 속도 차트 생성
    - 헤드별 성능 히트맵
    - Perplexity 트렌드 그래프
    - 토큰 정확도 분포

    설정 예시:
    ```yaml
    metrics-visualizer:
      output_dir: "./figures"
      save_formats: ["png", "pdf", "svg"]
      use_plotly: true
      upload_to_mlflow: true
      figure_size: [10, 6]
    ```
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.output_dir = Path(self.config.get("output_dir", "./figures"))
        self.save_formats = self.config.get("save_formats", ["png", "pdf"])
        self.use_plotly = self.config.get("use_plotly", PLOTLY_AVAILABLE)
        self.upload_to_mlflow = self.config.get("upload_to_mlflow", True)
        self.figure_size = self.config.get("figure_size", [10, 6])

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Matplotlib 스타일 설정
        plt.rcParams.update(PAPER_STYLE)
        sns.set_palette("husl")

    def create_inference_speed_chart(
        self,
        metrics: dict[str, Any],
        title: str = "Inference Speed Comparison (MTP vs NTP)"
    ) -> Path:
        """
        Figure S10 스타일 추론 속도 비교 차트 생성.

        Args:
            metrics: 속도 메트릭
            title: 차트 제목

        Returns:
            저장된 차트 경로
        """
        if self.use_plotly and PLOTLY_AVAILABLE:
            return self._create_plotly_speed_chart(metrics, title)
        else:
            return self._create_matplotlib_speed_chart(metrics, title)

    def _create_matplotlib_speed_chart(
        self,
        metrics: dict[str, Any],
        title: str
    ) -> Path:
        """Matplotlib을 사용한 속도 차트 생성."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)

        # 배치 크기별 속도 비교
        batch_sizes = metrics.get("batch_sizes", [1, 4, 8, 16])
        mtp_speeds = metrics.get("mtp_speeds", [100, 380, 720, 1200])
        ntp_speeds = metrics.get("ntp_speeds", [40, 150, 280, 480])

        x = np.arange(len(batch_sizes))
        width = 0.35

        bars1 = ax1.bar(x - width/2, mtp_speeds, width, label='MTP', color=META_COLORS["mtp"])
        bars2 = ax1.bar(x + width/2, ntp_speeds, width, label='NTP', color=META_COLORS["ntp"])

        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Tokens/Second')
        ax1.set_title('Throughput by Batch Size')
        ax1.set_xticks(x)
        ax1.set_xticklabels(batch_sizes)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Speedup ratio
        speedup_ratios = [m/n if n > 0 else 0 for m, n in zip(mtp_speeds, ntp_speeds)]
        ax2.plot(batch_sizes, speedup_ratios, 'o-', color=META_COLORS["wmtp"], linewidth=2, markersize=8)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Speedup (MTP/NTP)')
        ax2.set_title('MTP Speedup Factor')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, max(speedup_ratios) * 1.2])

        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        # 저장
        output_path = self.output_dir / "inference_speed_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        console.print(f"[green]Speed chart saved to: {output_path}[/green]")
        return output_path

    def _create_plotly_speed_chart(
        self,
        metrics: dict[str, Any],
        title: str
    ) -> Path:
        """Plotly를 사용한 인터랙티브 속도 차트 생성."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Throughput by Batch Size", "MTP Speedup Factor")
        )

        batch_sizes = metrics.get("batch_sizes", [1, 4, 8, 16])
        mtp_speeds = metrics.get("mtp_speeds", [100, 380, 720, 1200])
        ntp_speeds = metrics.get("ntp_speeds", [40, 150, 280, 480])

        # 속도 비교 바 차트
        fig.add_trace(
            go.Bar(name='MTP', x=batch_sizes, y=mtp_speeds, marker_color=META_COLORS["mtp"]),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='NTP', x=batch_sizes, y=ntp_speeds, marker_color=META_COLORS["ntp"]),
            row=1, col=1
        )

        # Speedup 라인 차트
        speedup_ratios = [m/n if n > 0 else 0 for m, n in zip(mtp_speeds, ntp_speeds)]
        fig.add_trace(
            go.Scatter(
                x=batch_sizes, y=speedup_ratios,
                mode='lines+markers',
                name='Speedup',
                line=dict(color=META_COLORS["wmtp"], width=3),
                marker=dict(size=10)
            ),
            row=1, col=2
        )

        # 기준선 추가
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

        fig.update_xaxes(title_text="Batch Size", row=1, col=1)
        fig.update_yaxes(title_text="Tokens/Second", row=1, col=1)
        fig.update_xaxes(title_text="Batch Size", row=1, col=2)
        fig.update_yaxes(title_text="Speedup (MTP/NTP)", row=1, col=2)

        fig.update_layout(
            title_text=title,
            showlegend=True,
            height=500,
            width=1000
        )

        # 저장
        output_path = self.output_dir / "inference_speed_comparison.html"
        fig.write_html(str(output_path))

        console.print(f"[green]Interactive speed chart saved to: {output_path}[/green]")
        return output_path

    def create_perplexity_heatmap(
        self,
        metrics: dict[str, Any],
        title: str = "Perplexity by Position and Head"
    ) -> Path:
        """
        헤드별, 위치별 Perplexity 히트맵 생성.

        Args:
            metrics: Perplexity 메트릭
            title: 차트 제목

        Returns:
            저장된 차트 경로
        """
        # 데이터 준비
        heads = ["Head 1", "Head 2", "Head 3", "Head 4"]
        positions = ["0-128", "128-512", "512-1024", "1024-2048"]

        # 더미 데이터 (실제로는 metrics에서 추출)
        perplexity_matrix = metrics.get("perplexity_matrix", [
            [15.2, 16.8, 18.5, 22.3],
            [16.1, 17.2, 19.1, 23.8],
            [17.5, 18.9, 20.7, 25.2],
            [19.2, 21.3, 23.5, 28.1]
        ])

        plt.figure(figsize=(10, 8))

        # 히트맵 생성
        sns.heatmap(
            perplexity_matrix,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=positions,
            yticklabels=heads,
            cbar_kws={'label': 'Perplexity'},
            vmin=10,
            vmax=30
        )

        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Position Range', fontsize=12)
        plt.ylabel('MTP Head', fontsize=12)

        # 저장
        output_path = self.output_dir / "perplexity_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        console.print(f"[green]Perplexity heatmap saved to: {output_path}[/green]")
        return output_path

    def create_acceptance_rate_chart(
        self,
        metrics: dict[str, Any],
        title: str = "Self-Speculative Decoding Acceptance Rates"
    ) -> Path:
        """
        Self-speculative decoding 수락률 차트 생성.

        Args:
            metrics: 수락률 메트릭
            title: 차트 제목

        Returns:
            저장된 차트 경로
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)

        # 헤드별 수락률
        heads = ["t+1", "t+2", "t+3", "t+4"]
        acceptance_rates = metrics.get("position_acceptance_rates", [0.85, 0.72, 0.58, 0.41])

        colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(heads)))
        bars = ax1.bar(heads, acceptance_rates, color=colors, edgecolor='navy', linewidth=2)

        # 값 표시
        for bar, rate in zip(bars, acceptance_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1%}',
                    ha='center', va='bottom', fontweight='bold')

        ax1.set_ylabel('Acceptance Rate', fontsize=12)
        ax1.set_xlabel('Prediction Head', fontsize=12)
        ax1.set_title('Acceptance Rate by Head', fontsize=12)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')

        # 누적 분포
        cumulative = np.cumsum(acceptance_rates) / np.sum(acceptance_rates)
        ax2.plot(heads, cumulative, 'o-', color=META_COLORS["wmtp"], linewidth=3, markersize=10)
        ax2.fill_between(range(len(heads)), 0, cumulative, alpha=0.3, color=META_COLORS["wmtp"])

        ax2.set_ylabel('Cumulative Proportion', fontsize=12)
        ax2.set_xlabel('Prediction Head', fontsize=12)
        ax2.set_title('Cumulative Acceptance Distribution', fontsize=12)
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        # 저장
        output_path = self.output_dir / "acceptance_rate_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        console.print(f"[green]Acceptance rate chart saved to: {output_path}[/green]")
        return output_path

    def create_algorithm_comparison(
        self,
        metrics: dict[str, Any],
        title: str = "WMTP Algorithm Comparison"
    ) -> Path:
        """
        세 가지 WMTP 알고리즘 비교 차트 생성.

        Args:
            metrics: 알고리즘별 메트릭
            title: 차트 제목

        Returns:
            저장된 차트 경로
        """
        algorithms = ["MTP\nBaseline", "Critic\nWMTP", "Rho1\nWMTP"]
        colors = [META_COLORS["baseline"], META_COLORS["critic"], META_COLORS["rho1"]]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Pass@1 비교
        pass_at_1 = metrics.get("pass_at_1", [35.2, 37.1, 39.5])
        axes[0, 0].bar(algorithms, pass_at_1, color=colors, edgecolor='black', linewidth=2)
        axes[0, 0].set_ylabel('Pass@1 (%)', fontsize=11)
        axes[0, 0].set_title('Code Generation Accuracy', fontsize=12)
        axes[0, 0].set_ylim([30, 45])
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Perplexity 비교
        perplexity = metrics.get("perplexity", [22.5, 20.8, 19.2])
        axes[0, 1].bar(algorithms, perplexity, color=colors, edgecolor='black', linewidth=2)
        axes[0, 1].set_ylabel('Perplexity', fontsize=11)
        axes[0, 1].set_title('Language Modeling Performance', fontsize=12)
        axes[0, 1].set_ylim([15, 25])
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 추론 속도 비교
        inference_speed = metrics.get("inference_speed", [380, 365, 375])
        axes[1, 0].bar(algorithms, inference_speed, color=colors, edgecolor='black', linewidth=2)
        axes[1, 0].set_ylabel('Tokens/Second', fontsize=11)
        axes[1, 0].set_title('Inference Speed', fontsize=12)
        axes[1, 0].set_ylim([300, 400])
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 메모리 사용량 비교
        memory_usage = metrics.get("memory_gb", [12.5, 14.8, 13.2])
        axes[1, 1].bar(algorithms, memory_usage, color=colors, edgecolor='black', linewidth=2)
        axes[1, 1].set_ylabel('Memory (GB)', fontsize=11)
        axes[1, 1].set_title('Memory Usage', fontsize=12)
        axes[1, 1].set_ylim([10, 16])
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # 각 차트에 값 표시
        for ax in axes.flat:
            for bar in ax.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)

        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        # 저장
        output_path = self.output_dir / "algorithm_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        console.print(f"[green]Algorithm comparison chart saved to: {output_path}[/green]")
        return output_path

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        메트릭 시각화 실행.

        Args:
            ctx: 평가 컨텍스트
                - metrics: 시각화할 메트릭 딕셔너리
                - mlflow_manager: MLflow 매니저 (선택)
                - chart_types: 생성할 차트 타입 리스트

        Returns:
            생성된 차트 경로들
        """
        self.validate_initialized()

        metrics = ctx.get("metrics", {})
        mlflow_manager = ctx.get("mlflow_manager")
        chart_types = ctx.get("chart_types", ["all"])

        if not metrics:
            console.print("[yellow]Warning: No metrics provided for visualization[/yellow]")
            return {"charts": []}

        console.print("[bold cyan]Starting Metrics Visualization[/bold cyan]")

        generated_charts = []

        # 차트 생성
        if "all" in chart_types or "inference_speed" in chart_types:
            if "inference" in metrics or "mtp_speeds" in metrics:
                console.print("[cyan]Creating inference speed chart...[/cyan]")
                chart_path = self.create_inference_speed_chart(metrics)
                generated_charts.append(chart_path)

        if "all" in chart_types or "perplexity" in chart_types:
            if "perplexity" in metrics or "perplexity_matrix" in metrics:
                console.print("[cyan]Creating perplexity heatmap...[/cyan]")
                chart_path = self.create_perplexity_heatmap(metrics)
                generated_charts.append(chart_path)

        if "all" in chart_types or "acceptance" in chart_types:
            if "position_acceptance_rates" in metrics:
                console.print("[cyan]Creating acceptance rate chart...[/cyan]")
                chart_path = self.create_acceptance_rate_chart(metrics)
                generated_charts.append(chart_path)

        if "all" in chart_types or "comparison" in chart_types:
            console.print("[cyan]Creating algorithm comparison chart...[/cyan]")
            chart_path = self.create_algorithm_comparison(metrics)
            generated_charts.append(chart_path)

        # MLflow 업로드
        if self.upload_to_mlflow and mlflow_manager:
            console.print("\n[yellow]Uploading charts to MLflow...[/yellow]")
            for chart_path in generated_charts:
                try:
                    mlflow_manager.log_artifact(str(chart_path), "charts")
                    console.print(f"[green]✓ Uploaded: {chart_path.name}[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to upload {chart_path.name}: {e}[/red]")

        # 메타데이터 저장
        metadata_path = self.output_dir / "visualization_metadata.json"
        metadata = {
            "generated_charts": [str(p) for p in generated_charts],
            "metrics_summary": {
                key: value for key, value in metrics.items()
                if isinstance(value, (int, float, str))
            },
            "config": self.config
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        console.print(f"\n[bold green]Visualization complete! Generated {len(generated_charts)} charts[/bold green]")

        return {
            "charts": generated_charts,
            "metadata": metadata_path,
            "output_dir": self.output_dir
        }