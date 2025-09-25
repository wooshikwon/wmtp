"""
WMTP 평가 파이프라인: 학습된 모델의 성능 검증 시스템

WMTP 연구 맥락:
WMTP의 핵심 가설 "Not All Tokens Are What You Need"가 실제로
성능 향상을 가져오는지 검증하는 중요한 모듈입니다.
MBPP와 CodeContests 벤치마크에서 pass@k 메트릭을 측정하여
토큰 가중치의 효과를 정량적으로 평가합니다.

핵심 기능:
- 체크포인트 로딩: 학습된 WMTP 모델 복원
- 벤치마크 평가: MBPP, CodeContests에서 코드 생성 능력 측정
- pass@k 계산: Chen et al. (2021)의 unbiased estimator 적용
- MLflow 통합: 평가 결과 자동 기록 및 비교 분석

WMTP 알고리즘과의 연결:
- Baseline MTP: 균등 가중치의 기준 성능 제공
- Critic-WMTP: Value 기반 가중치의 효과 측정
- Rho1-WMTP: CE 차이 기반 가중치의 성능 검증

사용 예시:
    >>> pipeline = EvaluationPipeline(config, recipe)
    >>> results = pipeline.run(
    >>>     checkpoint=Path("checkpoints/rho1_epoch_10.pt"),
    >>>     datasets=["mbpp", "codecontests"],
    >>>     save_predictions=True
    >>> )
    >>> print(f"MBPP pass@1: {results['mbpp']['pass@1']}")

성능 최적화:
- 배치 추론으로 GPU 활용도 극대화
- 토큰 생성 시 KV 캐시 활용
- 병렬 코드 실행으로 평가 시간 단축

디버깅 팁:
- 체크포인트 로드 실패: 모델 구조와 체크포인트 호환성 확인
- OOM 오류: sampling.batch_size 감소
- 평가 속도 느림: sampling.num_samples 조정
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from src.factory.component_factory import ComponentFactory
from src.settings import Config, Recipe
from src.utils import create_mlflow_manager, set_seed

console = Console()


class EvaluationPipeline:
    """
    코딩 벤치마크에서 학습된 WMTP 모델을 평가하는 파이프라인.

    WMTP 연구 맥락:
    토큰 가중치가 실제로 코드 생성 능력을 향상시키는지 검증합니다.
    Baseline, Critic, Rho1 세 알고리즘의 성능을 동일한 조건에서
    공정하게 비교하여 연구 가설을 입증합니다.

    설계 패턴:
    - ComponentFactory로 평가기 생성 (일관성 유지)
    - setup() → run() 패턴으로 컴포넌트 실행
    - MLflow로 실험 결과 자동 추적

    평가 프로토콜:
    - Meta MTP 논문의 프로토콜 준수
    - HumanEval/MBPP에서 pass@k 측정
    - Temperature 0.8, top-p 0.95 샘플링
    """

    def __init__(self, config: Config, recipe: Recipe):
        """
        평가 파이프라인 초기화.

        WMTP 맥락:
        알고리즘별로 다른 체크포인트 구조를 처리합니다.
        Critic-WMTP는 value_head.pt를 추가로 로드해야 합니다.

        Args:
            config: 환경 설정 (GPU, 저장소, MLflow)
            recipe: 레시피 설정 (모델, 평가 프로토콜)
        """
        self.config = config
        self.recipe = recipe
        self.mlflow = None

        # Set seed for reproducible evaluation
        set_seed(config.seed)

    def run(
        self,
        checkpoint: Path,
        datasets: list[str] | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
        save_predictions: bool = False,
        save_report: bool = False,
    ) -> dict[str, Any]:
        """
        완전한 평가 파이프라인 실행.

        WMTP 연구 맥락:
        학습된 모델이 실제로 "중요한 토큰"을 잘 학습했는지 검증합니다.
        코드 생성 태스크는 syntax와 logic이 모두 정확해야 하므로
        토큰 가중치의 효과를 명확히 보여줄 수 있습니다.

        구체적 동작:
        1. 체크포인트에서 모델 복원
        2. 벤치마크 데이터셋 로드 (MBPP/CodeContests)
        3. 코드 생성 및 실행
        4. pass@k 메트릭 계산
        5. MLflow에 결과 기록

        매개변수:
            checkpoint: 평가할 모델 체크포인트 경로
                - 예: "checkpoints/rho1_epoch_10.pt"
            datasets: 평가 데이터셋 리스트 (None = recipe 기본값)
                - ["mbpp", "codecontests", "humaneval"]
            run_name: MLflow 실행 이름 (None = "eval_{recipe.run.name}")
            tags: MLflow 태그 추가
            save_predictions: 생성된 코드 샘플 저장 여부
            save_report: 상세 평가 보고서 생성 여부

        반환값:
            dict: 평가 결과와 메트릭
                - checkpoint: 평가된 체크포인트 경로
                - datasets: 사용된 데이터셋
                - results: pass@k 등 성능 지표
                - config: 평가 설정

        예시:
            >>> results = pipeline.run(
            >>>     checkpoint=Path("model.pt"),
            >>>     datasets=["mbpp"],
            >>>     save_predictions=True
            >>> )
            >>> print(f"pass@1: {results['results']['mbpp']['pass@1']:.2%}")

        주의사항:
            - 체크포인트와 recipe의 모델 설정이 일치해야 함
            - GPU 메모리 부족 시 sampling.batch_size 감소
            - 코드 실행 타임아웃 기본값: 10초

        디버깅 팁:
            - FileNotFoundError: 체크포인트 경로 확인
            - CUDA OOM: 배치 크기 감소 또는 fp16 사용
            - 낮은 pass@k: 샘플링 temperature 조정

        WMTP 알고리즘별 기대 성능:
            - Baseline: Meta MTP 논문 수준
            - Critic: Baseline 대비 +2-5%p 개선 기대
            - Rho1: Baseline 대비 +3-7%p 개선 기대
        """
        console.print("[bold blue]Starting WMTP Evaluation Pipeline[/bold blue]")
        console.print(f"Checkpoint: {checkpoint}")

        # Validate checkpoint exists
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        # Initialize MLflow
        self.mlflow = create_mlflow_manager(self.config.model_dump())
        tag_map = {str(i): t for i, t in enumerate(tags or [])}
        self.mlflow.start_run(
            run_name=run_name or f"eval_{self.recipe.run.name}", tags=tag_map
        )

        try:
            # Load model from checkpoint
            console.print("[cyan]Loading model from checkpoint...[/cyan]")
            model, tokenizer = self._load_checkpoint(checkpoint)

            # Prepare datasets
            console.print("[cyan]Preparing datasets...[/cyan]")
            dataset_sources = datasets or self.recipe.data.eval.sources
            loaded_datasets = self._load_datasets(dataset_sources)

            # Create and run evaluator
            console.print("[cyan]Running evaluation...[/cyan]")
            evaluator = ComponentFactory.create_evaluator(self.recipe, self.config)
            evaluator.setup(
                {
                    "sampling": self.recipe.eval.sampling.model_dump(),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                }
            )

            # Prepare evaluation context
            eval_ctx = {
                "model": model,
                "tokenizer": tokenizer,
                **loaded_datasets,  # mbpp_dataset, contest_dataset etc.
            }

            # Run evaluation
            results = evaluator.run(eval_ctx)

            # Log results to MLflow with enhanced options
            self._log_results(results, save_predictions, save_report, checkpoint)

            # Create summary
            summary = {
                "checkpoint": str(checkpoint),
                "datasets": dataset_sources,
                "algorithm": self.recipe.train.algo,
                "results": results,
                "config": {
                    "model_id": self.recipe.model.base_id,
                    "mtp_heads": self.recipe.model.mtp.n_heads,
                    "horizon": self.recipe.model.mtp.horizon,
                    "eval_protocol": self.recipe.eval.protocol,
                    "sampling": self.recipe.eval.sampling.model_dump(),
                },
            }

            console.print("[green]Evaluation completed successfully![/green]")
            self.mlflow.end_run("FINISHED")
            return summary

        except Exception as e:
            console.print(f"[red]Evaluation failed: {e}[/red]")
            if self.mlflow:
                self.mlflow.end_run("FAILED")
            raise

    def _load_checkpoint(self, checkpoint_path: Path) -> tuple[Any, Any]:
        """
        Load model and tokenizer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (model, tokenizer)
        """
        # Load base model and tokenizer using existing infrastructure
        model_loader = ComponentFactory.create_model_loader(self.config)
        model_loader.setup({})

        # Load all models (base, ref, rm if needed)
        models = model_loader.run({"load_all_models": True})["models"]
        base_model = models["base"]["model"]
        tokenizer = models["base"]["tokenizer"]

        # Load checkpoint state
        console.print(f"Loading checkpoint state from {checkpoint_path}")
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")

        # Load model weights
        if "model_state_dict" in checkpoint_state:
            base_model.load_state_dict(checkpoint_state["model_state_dict"])
        elif "state_dict" in checkpoint_state:
            base_model.load_state_dict(checkpoint_state["state_dict"])
        else:
            # Assume the checkpoint is just the model state dict
            base_model.load_state_dict(checkpoint_state)

        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = base_model.to(device)
        base_model.eval()

        console.print(f"[green]Model loaded on {device}[/green]")
        return base_model, tokenizer

    def _load_datasets(self, dataset_sources: list[str]) -> dict[str, Any]:
        """
        Load evaluation datasets.

        Args:
            dataset_sources: List of dataset names to load

        Returns:
            Dictionary mapping dataset names to loaded datasets
        """
        loaded_datasets = {}

        for source in dataset_sources:
            console.print(f"Loading dataset: {source}")

            try:
                # Create data loader for this source
                data_loader = ComponentFactory.create_data_loader(source, self.config)
                data_loader.setup({})

                # Load evaluation split
                dataset = data_loader.run(
                    {
                        "split": "test",  # Use test split for evaluation
                        "max_length": getattr(
                            self.recipe.data.eval, "max_length", 2048
                        ),
                        "add_solution": False,  # Don't need solutions for evaluation
                    }
                )["dataset"]

                # Map to expected context keys for evaluator
                if source == "mbpp":
                    loaded_datasets["mbpp_dataset"] = dataset
                elif source == "contest":
                    loaded_datasets["contest_dataset"] = dataset
                else:
                    loaded_datasets[f"{source}_dataset"] = dataset

                console.print(
                    f"[green]Loaded {len(dataset)} samples from {source}[/green]"
                )

            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load {source}: {e}[/yellow]")
                # Continue with other datasets

        return loaded_datasets

    def _log_results(
        self,
        results: dict[str, Any],
        save_predictions: bool,
        save_report: bool,
        checkpoint: Path,
    ) -> None:
        """
        Log evaluation results to MLflow with enhanced artifacts.

        Args:
            results: Results dictionary from evaluator
            save_predictions: Whether to save prediction samples
            save_report: Whether to generate evaluation report
            checkpoint: Path to checkpoint being evaluated
        """
        if not self.mlflow:
            return

        import json
        from datetime import datetime

        metrics = results.get("metrics", {})

        # Log all metrics with enhanced categorization
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, int | float):
                self.mlflow.log_metric(metric_name, metric_value)

        # Add summary metrics
        if metrics:
            # Calculate average performance across all metrics
            numeric_metrics = [
                v for v in metrics.values() if isinstance(v, int | float)
            ]
            if numeric_metrics:
                avg_performance = sum(numeric_metrics) / len(numeric_metrics)
                self.mlflow.log_metric("avg_performance", avg_performance)

        # Log evaluation config as parameters
        eval_params = {
            "eval_protocol": self.recipe.eval.protocol,
            "eval_temperature": self.recipe.eval.sampling.temperature,
            "eval_top_p": self.recipe.eval.sampling.top_p,
            "eval_n_samples": self.recipe.eval.sampling.n,
            "batch_size": self.recipe.data.eval.batch_size,
            "checkpoint_path": str(checkpoint),
            "algorithm": self.recipe.train.algo,
            "model_id": self.recipe.model.base_id,
            "mtp_heads": self.recipe.model.mtp.n_heads,
        }

        for param_name, param_value in eval_params.items():
            self.mlflow.log_param(param_name, param_value)

        # Save enhanced results as artifact
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Save complete evaluation results
        results_file = Path(f"evaluation_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(
                {
                    "checkpoint": str(checkpoint),
                    "timestamp": timestamp,
                    "algorithm": self.recipe.train.algo,
                    "metrics": metrics,
                    "config": eval_params,
                },
                f,
                indent=2,
            )
        self.mlflow.log_artifact(str(results_file), "evaluation")
        results_file.unlink()

        # 2. Save prediction samples if requested
        if save_predictions:
            self._save_prediction_samples(results, timestamp)

        # 3. Save weight distribution statistics if available
        self._save_weight_statistics(results, timestamp)

        # 4. Generate and save evaluation report if requested
        if save_report:
            self._generate_evaluation_report(results, metrics, checkpoint, timestamp)

        console.print("[green]Results and artifacts logged to MLflow[/green]")

    def _save_prediction_samples(self, results: dict[str, Any], timestamp: str) -> None:
        """
        Save prediction samples as MLflow artifacts.

        Args:
            results: Evaluation results dictionary
            timestamp: Timestamp for file naming
        """
        import json

        # Extract predictions if available
        predictions = results.get("predictions", [])
        references = results.get("references", [])

        if not predictions:
            console.print("[yellow]No predictions available to save[/yellow]")
            return

        # Save sample predictions (first 10)
        samples = []
        for i, (pred, ref) in enumerate(zip(predictions[:10], references[:10])):
            samples.append(
                {
                    "sample_id": i,
                    "prediction": pred if isinstance(pred, str) else str(pred),
                    "reference": ref if isinstance(ref, str) else str(ref),
                }
            )

        if samples:
            samples_file = Path(f"prediction_samples_{timestamp}.json")
            with open(samples_file, "w") as f:
                json.dump(
                    {
                        "total_samples": len(predictions),
                        "shown_samples": len(samples),
                        "samples": samples,
                    },
                    f,
                    indent=2,
                )

            self.mlflow.log_artifact(str(samples_file), "predictions")
            samples_file.unlink()
            console.print(f"[green]Saved {len(samples)} prediction samples[/green]")

    def _save_weight_statistics(self, results: dict[str, Any], timestamp: str) -> None:
        """
        Save weight distribution statistics as MLflow artifacts.

        Args:
            results: Evaluation results dictionary
            timestamp: Timestamp for file naming
        """
        import json

        import numpy as np

        # Extract weight statistics if available
        weight_stats = results.get("weight_statistics", {})

        if not weight_stats and hasattr(self, "_last_scorer_output"):
            weight_stats = getattr(self, "_last_scorer_output", {}).get(
                "statistics", {}
            )

        if weight_stats:
            # Calculate additional statistics if raw weights are available
            weights = results.get("weights", [])
            if weights:
                try:
                    weights_np = np.array(weights)
                    weight_stats.update(
                        {
                            "percentiles": {
                                "p10": float(np.percentile(weights_np, 10)),
                                "p25": float(np.percentile(weights_np, 25)),
                                "p50": float(np.percentile(weights_np, 50)),
                                "p75": float(np.percentile(weights_np, 75)),
                                "p90": float(np.percentile(weights_np, 90)),
                                "p95": float(np.percentile(weights_np, 95)),
                                "p99": float(np.percentile(weights_np, 99)),
                            },
                            "distribution": {
                                "mean": float(np.mean(weights_np)),
                                "std": float(np.std(weights_np)),
                                "min": float(np.min(weights_np)),
                                "max": float(np.max(weights_np)),
                                "variance": float(np.var(weights_np)),
                            },
                        }
                    )
                except Exception:
                    pass

            # Save weight statistics
            stats_file = Path(f"weight_statistics_{timestamp}.json")
            with open(stats_file, "w") as f:
                json.dump(
                    {
                        "algorithm": str(self.recipe.train.algo)
                        if hasattr(self.recipe.train, "algo")
                        else "unknown",
                        "statistics": weight_stats,
                        "config": {
                            "temperature": float(
                                getattr(self.recipe.loss, "temperature", 0.7)
                            ),
                            "lambda": float(getattr(self.recipe.loss, "lambda", 0.3)),
                        },
                    },
                    f,
                    indent=2,
                )

            self.mlflow.log_artifact(str(stats_file), "weights")
            stats_file.unlink()
            console.print("[green]Saved weight distribution statistics[/green]")

    def _generate_evaluation_report(
        self,
        results: dict[str, Any],
        metrics: dict[str, float],
        checkpoint: Path,
        timestamp: str,
    ) -> None:
        """
        Generate comprehensive evaluation report.

        Args:
            results: Complete evaluation results
            metrics: Extracted metrics
            checkpoint: Checkpoint path
            timestamp: Timestamp for file naming
        """
        from datetime import datetime

        # Create markdown report
        report_lines = [
            "# WMTP Evaluation Report",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Checkpoint**: `{checkpoint}`",
            f"\n**Algorithm**: {self.recipe.train.algo}",
            f"\n**Model**: {self.recipe.model.base_id}",
            "\n## Configuration",
            f"- **MTP Heads**: {self.recipe.model.mtp.n_heads}",
            f"- **Horizon**: {self.recipe.model.mtp.horizon}",
            f"- **Evaluation Protocol**: {self.recipe.eval.protocol}",
            f"- **Sampling Temperature**: {self.recipe.eval.sampling.temperature}",
            f"- **Top-p**: {self.recipe.eval.sampling.top_p}",
            "\n## Results Summary",
        ]

        # Add metrics table
        if metrics:
            report_lines.append("\n### Performance Metrics\n")
            report_lines.append("| Metric | Score |")
            report_lines.append("|--------|-------|")

            # Group by dataset
            mbpp_metrics = {}
            contest_metrics = {}
            other_metrics = {}

            for name, value in metrics.items():
                if "mbpp" in name.lower():
                    mbpp_metrics[name] = value
                elif "contest" in name.lower():
                    contest_metrics[name] = value
                else:
                    other_metrics[name] = value

            # Add MBPP metrics
            if mbpp_metrics:
                report_lines.append("| **MBPP** | |")
                for name, value in mbpp_metrics.items():
                    if isinstance(value, float) and 0 <= value <= 1:
                        report_lines.append(f"| {name} | {value:.2%} |")
                    else:
                        report_lines.append(f"| {name} | {value:.4f} |")

            # Add CodeContests metrics
            if contest_metrics:
                report_lines.append("| **CodeContests** | |")
                for name, value in contest_metrics.items():
                    if isinstance(value, float) and 0 <= value <= 1:
                        report_lines.append(f"| {name} | {value:.2%} |")
                    else:
                        report_lines.append(f"| {name} | {value:.4f} |")

            # Add other metrics
            if other_metrics:
                report_lines.append("| **Other** | |")
                for name, value in other_metrics.items():
                    if isinstance(value, float) and 0 <= value <= 1:
                        report_lines.append(f"| {name} | {value:.2%} |")
                    else:
                        report_lines.append(f"| {name} | {value:.4f} |")

        # Add weight statistics if available
        weight_stats = results.get("weight_statistics", {})
        if weight_stats:
            report_lines.extend(
                [
                    "\n### Weight Distribution Statistics",
                    f"- **Mean Weight**: {weight_stats.get('mean_weight', 'N/A')}",
                    f"- **Std Weight**: {weight_stats.get('std_weight', 'N/A')}",
                    f"- **Min Weight**: {weight_stats.get('min_weight', 'N/A')}",
                    f"- **Max Weight**: {weight_stats.get('max_weight', 'N/A')}",
                ]
            )

        # Add algorithm-specific details
        if self.recipe.train.algo == "critic-wmtp":
            report_lines.extend(
                [
                    "\n### Critic-WMTP Details",
                    "- **Value Head**: Trained with GAE",
                    "- **Delta Mode**: TD",
                    f"- **Temperature**: {getattr(self.recipe.loss, 'temperature', 0.7)}",
                ]
            )
        elif self.recipe.train.algo == "rho1-wmtp":
            report_lines.extend(
                [
                    "\n### Rho-1-WMTP Details",
                    "- **Scoring Method**: Absolute CE Excess",
                    f"- **Top Percentile**: {getattr(self.recipe, 'rho1', {}).get('percentile_top_p', 0.2) * 100}%",
                    f"- **Temperature**: {getattr(self.recipe.loss, 'temperature', 0.7)}",
                ]
            )

        # Save report
        report_content = "\n".join(report_lines)
        report_file = Path(f"evaluation_report_{timestamp}.md")
        with open(report_file, "w") as f:
            f.write(report_content)

        self.mlflow.log_artifact(str(report_file), "reports")
        report_file.unlink()
        console.print("[green]Generated comprehensive evaluation report[/green]")
