"""WMTP 통합 평가 파이프라인 - Meta 2024 논문 기준 평가 엔진.

이 파이프라인은 training_pipeline.py와 동일한 구조로 설계되어
Meta MTP 논문의 모든 평가 메트릭을 재현할 수 있는 통합 평가 엔진입니다.

파이프라인 설계 원칙:
  1. 어셈블리 전용: 복잡한 로직은 Factory와 Registry에 위임
  2. 모듈화된 컴포넌트: 각 평가 타입별 특화된 Evaluator 사용
  3. 조건부 데이터 로딩: 평가 타입에 따라 필요한 데이터셋만 선택적 로드
  4. 단계적 실행: 체크포인트 로드 → 평가 실행 → 결과 반환

평가 타입별 컴포넌트 조합:
  - meta-mtp: Pass@k 메트릭 (HumanEval, MBPP, CodeContests)
  - inference-speed: MTP vs NTP 추론 속도 비교
  - per-head-analysis: 헤드별(t+1~t+4) 성능 분석
  - token-accuracy: 토큰 위치별 예측 정확도

이 통합 접근법으로 연구자는 Meta 논문 결과를 완벽하게 재현할 수 있습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from src.factory.component_factory import ComponentFactory
from src.settings import Config, Recipe
from src.utils import create_mlflow_manager, set_seed

console = Console()


@dataclass
class EvaluationOutputs:
    """파이프라인 평가 결과를 담는 데이터 클래스.

    평가 완료 후 메트릭을 포함한 결과를 반환하여
    CLI나 다른 모듈에서 평가 성과를 확인할 수 있습니다.

    Attributes:
        metrics: 평가 과정에서 수집된 각종 메트릭
        algorithm: 평가된 알고리즘 타입
        checkpoint: 평가에 사용된 체크포인트 경로
    """

    metrics: dict[str, Any]
    algorithm: str
    checkpoint: str


def run_evaluation_pipeline(
    config: Config,
    recipe: Recipe,
    checkpoint_path: Path,
    eval_types: list[str] | None = None,
    dry_run: bool = False,
) -> EvaluationOutputs:
    """WMTP 통합 평가 파이프라인 - Meta 2024 논문 기준 메인 평가 함수.

    training_pipeline.py와 동일한 구조로 설계되어 일관성을 유지합니다.
    Factory 패턴을 통해 다양한 평가 타입을 동적으로 조합합니다.

    Args:
        config: 환경 설정 (GPU, 저장소, MLflow)
        recipe: 평가 레시피 (모델, 평가 프로토콜)
        checkpoint_path: 평가할 모델 체크포인트 경로
        eval_types: 평가 타입 리스트 (None = ["meta-mtp"])
        dry_run: 검증 모드 (실제 평가 X)

    Returns:
        EvaluationOutputs: 평가 메트릭이 포함된 결과 객체

    Raises:
        ValueError: 잘못된 설정값이나 지원되지 않는 평가 타입
        RuntimeError: 모델 로딩 실패나 평가 중 오류
    """
    # 파이프라인 실행 단계 추적 시작
    console.print("[bold blue]🚀 평가 파이프라인 실행 시작[/bold blue]")
    console.print(f"[dim]🔍 체크포인트: {checkpoint_path}[/dim]")

    # Step 0: 실험 추적 및 재현성 설정
    set_seed(config.seed)

    # Step 1: MLflow 실험 추적 초기화
    mlflow = create_mlflow_manager(config.model_dump())
    tag_map = {str(i): t for i, t in enumerate(recipe.run.tags)}
    mlflow.start_run(run_name=f"eval_{recipe.run.name}", tags=tag_map)

    console.print(f"[dim]🔍 MLflow 실험 추적 초기화 완료: run_name=eval_{recipe.run.name}[/dim]")

    # Step 2: 체크포인트 로딩
    checkpoint_loader = ComponentFactory.create_checkpoint_loader(config)
    checkpoint_loader.setup({})
    checkpoint_result = checkpoint_loader.run({
        "model_path": str(checkpoint_path),
        "load_metadata": True
    })

    model = checkpoint_result["model"]
    model.eval()  # 평가 모드 설정

    console.print(f"[dim]🔍 체크포인트 로딩 완료: {checkpoint_path}[/dim]")

    # Step 3: 토크나이저 생성
    tokenizer_component = ComponentFactory.create_tokenizer(recipe, config)
    tokenizer_component.setup({"config": config})
    tokenizer_result = tokenizer_component.run({})
    tokenizer = tokenizer_result["tokenizer"]

    console.print(f"[dim]🔍 토크나이저 생성 완료[/dim]")

    # Step 4: 평가 타입 결정
    if eval_types is None:
        eval_types = ["meta-mtp"]  # 기본값: Meta MTP 논문 평가
    elif "all" in eval_types:
        eval_types = ["meta-mtp", "inference-speed", "per-head-analysis", "token-accuracy"]

    console.print(f"[dim]🔍 평가 타입: {eval_types}[/dim]")

    # Step 5: 실행 모드 분기 (dry run)
    if dry_run:
        mlflow.end_run("FINISHED")
        return EvaluationOutputs(
            metrics={"dry_run": True},
            algorithm=recipe.train.algo,
            checkpoint=str(checkpoint_path)
        )

    # Step 6: 평가 타입별 데이터셋 로딩 (조건부)
    datasets = {}

    # meta-mtp 평가시 데이터셋 로딩
    if "meta-mtp" in eval_types:
        # MBPP 데이터셋 로딩
        if "mbpp" in recipe.data.eval.sources:
            mbpp_recipe = recipe.model_copy(deep=True)
            mbpp_recipe.data.train.sources = ["mbpp"]
            mbpp_loader = ComponentFactory.create_data_loader(mbpp_recipe, config)
            mbpp_loader.setup({})
            mbpp_result = mbpp_loader.run({
                "split": "test",
                "max_length": recipe.data.eval.max_length
            })
            datasets["mbpp_dataset"] = mbpp_result["dataset"]
            console.print(f"[dim]🔍 MBPP 데이터셋 로딩 완료[/dim]")

        # CodeContests 데이터셋 로딩
        if "codecontests" in recipe.data.eval.sources:
            contest_recipe = recipe.model_copy(deep=True)
            contest_recipe.data.train.sources = ["contest"]
            contest_loader = ComponentFactory.create_data_loader(contest_recipe, config)
            contest_loader.setup({})
            contest_result = contest_loader.run({
                "split": "test",
                "max_length": recipe.data.eval.max_length
            })
            datasets["contest_dataset"] = contest_result["dataset"]
            console.print(f"[dim]🔍 CodeContests 데이터셋 로딩 완료[/dim]")

    # Step 7: 평가 타입별 실행 및 메트릭 수집
    all_metrics = {}

    for eval_type in eval_types:
        console.print(f"[cyan]⚡ {eval_type} 평가 실행 중...[/cyan]")

        # 평가기 생성
        evaluator = ComponentFactory.create_evaluator_by_type(eval_type, recipe, config)

        # 평가기 초기화
        evaluator.setup({
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "sampling": recipe.eval.sampling.model_dump()
        })

        # 평가 컨텍스트 준비
        eval_context = {
            "model": model,
            "tokenizer": tokenizer,
            **datasets  # 필요한 데이터셋 포함
        }

        # 평가 실행
        eval_result = evaluator.run(eval_context)

        # 메트릭 수집
        if "metrics" in eval_result:
            for metric_name, metric_value in eval_result["metrics"].items():
                all_metrics[f"{eval_type}.{metric_name}"] = metric_value

                # MLflow에 메트릭 기록
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"{eval_type}.{metric_name}", metric_value)

        console.print(f"[green]✓ {eval_type} 평가 완료[/green]")

    # Step 8: MLflow 파라미터 기록
    mlflow.log_params({
        "checkpoint": str(checkpoint_path),
        "algorithm": recipe.train.algo,
        "model_id": recipe.model.base_id,
        "mtp_heads": recipe.model.mtp.n_heads,
        "eval_protocol": recipe.eval.protocol,
        "eval_types": ",".join(eval_types)
    })

    # Step 9: 실험 종료 및 결과 반환
    mlflow.end_run("FINISHED")

    console.print("[bold green]🏁 평가 파이프라인 실행 완료[/bold green]")
    console.print(f"[dim]🔍 수집된 메트릭 수: {len(all_metrics)}[/dim]")

    return EvaluationOutputs(
        metrics=all_metrics,
        algorithm=recipe.train.algo,
        checkpoint=str(checkpoint_path)
    )