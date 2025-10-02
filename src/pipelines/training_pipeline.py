"""WMTP 통합 훈련 파이프라인 - 모든 알고리즘의 핵심 실행 엔진.

이 파이프라인은 WMTP의 핵심 아이디어를 실현하는 통합 실행 엔진입니다.
세 가지 알고리즘(mtp-baseline, critic-wmtp, rho1-wmtp) 모두가 동일한
파이프라인을 사용하되, Factory 패턴을 통해 다른 컴포넌트를 조합합니다.

파이프라인 설계 원칙:
  1. 어셈블리 전용: 복잡한 로직은 Factory와 Registry에 위임
  2. 모듈화된 컴포넌트: 각 알고리즘별 특화된 Trainer 사용 (v2.1.0+)
  3. 조건부 모델 로딩: 알고리즘에 따라 필요한 모델만 선택적 로드
  4. 단계적 실행: Stage1(선택적) → Stage2(메인 훈련) → 결과 반환

알고리즘별 컴포넌트 조합:
  - mtp-baseline: Base Model + BaselineMtpTrainer (Uniform Weighting)
  - critic-wmtp: Base + RM + CriticWmtpTrainer + Stage1 Pretraining
  - rho1-wmtp: Base + Ref + Rho1WmtpTrainer (Dynamic Weighting)

이 통합 접근법으로 연구자는 알고리즘 간 성능을 공정하게 비교할 수 있습니다.
"""

from __future__ import annotations  # Python 3.10+ 타입 힌트 호환성

from dataclasses import dataclass  # 간단한 데이터 클래스 생성용
from pathlib import Path  # 파일 경로 처리
from typing import Any  # 범용 타입 힌트

from rich.console import Console  # Rich 콘솔 출력
from torch.utils.data import DataLoader  # 데이터셋을 배치로 로드하는 도구
from torch.utils.data.distributed import (
    DistributedSampler,  # 분산 훈련을 위한 데이터 분배기
)

# from transformers import default_data_collator  # ← utils MTP collator로 대체됨
from src.factory.component_factory import (
    ComponentFactory,  # 알고리즘별 컴포넌트 생성 팩토리
)
from src.settings import Config, Recipe  # Pydantic 기반 설정 모델들
from src.utils import (  # MLflow 추적과 재현성 보장 유틸
    create_mlflow_manager,
    get_dist_manager,
    set_seed,
)
from src.utils.mtp_collator import create_mtp_collator  # WMTP 전용 단순화된 collator

console = Console()


@dataclass
class RunOutputs:
    """파이프라인 실행 결과를 담는 데이터 클래스.

    훈련 완료 후 메트릭을 포함한 결과를 반환하여
    CLI나 다른 모듈에서 훈련 성과를 확인할 수 있습니다.

    Attributes:
        trainer_metrics: 훈련 과정에서 수집된 각종 메트릭 (loss, accuracy 등)
    """

    trainer_metrics: dict[str, Any]  # 훈련 메트릭 딕셔너리


def run_training_pipeline(
    config: Config,  # 환경 설정 (GPU, 분산훈련, S3 등)
    recipe: Recipe,  # 훈련 레시피 (알고리즘, 모델, 데이터셋)
    dry_run: bool = False,  # 검증 모드 (실제 훈련 X)
    resume_checkpoint: str | Path | None = None,  # 재개용 체크포인트 (선택적)
) -> RunOutputs:
    """WMTP 통합 훈련 파이프라인 - 모든 알고리즘의 메인 실행 함수.

    Returns:
        RunOutputs: 훈련 메트릭이 포함된 결과 객체

    Raises:
        ValueError: 잘못된 설정값이나 지원되지 않는 알고리즘
        RuntimeError: 모델 로딩 실패나 훈련 중 오류
    """
    # 파이프라인 실행 단계 추적 시작
    console.print("[bold green]🚀 파이프라인 실행 시작[/bold green]")
    console.print("[dim]🔍 파이프라인 단계 추적 시작...[/dim]")

    # Step 0: 실험 추적 및 재현성 설정
    set_seed(config.seed)  # 동일한 시드로 재현 가능한 실험 보장

    # Step 0.5: Config 기반 분산 초기화
    dist_manager = get_dist_manager(config.devices.distributed)
    dist_manager.setup()  # Config 기반 자동 초기화

    # 재개 처리 로직 - ComponentFactory 통합 (한 번만 로딩)
    start_epoch = 0
    start_step = 0
    resume_run_id = None
    checkpoint_data = None

    if resume_checkpoint:
        # 체크포인트 전용 로더 생성 - 한 번만 로드하여 모든 정보 추출
        checkpoint_loader = ComponentFactory.create_checkpoint_loader(config)
        checkpoint_loader.setup({})

        # 체크포인트 로딩 및 메타데이터 추출
        checkpoint_result = checkpoint_loader.run(
            {"model_path": resume_checkpoint, "load_metadata": True}
        )

        # 메타데이터와 체크포인트 데이터 모두 추출
        if checkpoint_result.get("checkpoint_data") is not None:
            checkpoint_data = checkpoint_result["checkpoint_data"]
            start_epoch = checkpoint_result.get("epoch", 0)
            start_step = checkpoint_result.get("step", 0)
            resume_run_id = checkpoint_result.get("mlflow_run_id")

    console.print(
        f"[dim]🔍 체크포인트 로딩 완료: epoch={start_epoch}, step={start_step}[/dim]"
    )
    console.print(f"[dim]🔍 MLflow Run ID: {resume_run_id}[/dim]")

    # Step 1: MLflow 실험 추적 초기화
    # 실험 메트릭과 아티팩트를 체계적으로 추적하기 위한 MLflow 설정
    mlflow = create_mlflow_manager(config.model_dump())
    tag_map = {
        str(i): t for i, t in enumerate(recipe.run.tags)
    }  # 태그를 MLflow 형식으로 변환

    if resume_run_id:
        mlflow.start_run(run_id=resume_run_id, resume=True)
    else:
        mlflow.start_run(run_name=recipe.run.name, tags=tag_map)

    console.print(
        f"[dim]🔍 MLflow 실험 추적 초기화 완료: run_name={recipe.run.name}[/dim]"
    )

    # Step 2: Base 모델 로딩
    # Facebook native MTP 모델 - 4개 head가 내장된 WMTP의 핵심 아키텍처
    base_loader = ComponentFactory.create_model_loader(config, recipe, "base")
    base_loader.setup({})
    base_result = base_loader.run({})  # model_path는 Factory에서 이미 설정됨
    base = base_result["model"]

    console.print(f"[dim]🔍 Base 모델 로딩 완료: {config.paths.models.base}[/dim]")

    # Step 3: 토크나이저 생성
    # HuggingFace 호환 통합 토크나이저 - 모든 WMTP 모델이 공유하는 어휘 체계
    tokenizer_component = ComponentFactory.create_tokenizer(recipe, config)
    tokenizer_component.setup({"config": config})
    tokenizer_result = tokenizer_component.run({})
    tokenizer = tokenizer_result["tokenizer"]

    console.print(f"[dim]🔍 토크나이저 생성 완료: {config.paths.models.base}[/dim]")

    # Step 4: Auxiliary 모델 로딩 (알고리즘에 따라 자동 선택)
    # Factory에서 알고리즘에 맞는 보조 모델을 자동으로 선택
    ref_model = None  # Rho-1에서 사용할 참조 모델
    rm_model = None  # Critic에서 사용할 보상 모델

    aux_loader = ComponentFactory.create_model_loader(config, recipe, "aux")
    if aux_loader:  # baseline-mtp는 None 반환
        aux_loader.setup({})
        aux_result = aux_loader.run({})  # model_path는 Factory에서 이미 설정됨

        # 알고리즘에 따라 적절한 변수에 할당
        if recipe.train.algo == "rho1-wmtp":
            ref_model = aux_result["model"]
        elif recipe.train.algo == "critic-wmtp":
            rm_model = aux_result["model"]

    console.print(f"[dim]🔍 알고리즘별 추가 모델 로딩 완료: {recipe.train.algo}[/dim]")

    # Step 5: 옵티마이저 생성 (setup은 데이터셋 크기 확정 후 수행)
    # AdamW + BF16 + FSDP 조합으로 대규모 모델 훈련 최적화
    optimizer = ComponentFactory.create_optimizer(recipe, base.parameters())

    console.print(f"[dim]🔍 옵티마이저 생성 완료: {recipe.train.algo}[/dim]")

    # Step 6: 데이터셋 로딩
    # MBPP, CodeContests, HumanEval 등 코드 생성 벤치마크 지원
    train_loader_comp = ComponentFactory.create_data_loader(recipe, config)
    train_loader_comp.setup({})
    train_ds = train_loader_comp.run(
        {
            "split": "train",
            "max_length": recipe.data.train.max_length,
            "add_solution": True,
        }
    )["dataset"]

    # Step 7: 데이터셋 토크나이징
    # HuggingFace 호환 토크나이저로 텍스트를 모델 입력 형식으로 변환
    tokenized = tokenizer.tokenize_dataset(
        dataset=train_ds,
        max_length=recipe.data.train.max_length,
        remove_columns=train_ds.column_names,
        load_from_cache_file=True,
        num_proc=config.devices.num_proc,
    )

    console.print("[green]✅ 데이터셋 토크나이징 완료[/green]")

    # Step 8: 분산 훈련용 데이터 샘플러 설정
    # 다중 GPU 환경에서 데이터를 효율적으로 분배하기 위한 샘플러 구성
    sampler = None  # 분산 훈련용 데이터 샘플러 (단일 GPU에서는 None)
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(tokenized, shuffle=True)
    except Exception:
        sampler = None

    # Step 9-1: Data Collator 생성 (구조적 해결)
    # 모든 WMTP 알고리즘이 MTPDataCollator 사용하므로 직접 생성

    # 구조적 해결: tokenizer component의 명확한 인터페이스 사용
    # 복잡한 추출 로직 대신 get_hf_tokenizer() 메서드 활용
    hf_tokenizer = tokenizer.get_hf_tokenizer()

    # 모든 WMTP 알고리즘에 MTP collator 사용 (horizon=4)
    collator = create_mtp_collator(
        tokenizer=hf_tokenizer,
        horizon=4,  # Meta 논문 기준
        pad_to_multiple_of=8,  # GPU 효율성 최적화
    )

    console.print(f"[dim]🔍 Data Collator 생성 완료: {type(collator).__name__}[/dim]")

    # Step 9-2: PyTorch DataLoader 생성 (단순화된 utils collator 사용)
    train_dl = DataLoader(
        tokenized,
        batch_size=recipe.data.train.batch_size or 1,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collator,  # ← Factory에서 생성된 collator
        num_workers=0,  # fork 방지, tokenizer 병렬화 우선
        pin_memory=torch.cuda.is_available(),
    )

    console.print(f"[dim]🔍 PyTorch DataLoader 생성 완료: {recipe.train.algo}[/dim]")

    # Step 9-3: 옵티마이저 setup (데이터셋 크기 기반 num_training_steps 계산)
    # LR scheduler가 올바른 total steps로 동작하도록 정확한 계산 필요
    dataset_size = len(tokenized)
    num_epochs = recipe.train.num_epochs
    max_steps = recipe.train.max_steps

    if max_steps is None:
        # max_steps가 None이면 전체 epoch 기준
        num_training_steps = dataset_size * num_epochs
    else:
        # max_steps와 전체 epoch 중 작은 값
        num_training_steps = min(max_steps, dataset_size * num_epochs)

    optimizer.setup({"num_training_steps": num_training_steps})

    console.print(
        f"[dim]🔍 옵티마이저 setup 완료: {num_training_steps} steps "
        f"(dataset={dataset_size}, epochs={num_epochs}, max_steps={max_steps})[/dim]"
    )

    # Step 10: Stage1 사전훈련 (Critic 전용, 조건부)
    # Critic 알고리즘의 특별한 2단계 학습 - Value Head 훈련을 S3에 직접 저장
    value_head_path = None  # Stage 2에 전달할 경로

    if recipe.train.algo == "critic-wmtp" and rm_model is not None and not dry_run:
        console.print("[bold cyan]🚀 Stage 1 시작: Value Head 사전학습[/bold cyan]")

        pretrainer = ComponentFactory.create_pretrainer(recipe)
        pretrainer.setup({})

        # Stage 1 실행
        stage1_result = pretrainer.run(
            {
                "base_model": base,
                "rm_model": rm_model,
                "train_dataloader": train_dl,
                "run_name": recipe.run.name or "default",  # S3 경로 생성용 실행 이름
                "config": config,  # Log interval 등 전역 설정 전달
            }
        )

        # Stage 1에서 저장된 Value Head 경로 추출
        if stage1_result.get("saved"):
            value_head_path = stage1_result["saved"]
        else:
            console.print(
                "[yellow]⚠️ Stage 1 skipped or failed, proceeding without pretrained Value Head[/yellow]"
            )

        # Early stopping 결과 확인
        if stage1_result.get("early_stopped"):
            console.print(
                f"[yellow]⚠️ Stage 1 early stopped: {stage1_result.get('stop_reason')}[/yellow]"
            )

    # Step 11: 메인 Trainer 생성 및 초기화
    # 모든 WMTP 알고리즘의 독립된 Trainer 생성
    trainer = ComponentFactory.create_trainer(recipe, config)

    # Setup 컨텍스트 구성
    setup_ctx = {
        "model": base,
        "optimizer": optimizer,
        "mlflow_manager": mlflow,
        "ref_model": ref_model,
        "base_tokenizer": tokenizer,
        "rm_model": rm_model,
        "recipe": recipe,
        # 이미 로드된 체크포인트 데이터와 메타데이터 전달
        "checkpoint_data": checkpoint_data,
        "start_epoch": start_epoch,
        "start_step": start_step,
    }

    # 🔗 Critic-WMTP의 경우 Stage 1에서 학습된 Value Head 경로 전달
    if recipe.train.algo == "critic-wmtp" and value_head_path:
        setup_ctx["value_head_path"] = value_head_path
        console.print("[cyan]📎 Passing Stage 1 Value Head to Stage 2 trainer[/cyan]")

    trainer.setup(setup_ctx)

    console.print(
        f"[dim]🔍 메인 Trainer 생성 및 초기화 완료: {recipe.train.algo}[/dim]"
    )

    # Step 12: 실행 모드 분기
    # Dry run 모드에서는 설정 검증만 수행하고 실제 훈련은 건너뛰기
    if dry_run:
        mlflow.end_run("FINISHED")
        return RunOutputs(trainer_metrics={"dry_run": True})

    console.print(f"[dim]🔍 실행 모드 분기 완료: {recipe.train.algo}[/dim]")

    # Step 13: 메인 WMTP 훈련 실행
    # L_WMTP = Σ w_{t+k} × CE_k 공식으로 토큰별 중요도 반영 훈련
    metrics = trainer.run(
        {
            "train_dataloader": train_dl,
            "num_epochs": recipe.train.num_epochs,
            "max_steps": recipe.train.max_steps,
            "config": config,  # Config 객체 전달 (log_interval 등)
        }
    )

    console.print(f"[dim]🔍 메인 WMTP 훈련 실행 완료: {recipe.train.algo}[/dim]")

    # Step 14: 실험 종료 및 결과 반환
    # MLflow 추적 종료 및 훈련 메트릭 반환
    mlflow.end_run("FINISHED")

    console.print("[bold green]🏁 파이프라인 실행 완료[/bold green]")
    console.print(f"[dim]🔍 파이프라인 실행 결과: {metrics}[/dim]")

    return RunOutputs(trainer_metrics=metrics)
