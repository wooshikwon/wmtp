"""WMTP 통합 훈련 파이프라인 - 모든 알고리즘의 핵심 실행 엔진.

연구 철학 구현: "Not All Tokens Are What You Need"
===============================================

이 파이프라인은 WMTP의 핵심 아이디어를 실현하는 통합 실행 엔진입니다.
세 가지 알고리즘(mtp-baseline, critic-wmtp, rho1-wmtp) 모두가 동일한
파이프라인을 사용하되, Factory 패턴을 통해 다른 컴포넌트를 조합합니다.

파이프라인 설계 원칙:
  1. 어셈블리 전용: 복잡한 로직은 Factory와 Registry에 위임
  2. 모듈화된 컴포넌트: 각 알고리즘별 특화된 Scorer, Trainer 사용
  3. 조건부 모델 로딩: 알고리즘에 따라 필요한 모델만 선택적 로드
  4. 단계적 실행: Stage1(선택적) → Stage2(메인 훈련) → 결과 반환

알고리즘별 컴포넌트 조합:
  - mtp-baseline: Base Model + No Scorer + Uniform Weighting
  - critic-wmtp: Base + RM + CriticScorer + Stage1 Pretraining
  - rho1-wmtp: Base + Ref + Rho1Scorer + Dynamic Weighting

이 통합 접근법으로 연구자는 알고리즘 간 성능을 공정하게 비교할 수 있습니다.
"""

from __future__ import annotations  # Python 3.10+ 타입 힌트 호환성

from dataclasses import dataclass  # 간단한 데이터 클래스 생성용
from typing import Any  # 범용 타입 힌트

from torch.utils.data import DataLoader  # 데이터셋을 배치로 로드하는 도구
from torch.utils.data.distributed import (
    DistributedSampler,  # 분산 훈련을 위한 데이터 분배기
)
from transformers import default_data_collator  # HuggingFace의 기본 데이터 배치 생성기

from src.factory.component_factory import (
    ComponentFactory,  # 알고리즘별 컴포넌트 생성 팩토리
)
from src.settings import Config, Recipe  # Pydantic 기반 설정 모델들
from src.utils import create_mlflow_manager, set_seed  # MLflow 추적과 재현성 보장 유틸


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
    run_name: str | None = None,  # MLflow 실험 이름 (선택적)
    tags: list[str] | None = None,  # 실험 분류용 태그 (선택적)
    dry_run: bool = False,  # 검증 모드 (실제 훈련 X)
    max_steps: int | None = None,  # 최대 훈련 스텝 (제한용)
    resume_checkpoint: Path | None = None,  # 재개용 체크포인트 (선택적)
) -> RunOutputs:
    """WMTP 통합 훈련 파이프라인 - 모든 알고리즘의 메인 실행 함수.

    연구 철학 "Not All Tokens Are What You Need" 구현:
        이 함수는 세 가지 WMTP 알고리즘을 통합된 파이프라인으로 실행합니다.
        각 알고리즘은 토큰 가중치 계산 방식이 다르지만, 동일한 구조로
        공정한 성능 비교가 가능합니다.

    파이프라인 실행 단계:
        1. 실험 추적 설정 (MLflow + 시드 고정)
        2. 알고리즘별 모델 로딩:
           - Base: 항상 로드 (Facebook native MTP)
           - Ref: rho1-wmtp에서만 사용
           - RM: critic-wmtp에서만 사용
        3. 옵티마이저 설정 (대부분 AdamW + BF16)
        4. 데이터셋 로딩 및 토크나이징
        5. 분산 훈련용 데이터로더 설정
        6. Stage1 사전훈련 (critic-wmtp만 해당)
        7. 메인 훈련 실행 (모든 알고리즘 공통)
        8. 결과 반환 및 실험 종료

    Args:
        config: GPU, 메모리, S3 등 환경 설정
        recipe: 알고리즘, 하이퍼파라미터, 데이터 설정
        run_name: MLflow 실험명 (None시 recipe.run.name 사용)
        tags: 실험 분류용 태그들 (예: ["exp1", "critic", "mbpp"])
        dry_run: True시 설정 검증만 하고 실제 훈련은 skip
        max_steps: 훈련 스텝 제한 (None시 recipe 설정 따름)

    Returns:
        RunOutputs: 훈련 메트릭이 포함된 결과 객체

    Raises:
        ValueError: 잘못된 설정값이나 지원되지 않는 알고리즘
        RuntimeError: 모델 로딩 실패나 훈련 중 오류
    """
    # Step 1: 실험 추적 및 재현성 설정
    set_seed(config.seed)  # 동일한 시드로 재현 가능한 실험 보장

    # 재개 처리 로직
    start_epoch = 0
    start_step = 0
    resume_run_id = None

    if resume_checkpoint:
        import torch
        from rich.console import Console

        console = Console()
        checkpoint_data = None

        # S3 또는 로컬 체크포인트 로드 처리
        if isinstance(resume_checkpoint, str) and resume_checkpoint.startswith("s3://"):
            # S3에서 직접 로드
            from src.utils.s3 import S3Manager
            s3_manager = S3Manager()
            s3_key = resume_checkpoint.replace("s3://wmtp/", "")
            try:
                checkpoint_bytes = s3_manager.stream_model(s3_key)
                checkpoint_data = torch.load(checkpoint_bytes, map_location="cpu")
                console.print(f"[green]Loading checkpoint from S3: {resume_checkpoint}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load S3 checkpoint: {e}[/red]")
        elif hasattr(resume_checkpoint, "exists") and resume_checkpoint.exists():
            # 로컬 파일 로드 (Path 객체)
            checkpoint_data = torch.load(resume_checkpoint, map_location="cpu")
            console.print(f"[green]Loading checkpoint from local: {resume_checkpoint}[/green]")
        elif isinstance(resume_checkpoint, str):
            # 로컬 파일 경로 (문자열)
            from pathlib import Path
            checkpoint_path = Path(resume_checkpoint)
            if checkpoint_path.exists():
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
                console.print(f"[green]Loading checkpoint from local: {checkpoint_path}[/green]")
            else:
                console.print(f"[yellow]Checkpoint not found: {resume_checkpoint}[/yellow]")

        if checkpoint_data:
            start_epoch = checkpoint_data.get("epoch", 0)
            start_step = checkpoint_data.get("step", 0)
            resume_run_id = checkpoint_data.get("mlflow_run_id")
            console.print(
                f"[green]Resuming from epoch {start_epoch}, step {start_step}[/green]"
            )

    # Step 1: MLflow 실험 추적 초기화
    # 실험 메트릭과 아티팩트를 체계적으로 추적하기 위한 MLflow 설정
    mlflow = create_mlflow_manager(config.model_dump())
    tag_map = {
        str(i): t for i, t in enumerate(tags or [])
    }  # 태그를 MLflow 형식으로 변환

    if resume_run_id:
        mlflow.start_run(run_id=resume_run_id, resume=True)
    else:
        mlflow.start_run(run_name=run_name or recipe.run.name, tags=tag_map)

    # Step 2: Base 모델 로딩
    # Facebook native MTP 모델 - 4개 head가 내장된 WMTP의 핵심 아키텍처
    base_loader = ComponentFactory.create_model_loader(config, recipe)
    base_loader.setup({})
    base_result = base_loader.run({
        "model_path": str(config.paths.models.base)
    })
    base = base_result["model"]

    # Step 3: 토크나이저 생성
    # HuggingFace 호환 통합 토크나이저 - 모든 WMTP 모델이 공유하는 어휘 체계
    tokenizer_component = ComponentFactory.create_tokenizer(recipe, config)
    tokenizer_component.setup({"config": config})
    tokenizer_result = tokenizer_component.run({})
    tokenizer = tokenizer_result["tokenizer"]

    # Step 4: 알고리즘별 추가 모델 로딩 (조건부)
    # 각 WMTP 알고리즘은 서로 다른 보조 모델을 필요로 함
    ref_model = None  # Rho-1에서 사용할 참조 모델
    rm_model = None  # Critic에서 사용할 보상 모델

    if recipe.train.algo == "rho1-wmtp":
        # Rho-1: Reference Model 로딩 - |CE^ref_t - CE^base_t| 계산용
        ref_loader = ComponentFactory.create_model_loader(config)
        ref_loader.setup({})
        ref_result = ref_loader.run({
            "model_path": str(config.paths.models.ref)
        })
        ref_model = ref_result["model"]

    elif recipe.train.algo == "critic-wmtp":
        # Critic: Reward Model 로딩 - Stage1 Value Head 훈련용
        rm_loader = ComponentFactory.create_model_loader(config)
        rm_loader.setup({})
        rm_result = rm_loader.run({
            "model_path": str(config.paths.models.rm)
        })
        rm_model = rm_result["model"]

    # mtp-baseline은 추가 모델 불필요 - Base 모델만으로 균등 가중치 MTP 수행

    # Step 5: 옵티마이저 설정 (예외: .run() 없는 패턴)
    # AdamW + BF16 + FSDP 조합으로 대규모 모델 훈련 최적화
    optimizer = ComponentFactory.create_optimizer(recipe, base.parameters())
    optimizer.setup({
        "num_training_steps": max_steps or 0
    })

    # Step 6: 데이터셋 로딩
    # MBPP, CodeContests, HumanEval 등 코드 생성 벤치마크 지원
    train_loader_comp = ComponentFactory.create_data_loader(recipe, config)
    train_loader_comp.setup({})
    train_ds = train_loader_comp.run({
        "split": "train",
        "max_length": recipe.data.train.max_length,
        "add_solution": True,
    })["dataset"]

    # Step 7: 데이터셋 토크나이징
    # HuggingFace 호환 토크나이저로 텍스트를 모델 입력 형식으로 변환
    tokenized = tokenizer.tokenize_dataset(
        dataset=train_ds,
        max_length=recipe.data.train.max_length,
        remove_columns=train_ds.column_names,
        load_from_cache_file=True,
    )

    # Step 8: 분산 훈련용 데이터 샘플러 설정
    # 다중 GPU 환경에서 데이터를 효율적으로 분배하기 위한 샘플러 구성
    sampler = None  # 분산 훈련용 데이터 샘플러 (단일 GPU에서는 None)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(tokenized, shuffle=True)
    except Exception:
        sampler = None

    # Step 9: PyTorch DataLoader 생성
    # 토큰화된 데이터를 배치 단위로 모델에 공급하기 위한 데이터 로더 구성
    train_dl = DataLoader(
        tokenized,
        batch_size=recipe.data.train.batch_size or 1,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Step 10: Stage1 사전훈련 (Critic 전용, 조건부)
    # Critic 알고리즘의 특별한 2단계 학습 - Value Head 훈련 단계
    if recipe.train.algo == "critic-wmtp" and rm_model is not None and not dry_run:
        from pathlib import Path

        pretrainer = ComponentFactory.create_pretrainer(recipe)
        pretrainer.setup({})
        pretrainer.run({
            "base_model": base,
            "rm_model": rm_model,
            "train_dataloader": train_dl,
            "cache_root": Path(config.paths.cache) / "critic" / (recipe.run.name or "default"),
        })

    # Step 11: 메인 Trainer 생성 및 초기화
    # 모든 WMTP 알고리즘의 통합 실행 엔진 - scorer에 따라 가중치 방식 결정
    trainer = ComponentFactory.create_trainer(recipe, config)
    trainer.setup({
        "model": base,
        "optimizer": optimizer,
        "mlflow_manager": mlflow,
        "ref_model": ref_model,
        "base_tokenizer": tokenizer,
        "rm_model": rm_model,
        "recipe": recipe,
        "resume_checkpoint": resume_checkpoint,
    })

    # Step 12: 실행 모드 분기
    # Dry run 모드에서는 설정 검증만 수행하고 실제 훈련은 건너뛰기
    if dry_run:
        mlflow.end_run("FINISHED")
        return RunOutputs(trainer_metrics={"dry_run": True})

    # Step 13: 메인 WMTP 훈련 실행
    # L_WMTP = Σ w_{t+k} × CE_k 공식으로 토큰별 중요도 반영 훈련
    metrics = trainer.run({
        "train_dataloader": train_dl,
        "max_steps": max_steps
    })

    # Step 14: 실험 종료 및 결과 반환
    # MLflow 추적 종료 및 훈련 메트릭 반환
    mlflow.end_run("FINISHED")
    return RunOutputs(trainer_metrics=metrics)
