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
    get_console_output,
    get_dist_manager,
    set_seed,
)
from src.utils.mtp_collator import create_mtp_collator  # WMTP 전용 단순화된 collator

console_out = get_console_output()


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
    # Phase 0: 환경 초기화
    with console_out.phase("환경 초기화"):
        console_out.task("시드 설정")
        set_seed(config.seed)

        console_out.task("분산 환경 초기화")
        dist_manager = get_dist_manager(config.devices.distributed)
        dist_manager.setup()

        # 체크포인트 재개 처리
        start_epoch = 0
        start_step = 0
        resume_run_id = None
        checkpoint_data = None

        if resume_checkpoint:
            console_out.task("체크포인트 로딩")
            checkpoint_loader = ComponentFactory.create_checkpoint_loader(config)
            checkpoint_loader.setup({})

            checkpoint_result = checkpoint_loader.run(
                {"model_path": resume_checkpoint, "load_metadata": True}
            )

            if checkpoint_result.get("checkpoint_data") is not None:
                checkpoint_data = checkpoint_result["checkpoint_data"]
                start_epoch = checkpoint_result.get("epoch", 0)
                start_step = checkpoint_result.get("step", 0)
                resume_run_id = checkpoint_result.get("mlflow_run_id")
                console_out.detail(f"epoch={start_epoch}, step={start_step}")

        console_out.task("MLflow 실험 추적 초기화")
        mlflow = create_mlflow_manager(config.model_dump())
        tag_map = {str(i): t for i, t in enumerate(recipe.run.tags)}

        if resume_run_id:
            mlflow.start_run(run_id=resume_run_id, resume=True)
        else:
            mlflow.start_run(run_name=recipe.run.name, tags=tag_map)

    # Phase 1: 모델 & 토크나이저 준비
    with console_out.phase("모델 & 토크나이저 준비"):
        console_out.task("Base 모델 로딩")
        base_loader = ComponentFactory.create_model_loader(config, recipe, "base")
        base_loader.setup({})
        base_result = base_loader.run({})
        base = base_result["model"]

        console_out.task("토크나이저 생성")
        tokenizer_component = ComponentFactory.create_tokenizer(recipe, config)
        tokenizer_component.setup({"config": config})
        tokenizer_result = tokenizer_component.run({})
        tokenizer = tokenizer_result["tokenizer"]

        ref_model = None
        rm_model = None
        aux_loader = ComponentFactory.create_model_loader(config, recipe, "aux")
        if aux_loader:
            console_out.task(f"Aux 모델 로딩 ({recipe.train.algo})")
            aux_loader.setup({})
            aux_result = aux_loader.run({})

            if recipe.train.algo == "rho1-wmtp":
                ref_model = aux_result["model"]
            elif recipe.train.algo == "critic-wmtp":
                rm_model = aux_result["model"]

        console_out.task("옵티마이저 생성")
        optimizer = ComponentFactory.create_optimizer(recipe, base.parameters())

    # Phase 2: 데이터셋 준비
    with console_out.phase("데이터셋 준비"):
        console_out.task("데이터셋 로딩")
        train_loader_comp = ComponentFactory.create_data_loader(recipe, config)
        train_loader_comp.setup({})
        train_ds = train_loader_comp.run(
            {
                "split": "train",
                "max_length": recipe.data.train.max_length,
                "add_solution": True,
            }
        )["dataset"]

        console_out.task("데이터셋 토크나이징")
        tokenized = tokenizer.tokenize_dataset(
            dataset=train_ds,
            max_length=recipe.data.train.max_length,
            remove_columns=train_ds.column_names,
            load_from_cache_file=True,
            num_proc=config.devices.num_proc,
        )

    # Phase 3: 훈련 설정
    with console_out.phase("훈련 설정"):
        console_out.task("분산 샘플러 설정")
        sampler = None
        try:
            import torch
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                sampler = DistributedSampler(tokenized, shuffle=True)
        except Exception:
            sampler = None

        console_out.task("Data Collator 생성")
        hf_tokenizer = tokenizer.get_hf_tokenizer()
        collator = create_mtp_collator(
            tokenizer=hf_tokenizer,
            horizon=4,
            pad_to_multiple_of=8,
        )

        console_out.task("PyTorch DataLoader 생성")
        train_dl = DataLoader(
            tokenized,
            batch_size=recipe.data.train.batch_size or 1,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        console_out.task("옵티마이저 setup")
        dataset_size = len(tokenized)
        num_epochs = recipe.train.num_epochs
        max_steps = recipe.train.max_steps

        if max_steps is None:
            num_training_steps = dataset_size * num_epochs
        else:
            num_training_steps = min(max_steps, dataset_size * num_epochs)

        optimizer.setup({"num_training_steps": num_training_steps})
        console_out.detail(
            f"{num_training_steps} steps (dataset={dataset_size}, epochs={num_epochs})"
        )

    # Phase 4: Stage 1 사전학습 (Critic 전용, 조건부)
    value_head_path = None
    if recipe.train.algo == "critic-wmtp" and rm_model is not None and not dry_run:
        with console_out.phase("Stage 1 사전학습 (Value Head)"):
            console_out.task("Pretrainer 초기화")
            pretrainer = ComponentFactory.create_pretrainer(recipe)
            pretrainer.setup({})

            console_out.task("Value Head 훈련")
            stage1_result = pretrainer.run(
                {
                    "base_model": base,
                    "rm_model": rm_model,
                    "train_dataloader": train_dl,
                    "run_name": recipe.run.name or "default",
                    "config": config,
                }
            )

            if stage1_result.get("saved"):
                value_head_path = stage1_result["saved"]
                console_out.detail(f"저장 경로: {value_head_path}")
            else:
                console_out.warning("Stage 1 skipped or failed")

            if stage1_result.get("early_stopped"):
                console_out.warning(
                    f"Early stopped: {stage1_result.get('stop_reason')}"
                )

    # Phase 5: 메인 훈련
    with console_out.phase("메인 훈련"):
        console_out.task("Trainer 초기화")
        trainer = ComponentFactory.create_trainer(recipe, config)

        setup_ctx = {
            "model": base,
            "optimizer": optimizer,
            "mlflow_manager": mlflow,
            "ref_model": ref_model,
            "base_tokenizer": tokenizer,
            "rm_model": rm_model,
            "recipe": recipe,
            "checkpoint_data": checkpoint_data,
            "start_epoch": start_epoch,
            "start_step": start_step,
        }

        if recipe.train.algo == "critic-wmtp" and value_head_path:
            setup_ctx["value_head_path"] = value_head_path
            console_out.detail("Stage 1 Value Head 연결")

        trainer.setup(setup_ctx)

        if dry_run:
            console_out.info("DRY RUN - 훈련 스킵")
            mlflow.end_run("FINISHED")
            return RunOutputs(trainer_metrics={"dry_run": True})

        console_out.task("훈련 실행")
        metrics = trainer.run(
            {
                "train_dataloader": train_dl,
                "num_epochs": recipe.train.num_epochs,
                "max_steps": recipe.train.max_steps,
                "config": config,
            }
        )

    # 실험 종료
    mlflow.end_run("FINISHED")
    console_out.pipeline_end()
    console_out.info(f"최종 메트릭: {metrics}")

    return RunOutputs(trainer_metrics=metrics)
