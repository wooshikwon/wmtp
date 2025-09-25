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

    if resume_checkpoint and resume_checkpoint.exists():
        import torch
        from rich.console import Console

        console = Console()
        checkpoint_data = torch.load(resume_checkpoint, map_location="cpu")
        start_epoch = checkpoint_data.get("epoch", 0)
        start_step = checkpoint_data.get("step", 0)
        resume_run_id = checkpoint_data.get("mlflow_run_id")

        console.print(
            f"[green]Resuming from epoch {start_epoch}, step {start_step}[/green]"
        )

    # MLflow 실험 추적 매니저 초기화 및 실행 시작/재개
    mlflow = create_mlflow_manager(config.model_dump())
    tag_map = {
        str(i): t for i, t in enumerate(tags or [])
    }  # 태그를 MLflow 형식으로 변환

    if resume_run_id:
        mlflow.start_run(run_id=resume_run_id, resume=True)
    else:
        mlflow.start_run(run_name=run_name or recipe.run.name, tags=tag_map)

    # Step 2: 기본 모델 로딩 (모든 알고리즘에서 공통으로 필요)
    # Facebook의 native MTP 모델 - 4개 head가 내장된 아키텍처 사용
    base_loader = ComponentFactory.create_model_loader(config, recipe)
    base_loader.setup({})  # 로더 초기화

    # Base 모델은 항상 필요 - WMTP의 핵심이 되는 Multi-Token Prediction 모델
    base_result = base_loader.run(
        {
            "model_path": str(config.paths.models.base_local)  # 로컬에 캐시된 모델 경로
        }
    )
    base = base_result["model"]  # Facebook MTP 모델 인스턴스
    tokenizer = base_result["tokenizer"]  # 모델과 호환되는 토크나이저

    # Step 3: 알고리즘별 추가 모델 로딩 (조건부)
    # 각 WMTP 알고리즘은 서로 다른 보조 모델을 필요로 함
    ref_model = None  # Rho-1에서 사용할 참조 모델
    rm_model = None  # Critic에서 사용할 보상 모델

    if recipe.train.algo == "rho1-wmtp":
        # Rho-1 알고리즘: Reference Model이 필요
        # |CE^ref_t - CE^base_t| 계산을 위해 참조 모델의 CE 값 필요
        ref_loader = ComponentFactory.create_model_loader(
            config
        )  # Recipe 없으면 HF 로더
        ref_loader.setup({})
        ref_result = ref_loader.run(
            {
                "model_path": str(
                    config.paths.models.ref_local
                )  # CodeLlama 등 참조 모델
            }
        )
        ref_model = ref_result["model"]

    elif recipe.train.algo == "critic-wmtp":
        # Critic 알고리즘: Reward Model이 필요
        # Stage1에서 시퀀스 레벨 보상 계산 및 Value Head 훈련에 사용
        rm_loader = ComponentFactory.create_model_loader(
            config
        )  # Recipe 없으면 HF 로더
        rm_loader.setup({})
        rm_result = rm_loader.run(
            {
                "model_path": str(config.paths.models.rm_local)  # Llama RM 등 보상 모델
            }
        )
        rm_model = rm_result["model"]

    # mtp-baseline은 추가 모델 불필요 - Base 모델만으로 균등 가중치 MTP 수행

    # Step 4: 옵티마이저 설정
    # 대부분의 경우 AdamW + BF16 + FSDP 조합 사용
    optimizer = ComponentFactory.create_optimizer(recipe, base.parameters())
    optimizer.setup(
        {"num_training_steps": max_steps or 0}
    )  # 스케줄러를 위한 총 스텝 수

    # 📝 중요: Facebook native MTP 모델은 4개의 horizon head가 내장되어 있음
    # 별도의 MTPWrapper 불필요 - native implementation 직접 사용
    # 이는 성능과 메모리 효율성 측면에서 유리

    # Step 5: 데이터셋 로딩 및 전처리
    # 지원 데이터셋: MBPP, CodeContests, HumanEval, Custom
    train_source = recipe.data.train.sources[0]  # 첫 번째 훈련 소스 사용
    train_loader_comp = ComponentFactory.create_data_loader(train_source, config)
    train_loader_comp.setup({})

    # 훈련 데이터셋 로드 - 문제와 솔루션이 포함된 형태
    train_ds = train_loader_comp.run(
        {
            "split": "train",  # 훈련 분할 사용
            "max_length": recipe.data.train.max_length,  # 최대 시퀀스 길이
            "add_solution": True,  # 솔루션 포함 (코드 생성 태스크)
        }
    )["dataset"]

    # Step 6: 토크나이징 - 텍스트를 모델이 이해할 수 있는 숫자로 변환
    def _tokenize_function(example: dict[str, Any]) -> dict[str, Any]:
        """개별 데이터 샘플을 토큰화하는 내부 함수.

        Args:
            example: 데이터셋의 한 샘플 (딕셔너리 형태)

        Returns:
            토큰화된 결과 (input_ids, attention_mask, labels 포함)
        """
        # 텍스트 추출 - 데이터셋 형식에 따라 다른 키 사용 가능
        text = example.get("full_text") or example.get("prompt") or ""

        # 토크나이저로 텍스트를 숫자 시퀀스로 변환
        tok = tokenizer(
            text,
            truncation=True,  # 최대 길이 초과시 자르기
            max_length=recipe.data.train.max_length,  # 최대 시퀀스 길이
            padding=False,  # 배치에서 패딩 (여기서는 하지 않음)
        )
        # 라벨은 input_ids와 동일 (언어모델은 다음 토큰 예측)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    # 전체 데이터셋에 토크나이징 적용
    tokenized = train_ds.map(
        _tokenize_function,
        remove_columns=train_ds.column_names,  # 원본 텍스트 컬럼 제거 (메모리 절약)
        desc="훈련 데이터 토크나이징",  # 진행률 표시용 설명
        load_from_cache_file=True,  # 캐시 사용으로 재실행시 속도 향상
    )

    # Step 7: 분산 훈련을 위한 데이터 샘플러 설정
    sampler = None  # 기본값: 샘플러 없음
    try:
        import torch.distributed as dist

        # 분산 훈련이 활성화되어 있는지 확인
        if dist.is_available() and dist.is_initialized():
            # DistributedSampler: 각 GPU가 다른 데이터 부분을 처리하도록 분배
            sampler = DistributedSampler(tokenized, shuffle=True)
    except Exception:
        # 분산 훈련이 설정되지 않은 경우 None 유지
        sampler = None

    # Step 8: PyTorch DataLoader 생성 - 배치 단위로 데이터 공급
    train_dl = DataLoader(
        tokenized,  # 토큰화된 데이터셋
        batch_size=recipe.data.train.batch_size or 1,  # 배치 크기 (메모리에 따라 조정)
        shuffle=(sampler is None),  # 분산 훈련이 아닐 때만 셔플
        sampler=sampler,  # 분산 훈련용 샘플러 (있는 경우)
        collate_fn=default_data_collator,  # HuggingFace의 기본 배치 생성기
        num_workers=2,  # 데이터 로딩용 워커 프로세스 수
        pin_memory=torch.cuda.is_available(),  # GPU 사용시 메모리 핀닝으로 속도 향상
    )

    # Step 9: Stage1 사전훈련 (critic-wmtp 전용)
    # Critic 알고리즘만의 특별한 2단계 학습 과정
    if recipe.train.algo == "critic-wmtp" and rm_model is not None and not dry_run:
        from pathlib import Path

        from src.components.registry import trainer_registry

        # Stage1 설정: Value Head 훈련을 위한 파라미터들
        pre_cfg = {
            # 보상 타겟: "rm_sequence" (시퀀스 레벨 보상 사용)
            "target": getattr(recipe.critic, "target", "rm_sequence")
            if hasattr(recipe, "critic")
            else "rm_sequence",
            # 토큰 확산 방식: "gae" (Generalized Advantage Estimation)
            "token_spread": getattr(recipe.critic, "token_spread", "gae")
            if hasattr(recipe, "critic")
            else "gae",
            # 델타 계산 모드: "td" (Temporal Difference)
            "delta_mode": getattr(recipe.critic, "delta_mode", "td")
            if hasattr(recipe, "critic")
            else "td",
            # 정규화 방식: "zscore" (표준화)
            "normalize": getattr(recipe.critic, "normalize", "zscore")
            if hasattr(recipe, "critic")
            else "zscore",
            "temperature": recipe.loss.temperature,  # 소프트맥스 온도
            "lr": 1e-4,  # Stage1 전용 학습률 (보통 메인보다 낮음)
        }

        # Stage1 전용 trainer 생성 및 실행
        pretrainer = trainer_registry.create("critic-stage1-pretrainer-v1", pre_cfg)
        cache_root = (
            Path(config.paths.cache) / "critic" / (recipe.run.name or "default")
        )
        pretrainer.setup({})

        # Stage1 실행: Value Head 훈련
        # RM 모델로부터 시퀀스 레벨 보상을 받아 Value Function 학습
        pretrainer.run(
            {
                "base_model": base,  # 기본 MTP 모델
                "rm_model": rm_model,  # 보상 점수 제공 모델
                "train_dataloader": train_dl,  # 훈련 데이터
                "cache_root": cache_root,  # Value Head 체크포인트 저장 위치
            }
        )

    # Step 10: 메인 Trainer 생성 - 알고리즘별 다른 설정
    # WMTP의 핵심: 동일한 트레이너 구조에 다른 Scorer 조합
    if recipe.train.algo == "mtp-baseline":
        # Baseline: Scorer 없음 - 순수 MTP (균등 가중치)
        # 모든 토큰에 동일한 가중치 1.0 적용
        scorer = None
        trainer = ComponentFactory.create_trainer(recipe, config, scorer)
    else:
        # Weighted 방식: Scorer 사용 - 토큰별 중요도 계산
        # critic-wmtp 또는 rho1-wmtp에서 동적 가중치 적용
        scorer = ComponentFactory.create_scorer(recipe)

        # Critic의 경우: Stage1에서 훈련된 Value Head 경로 제공
        try:
            from pathlib import Path

            # Stage1에서 저장된 value_head.pt 파일 경로
            vh_path = (
                Path(config.paths.cache)
                / "critic"
                / (recipe.run.name or "default")
                / "value_head.pt"
            )
            if vh_path.exists():
                # Value Head가 존재하면 Scorer에 경로 제공
                scorer.setup({"value_head_path": vh_path})
            else:
                # Value Head가 없으면 기본 설정으로 진행
                scorer.setup({})
        except Exception:
            # 오류 발생시 기본 설정 사용
            scorer.setup({})

        # 최종 Trainer 생성 - Scorer가 포함된 가중치 기반 훈련
        trainer = ComponentFactory.create_trainer(recipe, config, scorer)
    # Step 11: Trainer 초기화 - 모든 필요한 컴포넌트 연결
    trainer.setup(
        {
            "model": base,  # Facebook native MTP 모델
            "optimizer": optimizer,  # AdamW 등 최적화기
            "mlflow_manager": mlflow,  # 실험 추적 매니저
            "ref_model": ref_model,  # Rho-1용 참조 모델 (해당시)
            "base_tokenizer": tokenizer,  # 토크나이저
            "rm_model": rm_model,  # Critic용 보상 모델 (해당시)
            "recipe": recipe,  # 체크포인트 설정을 위한 Recipe 전달
            "resume_checkpoint": resume_checkpoint,  # 재개용 체크포인트
        }
    )

    # Step 12: 실행 모드에 따른 처리
    if dry_run:
        # 검증 모드: 설정만 확인하고 실제 훈련은 건너뛰기
        mlflow.end_run("FINISHED")  # MLflow 실행 종료
        return RunOutputs(trainer_metrics={"dry_run": True})

    # Step 13: 메인 훈련 실행
    # 여기서 실제 WMTP 훈련이 수행됨 - 알고리즘에 따라 다른 가중치 적용
    # L_WMTP = Σ w_{t+k} × CE_k (k=1,2,3,4)
    metrics = trainer.run({"train_dataloader": train_dl, "max_steps": max_steps})

    # Step 14: 실험 종료 및 결과 반환
    mlflow.end_run("FINISHED")  # MLflow 추적 종료
    return RunOutputs(trainer_metrics=metrics)  # 훈련 메트릭 반환
