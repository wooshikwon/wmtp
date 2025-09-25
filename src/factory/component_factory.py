"""WMTP 컴포넌트 팩토리 - 설정 기반 알고리즘 컴포넌트 생성.

연구 철학 지원: "Not All Tokens Are What You Need"
===============================================

이 팩토리는 WMTP의 핵심 설계 철학을 구현합니다:
동일한 파이프라인 구조에서 서로 다른 컴포넌트 조합을 통해
세 가지 알고리즘(mtp-baseline, critic-wmtp, rho1-wmtp)을 지원합니다.

팩토리 패턴의 장점:
  1. 설정 기반 생성: YAML 설정에서 자동으로 적합한 컴포넌트 선택
  2. 알고리즘 분리: 각 알고리즘의 특화된 로직을 컴포넌트 내부에 캡슐화
  3. 확장성: 새로운 알고리즘 추가 시 Registry에만 등록하면 됨
  4. 일관성: 모든 알고리즘이 동일한 인터페이스 사용

컴포넌트 조합 전략:
  - mtp-baseline: BaselineMtpTrainer → 균등 가중치
  - critic-wmtp: CriticWmtpTrainer → Value Head 직접 통합
  - rho1-wmtp: Rho1WmtpTrainer → Reference Model 차이 계산

이를 통해 연구자는 알고리즘 간 공정한 성능 비교가 가능합니다.
"""

from pathlib import Path  # 경로 조작용
from typing import Any  # 범용 타입 힌트

# WMTP 컴포넌트 베이스 클래스들 - 모든 구현체가 상속받는 추상 인터페이스
from src.components.base import (
    Evaluator,  # 평가 수행 인터페이스 (HumanEval, MBPP 등)
    Loader,  # 데이터/모델 로딩 인터페이스
    Optimizer,  # 최적화기 인터페이스 (AdamW, Lion 등)
    # Scorer 제거됨 (v2.1.0) - 모든 로직이 Trainer로 통합
    Trainer,  # 훈련 실행 인터페이스 (WMTP 통합 트레이너)
)

# 각 컴포넌트 타입별 Registry - 구현체들을 키로 등록/조회하는 저장소
from src.components.registry import (
    evaluator_registry,  # 평가기 구현체들 (meta-mtp, mbpp-v1 등)
    loader_registry,  # 로더 구현체들 (hf-model, mtp-native 등)
    optimizer_registry,  # 옵티마이저 구현체들 (adamw-bf16-fused 등)
    registry,  # 통합 레지스트리 (직접 접근용)
    # scorer_registry 제거됨 (v2.1.0) - 모든 scorer 로직이 trainer로 통합
    tokenizer_registry,  # 토크나이저 구현체들 (unified-sentencepiece 등)
    trainer_registry,  # 트레이너 구현체들 (mtp-weighted-ce-trainer 등)
)
from src.settings import Config, Recipe  # Pydantic 설정 모델들


class ComponentFactory:
    """WMTP 알고리즘별 컴포넌트 생성 팩토리.

    연구 철학 "Not All Tokens Are What You Need" 구현의 핵심:
        이 클래스는 설정 파일(recipe.yaml)의 알고리즘 선택에 따라
        적합한 컴포넌트 조합을 자동으로 생성합니다.

        Phase 2 리팩토링: 각 WMTP 알고리즘마다 독립된 트레이너 클래스를 사용합니다.
        - BaselineMtpTrainer: 균등 가중치
        - CriticWmtpTrainer: Critic 기반 가중치  
        - Rho1WmtpTrainer: Reference 모델 기반 가중치

    설계 원칙:
        1. 하드코딩 방지: 모든 매핑 정보를 클래스 상수로 관리
        2. Registry 패턴: 실제 구현체는 별도 Registry에서 조회
        3. 설정 주도: recipe.yaml의 값이 컴포넌트 선택을 결정
        4. 오류 처리: 잘못된 설정에 대한 명확한 에러 메시지 제공
    """

    # 🎯 직접 호출 방식: YAML 키가 곧 Registry 키
    # 매핑 딕셔너리 제거 - Pydantic 스키마와 Registry 키 완전 일치

    # create_scorer 메서드는 v2.1.0부터 제거됨
    # 모든 scorer 로직이 각각의 Trainer 클래스로 통합되었습니다.
    # - BaselineMtpTrainer: 균등 가중치 (scorer 불필요)
    # - CriticWmtpTrainer: Value Head 직접 관리
    # - Rho1WmtpTrainer: Reference Model 차이 직접 계산

    @staticmethod
    def create_trainer(recipe: Recipe, config: Config) -> Trainer:
        """트레이너 생성 - recipe/config만 사용, scorer 의존성 자동 관리.

        WMTP 설계의 우아함: "One Trainer, Multiple Scorers"
            이 메서드는 WMTP의 핵심 설계 철학을 보여줍니다.
            Phase 2 리팩토링으로 각 알고리즘마다 독립된 트레이너를 사용합니다.
            공통 로직은 BaseWmtpTrainer에 추상화되어 있습니다.

            이 통합 접근법의 장점:
            1. 공정한 비교: 알고리즘 간 차이는 오직 가중치 계산 방식
            2. 코드 중복 제거: 훈련 로직은 한 곳에만 구현
            3. 유지보수성: 새 알고리즘 추가시 Scorer만 개발
            4. 버그 최소화: 공통 로직은 한 번만 테스트

        알고리즘별 Trainer 매핑:
            - baseline-mtp: BaselineMtpTrainer → 균등 가중치
            - critic-wmtp: CriticWmtpTrainer → Value Head 직접 통합 (v2.1.0+)
            - rho1-wmtp: Rho1WmtpTrainer → Reference Model 차이 직접 계산

        Args:
            recipe: 훈련 레시피 (알고리즘, MTP 설정, 손실함수 등)
            config: 환경 설정 (GPU, 분산훈련, 메모리 최적화 등)

        Returns:
            알고리즘별 독립 Trainer 인스턴스 (BaseWmtpTrainer 상속)

        Raises:
            ValueError: 지원되지 않는 알고리즘 요청시
        """
        # Trainer 설정 구성
        trainer_config = {
            # MTP 모델 관련 설정
            "n_heads": recipe.model.mtp.n_heads,  # 예측 헤드 개수 (보통 4)
            "horizon": recipe.model.mtp.horizon,  # 예측 범위 (t+1, t+2, t+3, t+4)
            # 손실 함수 설정 - WMTP 공식 L_WMTP = Σ w_{t+k} × CE_k
            "loss_config": {
                "weight_norm": recipe.loss.weight_norm,  # 가중치 정규화 방식
                "lambda": recipe.loss.lambda_weight,  # 정규화 강도 λ
                "temperature": recipe.loss.temperature,  # 소프트맥스 온도
                "epsilon": recipe.loss.epsilon,  # 수치 안정성용 엡실론
                "max_weight": recipe.loss.max_weight,  # 최대 가중치 제한
            },
            # 훈련 방식 설정
            "full_finetune": recipe.train.full_finetune,  # 전체 파인튜닝 전용
            # 분산 훈련 및 메모리 최적화
            "mixed_precision": config.devices.mixed_precision,  # BF16/FP16 혼합 정밀도
            # FSDP (Fully Sharded Data Parallel) 설정
            "fsdp_config": config.devices.fsdp.model_dump()
            if config.devices.fsdp.enabled
            else None
        }

        # Registry에서 Trainer 인스턴스 생성 및 반환
        return trainer_registry.create(recipe.train.algo, trainer_config)

    @staticmethod
    def create_optimizer(recipe: Recipe, model_params: Any) -> Optimizer:
        """최적화기(Optimizer) 생성 - 모델 파라미터 업데이트 담당.

        현재 WMTP에서는 AdamW + BFloat16 + Fused 조합을 주로 사용합니다.
        이는 대규모 언어모델 훈련에서 검증된 안정적이고 효율적인 조합입니다.

        지원 최적화기:
            - adamw: AdamW + BF16 + Fused (추천, 메모리 효율적)
            - lion: Lion 옵티마이저 (미구현, 향후 추가 예정)
            - sgd: SGD with momentum (미구현, 향후 추가 예정)

        Args:
            recipe: 훈련 레시피 (학습률, 가중치 감쇠 등 옵티마이저 설정)
            model_params: 최적화할 모델 파라미터 (보통 model.parameters())

        Returns:
            설정된 Optimizer 인스턴스

        Raises:
            ValueError: 지원되지 않는 옵티마이저 요청시
        """
        # 직접 호출: YAML optimizer 값이 곧 Registry 키

        # 옵티마이저 설정 구성
        optimizer_config = {
            "params": model_params,  # 최적화할 파라미터
            "lr": recipe.optim.lr,  # 학습률
            "weight_decay": recipe.optim.weight_decay,  # L2 정규화 (가중치 감쇠)
            "betas": recipe.optim.betas,  # Adam 모멘텀 계수 (β₁, β₂)
            "grad_clip": recipe.optim.grad_clip,  # 그래디언트 클리핑 (폭발 방지)
            "scheduler": recipe.optim.scheduler,  # 학습률 스케줄러 타입
            "warmup_ratio": recipe.optim.warmup_ratio,  # 워밍업 비율
        }

        # Registry에서 Optimizer 인스턴스 생성 및 반환
        return optimizer_registry.create(recipe.optim.optimizer, optimizer_config)

    @staticmethod
    def create_data_loader(recipe: Recipe, config: Config) -> Loader:
        """데이터 로더 생성 - recipe/config만 사용하는 통합 패턴.

        WMTP는 다양한 코드 생성 벤치마크를 지원합니다:
            - MBPP: Python 기본 프로그래밍 문제
            - CodeContests: 알고리즘 경진 대회 문제
            - HumanEval: 함수 구현 평가 (OpenAI)
            - Custom: 사용자 정의 데이터셋

        Args:
            recipe: 훈련 레시피 (data.train.sources 필드 포함)
            config: 환경 설정

        Returns:
            UnifiedDataLoader 인스턴스
        """
        # 1. source를 recipe에서 자동 추출 (더 이상 별도 인자 불필요)
        source = recipe.data.train.sources[0]  # 첫 번째 훈련 소스 사용

        # 2. 소스별 데이터셋 경로 결정 (기존 로직 유지)
        dataset_path = None
        if source == "mbpp":
            dataset_path = str(config.paths.datasets.mbpp)
        elif source in ["contest", "codecontests"]:
            dataset_path = str(config.paths.datasets.contest)
        else:
            # Custom 또는 기타는 source를 그대로 경로로 사용
            dataset_path = source

        # 3. 통합 데이터 로더 설정 (기존 로직 유지)
        loader_config = {
            "storage": config.storage.model_dump(),
            "paths": config.paths.model_dump(),
            "split": "train",  # 기본 분할
            "dataset_type": source,  # 명시적 타입 지정
            "dataset_path": dataset_path,  # 경로 추가
        }

        # 4. UnifiedDataLoader 생성
        return loader_registry.create("unified-data-loader", loader_config)

    @staticmethod
    def create_model_loader(config: Config, recipe: Recipe = None) -> Loader:
        """통합 모델 로더만 반환 - Phase 2 리팩토링 적용.

        WMTP는 Facebook의 native MTP 모델을 기본으로 사용하되,
        다양한 모델 소스와 포맷을 지원합니다:
            - mtp-native: Facebook native MTP (consolidated.pth)
            - hf-model: HuggingFace 변환된 모델
            - checkpoint: 훈련 중단점 파일 (.pt/.pth)
            - sheared-llama: Princeton 경량화 모델
            - starling-rm: Berkeley 보상 모델

        Args:
            config: 환경 설정 (모델 경로, GPU 설정 등)
            recipe: 훈련 레시피 (선택)

        Returns:
            UnifiedModelLoader 인스턴스
        """
        # 통합 모델 로더 설정
        loader_config = config.model_dump()

        # UnifiedModelLoader 생성 - 모든 모델 타입을 하나의 로더로 처리
        return loader_registry.create("unified-model-loader", loader_config)

    @staticmethod
    def create_checkpoint_loader(config: Config) -> Loader:
        """체크포인트 전용 로더 생성 - 훈련 재개를 위한 특화된 인터페이스.

        훈련 재개 시나리오를 위한 전용 로더:
            - 메타데이터 자동 추출 (epoch, step, mlflow_run_id)
            - S3/로컬 통합 지원
            - Rich Console 기반 진행상황 표시
            - 견고한 오류 처리

        UnifiedModelLoader와의 차이점:
            - 체크포인트 전용 최적화
            - 훈련 메타데이터 자동 파싱
            - 재개 전용 인터페이스

        Args:
            config: 환경 설정 (S3, 경로, GPU 설정)

        Returns:
            CheckpointLoader 인스턴스

        Usage:
            ```python
            checkpoint_loader = ComponentFactory.create_checkpoint_loader(config)
            checkpoint_loader.setup({})
            result = checkpoint_loader.run({
                "model_path": "s3://wmtp/checkpoints/model.pt",
                "load_metadata": True
            })
            epoch = result["epoch"]
            step = result["step"]
            mlflow_run_id = result["mlflow_run_id"]
            ```
        """
        # 체크포인트 로더 설정 구성
        loader_config = config.model_dump()

        # CheckpointLoader 생성 - 체크포인트 전용 특화 기능
        return loader_registry.create("checkpoint-loader", loader_config)

    @staticmethod
    def create_evaluator(recipe: Recipe, config: Config) -> Evaluator:
        """평가 프로토콜별 특화된 평가기 생성.

        각 벤치마크마다 다른 평가 방식과 메트릭이 필요합니다:
            - meta-mtp: Meta MTP 논문 방식 (pass@k, 추론 속도)
            - mbpp: MBPP 테스트 케이스 실행 기반 평가
            - codecontests: 경진 대회 문제 정답 비교

        Args:
            recipe: 훈련 레시피 (평가 설정, 샘플링 파라미터)
            config: 환경 설정 (배치 크기, GPU 설정)

        Returns:
            프로토콜에 맞는 Evaluator 인스턴스

        Raises:
            ValueError: 지원되지 않는 평가 프로토콜
        """
        # 직접 호출: YAML protocol 값이 곧 Registry 키
        protocol = recipe.eval.protocol

        # 평가기 설정 구성
        evaluator_config = {
            "sampling": recipe.eval.sampling.model_dump(),  # 샘플링 파라미터
            "metrics": recipe.eval.metrics,  # 평가 메트릭 리스트
            "batch_size": recipe.data.eval.batch_size,  # 평가 배치 크기
        }

        # Registry에서 특화된 평가기 생성
        return evaluator_registry.create(protocol, evaluator_config)

    @staticmethod
    def create_pretrainer(recipe: Recipe) -> Any:
        """알고리즘별 사전훈련기 생성 - ComponentFactory 패턴 일관성 유지.

        현재는 critic-wmtp의 Stage1 pretrainer만 지원하지만,
        향후 다른 알고리즘의 multi-stage 학습을 위해 확장 가능한 구조로 설계.

        알고리즘별 Pretrainer 매핑:
            - critic-wmtp: Stage1 Value Head 훈련기
            - rho1-wmtp: 현재 미지원 (단일 스테이지)
            - mtp-baseline: 현재 미지원 (단일 스테이지)

        Args:
            recipe: 훈련 레시피 설정 (알고리즘 및 critic 설정 포함)

        Returns:
            선택된 알고리즘에 맞는 Pretrainer 인스턴스

        Raises:
            ValueError: 지원되지 않는 알고리즘이 요청된 경우
        """
        algo = recipe.train.algo

        if algo == "critic-wmtp":
            # Critic: Stage1 Value Head 훈련을 위한 설정
            pretrainer_config = {
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

            # Registry에서 Stage1 Pretrainer 인스턴스 생성 및 반환
            from src.components.registry import pretrainer_registry
            return pretrainer_registry.create("critic-head-pretrainer", pretrainer_config)

        else:
            # 다른 알고리즘들은 단일 스테이지이므로 pretrainer 불필요
            raise ValueError(
                f"Algorithm '{algo}' does not support multi-stage training. "
                f"Only 'critic-wmtp' currently requires pretrainer."
            )

    @staticmethod
    def create_aux_model_loader(recipe: Recipe, config: Config, aux_type: str) -> Loader:
        """알고리즘별 보조 모델 로더 생성 - ref/rm 모델 전용.

        WMTP 알고리즘별 보조 모델 처리를 위한 특화된 로더:
            - rho1-wmtp: Reference Model 로딩 (CE 차이 계산용)
            - critic-wmtp: Reward Model 로딩 (Value Head 훈련용)
            - mtp-baseline: 보조 모델 불필요 (에러 발생)

        create_model_loader와의 차이점:
            1. 알고리즘별 특화: recipe.train.algo에 따른 검증
            2. 보조 모델 전용: base 모델과 구분된 처리
            3. 경로 자동 매핑: aux_type에 따른 config 경로 자동 선택
            4. 타입 안전성: 잘못된 알고리즘-모델 조합 차단

        Args:
            recipe: 훈련 레시피 (알고리즘 타입 확인용)
            config: 환경 설정 (모델 경로들 포함)
            aux_type: 보조 모델 타입 ("ref" | "rm")

        Returns:
            알고리즘에 맞는 UnifiedModelLoader 인스턴스

        Raises:
            ValueError: 잘못된 알고리즘-보조모델 조합

        Usage:
            ```python
            # Rho1 알고리즘의 참조 모델
            ref_loader = ComponentFactory.create_aux_model_loader(
                recipe, config, "ref"
            )

            # Critic 알고리즘의 보상 모델
            rm_loader = ComponentFactory.create_aux_model_loader(
                recipe, config, "rm"
            )
            ```
        """
        algo = recipe.train.algo

        # 알고리즘별 보조 모델 필요성 검증
        if algo == "baseline-mtp":
            raise ValueError(
                f"Algorithm '{algo}' does not require auxiliary models. "
                f"Use create_model_loader() for base model only."
            )
        elif algo == "rho1-wmtp" and aux_type != "ref":
            raise ValueError(
                f"Algorithm '{algo}' only supports aux_type='ref' for reference model. "
                f"Got aux_type='{aux_type}'"
            )
        elif algo == "critic-wmtp" and aux_type != "rm":
            raise ValueError(
                f"Algorithm '{algo}' only supports aux_type='rm' for reward model. "
                f"Got aux_type='{aux_type}'"
            )
        elif algo not in ["rho1-wmtp", "critic-wmtp"]:
            raise ValueError(
                f"Algorithm '{algo}' is not supported for auxiliary model loading. "
                f"Supported algorithms: 'rho1-wmtp', 'critic-wmtp'"
            )

        # aux_type에 따른 모델 경로 자동 매핑
        aux_model_paths = {
            "ref": config.paths.models.ref,
            "rm": config.paths.models.rm,
        }

        if aux_type not in aux_model_paths:
            raise ValueError(
                f"Invalid aux_type '{aux_type}'. "
                f"Supported types: {list(aux_model_paths.keys())}"
            )

        # UnifiedModelLoader 설정 - base loader와 동일하나 경로 및 메타데이터 차별화
        loader_config = config.model_dump()

        # 보조 모델 메타데이터 추가
        loader_config["_aux_model_info"] = {
            "algorithm": algo,
            "aux_type": aux_type,
            "model_path": str(aux_model_paths[aux_type]),
            "description": f"{algo} auxiliary {aux_type} model loader",
        }

        # UnifiedModelLoader 생성 - 동일한 로더 클래스 사용
        return loader_registry.create("unified-model-loader", loader_config)

    @staticmethod
    def create_tokenizer(recipe: Recipe, config: Config) -> Any:
        """토크나이저 생성 - recipe/config만 사용하는 통합 패턴.

        두 가지 토크나이저 중 recipe 설정에 따라 자동 선택:
        1. "hf": HfSentencePieceTokenizer - HuggingFace 호환 인터페이스
        2. "raw": SentencePieceTokenizer - Raw SentencePiece 인터페이스

        Args:
            recipe: 훈련 레시피 (tokenizer_type 필드 포함)
            config: 환경 설정 (토크나이저 경로 정보 포함)

        Returns:
            토크나이저 BaseComponent 인스턴스

        Raises:
            ValueError: 지원되지 않는 tokenizer_type
        """
        # 1. tokenizer_type을 recipe에서 가져옴 (더 이상 별도 인자 불필요)
        tokenizer_type = recipe.model.tokenizer_type

        # 2. Registry 키 결정 - recipe 기반 tokenizer_type
        if tokenizer_type in ["hf", "huggingface", "hf-sentencepiece"]:
            registry_key = "hf"
        elif tokenizer_type in ["raw", "sentencepiece", "default"]:
            registry_key = "default"
        else:
            raise ValueError(
                f"지원되지 않는 tokenizer_type: {tokenizer_type}. "
                f"사용 가능한 옵션: 'hf', 'huggingface', 'raw', 'sentencepiece'"
            )

        # 3. 설정 구성 - config 값 직접 사용
        tokenizer_config = config.model_dump()

        # 4. Registry 생성 및 반환 - 표준 패턴
        return tokenizer_registry.create(registry_key, tokenizer_config)

    @staticmethod
    def create_evaluator_by_type(eval_type: str, recipe: Recipe, config: Config) -> Evaluator:
        """평가 타입별 특화된 평가기 생성 (Meta 논문 지원).

        Meta 2024 MTP 논문의 모든 평가 항목을 재현하기 위한
        평가기 동적 생성 메서드. evaluation_pipeline.py에서 사용됩니다.

        Args:
            eval_type: 평가 타입
                - "meta-mtp": Pass@k 메트릭
                - "inference-speed": 추론 속도 비교
                - "per-head-analysis": 헤드별 성능 분석
                - "token-accuracy": 토큰 위치별 정확도
            recipe: 평가 레시피 설정
            config: 환경 설정

        Returns:
            평가 타입에 맞는 Evaluator 인스턴스

        Raises:
            ValueError: 지원되지 않는 평가 타입
        """
        # torch import for CUDA check
        import torch

        # 평가 타입별 설정 구성
        eval_configs = {
            "meta-mtp": {
                "metrics": recipe.eval.metrics,
                "sampling": recipe.eval.sampling.model_dump(),
                "batch_size": recipe.data.eval.batch_size,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "inference-speed": {
                "batch_sizes": [1, 4, 8, 16],
                "sequence_lengths": [512, 1024, 2048],
                "num_trials": 10,
                "warmup_steps": 3,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "per-head-analysis": {
                "analyze_positions": True,
                "compute_confidence": True,
                "head_comparison": True,
                "position_buckets": [(0, 128), (128, 512), (512, 1024), (1024, 2048)],
                "batch_size": recipe.data.eval.batch_size,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "token-accuracy": {
                "position_range": (0, 100),
                "token_types": ["code", "text", "special"],
                "accuracy_threshold": 0.5,
                "granularity": 10,
                "analyze_token_types": True,
                "batch_size": recipe.data.eval.batch_size,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            # Phase 2 평가기 추가
            "self-speculative": {
                "num_sequences": 100,
                "max_tokens": 512,
                "temperature": recipe.eval.sampling.temperature if hasattr(recipe.eval.sampling, 'temperature') else 0.8,
                "top_p": recipe.eval.sampling.top_p if hasattr(recipe.eval.sampling, 'top_p') else 0.95,
                "measure_speedup": True,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "perplexity-measurer": {
                "batch_size": recipe.data.eval.batch_size,
                "max_length": 2048,
                "position_buckets": [[0, 128], [128, 512], [512, 1024], [1024, 2048]],
                "analyze_token_types": True,
                "compute_head_perplexity": True,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "metrics-visualizer": {
                "output_dir": "./figures",
                "save_formats": ["png", "pdf"],
                "use_plotly": True,
                "upload_to_mlflow": True,
                "figure_size": [10, 6]
            }
        }

        if eval_type not in eval_configs:
            raise ValueError(
                f"지원되지 않는 평가 타입: {eval_type}. "
                f"사용 가능한 옵션: {list(eval_configs.keys())}"
            )

        # 평가기 설정 가져오기
        eval_config = eval_configs[eval_type]

        # Registry에서 평가기 생성
        return evaluator_registry.create(eval_type, eval_config)
