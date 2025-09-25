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
  - mtp-baseline: Trainer(scorer=None) → 균등 가중치
  - critic-wmtp: Trainer(CriticScorer) → Value Function 기반 가중치
  - rho1-wmtp: Trainer(Rho1Scorer) → Reference Model 차이 기반 가중치

이를 통해 연구자는 알고리즘 간 공정한 성능 비교가 가능합니다.
"""

from pathlib import Path  # 경로 조작용
from typing import Any  # 범용 타입 힌트

# WMTP 컴포넌트 베이스 클래스들 - 모든 구현체가 상속받는 추상 인터페이스
from src.components.base import (
    Evaluator,  # 평가 수행 인터페이스 (HumanEval, MBPP 등)
    Loader,  # 데이터/모델 로딩 인터페이스
    Optimizer,  # 최적화기 인터페이스 (AdamW, Lion 등)
    Scorer,  # 토큰 가중치 계산 인터페이스 (Critic, Rho1 등)
    Trainer,  # 훈련 실행 인터페이스 (WMTP 통합 트레이너)
)

# 각 컴포넌트 타입별 Registry - 구현체들을 키로 등록/조회하는 저장소
from src.components.registry import (
    evaluator_registry,  # 평가기 구현체들 (meta-mtp, mbpp-v1 등)
    loader_registry,  # 로더 구현체들 (hf-model, mtp-native 등)
    optimizer_registry,  # 옵티마이저 구현체들 (adamw-bf16-fused 등)
    scorer_registry,  # 스코어러 구현체들 (critic-delta-v1, rho1-excess-v1 등)
    tokenizer_registry,  # 토크나이저 구현체들 (unified-sentencepiece 등)
    trainer_registry,  # 트레이너 구현체들 (mtp-weighted-ce-trainer 등)
)
from src.settings import Config, Recipe  # Pydantic 설정 모델들


class ComponentFactory:
    """WMTP 알고리즘별 컴포넌트 생성 팩토리.

    연구 철학 "Not All Tokens Are What You Need" 구현의 핵심:
        이 클래스는 설정 파일(recipe.yaml)의 알고리즘 선택에 따라
        적합한 컴포넌트 조합을 자동으로 생성합니다.

        모든 WMTP 알고리즘은 동일한 MTPWeightedCETrainer를 사용하되,
        서로 다른 Scorer를 조합하여 토큰 가중치 계산 방식을 차별화합니다.

    설계 원칙:
        1. 하드코딩 방지: 모든 매핑 정보를 클래스 상수로 관리
        2. Registry 패턴: 실제 구현체는 별도 Registry에서 조회
        3. 설정 주도: recipe.yaml의 값이 컴포넌트 선택을 결정
        4. 오류 처리: 잘못된 설정에 대한 명확한 에러 메시지 제공
    """

    # 🎯 직접 호출 방식: YAML 키가 곧 Registry 키
    # 매핑 딕셔너리 제거 - Pydantic 스키마와 Registry 키 완전 일치

    @staticmethod
    def create_scorer(recipe: Recipe) -> Scorer:
        """알고리즘별 토큰 가중치 계산 Scorer 생성.

        WMTP 핵심 철학 구현: "Not All Tokens Are What You Need"
            이 메서드는 각 알고리즘의 토큰 중요도 계산 방식을 구현한
            Scorer 인스턴스를 생성합니다. 이것이 WMTP와 기존 MTP의
            핵심적인 차이점입니다.

        알고리즘별 Scorer 매핑:
            - mtp-baseline: None → 모든 토큰에 가중치 1.0 (균등)
            - critic-wmtp: CriticDeltaScorer → δ_t = V_t - V_{t-1}
            - rho1-wmtp: Rho1ExcessScorer → |CE^ref_t - CE^base_t|

        Args:
            recipe: 훈련 레시피 설정 (알고리즘 및 하이퍼파라미터 포함)

        Returns:
            선택된 알고리즘에 맞는 Scorer 인스턴스 또는 baseline용 None

        Raises:
            ValueError: 지원되지 않는 알고리즘이 요청된 경우
        """
        algo = recipe.train.algo

        # Baseline: Scorer 없음 → 균등 가중치 (모든 토큰 = 1.0)
        if algo == "mtp-baseline":
            return None

        # 직접 호출: YAML algo 값이 곧 Registry 키

        # 알고리즘별 Scorer 설정 준비
        if algo == "critic-wmtp":
            # Critic: Value Function 기반 토큰 가중치 계산
            scorer_config = {
                "target": recipe.critic.target,  # 보상 타겟 ("rm_sequence")
                "token_spread": recipe.critic.token_spread,  # 확산 방식 ("gae")
                "delta_mode": recipe.critic.delta_mode,  # 델타 계산 ("td")
                "normalize": recipe.critic.normalize,  # 정규화 방식 ("zscore")
                "temperature": recipe.loss.temperature,  # 소프트맥스 온도
            }
        elif algo == "rho1-wmtp":
            # Rho1: Reference Model과의 CE 차이 기반 가중치
            scorer_config = {
                "score": recipe.rho1.score,  # 점수 계산 방식
                "percentile_top_p": recipe.rho1.percentile_top_p,  # 상위 백분위수
                "refresh_per_epoch": recipe.rho1.refresh_per_epoch,  # 에포크별 갱신 여부
                "temperature": recipe.loss.temperature,  # 소프트맥스 온도
            }
        else:
            # 예상치 못한 알고리즘의 경우 빈 설정
            scorer_config = {}

        # Registry에서 Scorer 인스턴스 생성 및 반환
        return scorer_registry.create(algo, scorer_config)

    @staticmethod
    def create_trainer(recipe: Recipe, config: Config) -> Trainer:
        """트레이너 생성 - recipe/config만 사용, scorer 의존성 자동 관리.

        WMTP 설계의 우아함: "One Trainer, Multiple Scorers"
            이 메서드는 WMTP의 핵심 설계 철학을 보여줍니다.
            모든 알고리즘이 동일한 MTPWeightedCETrainer를 사용하되,
            서로 다른 Scorer를 조합하여 차별화됩니다.

            이 통합 접근법의 장점:
            1. 공정한 비교: 알고리즘 간 차이는 오직 가중치 계산 방식
            2. 코드 중복 제거: 훈련 로직은 한 곳에만 구현
            3. 유지보수성: 새 알고리즘 추가시 Scorer만 개발
            4. 버그 최소화: 공통 로직은 한 번만 테스트

        알고리즘별 Trainer + Scorer 조합:
            - mtp-baseline: MTPWeightedCETrainer + None → 균등 가중치
            - critic-wmtp: MTPWeightedCETrainer + CriticDeltaScorer → δ 기반
            - rho1-wmtp: MTPWeightedCETrainer + Rho1ExcessScorer → CE 차이 기반

        Args:
            recipe: 훈련 레시피 (알고리즘, MTP 설정, 손실함수 등)
            config: 환경 설정 (GPU, 분산훈련, 메모리 최적화 등)

        Returns:
            설정된 MTPWeightedCETrainer 인스턴스

        Raises:
            ValueError: 지원되지 않는 알고리즘 요청시
        """
        # 1. scorer를 내부에서 자동 생성 (더 이상 별도 인자 불필요)
        if recipe.train.algo == "mtp-baseline":
            scorer = None  # Baseline: 균등 가중치
        else:
            scorer = ComponentFactory.create_scorer(recipe)  # 자동으로 적합한 scorer 생성

        # 2. trainer 설정 구성 (기존 로직 유지)
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
            "full_finetune": recipe.train.full_finetune,  # 전체 파인튜닝 vs LoRA
            # LoRA 설정 (메모리 효율적 파인튜닝)
            "lora_config": recipe.train.lora.model_dump()
            if recipe.train.lora.enabled
            else None,
            # 분산 훈련 및 메모리 최적화
            "mixed_precision": config.devices.mixed_precision,  # BF16/FP16 혼합 정밀도
            # FSDP (Fully Sharded Data Parallel) 설정
            "fsdp_config": config.devices.fsdp.model_dump()
            if config.devices.fsdp.enabled
            else None,
            # 🎯 핵심: 알고리즘별 차별화 요소 (자동 생성된 scorer)
            "scorer": scorer,  # 자동 생성된 scorer 포함
        }

        # 3. registry 생성 및 반환
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
            return pretrainer_registry.create("critic-stage1-pretrainer-v1", pretrainer_config)

        else:
            # 다른 알고리즘들은 단일 스테이지이므로 pretrainer 불필요
            raise ValueError(
                f"Algorithm '{algo}' does not support multi-stage training. "
                f"Only 'critic-wmtp' currently requires pretrainer."
            )

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
