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

    # 🎯 핵심 매핑 테이블들 - WMTP 알고리즘별 컴포넌트 선택 규칙

    # 알고리즘 → Scorer Registry 키 매핑
    # WMTP의 핵심: 각 알고리즘은 고유한 토큰 가중치 계산 방식을 가짐
    ALGO_TO_SCORER = {
        "critic-wmtp": "critic-delta-v1",  # δ_t = V_t - V_{t-1} 차분값 기반
        "rho1-wmtp": "rho1-excess-v1",  # |CE^ref_t - CE^base_t| 차이 기반
        # "mtp-baseline"은 scorer=None - 균등 가중치(1.0)
    }

    # 옵티마이저 이름 → Registry 키 매핑
    # 현재는 AdamW + BF16 + Fused 조합만 구현됨
    OPTIMIZER_MAP = {
        "adamw": "adamw-bf16-fused",  # AdamW + BFloat16 + 융합 최적화
        # "lion": "lion-optimizer",          # Lion 옵티마이저 (미구현)
        # "sgd": "sgd-optimizer",            # SGD 옵티마이저 (미구현)
    }

    # 🔑 통합 설계의 핵심: 모든 알고리즘이 동일한 Trainer 사용
    # 차이점은 Scorer 조합뿐 - 이것이 WMTP의 우아한 설계
    ALGO_TO_TRAINER = {
        "mtp-baseline": "mtp-weighted-ce-trainer",  # scorer=None (균등)
        "critic-wmtp": "mtp-weighted-ce-trainer",  # CriticDeltaScorer 조합
        "rho1-wmtp": "mtp-weighted-ce-trainer",  # Rho1ExcessScorer 조합
    }

    # 평가 프로토콜 → Evaluator Registry 키 매핑
    # 각 벤치마크별 특화된 평가 방식 제공
    EVALUATOR_MAP = {
        "meta-mtp": "meta-mtp-evaluator",  # Meta MTP 논문 평가 방식
        "mbpp": "mbpp-v1",  # MBPP 코드 생성 평가
        "codecontests": "codecontests-v1",  # CodeContests 경진 평가
    }

    # 토크나이저 타입 → Tokenizer Registry 키 매핑
    # 현재는 unified-sentencepiece만 지원 (모든 WMTP 모델 호환)
    TOKENIZER_MAP = {
        "unified": "unified-sentencepiece",  # 기본 통합 SentencePiece 토크나이저
        "sentencepiece": "unified-sentencepiece",  # 명시적 SentencePiece
        "default": "unified-sentencepiece",  # 기본값
    }

    @classmethod
    def create_scorer(cls, recipe: Recipe) -> Scorer:
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

        # Registry에서 알고리즘에 맞는 Scorer 키 조회
        scorer_key = cls.ALGO_TO_SCORER.get(algo)
        if not scorer_key:
            raise ValueError(
                f"'{algo}' 알고리즘에 대한 Scorer 매핑을 찾을 수 없습니다. "
                f"지원 알고리즘: {list(cls.ALGO_TO_SCORER.keys())}"
            )

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
        return scorer_registry.create(scorer_key, scorer_config)

    @classmethod
    def create_trainer(
        cls,
        recipe: Recipe,  # 훈련 레시피 설정
        config: Config,  # 환경 설정
        scorer: Scorer | None = None,  # 토큰 가중치 계산기 (create_scorer에서 생성)
    ) -> Trainer:
        """WMTP 통합 Trainer 생성 - 모든 알고리즘의 핵심 실행기.

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
            scorer: 토큰 가중치 계산기 (None이면 균등 가중치)

        Returns:
            설정된 MTPWeightedCETrainer 인스턴스

        Raises:
            ValueError: 지원되지 않는 알고리즘 요청시
        """
        # 알고리즘에 따른 Trainer Registry 키 조회 (모든 알고리즘이 동일함)
        trainer_key = cls.ALGO_TO_TRAINER.get(recipe.train.algo)

        if not trainer_key:
            raise ValueError(
                f"'{recipe.train.algo}' 알고리즘에 대한 Trainer 매핑을 찾을 수 없습니다. "
                f"지원 알고리즘: {list(cls.ALGO_TO_TRAINER.keys())}"
            )

        # Trainer 설정 구성 - 모든 필요한 하이퍼파라미터와 컴포넌트
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
            # 🎯 핵심: 알고리즘별 차별화 요소
            "scorer": scorer,  # None(baseline) / CriticScorer / Rho1Scorer
        }

        # Registry에서 설정된 Trainer 인스턴스 생성 및 반환
        return trainer_registry.create(trainer_key, trainer_config)

    @classmethod
    def create_optimizer(cls, recipe: Recipe, model_params: Any) -> Optimizer:
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
        # 옵티마이저 이름으로 Registry 키 조회
        optimizer_key = cls.OPTIMIZER_MAP.get(recipe.optim.optimizer)

        if not optimizer_key:
            raise ValueError(
                f"'{recipe.optim.optimizer}' 옵티마이저는 지원되지 않습니다. "
                f"사용 가능한 옵티마이저: {list(cls.OPTIMIZER_MAP.keys())}"
            )

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
        return optimizer_registry.create(optimizer_key, optimizer_config)

    @classmethod
    def create_data_loader(cls, source: str, config: Config) -> Loader:
        """통합 데이터 로더만 반환 - Phase 2 리팩토링 적용.

        WMTP는 다양한 코드 생성 벤치마크를 지원합니다:
            - MBPP: Python 기본 프로그래밍 문제
            - CodeContests: 알고리즘 경진 대회 문제
            - HumanEval: 함수 구현 평가 (OpenAI)
            - Custom: 사용자 정의 데이터셋

        Args:
            source: 데이터 소스명 (mbpp/codecontests/humaneval/custom)
            config: 환경 설정

        Returns:
            UnifiedDataLoader 인스턴스
        """
        # 소스별 데이터셋 경로 결정
        dataset_path = None
        if source == "mbpp":
            dataset_path = str(config.paths.datasets.mbpp)
        elif source in ["contest", "codecontests"]:
            dataset_path = str(config.paths.datasets.contest)
        else:
            # Custom 또는 기타는 source를 그대로 경로로 사용
            dataset_path = source

        # 통합 데이터 로더 설정
        loader_config = {
            "storage": config.storage.model_dump(),
            "paths": config.paths.model_dump(),
            "split": "train",  # 기본 분할
            "dataset_type": source,  # 명시적 타입 지정
        }

        # UnifiedDataLoader 생성
        return loader_registry.create("unified-data-loader", loader_config)

    @classmethod
    def create_model_loader(cls, config: Config, recipe: Recipe = None) -> Loader:
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

    @classmethod
    def create_evaluator(cls, recipe: Recipe, config: Config) -> Evaluator:
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
        protocol = recipe.eval.protocol
        evaluator_key = cls.EVALUATOR_MAP.get(protocol)

        if not evaluator_key:
            raise ValueError(
                f"'{protocol}' 평가 프로토콜은 지원되지 않습니다. "
                f"사용 가능한 프로토콜: {list(cls.EVALUATOR_MAP.keys())}"
            )

        # 평가기 설정 구성
        evaluator_config = {
            "sampling": recipe.eval.sampling.model_dump(),  # 샘플링 파라미터
            "metrics": recipe.eval.metrics,  # 평가 메트릭 리스트
            "batch_size": recipe.data.eval.batch_size,  # 평가 배치 크기
        }

        # Registry에서 특화된 평가기 생성
        return evaluator_registry.create(evaluator_key, evaluator_config)

    @classmethod
    def create_tokenizer(cls, config: Config, tokenizer_type: str = "default") -> Any:
        """통합 토크나이저 생성 - ComponentFactory 패턴 완성.

        다른 create_* 메서드들과 동일한 Registry 패턴을 사용하여
        일관된 컴포넌트 생성 인터페이스를 제공합니다.

        현재 모든 WMTP 모델이 동일한 SentencePiece tokenizer.model을 사용하므로
        tokenizer_type에 관계없이 unified-sentencepiece가 선택됩니다.

        Args:
            config: 환경 설정 (모델 경로 정보 포함)
            tokenizer_type: 토크나이저 타입 (기본: "default", 모두 unified로 매핑)

        Returns:
            SentencePieceProcessor 인스턴스 (싱글톤)

        Raises:
            ValueError: 지원되지 않는 tokenizer_type
            FileNotFoundError: tokenizer.model 파일을 찾을 수 없는 경우
        """
        # Registry 키 결정 (현재는 모두 unified-sentencepiece로 매핑)
        tokenizer_key = cls.TOKENIZER_MAP.get(tokenizer_type, "unified-sentencepiece")

        # Config 기반 토크나이저 경로 결정 - 다른 로더들과 동일한 우선순위 로직
        tokenizer_path = cls._resolve_tokenizer_path(config)

        # Registry 패턴으로 토크나이저 설정 구성
        tokenizer_config = {
            "tokenizer_path": tokenizer_path,  # 우선순위 기반 경로
        }

        # Registry에서 토크나이저 컴포넌트 생성
        tokenizer_component = tokenizer_registry.create(tokenizer_key, tokenizer_config)

        # 컴포넌트 초기화 및 실행
        tokenizer_component.setup({"tokenizer_path": tokenizer_path})
        result = tokenizer_component.run({})

        # 다른 create_* 메서드들과의 일관성을 위해 실제 토크나이저 인스턴스만 반환
        return result["tokenizer"]

    @classmethod
    def _resolve_tokenizer_path(cls, config: Config) -> Path | None:
        """
        Config 기반 토크나이저 경로 해결.

        다른 create_model_loader()와 동일한 우선순위 패턴:
        1. config.paths.models.base_local 우선 확인
        2. 표준 경로들 순차 확인
        3. None 반환으로 컴포넌트의 자체 탐색에 위임
        """
        # config.paths.models.base_local 우선 확인 (다른 모델 로더와 동일 패턴)
        base_path = Path(config.paths.models.base_local)
        if (base_path / "tokenizer.model").exists():
            return base_path / "tokenizer.model"

        # 표준 경로들 순차 확인 (기존 default_paths 로직 재사용)
        default_paths = [
            Path("models/tokenizer.model"),
            Path("models/7b_1t_4/tokenizer.model"),
            Path(".cache/tokenizer.model"),
        ]

        for path in default_paths:
            if path.exists():
                return path

        return None  # SentencePieceTokenizer가 S3 또는 환경변수 경로를 자체 탐색

    # 📝 설계 변경 기록: build_pipeline_components 메서드 제거
    # 이유: 파이프라인에서 create_* 메서드들을 직접 호출하여 더 명확한 제어 제공
    # 각 컴포넌트의 생성 시점과 의존성을 파이프라인에서 명시적으로 관리
