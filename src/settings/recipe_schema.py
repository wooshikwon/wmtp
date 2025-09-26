"""
WMTP 실험 설정 스키마: recipe.yaml 구조 정의 및 알고리즘 선택

WMTP 연구 맥락:
recipe.yaml은 WMTP 실험의 핵심 설정으로, 세 가지 알고리즘(baseline/critic/rho1) 중
하나를 선택하고 하이퍼파라미터를 정의합니다. 이 파일의 설정에 따라
토큰 중요도 계산 방식과 손실 함수 가중치가 결정됩니다.

핵심 기능:
- 알고리즘 선택: baseline-mtp, critic-wmtp, rho1-wmtp 중 택일
- 모델 설정: MTP 헤드 수, 예측 horizon, 토크나이저 설정
- 학습 설정: 배치 크기, 학습률, epoch 수, 체크포인트 전략
- 손실 함수 설정: lambda, temperature 등 가중치 제어 파라미터

WMTP 알고리즘과의 연결:
- Baseline MTP: 표준 MTP 학습, 토큰 가중치 없음 (비교 기준)
- Critic-WMTP: Value head 학습 후 delta 기반 가중치 적용
- Rho1-WMTP: 참조 모델과의 CE 차이로 가중치 계산 (권장)

사용 예시:
    >>> from src.settings.recipe_schema import Recipe
    >>> from src.settings.loader import load_recipe
    >>>
    >>> # 레시피 로드 및 검증
    >>> recipe = load_recipe("configs/recipe_rho1.yaml")
    >>> print(f"알고리즘: {recipe.train.algo}")
    알고리즘: rho1-wmtp
    >>> print(f"MTP 헤드 수: {recipe.model.mtp.n_heads}")
    MTP 헤드 수: 4
    >>>
    >>> # 알고리즘별 필수 설정 확인
    >>> if recipe.train.algo == "critic-wmtp":
    >>>     assert recipe.model.rm_id is not None, "Critic-WMTP는 보상 모델 필요"
    >>> elif recipe.train.algo == "rho1-wmtp":
    >>>     assert recipe.loss.rho1 is not None, "Rho1-WMTP는 rho1 설정 필요"

성능 최적화:
- MTP n_heads=4: 메모리와 성능의 균형점 (Meta 연구 기준)
- 배치 크기: A100 기준 micro_batch=1, gradient_accumulation=16
- 학습률: 5e-6 (사전학습된 MTP 모델 미세조정 최적값)
- Lambda=0.5: Rho1에서 가중치 강도 (0.1~1.0 범위에서 조정)

디버깅 팁:
- ValidationError: 알고리즘과 필수 설정 불일치 확인
- CUDA OOM: micro_batch_size 감소 또는 gradient_accumulation 증가
- 학습 불안정: learning_rate 감소, warmup_steps 증가
- 가중치 효과 없음: lambda 증가, temperature 감소
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Run(BaseModel):
    """실행 메타데이터 설정: MLflow 실험 추적 정보

    WMTP 연구 맥락:
    각 실험은 MLflow에 기록되어 재현성을 보장합니다.
    알고리즘별로 다른 네이밍 규칙을 사용하여 실험을 구분합니다.

    Attributes:
        name: 실행 이름 (필수)
            - 형식: "{algo}_{dataset}_{date}"
            - 예: "rho1_mbpp_20241225", "critic_codecontests_exp3"
        tags: MLflow 태그 리스트
            - 실험 분류와 필터링에 사용
            - 예: ["production", "rho1", "7b", "mbpp"]

    예시:
        run:
          name: rho1_wmtp_final
          tags: ["rho1", "7b", "final", "paper"]
    """

    name: str = Field(..., description="실행 이름 (MLflow 추적용)")
    tags: list[str] = Field(default_factory=list, description="MLflow 태그")


# Phase 2 리팩토링: Model, MTPConfig 클래스 제거됨
# MTP 설정은 ComponentFactory.MTP_CONFIG 상수로 고정 (n_heads=4, horizon=4)


class Checkpointing(BaseModel):
    """체크포인트 저장 설정."""

    save_interval: int = Field(
        default=100, ge=1, description="N스텝마다 체크포인트 저장"
    )
    keep_last: int = Field(default=3, ge=1, description="최근 N개 체크포인트 유지")
    save_final: bool = Field(default=True, description="최종 모델 저장 여부")


class Stage1Config(BaseModel):
    """Stage 1 pretraining configuration for critic-wmtp."""

    enabled: bool = Field(default=True, description="Enable Stage 1 pretraining")
    max_steps: int = Field(default=2000, ge=1, description="Stage 1 max steps")
    lr: float = Field(default=1.0e-4, gt=0, description="Stage 1 learning rate")
    save_value_head: bool = Field(
        default=True, description="Save value head after Stage 1"
    )


class Train(BaseModel):
    """Training configuration."""

    algo: Literal["baseline-mtp", "critic-wmtp", "rho1-wmtp"] = Field(
        ..., description="Training algorithm"
    )
    full_finetune: bool = Field(default=True, description="Full fine-tuning mode")
    max_steps: int | None = Field(
        default=None, ge=1, description="Maximum training steps (None for unlimited)"
    )
    eval_interval: int = Field(default=500, ge=1, description="Evaluation interval")
    save_interval: int = Field(default=1000, ge=1, description="Save interval")
    checkpointing: Checkpointing | None = Field(
        default=None, description="체크포인트 저장 설정 (선택사항)"
    )
    stage1: Stage1Config | None = Field(
        default=None, description="Stage 1 configuration for critic-wmtp"
    )


class Optim(BaseModel):
    """Optimizer configuration."""

    optimizer: Literal["adamw", "lion", "sgd"] = Field(
        default="adamw", description="Optimizer type"
    )
    lr: float = Field(default=1.2e-5, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.1, ge=0, description="Weight decay")
    betas: list[float] = Field(default=[0.9, 0.95], description="Adam beta parameters")
    grad_clip: float = Field(default=1.0, gt=0, description="Gradient clipping")
    scheduler: Literal["cosine", "linear", "constant"] = Field(
        default="cosine", description="Learning rate scheduler"
    )
    warmup_ratio: float = Field(default=0.03, ge=0, le=0.5, description="Warmup ratio")

    @field_validator("betas")
    @classmethod
    def validate_betas(cls, v: list[float]) -> list[float]:
        """Validate beta values for Adam."""
        if len(v) != 2:
            raise ValueError("Betas must have exactly 2 values")
        if not (0 <= v[0] < 1 and 0 <= v[1] < 1):
            raise ValueError("Beta values must be in [0, 1)")
        return v


class DataConfig(BaseModel):
    """Data configuration for train or eval."""

    sources: list[str] = Field(..., description="Data sources")
    max_length: int = Field(default=2048, ge=128, description="Max sequence length")
    batch_size: int | None = Field(default=8, ge=1, description="Batch size")
    pack_sequences: bool = Field(
        default=True, description="Pack sequences for efficiency"
    )
    num_workers: int = Field(
        default=8, ge=0, description="Number of data loader workers"
    )

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        """Validate data sources."""
        valid_sources = ["mbpp", "contest", "humaneval", "custom"]
        for source in v:
            if source not in valid_sources:
                raise ValueError(
                    f"Invalid data source '{source}'. Must be one of {valid_sources}"
                )
        return v


class Data(BaseModel):
    """Data configuration."""

    train: DataConfig = Field(..., description="Training data configuration")
    eval: DataConfig = Field(..., description="Evaluation data configuration")


class Loss(BaseModel):
    """Loss configuration."""

    weight_norm: Literal["mean1.0_clip", "softmax", "none"] = Field(
        default="mean1.0_clip", description="Weight normalization strategy"
    )
    # Phase 2.2: lambda removed - main loss always 1.0 in critic-wmtp
    # Kept for baseline-mtp and rho1-wmtp compatibility
    lambda_weight: float = Field(
        default=0.3,
        gt=0,
        le=1,
        validation_alias="lambda",
        serialization_alias="lambda",
        description="Loss weight strength (not used in critic-wmtp)",
    )
    weight_temperature: float = Field(
        default=0.7,
        gt=0,
        description="Softmax temperature for weight calculation (not CE logits)",
        validation_alias="temperature",  # backward compatibility
    )
    epsilon: float = Field(default=0.05, gt=0, description="Minimum weight value")
    max_weight: float = Field(default=3.0, gt=1, description="Maximum weight value")

    model_config = ConfigDict(populate_by_name=True)


class Critic(BaseModel):
    """Critic-specific configuration."""

    target: Literal["rm_sequence"] = Field(
        default="rm_sequence", description="Critic target"
    )
    token_spread: Literal["uniform", "length", "attention", "gae"] = Field(
        default="gae", description="Token-level reward distribution method"
    )
    delta_mode: Literal["td", "diff"] = Field(
        default="td", description="Delta computation mode"
    )
    normalize: Literal["zscore", "minmax", "none"] = Field(
        default="zscore", description="Weight normalization method"
    )
    # Phase 2.1: TD error discount parameter (previously hardcoded)
    discount_lambda: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="TD error discount for delta computation (short-term discount)",
    )
    # GAE parameters for value function (long-term discount)
    gamma: float = Field(
        default=0.99, ge=0, le=1, description="Discount factor for rewards"
    )
    gae_lambda: float = Field(
        default=0.95, ge=0, le=1, description="GAE lambda parameter"
    )
    # Phase 2.2: Loss structure improvement - main loss fixed at 1.0
    auxiliary_loss_coef: float = Field(
        default=0.1,
        ge=0,
        description="Auxiliary value loss coefficient (main WMTP loss is always 1.0)",
        validation_alias="value_coef",  # backward compatibility for parsing
    )
    value_lr: float = Field(default=5e-5, gt=0, description="Value head learning rate")
    use_pseudo_rewards: bool = Field(
        default=True, description="Use pseudo rewards when RM is unavailable"
    )


class Rho1(BaseModel):
    """Rho-1 specific configuration."""

    # Selection mode: token_skip (original Rho-1) or weighted (current WMTP)
    selection_mode: Literal["token_skip", "weighted"] = Field(
        default="weighted", description="Token selection strategy"
    )

    # Token skip mode parameters
    skip_threshold_percentile: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Bottom percentile to skip in token_skip mode",
    )

    # CE difference threshold for noise filtering (Phase 1.2)
    min_ce_diff: float = Field(
        default=0.01,
        ge=0,
        description="Minimum CE difference threshold for noise filtering",
    )


class EvalSampling(BaseModel):
    """Evaluation sampling configuration."""

    temperature: float = Field(
        default=0.2, gt=0, le=2, description="Sampling temperature"
    )
    top_p: float = Field(default=0.95, gt=0, le=1, description="Top-p sampling")
    n: int = Field(default=1, ge=1, description="Number of samples")


class Eval(BaseModel):
    """Evaluation configuration."""

    protocol: Literal["meta-mtp"] = Field(
        default="meta-mtp", description="Evaluation protocol"
    )
    sampling: EvalSampling = Field(default_factory=EvalSampling)
    metrics: list[str] = Field(
        default=["mbpp_exact", "contest_pass@1", "contest_pass@5"],
        description="Metrics to compute",
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validate metric names."""
        valid_metrics = [
            "mbpp_exact",
            "humaneval_exact",
            "contest_exact",
            "contest_pass@1",
            "contest_pass@5",
            "contest_pass@10",
        ]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Invalid metric '{metric}'. Must be one of {valid_metrics}"
                )
        return v


class Recipe(BaseModel):
    """Root recipe schema for training configuration.

    Phase 2 리팩토링: model 섹션 제거됨
    - 모델 경로는 config.paths.models에서 관리
    - MTP 설정은 고정값 사용 (n_heads=4, horizon=4)
    - 토크나이저는 환경 기반 자동 선택
    """

    run: Run = Field(..., description="Run metadata")
    # model: Model = Field(...) <- Phase 2에서 제거됨
    train: Train = Field(..., description="Training configuration")
    optim: Optim = Field(..., description="Optimizer configuration")
    data: Data = Field(..., description="Data configuration")
    loss: Loss = Field(..., description="Loss configuration")
    critic: Critic | None = Field(
        default=None, description="Critic configuration (required for critic-wmtp)"
    )
    rho1: Rho1 | None = Field(
        default=None, description="Rho-1 configuration (required for rho1-wmtp)"
    )
    eval: Eval = Field(..., description="Evaluation configuration")

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )

    @model_validator(mode="after")
    def validate_algo_requirements(self):
        """알고리즘별 필수 설정 검증.

        Phase 2 리팩토링: 모델 관련 검증 제거
        - rm_id, ref_id 검증은 config에서 처리
        - recipe는 순수 실험 파라미터만 검증
        """
        algo = self.train.algo

        if algo == "critic-wmtp":
            if self.critic is None:
                raise ValueError("critic-wmtp는 critic 설정 필수")
        elif algo == "rho1-wmtp":
            if self.rho1 is None:
                raise ValueError("rho1-wmtp는 rho1 설정 필수")
        elif algo == "baseline-mtp":
            # baseline은 추가 설정 불필요
            pass
        else:
            raise ValueError(f"지원되지 않는 알고리즘: {algo}")

        return self
