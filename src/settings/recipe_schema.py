"""
WMTP 실험 설정 스키마: recipe.yaml 구조 정의 및 알고리즘 선택

WMTP 연구 맥락:
recipe.yaml은 WMTP 실험의 핵심 설정으로, 세 가지 알고리즘(baseline/critic/rho1) 중
하나를 선택하고 하이퍼파라미터를 정의합니다. 이 파일의 설정에 따라
토큰 중요도 계산 방식과 손실 함수 가중치가 결정됩니다.

핵심 기능:
- 알고리즘 선택: mtp-baseline, critic-wmtp, rho1-wmtp 중 택일
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


class MTPConfig(BaseModel):
    """다중토큰예측(MTP) 설정: WMTP의 핵심 구조 정의

    WMTP 연구 맥락:
    MTP는 한 번에 N개의 미래 토큰을 예측하는 Meta AI의 혁신적 방법입니다.
    WMTP는 이 N개 토큰에 각각 다른 중요도 가중치를 부여하여
    "Not All Tokens Are What You Need" 원칙을 구현합니다.

    구체적 동작:
    1. Shared trunk에서 은닉 상태 h_t 생성
    2. h_t를 n_heads개로 복사
    3. 각 헤드가 t+1, t+2, ..., t+n 토큰 예측
    4. WMTP: 각 헤드의 CE 손실에 w_{t+k} 가중치 적용

    Attributes:
        n_heads: MTP 헤드 개수 (1~8)
            - 4: Meta 논문 기준 최적값 (메모리/성능 균형)
            - 1: 표준 next-token 예측 (비교용)
            - 8: 최대 병렬 예측 (메모리 많이 사용)
        horizon: 예측 범위 (1~8)
            - 일반적으로 n_heads와 동일
            - 향후 확장: skip prediction 지원 예정

    WMTP 알고리즘별 활용:
        - Baseline: 모든 헤드에 균등 가중치 (w=1.0)
        - Critic: 각 헤드별 value delta로 가중
        - Rho1: 각 헤드별 CE excess로 가중

    예시:
        mtp:
          n_heads: 4  # 4개 토큰 동시 예측
          horizon: 4  # t+1 ~ t+4 예측

    주의사항:
        - n_heads > 4일 때 메모리 사용량 급증
        - GPU 메모리 부족 시 n_heads 감소 필요
    """

    n_heads: int = Field(default=4, ge=1, le=8, description="MTP 헤드 개수")
    horizon: int = Field(default=4, ge=1, le=8, description="예측 범위")

    @model_validator(mode="after")
    def validate_heads_horizon(self):
        """헤드 수와 예측 범위 일치성 검증.

        현재는 n_heads == horizon 강제하지만,
        향후 skip prediction 등 확장 가능성 염두.
        """
        if self.n_heads != self.horizon:
            raise ValueError(
                f"n_heads ({self.n_heads})와 horizon ({self.horizon})은 "
                f"일반적으로 같아야 합니다. 향후 skip prediction 지원 예정."
            )
        return self


class Model(BaseModel):
    """모델 설정: WMTP에서 사용하는 모든 모델 정의

    WMTP 연구 맥락:
    WMTP는 최대 3개의 모델을 조합하여 동작합니다:
    1. Base: MTP 사전학습된 주 모델 (학습 대상)
    2. RM: 보상 모델 (Critic-WMTP에서 value 학습)
    3. Ref: 참조 모델 (Rho1-WMTP에서 CE 비교)

    구체적 동작 흐름:
    1. Base 모델이 MTP 헤드로 다중 토큰 예측
    2. 알고리즘별 중요도 계산:
       - Critic: RM의 보상 → Value head → Delta
       - Rho1: Ref 모델 CE와 Base CE 차이
    3. 중요도를 가중치로 변환하여 손실에 적용

    Attributes:
        base_id: 기본 MTP 모델 식별자 (필수)
            - 로컬 경로: "models/7b_1t_4"
            - HF 모델: "facebook/multi-token-prediction"
            - MTP 사전학습 필수!

        rm_id: 보상 모델 식별자 (Critic-WMTP 필수)
            - "models/Llama_3_8B_RM"
            - 인간 선호도 학습된 모델
            - None이면 Critic 알고리즘 사용 불가

        ref_id: 참조 모델 식별자 (필수)
            - Rho1: CE 비교 기준 (CodeLlama-7B 권장)
            - Critic: 토크나이저 호환성 체크용
            - "models/codellama_7b" 또는 "models/sheared_llama_1.3b"

        tokenizer_pad_side: 토크나이저 패딩 방향
            - "right": 일반적 설정 (권장)
            - "left": 생성 태스크에서 가끔 사용

        mtp: MTP 구조 설정
            - n_heads, horizon 등

    WMTP 알고리즘별 필수 모델:
        - Baseline: base_id, ref_id (토크나이저용)
        - Critic: base_id, rm_id (필수!), ref_id
        - Rho1: base_id, ref_id (CE 계산 필수!)

    예시:
        # Rho1-WMTP 설정
        model:
          base_id: facebook/multi-token-prediction
          ref_id: codellama/CodeLlama-7b-Python-hf
          rm_id: null  # Rho1은 RM 불필요
          tokenizer_pad_side: right
          mtp:
            n_heads: 4

        # Critic-WMTP 설정
        model:
          base_id: models/7b_1t_4
          rm_id: models/Llama_3_8B_RM  # 필수!
          ref_id: models/codellama_7b
          tokenizer_pad_side: right

    디버깅 팁:
        - "not MTP model" 경고: base_id가 MTP 모델인지 확인
        - rm_id None 오류: Critic 선택 시 rm_id 필수
        - 토크나이저 불일치: base와 ref 모델의 vocab 호환성 확인
    """

    base_id: str = Field(..., description="기본 MTP 모델 식별자")
    rm_id: str | None = Field(None, description="보상 모델 식별자 (critic-wmtp 필수)")
    ref_id: str = Field(..., description="참조 모델 식별자")
    tokenizer_type: Literal["hf", "raw"] = Field(
        default="hf",
        description="Tokenizer interface type: hf=HuggingFace compatible, raw=SentencePiece direct"
    )
    tokenizer_pad_side: Literal["left", "right"] = Field(
        default="right", description="토크나이저 패딩 방향"
    )
    mtp: MTPConfig = Field(default_factory=MTPConfig)

    @field_validator("base_id")
    @classmethod
    def validate_base_is_mtp(cls, v: str) -> str:
        """Base 모델이 MTP 모델인지 검증.

        MTP 사전학습이 WMTP의 전제조건이므로
        모델 ID에서 MTP 키워드를 확인합니다.
        """
        # MTP 모델임을 나타내는 키워드들
        mtp_keywords = ["multi-token", "mtp", "multi_token", "7b_1t_4"]
        if not any(keyword in v.lower() for keyword in mtp_keywords):
            # 경고만 출력 (ID만으로는 100% 확신 불가)
            print(
                f"경고: base_id '{v}'가 MTP 모델이 아닐 수 있습니다. "
                f"예상 키워드: {mtp_keywords}"
            )
        return v


class Checkpointing(BaseModel):
    """체크포인트 저장 설정."""

    save_interval: int = Field(default=100, ge=1, description="N스텝마다 체크포인트 저장")
    keep_last: int = Field(default=3, ge=1, description="최근 N개 체크포인트 유지")
    save_final: bool = Field(default=True, description="최종 모델 저장 여부")


class Train(BaseModel):
    """Training configuration."""

    algo: Literal["mtp-baseline", "critic-wmtp", "rho1-wmtp"] = Field(
        ..., description="Training algorithm"
    )
    full_finetune: bool = Field(default=True, description="Full fine-tuning mode")
    max_steps: int | None = Field(
        default=None, ge=1, description="Maximum training steps (None for unlimited)"
    )
    checkpointing: Checkpointing | None = Field(
        default=None, description="체크포인트 저장 설정 (선택사항)"
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

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        """Validate data sources."""
        valid_sources = ["mbpp", "contest", "custom"]
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
    lambda_weight: float = Field(
        default=0.3,
        gt=0,
        le=1,
        validation_alias="lambda",
        serialization_alias="lambda",
        description="Loss weight strength",
    )
    temperature: float = Field(default=0.7, gt=0, description="Softmax temperature")
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


class Rho1(BaseModel):
    """Rho-1 specific configuration."""

    score: Literal["abs_excess_ce", "rel_excess_ce"] = Field(
        default="abs_excess_ce", description="Scoring method"
    )
    percentile_top_p: float = Field(
        default=0.2, gt=0, le=1, description="Top percentile to emphasize"
    )
    refresh_per_epoch: bool = Field(
        default=False, description="Refresh scores per epoch"
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
    """Root recipe schema for training configuration."""

    run: Run = Field(..., description="Run metadata")
    model: Model = Field(..., description="Model configuration")
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
    def validate_algo_config(self):
        """Ensure algorithm-specific configs are present and valid."""
        if self.train.algo == "critic-wmtp":
            if self.critic is None:
                raise ValueError(
                    "critic configuration is required when algo='critic-wmtp'"
                )
            if self.model.rm_id is None:
                raise ValueError(
                    "rm_id in model configuration is required when algo='critic-wmtp'"
                )
            # Rho1 config is ignored for critic algorithm
        elif self.train.algo == "rho1-wmtp":
            if self.rho1 is None:
                raise ValueError("rho1 configuration is required when algo='rho1-wmtp'")
            # Critic config is ignored for rho1 algorithm
        elif self.train.algo == "mtp-baseline":
            # No critic or rho1 config needed for baseline
            pass
        else:
            raise ValueError(f"Unknown algorithm: {self.train.algo}")
        return self
