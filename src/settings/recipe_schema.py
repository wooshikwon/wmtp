"""
Recipe schema for model training and evaluation settings.

This module defines the Pydantic models for recipe.yaml which contains
model, training, optimization, and evaluation configurations.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Run(BaseModel):
    """Run metadata configuration."""

    name: str = Field(..., description="Run name for tracking")
    tags: list[str] = Field(default_factory=list, description="Tags for MLflow")


class MTPConfig(BaseModel):
    """Multi-Token Prediction configuration."""

    n_heads: int = Field(default=4, ge=1, le=8, description="Number of MTP heads")
    horizon: int = Field(default=4, ge=1, le=8, description="Prediction horizon")

    @model_validator(mode="after")
    def validate_heads_horizon(self):
        """Ensure n_heads and horizon are consistent."""
        if self.n_heads != self.horizon:
            raise ValueError(
                f"n_heads ({self.n_heads}) should typically equal horizon ({self.horizon})"
            )
        return self


class Model(BaseModel):
    """Model configuration."""

    base_id: str = Field(..., description="Base MTP model identifier")
    rm_id: str = Field(..., description="Reward model identifier")
    ref_id: str = Field(..., description="Reference model identifier")
    tokenizer_pad_side: Literal["left", "right"] = Field(
        default="right", description="Tokenizer padding side"
    )
    mtp: MTPConfig = Field(default_factory=MTPConfig)

    @field_validator("base_id")
    @classmethod
    def validate_base_is_mtp(cls, v: str) -> str:
        """Validate base model is an MTP model."""
        # Check if model ID suggests it's an MTP model
        mtp_keywords = ["multi-token", "mtp", "multi_token", "7b_1t_4"]
        if not any(keyword in v.lower() for keyword in mtp_keywords):
            # Warning rather than error since we can't be 100% sure from ID alone
            print(
                f"Warning: base_id '{v}' doesn't appear to be an MTP model. "
                f"Expected keywords: {mtp_keywords}"
            )
        return v


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""

    enabled: bool = Field(default=False, description="Enable LoRA fine-tuning")
    r: int = Field(default=16, ge=1, description="LoRA rank")
    alpha: int = Field(default=32, ge=1, description="LoRA alpha")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: list[str] = Field(
        default=[
            "q_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target modules for LoRA",
    )


class Train(BaseModel):
    """Training configuration."""

    algo: Literal["critic-wmtp", "rho1-wmtp"] = Field(
        ..., description="Training algorithm"
    )
    full_finetune: bool = Field(default=True, description="Full fine-tuning mode")
    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    @model_validator(mode="after")
    def validate_finetune_mode(self):
        """Ensure either full fine-tune or LoRA is enabled."""
        if not self.full_finetune and not self.lora.enabled:
            raise ValueError("Either full_finetune or lora.enabled must be True")
        if self.full_finetune and self.lora.enabled:
            raise ValueError("Cannot enable both full_finetune and LoRA simultaneously")
        return self


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


class Batching(BaseModel):
    """Batching configuration."""

    global_batch_tokens: int = Field(
        default=4_000_000, gt=0, description="Global batch size in tokens"
    )
    micro_batch_size: int = Field(
        default=1, ge=1, description="Micro batch size per device"
    )
    grad_accum_steps: int = Field(
        default=64, ge=1, description="Gradient accumulation steps"
    )


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
    batching: Batching = Field(..., description="Batching configuration")
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
            # Rho1 config is ignored for critic algorithm
        elif self.train.algo == "rho1-wmtp":
            if self.rho1 is None:
                raise ValueError("rho1 configuration is required when algo='rho1-wmtp'")
            # Critic config is ignored for rho1 algorithm
        else:
            raise ValueError(f"Unknown algorithm: {self.train.algo}")
        return self
