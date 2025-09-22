"""
Configuration schema for environment settings.

This module defines the Pydantic models for config.yaml which contains
environment-specific settings like storage, paths, MLflow, and hardware configuration.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class S3Config(BaseModel):
    """S3 storage configuration."""

    bucket: str = Field(..., description="S3 bucket name")
    region: str = Field(default="ap-northeast-2", description="AWS region")
    prefix: str = Field(default="", description="S3 prefix for all objects")

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, v: str) -> str:
        """Validate S3 bucket name."""
        if not v:
            raise ValueError("S3 bucket name cannot be empty")
        if len(v) > 63:
            raise ValueError("S3 bucket name must be 63 characters or less")
        return v.lower()


class Storage(BaseModel):
    """Storage configuration."""

    mode: Literal["local", "s3"] = Field(..., description="Storage mode")
    s3: S3Config | None = None

    @model_validator(mode="after")
    def validate_s3_required(self):
        """Ensure S3 config is provided when mode is s3."""
        if self.mode == "s3" and not self.s3:
            raise ValueError("S3 configuration is required when storage mode is 's3'")
        return self


class ModelPaths(BaseModel):
    """Model paths configuration."""

    base_local: Path = Field(
        default=Path("models/7b_1t_4"),
        description="Local path for base MTP model",
    )
    rm_local: Path = Field(
        default=Path("models/Llama_3_8B_RM"),
        description="Local path for reward model",
    )
    ref_local: Path = Field(
        default=Path("models/sheared_llama_1.3B"),
        description="Local path for reference model",
    )


class DatasetPaths(BaseModel):
    """Dataset paths configuration."""

    mbpp_local: Path = Field(
        default=Path("dataset/mbpp"),
        description="Local path for MBPP dataset",
    )
    contest_local: Path = Field(
        default=Path("dataset/contest"),
        description="Local path for CodeContests dataset",
    )


class Paths(BaseModel):
    """All path configurations."""

    models: ModelPaths = Field(default_factory=ModelPaths)
    datasets: DatasetPaths = Field(default_factory=DatasetPaths)
    cache: Path = Field(default=Path(".cache"), description="Cache directory")


class MLflow(BaseModel):
    """MLflow configuration."""

    experiment: str = Field(
        default="mtp/wmtp",
        description="MLflow experiment name",
    )
    tracking_uri: str = Field(
        ...,
        description="MLflow tracking server URI (must be S3 for remote)",
    )
    registry_uri: str = Field(
        ...,
        description="MLflow model registry URI (must be S3 for remote)",
    )

    @field_validator("tracking_uri", "registry_uri")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        """Validate MLflow URIs are S3."""
        if not v.startswith(("s3://", "file://", "http://", "https://")):
            raise ValueError(
                f"Invalid MLflow URI: {v}. Must start with s3://, file://, http://, or https://"
            )
        return v


class LauncherResources(BaseModel):
    """Launcher resource requirements."""

    gpus: int = Field(default=4, ge=0, description="Number of GPUs")
    gpu_type: str = Field(default="A100", description="GPU type")
    cpus: int = Field(default=32, ge=1, description="Number of CPUs")
    memory_gb: int = Field(default=256, ge=1, description="Memory in GB")
    disk_gb: int = Field(default=500, ge=10, description="Disk space in GB")


class Launcher(BaseModel):
    """Launcher configuration."""

    target: Literal["local", "vessl"] = Field(..., description="Launcher target")
    resources: LauncherResources = Field(default_factory=LauncherResources)


class FSDPConfig(BaseModel):
    """Fully Sharded Data Parallel configuration."""

    enabled: bool = Field(default=True, description="Enable FSDP")
    auto_wrap: bool = Field(default=True, description="Enable auto wrapping")
    activation_ckpt: bool = Field(
        default=True, description="Enable activation checkpointing"
    )
    sharding: Literal["full", "shard_grad_op"] = Field(
        default="full", description="FSDP sharding strategy"
    )


class Devices(BaseModel):
    """Device and distributed training configuration."""

    mixed_precision: Literal["bf16", "fp16", "fp32"] = Field(
        default="bf16", description="Mixed precision mode"
    )
    fsdp: FSDPConfig = Field(default_factory=FSDPConfig)

    @field_validator("mixed_precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        """Validate precision is supported."""
        if v == "bf16":
            # Note: bf16 requires Ampere or newer GPUs (A100, A6000, etc.)
            pass  # Will validate at runtime based on actual hardware
        return v


class Config(BaseModel):
    """Root configuration schema for environment settings."""

    project: str = Field(default="mtp_ft", description="Project name")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    storage: Storage = Field(..., description="Storage configuration")
    paths: Paths = Field(default_factory=Paths, description="Path configurations")
    mlflow: MLflow = Field(..., description="MLflow configuration")
    launcher: Launcher = Field(..., description="Launcher configuration")
    devices: Devices = Field(
        default_factory=Devices, description="Device configuration"
    )

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        """Validate seed is positive."""
        if v < 0:
            raise ValueError("Seed must be non-negative")
        return v
