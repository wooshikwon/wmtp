"""Utility modules for S3, MLflow, HuggingFace, distributed training, and evaluation."""

# Import main utilities to prevent direct external imports
from .dist import (
    DistributedManager,
    FSDPConfig,
    compute_throughput,
    get_dist_manager,
    get_world_info,
    set_seed,
)
from .eval import EvaluationProtocol, aggregate_metrics
from .hf import (
    HFModelLoader,
    create_model_loader,
    get_dtype,
    get_model_size,
    resize_token_embeddings,
)
from .mlflow import (
    MLflowManager,
    auto_log_config,
    create_mlflow_manager,
    log_system_info,
)
from .s3 import S3Manager, compute_file_hash, create_s3_manager, ensure_s3_uri

__all__ = [
    # S3
    "S3Manager",
    "create_s3_manager",
    "compute_file_hash",
    "ensure_s3_uri",
    # HuggingFace
    "HFModelLoader",
    "create_model_loader",
    "resize_token_embeddings",
    "get_model_size",
    "get_dtype",
    # MLflow
    "MLflowManager",
    "create_mlflow_manager",
    "auto_log_config",
    "log_system_info",
    # Distributed
    "DistributedManager",
    "FSDPConfig",
    "set_seed",
    "get_world_info",
    "compute_throughput",
    "get_dist_manager",
    # Evaluation
    "EvaluationProtocol",
    "aggregate_metrics",
]
