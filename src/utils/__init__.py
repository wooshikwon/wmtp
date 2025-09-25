"""Utility modules for S3, MLflow, HuggingFace, distributed training, evaluation, and I/O."""

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
    get_dtype,
    get_model_size,
    resize_token_embeddings,
    safe_from_pretrained,
)
from .io import (
    ensure_path_exists,
    get_file_size,
    load_dataset_from_disk,
    load_dataset_from_json,
    read_jsonl,
    save_dataset_to_disk,
    save_dataset_to_json,
    write_jsonl,
)
from .mlflow import (
    MLflowManager,
    auto_log_config,
    create_mlflow_manager,
    log_system_info,
)
from .s3 import S3Manager, S3Utils, compute_file_hash, create_s3_manager, ensure_s3_uri

__all__ = [
    # S3
    "S3Manager",
    "S3Utils",
    "create_s3_manager",
    "compute_file_hash",
    "ensure_s3_uri",
    # HuggingFace
    "resize_token_embeddings",
    "get_model_size",
    "get_dtype",
    "safe_from_pretrained",
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
    # I/O
    "load_dataset_from_json",
    "load_dataset_from_disk",
    "save_dataset_to_json",
    "save_dataset_to_disk",
    "read_jsonl",
    "write_jsonl",
    "ensure_path_exists",
    "get_file_size",
]
