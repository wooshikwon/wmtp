"""Utility modules for S3, MLflow, HuggingFace, distributed training, evaluation, and I/O."""

# Import main utilities to prevent direct external imports
from .dist import (
    DistributedManager,
    get_dist_manager,
    set_seed,
)
from .eval import EvaluationProtocol
# from .hf import (
#     get_dtype,
#     get_model_size,
#     resize_token_embeddings,
#     safe_from_pretrained,
# )
# from .io import (
#     ensure_path_exists,
#     get_file_size,
#     load_dataset_from_disk,
#     load_dataset_from_json,
#     read_jsonl,
#     save_dataset_to_disk,
#     save_dataset_to_json,
#     write_jsonl,
# )
from .mlflow import (
    MLflowManager,
    create_mlflow_manager,
)
from .s3 import S3Manager, create_s3_manager

__all__ = [
    # S3
    "S3Manager",
    "create_s3_manager",
    # # HuggingFace
    # "resize_token_embeddings",
    # "get_model_size",
    # "get_dtype",
    # "safe_from_pretrained",
    # MLflow
    "MLflowManager",
    "create_mlflow_manager",
    # Distributed
    "DistributedManager",
    "set_seed",
    "get_dist_manager",
    # Evaluation
    "EvaluationProtocol",
    # # I/O
    # "load_dataset_from_json",
    # "load_dataset_from_disk",
    # "save_dataset_to_json",
    # "save_dataset_to_disk",
    # "read_jsonl",
    # "write_jsonl",
    # "ensure_path_exists",
    # "get_file_size",
]
