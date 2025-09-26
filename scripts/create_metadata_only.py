#!/usr/bin/env python3
"""
메타데이터만 생성하여 S3에 업로드
(대용량 파일 복사 없이 표준화 구조만 생성)
"""

import json
import os
from datetime import datetime
from pathlib import Path

import boto3
import yaml
from dotenv import load_dotenv
from rich.console import Console

console = Console()


def create_and_upload_metadata_only():
    """메타데이터만 생성하여 빠르게 테스트"""

    load_dotenv()

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="eu-north-1"
    )

    # Sheared-LLaMA 메타데이터
    metadata = {
        "wmtp_type": "base_model",
        "training_algorithm": "sheared_training",
        "horizon": 4,
        "n_heads": 4,
        "base_architecture": "llama",
        "model_size": "2.7b",
        "storage_version": "2.0",
        "created_by": "metadata_only_test",
        "standardization_date": datetime.now().isoformat(),
        "original_format": "sharded_pytorch_bins",
        "source_path": "s3://wmtp/models/Sheared-LLaMA-2.7B",
        "loading_strategy": {
            "loader_type": "huggingface",
            "model_class_name": None,
            "custom_module_file": None,
            "transformers_class": "AutoModelForCausalLM",
            "state_dict_mapping": {
                "remove_prefix": None,
                "add_prefix": None,
                "key_transforms": {}
            },
            "required_files": [
                "config.json",
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
                "pytorch_model.bin.index.json",
                "tokenizer.model",
                "metadata.json"
            ]
        },
        "algorithm_compatibility": ["baseline-mtp", "rho1-wmtp"]
    }

    # 로컬에 임시 저장
    temp_dir = Path("/tmp/metadata_test")
    temp_dir.mkdir(exist_ok=True)

    # metadata.json
    metadata_file = temp_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # MLflow 파일들
    mlflow_dir = temp_dir / "mlflow_model"
    mlflow_dir.mkdir(exist_ok=True)

    mlmodel = {
        "artifact_path": "model",
        "flavors": {
            "python_function": {
                "env": {"conda": "conda.yaml", "virtualenv": "python_env.yaml"},
                "loader_module": "mlflow.transformers",
                "model_path": "model",
                "python_version": "3.11.7"
            },
            "transformers": {
                "code": None,
                "framework": "pt",
                "model_path": "model",
                "task": "text-generation"
            }
        },
        "mlflow_version": "2.15.1",
        "model_size_bytes": 10794508288,  # 10GB
        "model_uuid": "sheared-llama-2.7b",
        "signature": {
            "inputs": '[{"type": "string"}]',
            "outputs": '[{"type": "string"}]'
        },
        "utc_time_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    }

    with open(mlflow_dir / "MLmodel", "w") as f:
        yaml.dump(mlmodel, f, default_flow_style=False, sort_keys=False)

    with open(mlflow_dir / "requirements.txt", "w") as f:
        f.write("transformers\ntorch")

    conda_env = {
        "name": "sheared_llama_env",
        "channels": ["defaults"],
        "dependencies": ["python=3.11.7", {"pip": ["transformers", "torch"]}]
    }

    with open(mlflow_dir / "conda.yaml", "w") as f:
        yaml.dump(conda_env, f, default_flow_style=False, sort_keys=False)

    # S3에 메타데이터만 업로드
    console.print("[yellow]📤 메타데이터 업로드 중...[/yellow]")

    target_prefix = "models_v2/sheared-llama-2.7b-test"

    # metadata.json
    s3_client.upload_file(
        str(metadata_file),
        "wmtp",
        f"{target_prefix}/metadata.json"
    )
    console.print("✅ metadata.json 업로드 완료")

    # MLflow 파일들
    for file_name in ["MLmodel", "requirements.txt", "conda.yaml"]:
        local_file = mlflow_dir / file_name
        s3_client.upload_file(
            str(local_file),
            "wmtp",
            f"{target_prefix}/mlflow_model/{file_name}"
        )
        console.print(f"✅ {file_name} 업로드 완료")

    # 기존 파일들에 대한 symbolic link 정보 생성
    link_info = {
        "model_files_location": "s3://wmtp/models/Sheared-LLaMA-2.7B/",
        "note": "실제 모델 파일은 원본 위치에서 직접 참조"
    }

    link_file = temp_dir / "link_info.json"
    with open(link_file, "w") as f:
        json.dump(link_info, f, indent=2)

    s3_client.upload_file(
        str(link_file),
        "wmtp",
        f"{target_prefix}/link_info.json"
    )

    console.print(f"\n[green]✅ 메타데이터 생성 완료![/green]")
    console.print(f"위치: s3://wmtp/{target_prefix}/")
    console.print("실제 모델 파일은 원본 위치에서 참조")

    return f"s3://wmtp/{target_prefix}/"


if __name__ == "__main__":
    path = create_and_upload_metadata_only()
    console.print(f"\n[bold green]테스트용 메타데이터 경로:[/bold green]")
    console.print(path)