#!/usr/bin/env python3
"""
WMTP S3 Single Model Standardization
S3 모델을 하나씩 안전하게 표준화하는 스크립트
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import boto3
from rich.console import Console

console = Console()


def standardize_sheared_llama():
    """Sheared LLaMA 모델 표준화 (분할 모델 → 표준화)"""
    console.print("[bold blue]Standardizing Sheared LLaMA 2.7B[/bold blue]")

    # AWS 클라이언트
    s3 = boto3.client("s3")
    bucket = "wmtp"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. 원본 모델 다운로드
        console.print("[yellow]Downloading original model...[/yellow]")
        source_prefix = "models/Sheared-LLaMA-2.7B/"
        local_source = temp_path / "original"
        local_source.mkdir()

        # 핵심 파일들만 다운로드 (캐시 파일 제외)
        core_files = [
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
            "pytorch_model.bin.index.json",
        ]

        for file_name in core_files:
            try:
                s3.download_file(
                    bucket, f"{source_prefix}{file_name}", str(local_source / file_name)
                )
                console.print(f"  ✅ Downloaded {file_name}")
            except Exception as e:
                console.print(f"  ⚠️  Failed to download {file_name}: {e}")

        # 2. 표준 형식으로 변환
        console.print("[yellow]Converting to standard format...[/yellow]")
        target_local = temp_path / "standardized"
        target_local.mkdir()

        # 설정 파일들 복사
        config_files = [
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]

        for file_name in config_files:
            src_file = local_source / file_name
            if src_file.exists():
                shutil.copy2(src_file, target_local / file_name)

        # 분할된 모델을 단일 safetensors로 변환
        console.print("[yellow]Merging sharded model to safetensors...[/yellow]")
        try:
            # 인덱스 파일 읽기
            with open(local_source / "pytorch_model.bin.index.json") as f:
                index_data = json.load(f)

            # 모든 shard 로드하여 병합
            import torch

            merged_state_dict = {}
            weight_map = index_data["weight_map"]

            for shard_file in set(weight_map.values()):
                shard_path = local_source / shard_file
                console.print(f"  Loading {shard_file}...")
                shard_state_dict = torch.load(
                    shard_path, map_location="cpu", weights_only=True
                )
                merged_state_dict.update(shard_state_dict)

            # 단일 safetensors 파일로 저장
            from safetensors.torch import save_file

            save_file(merged_state_dict, target_local / "model.safetensors")
            console.print("  ✅ Converted to single safetensors file")

        except Exception as e:
            console.print(f"  ❌ Model conversion failed: {e}")
            return

        # 3. WMTP 메타데이터 생성
        metadata = {
            "wmtp_type": "base_model",
            "training_algorithm": "sheared_training",
            "base_architecture": "llama",
            "model_size": "2.7b",
            "storage_version": "2.0",
            "created_by": "wmtp_s3_standardizer",
            "standardization_date": datetime.now().isoformat(),
            "original_format": "sharded_pytorch_bins",
            "conversion_metadata": {
                "converted_from": "s3://wmtp/models/Sheared-LLaMA-2.7B/",
                "conversion_date": datetime.now().isoformat(),
                "converter_version": "wmtp-v2.0",
            },
        }

        with open(target_local / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. MLflow 파일 생성
        mlflow_dir = target_local / "mlflow_model"
        mlflow_dir.mkdir()

        with open(mlflow_dir / "requirements.txt", "w") as f:
            f.write("torch>=2.0.0\ntransformers>=4.40.0\nsafetensors>=0.4.0\n")

        import yaml

        conda_env = {
            "channels": ["conda-forge", "pytorch"],
            "dependencies": ["python=3.12", "pytorch>=2.0.0"],
            "name": "wmtp_sheared_llama_env",
        }
        with open(mlflow_dir / "conda.yaml", "w") as f:
            yaml.dump(conda_env, f)

        # 5. S3에 업로드
        console.print("[yellow]Uploading standardized model to S3...[/yellow]")
        target_prefix = "models_v2/sheared-llama-2.7b/"

        for file_path in target_local.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(target_local)
                s3_key = f"{target_prefix}{relative_path}".replace("\\", "/")

                s3.upload_file(str(file_path), bucket, s3_key)
                console.print(f"  ✅ Uploaded {relative_path}")

        console.print(
            "[bold green]🎉 Sheared LLaMA standardization completed![/bold green]"
        )


if __name__ == "__main__":
    # AWS 환경변수 설정 - .env 파일에서 로드
    from dotenv import load_dotenv

    load_dotenv()

    standardize_sheared_llama()
