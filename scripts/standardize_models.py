#!/usr/bin/env python3
"""
WMTP Model Storage Standardization Tool

이 스크립트는 다양한 형식의 모델들을 WMTP 표준 저장 형식으로 변환합니다.
HuggingFace + MLflow 하이브리드 구조를 기반으로 합니다.

표준 구조:
model_name/
├── model.safetensors           # HF 표준 (보안)
├── config.json                # HF 표준 + WMTP 확장
├── tokenizer.json             # HF 표준
├── tokenizer_config.json      # HF 표준
├── special_tokens_map.json    # HF 표준
├── wmtp_metadata.json         # WMTP 전용 메타데이터
└── mlflow_model/              # MLflow 호환성
    ├── MLmodel               # MLflow 메타데이터
    ├── requirements.txt      # 의존성
    └── conda.yaml           # 환경
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import torch
from rich.console import Console
from rich.progress import Progress
from transformers import (
    AutoModelForCausalLM,
)

console = Console()


class WMTPModelStandardizer:
    """WMTP 모델 표준화 도구"""

    def __init__(
        self, aws_access_key: str, aws_secret_key: str, region: str = "eu-north-1"
    ):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region,
        )
        self.bucket = "wmtp"

    def standardize_s3_model(
        self,
        source_path: str,
        target_path: str,
        wmtp_type: str,
        model_info: dict[str, Any],
    ):
        """S3 모델을 표준 형식으로 변환"""
        console.print(
            f"[blue]Standardizing S3 model: {source_path} → {target_path}[/blue]"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 1. 원본 모델 다운로드
            self._download_s3_model(source_path, temp_path / "original")

            # 2. 표준 형식으로 변환
            standardized_path = temp_path / "standardized"
            self._convert_to_standard_format(
                temp_path / "original", standardized_path, wmtp_type, model_info
            )

            # 3. S3에 업로드
            self._upload_standardized_model(standardized_path, target_path)

        console.print(f"[green]✅ S3 model standardized: {target_path}[/green]")

    def standardize_local_model(
        self,
        source_path: Path,
        target_path: Path,
        wmtp_type: str,
        model_info: dict[str, Any],
    ):
        """로컬 모델을 표준 형식으로 변환"""
        console.print(
            f"[blue]Standardizing local model: {source_path} → {target_path}[/blue]"
        )

        target_path.mkdir(parents=True, exist_ok=True)

        self._convert_to_standard_format(
            source_path, target_path, wmtp_type, model_info
        )

        console.print(f"[green]✅ Local model standardized: {target_path}[/green]")

    def _download_s3_model(self, s3_path: str, local_path: Path):
        """S3에서 모델 다운로드"""
        local_path.mkdir(parents=True, exist_ok=True)

        # S3 객체 리스트 가져오기
        prefix = s3_path.replace("s3://wmtp/", "")

        with Progress() as progress:
            task = progress.add_task("Downloading from S3...", total=None)

            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    local_file = local_path / key.replace(prefix, "").lstrip("/")
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    self.s3_client.download_file(self.bucket, key, str(local_file))

            progress.update(task, completed=True)

    def _convert_to_standard_format(
        self,
        source_path: Path,
        target_path: Path,
        wmtp_type: str,
        model_info: dict[str, Any],
    ):
        """모델을 표준 형식으로 변환"""
        target_path.mkdir(parents=True, exist_ok=True)

        # 1. 모델 가중치 변환 (safetensors)
        self._convert_model_weights(source_path, target_path, wmtp_type)

        # 2. 설정 파일 표준화
        self._standardize_config_files(source_path, target_path)

        # 3. WMTP 메타데이터 생성
        self._generate_wmtp_metadata(target_path, wmtp_type, model_info)

        # 4. MLflow 호환성 파일 생성
        self._generate_mlflow_files(target_path, wmtp_type, model_info)

    def _convert_model_weights(
        self, source_path: Path, target_path: Path, wmtp_type: str
    ):
        """모델 가중치를 safetensors로 변환"""
        console.print("[yellow]Converting model weights to safetensors...[/yellow]")

        if wmtp_type == "mtp_native":
            # Meta consolidated.pth 변환
            consolidated_file = source_path / "consolidated.pth"
            if consolidated_file.exists():
                self._convert_consolidated_pth(consolidated_file, target_path)
            else:
                raise FileNotFoundError(f"consolidated.pth not found in {source_path}")

        elif (source_path / "pytorch_model.bin").exists():
            # HuggingFace pytorch_model.bin 변환
            self._convert_pytorch_bin(source_path, target_path)

        elif any(source_path.glob("pytorch_model-*.bin")):
            # 분할된 HuggingFace 모델 변환
            self._convert_sharded_pytorch_bins(source_path, target_path)

        elif (source_path / "model.safetensors").exists():
            # 이미 safetensors 형식
            shutil.copy2(
                source_path / "model.safetensors", target_path / "model.safetensors"
            )

        else:
            raise ValueError(f"No supported model weights found in {source_path}")

    def _convert_consolidated_pth(self, pth_file: Path, target_path: Path):
        """Meta consolidated.pth를 safetensors로 변환"""
        console.print("[yellow]Converting consolidated.pth to safetensors...[/yellow]")

        # PyTorch 파일 로드
        checkpoint = torch.load(pth_file, map_location="cpu", weights_only=True)

        # safetensors로 저장
        from safetensors.torch import save_file

        save_file(checkpoint, target_path / "model.safetensors")

        console.print("[green]✅ Converted consolidated.pth to safetensors[/green]")

    def _convert_pytorch_bin(self, source_path: Path, target_path: Path):
        """단일 pytorch_model.bin을 safetensors로 변환"""
        console.print("[yellow]Converting pytorch_model.bin to safetensors...[/yellow]")

        try:
            # HuggingFace 모델로 로드 후 safetensors로 저장
            model = AutoModelForCausalLM.from_pretrained(
                source_path, torch_dtype=torch.float32, trust_remote_code=True
            )
            model.save_pretrained(target_path, safe_serialization=True)

            # pytorch_model.bin은 제거 (safetensors만 유지)
            bin_file = target_path / "pytorch_model.bin"
            if bin_file.exists():
                bin_file.unlink()

        except Exception as e:
            console.print(f"[red]Failed to convert via HuggingFace: {e}[/red]")
            console.print("[yellow]Attempting direct conversion...[/yellow]")

            # 직접 변환
            state_dict = torch.load(
                source_path / "pytorch_model.bin", map_location="cpu", weights_only=True
            )
            from safetensors.torch import save_file

            save_file(state_dict, target_path / "model.safetensors")

        console.print("[green]✅ Converted pytorch_model.bin to safetensors[/green]")

    def _convert_sharded_pytorch_bins(self, source_path: Path, target_path: Path):
        """분할된 pytorch_model-*.bin을 단일 safetensors로 병합"""
        console.print("[yellow]Merging sharded pytorch models...[/yellow]")

        # 인덱스 파일 읽기
        index_file = source_path / "pytorch_model.bin.index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        with open(index_file) as f:
            index_data = json.load(f)

        # 모든 shard 로드하여 병합
        merged_state_dict = {}
        weight_map = index_data["weight_map"]

        for shard_file in set(weight_map.values()):
            shard_path = source_path / shard_file
            shard_state_dict = torch.load(
                shard_path, map_location="cpu", weights_only=True
            )
            merged_state_dict.update(shard_state_dict)

        # 단일 safetensors 파일로 저장
        from safetensors.torch import save_file

        save_file(merged_state_dict, target_path / "model.safetensors")

        console.print("[green]✅ Merged sharded models into single safetensors[/green]")

    def _standardize_config_files(self, source_path: Path, target_path: Path):
        """설정 파일들을 표준화"""
        console.print("[yellow]Standardizing config files...[/yellow]")

        # 필수 HuggingFace 파일들 복사
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",  # 선택적
            "vocab.json",  # GPT-2 계열
            "merges.txt",  # GPT-2 계열
            "tokenizer.model",  # LLaMA 계열
        ]

        for file_name in required_files:
            source_file = source_path / file_name
            if source_file.exists():
                shutil.copy2(source_file, target_path / file_name)

        # config.json에 WMTP 정보가 있다면 보존
        config_file = target_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)

            # WMTP 관련 설정 보존
            if "mtp_config" in config:
                console.print("[green]✅ Preserved MTP config in config.json[/green]")

        console.print("[green]✅ Config files standardized[/green]")

    def _generate_wmtp_metadata(
        self, target_path: Path, wmtp_type: str, model_info: dict[str, Any]
    ):
        """WMTP 메타데이터 파일 생성"""
        console.print("[yellow]Generating WMTP metadata...[/yellow]")

        metadata = {
            "wmtp_type": wmtp_type,
            "training_algorithm": model_info.get("training_algorithm", "unknown"),
            "horizon": model_info.get("horizon", 4),
            "n_heads": model_info.get("n_heads", 4),
            "base_architecture": model_info.get("architecture", "unknown"),
            "model_size": model_info.get("size", "unknown"),
            "storage_version": "2.0",
            "created_by": "wmtp_standardizer",
            "standardization_date": datetime.now().isoformat(),
            "original_format": model_info.get("original_format", "unknown"),
            "conversion_metadata": {
                "converted_from": model_info.get("source_path", "unknown"),
                "conversion_date": datetime.now().isoformat(),
                "converter_version": "wmtp-v2.0",
            },
        }

        with open(target_path / "wmtp_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        console.print("[green]✅ WMTP metadata generated[/green]")

    def _generate_mlflow_files(
        self, target_path: Path, wmtp_type: str, model_info: dict[str, Any]
    ):
        """MLflow 호환성 파일들 생성"""
        console.print("[yellow]Generating MLflow compatibility files...[/yellow]")

        mlflow_dir = target_path / "mlflow_model"
        mlflow_dir.mkdir(exist_ok=True)

        # MLmodel 파일
        mlmodel_content = {
            "artifact_path": "model",
            "flavors": {
                "python_function": {
                    "env": "conda.yaml",
                    "loader_module": "mlflow.pytorch",
                    "model_data": "../model.safetensors",
                    "python_version": "3.12",
                },
                "pytorch": {
                    "model_data": "../model.safetensors",
                    "pytorch_version": "2.4.1",
                },
            },
            "model_uuid": f"wmtp_{wmtp_type}_{model_info.get('size', 'unknown')}",
            "mlflow_version": "2.15.1",
            "signature": {
                "inputs": '[{"name": "input_ids", "type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1, -1]}}]',
                "outputs": '[{"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, -1, -1]}}]',
            },
        }

        with open(mlflow_dir / "MLmodel", "w") as f:
            import yaml

            yaml.dump(mlmodel_content, f, default_flow_style=False)

        # requirements.txt
        requirements = [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "safetensors>=0.4.0",
            "accelerate>=0.20.0",
        ]

        with open(mlflow_dir / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))

        # conda.yaml
        conda_env = {
            "channels": ["conda-forge", "pytorch"],
            "dependencies": ["python=3.12", "pytorch>=2.0.0", {"pip": requirements}],
            "name": f"wmtp_{wmtp_type}_env",
        }

        with open(mlflow_dir / "conda.yaml", "w") as f:
            import yaml

            yaml.dump(conda_env, f, default_flow_style=False)

        console.print("[green]✅ MLflow compatibility files generated[/green]")

    def _upload_standardized_model(self, local_path: Path, s3_path: str):
        """표준화된 모델을 S3에 업로드"""
        console.print(f"[yellow]Uploading standardized model to {s3_path}...[/yellow]")

        prefix = s3_path.replace("s3://wmtp/", "")

        with Progress() as progress:
            task = progress.add_task("Uploading to S3...", total=None)

            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{prefix}/{relative_path}".replace("\\", "/")

                    self.s3_client.upload_file(str(file_path), self.bucket, s3_key)

            progress.update(task, completed=True)


def main():
    """메인 실행 함수"""
    console.print("[bold blue]WMTP Model Storage Standardization[/bold blue]")

    # AWS 인증 정보 - 환경변수에서 로드
    import os

    from dotenv import load_dotenv

    load_dotenv()

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    standardizer = WMTPModelStandardizer(aws_access_key, aws_secret_key)

    # S3 모델 표준화 정의
    s3_models = [
        {
            "source": "s3://wmtp/models/7b_1t_4/",
            "target": "s3://wmtp/models_v2/llama-7b-mtp/",
            "wmtp_type": "mtp_native",
            "info": {
                "architecture": "llama",
                "size": "7b",
                "training_algorithm": "mtp",
                "horizon": 4,
                "n_heads": 4,
                "original_format": "consolidated_pth",
                "source_path": "s3://wmtp/models/7b_1t_4/",
            },
        },
        {
            "source": "s3://wmtp/models/Starling-RM-7B-alpha/",
            "target": "s3://wmtp/models_v2/starling-rm-7b/",
            "wmtp_type": "reward_model",
            "info": {
                "architecture": "llama",
                "size": "7b",
                "training_algorithm": "reward_modeling",
                "original_format": "pytorch_bin",
                "source_path": "s3://wmtp/models/Starling-RM-7B-alpha/",
            },
        },
        {
            "source": "s3://wmtp/models/Sheared-LLaMA-2.7B/",
            "target": "s3://wmtp/models_v2/sheared-llama-2.7b/",
            "wmtp_type": "base_model",
            "info": {
                "architecture": "llama",
                "size": "2.7b",
                "training_algorithm": "sheared_training",
                "original_format": "sharded_pytorch_bins",
                "source_path": "s3://wmtp/models/Sheared-LLaMA-2.7B/",
            },
        },
    ]

    # S3 모델들 표준화
    for model in s3_models:
        try:
            standardizer.standardize_s3_model(
                model["source"], model["target"], model["wmtp_type"], model["info"]
            )
        except Exception as e:
            console.print(f"[red]❌ Failed to standardize {model['source']}: {e}[/red]")

    # 로컬 테스트 모델들 표준화
    local_models = [
        {
            "source": Path("tests/tiny_models/distilgpt2-mtp"),
            "target": Path("tests/tiny_models_v2/distilgpt2-mtp"),
            "wmtp_type": "mtp_huggingface",
            "info": {
                "architecture": "gpt2",
                "size": "120m",
                "training_algorithm": "mtp",
                "horizon": 4,
                "n_heads": 4,
                "original_format": "huggingface_safetensors",
            },
        },
        {
            "source": Path("tests/tiny_models/tiny-reward-model"),
            "target": Path("tests/tiny_models_v2/tiny-reward-model"),
            "wmtp_type": "reward_model",
            "info": {
                "architecture": "gpt2",
                "size": "120m",
                "training_algorithm": "reward_modeling",
                "original_format": "huggingface_safetensors",
            },
        },
        {
            "source": Path("tests/tiny_models/distilgpt2"),
            "target": Path("tests/tiny_models_v2/distilgpt2"),
            "wmtp_type": "base_model",
            "info": {
                "architecture": "gpt2",
                "size": "120m",
                "training_algorithm": "base_training",
                "original_format": "huggingface_safetensors",
            },
        },
    ]

    # 로컬 모델들 표준화
    for model in local_models:
        try:
            standardizer.standardize_local_model(
                model["source"], model["target"], model["wmtp_type"], model["info"]
            )
        except Exception as e:
            console.print(f"[red]❌ Failed to standardize {model['source']}: {e}[/red]")

    console.print("[bold green]🎉 Model standardization completed![/bold green]")


if __name__ == "__main__":
    main()
