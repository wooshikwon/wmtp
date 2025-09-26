#!/usr/bin/env python3
"""
S3 In-Place 표준화 - 다운로드 없이 S3에서 직접 처리

주요 특징:
1. 대용량 파일은 S3에서 S3로 직접 복사 (다운로드 없음)
2. 메타데이터와 작은 설정 파일만 로컬에서 생성 후 업로드
3. 10GB 모델도 몇 분 내에 처리 가능
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import boto3
import yaml
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class S3InPlaceStandardizer:
    """S3에서 직접 모델 표준화 (다운로드 최소화)"""

    def __init__(
        self,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region: str = "eu-north-1",
    ):
        # 환경변수에서 AWS 인증 정보 로드
        if not aws_access_key:
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        if not aws_secret_key:
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS credentials not found")

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region,
        )
        self.bucket = "wmtp"

        console.print("[green]✅ S3 In-Place Standardizer 초기화[/green]")

    def standardize_model(
        self,
        source_prefix: str,
        target_prefix: str,
        wmtp_type: str = "base_model",
        model_info: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        S3에서 직접 모델 표준화
        1. 대용량 파일은 S3 내부 복사
        2. 메타데이터만 생성하여 업로드
        """
        console.print(f"\n[bold blue]🚀 S3 In-Place 표준화 시작[/bold blue]")
        console.print(f"소스: s3://{self.bucket}/{source_prefix}")
        console.print(f"대상: s3://{self.bucket}/{target_prefix}")

        if model_info is None:
            model_info = {}

        try:
            # 1. S3 파일 목록 조회
            console.print("\n[yellow]📋 Step 1/3: 파일 목록 조회[/yellow]")
            files = self._list_s3_files(source_prefix)

            if not files:
                console.print("[red]❌ 소스에 파일이 없습니다[/red]")
                return False

            # 파일 분류
            large_files = []  # S3 복사 대상
            small_files = []  # 다운로드 필요 (config 등)
            total_size = 0

            for file_key, size in files:
                total_size += size
                file_name = file_key.split("/")[-1]

                # 1MB 이하 파일만 다운로드 (config, tokenizer 등)
                if size <= 1024 * 1024 and file_name in [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "generation_config.json",
                    "vocab.json",
                    "merges.txt",
                    "tokenizer.model",
                ]:
                    small_files.append((file_key, size))
                else:
                    large_files.append((file_key, size))

            console.print(
                f"총 {len(files)}개 파일, {total_size / (1024**3):.2f}GB"
            )
            console.print(f"  - S3 복사: {len(large_files)}개 대용량 파일")
            console.print(f"  - 다운로드: {len(small_files)}개 설정 파일")

            # 2. 대용량 파일 S3 내부 복사
            console.print("\n[yellow]⚡ Step 2/3: S3 내부 고속 복사[/yellow]")
            self._copy_large_files_s3_to_s3(
                large_files, source_prefix, target_prefix
            )

            # 3. 설정 파일 처리 및 메타데이터 생성
            console.print("\n[yellow]📝 Step 3/3: 메타데이터 생성[/yellow]")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # 작은 파일들만 다운로드
                for file_key, _ in small_files:
                    self._download_small_file(file_key, temp_path)

                # 메타데이터 생성
                self._create_and_upload_metadata(
                    temp_path, target_prefix, wmtp_type, model_info
                )

                # 작은 파일들 업로드
                self._upload_small_files(temp_path, target_prefix)

            console.print(
                f"\n[bold green]✅ 표준화 완료! (다운로드 없이 처리)[/bold green]"
            )
            return True

        except Exception as e:
            console.print(f"\n[red]❌ 표준화 실패: {e}[/red]")
            return False

    def _list_s3_files(self, prefix: str) -> list[tuple[str, int]]:
        """S3 파일 목록 조회"""
        files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                if not obj["Key"].endswith("/"):  # 디렉토리 제외
                    files.append((obj["Key"], obj["Size"]))

        return files

    def _copy_large_files_s3_to_s3(
        self, files: list[tuple[str, int]], source_prefix: str, target_prefix: str
    ):
        """S3 내부에서 대용량 파일 직접 복사 (다운로드 없음)"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for file_key, size in files:
                file_name = file_key.split("/")[-1]
                relative_path = file_key.replace(source_prefix, "").lstrip("/")
                target_key = f"{target_prefix}/{relative_path}"

                # 대용량 파일 이름 변환 (필요시)
                if "pytorch_model" in file_name and file_name.endswith(".bin"):
                    # safetensors로 변환은 나중에 별도 처리
                    # 지금은 그대로 복사
                    pass

                task = progress.add_task(
                    f"복사 중: {file_name} ({size / (1024**2):.1f}MB)"
                )

                # S3 내부 복사 (서버사이드, 매우 빠름)
                copy_source = {"Bucket": self.bucket, "Key": file_key}

                try:
                    # 5GB 이상은 멀티파트 복사
                    if size > 5 * 1024**3:
                        self._multipart_copy(copy_source, target_key, size)
                    else:
                        self.s3_client.copy_object(
                            CopySource=copy_source,
                            Bucket=self.bucket,
                            Key=target_key,
                        )

                    progress.update(task, completed=100)
                    console.print(f"  ✅ {file_name} 복사 완료")

                except Exception as e:
                    console.print(f"  ❌ {file_name} 복사 실패: {e}")

    def _multipart_copy(self, copy_source: dict, target_key: str, size: int):
        """5GB 이상 파일의 멀티파트 복사"""
        # 멀티파트 업로드 시작
        mpu = self.s3_client.create_multipart_upload(
            Bucket=self.bucket, Key=target_key
        )
        mpu_id = mpu["UploadId"]

        # 500MB 청크로 분할
        chunk_size = 500 * 1024 * 1024
        parts = []

        try:
            for i, offset in enumerate(
                range(0, size, chunk_size), 1
            ):
                end_byte = min(offset + chunk_size - 1, size - 1)

                part = self.s3_client.upload_part_copy(
                    Bucket=self.bucket,
                    Key=target_key,
                    CopySource=copy_source,
                    CopySourceRange=f"bytes={offset}-{end_byte}",
                    PartNumber=i,
                    UploadId=mpu_id,
                )

                parts.append({
                    "ETag": part["CopyPartResult"]["ETag"],
                    "PartNumber": i
                })

            # 멀티파트 완료
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=target_key,
                UploadId=mpu_id,
                MultipartUpload={"Parts": parts},
            )

        except Exception as e:
            # 실패시 정리
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket, Key=target_key, UploadId=mpu_id
            )
            raise e

    def _download_small_file(self, file_key: str, local_path: Path):
        """작은 파일만 다운로드"""
        file_name = file_key.split("/")[-1]
        local_file = local_path / file_name

        try:
            self.s3_client.download_file(self.bucket, file_key, str(local_file))
        except Exception as e:
            console.print(f"[yellow]⚠️ {file_name} 다운로드 실패: {e}[/yellow]")

    def _create_and_upload_metadata(
        self,
        temp_path: Path,
        target_prefix: str,
        wmtp_type: str,
        model_info: dict[str, Any],
    ):
        """메타데이터 생성 및 업로드"""

        # loading_strategy 결정
        loading_strategy = self._determine_loading_strategy(wmtp_type, model_info)

        # algorithm_compatibility 결정
        algorithm_compatibility = self._determine_algorithm_compatibility(
            wmtp_type, model_info
        )

        # metadata.json 생성
        metadata = {
            "wmtp_type": wmtp_type,
            "training_algorithm": model_info.get("training_algorithm", "unknown"),
            "horizon": model_info.get("horizon", 4),
            "n_heads": model_info.get("n_heads", 4),
            "base_architecture": model_info.get("architecture", "unknown"),
            "model_size": model_info.get("size", "unknown"),
            "storage_version": "2.0",
            "created_by": "s3_inplace_standardizer",
            "standardization_date": datetime.now().isoformat(),
            "original_format": model_info.get("original_format", "unknown"),
            "source_path": f"s3://{self.bucket}/{target_prefix}",
            "loading_strategy": loading_strategy,
            "algorithm_compatibility": algorithm_compatibility,
        }

        # 로컬에 저장
        metadata_file = temp_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # S3에 업로드
        self.s3_client.upload_file(
            str(metadata_file),
            self.bucket,
            f"{target_prefix}/metadata.json",
        )

        # MLflow 파일들 생성 및 업로드
        self._create_mlflow_files(temp_path, target_prefix)

        console.print("✅ 메타데이터 생성 및 업로드 완료")

    def _determine_loading_strategy(
        self, wmtp_type: str, model_info: dict[str, Any]
    ) -> dict[str, Any]:
        """로딩 전략 결정"""
        is_mtp = (
            model_info.get("training_algorithm") == "mtp" or wmtp_type == "mtp_native"
        )

        if is_mtp:
            return {
                "loader_type": "custom_mtp",
                "model_class_name": "GPTMTPForCausalLM",
                "custom_module_file": "modeling.py",
                "transformers_class": None,
                "state_dict_mapping": {
                    "remove_prefix": "base_model." if wmtp_type == "mtp_native" else None,
                    "add_prefix": None,
                    "key_transforms": {},
                },
                "required_files": [
                    "config.json",
                    "model.safetensors",
                    "modeling.py",
                    "metadata.json",
                ],
            }
        else:
            transformers_class = (
                "AutoModel" if wmtp_type == "reward_model" else "AutoModelForCausalLM"
            )
            return {
                "loader_type": "huggingface",
                "model_class_name": None,
                "custom_module_file": None,
                "transformers_class": transformers_class,
                "state_dict_mapping": {
                    "remove_prefix": None,
                    "add_prefix": None,
                    "key_transforms": {},
                },
                "required_files": ["config.json", "model.safetensors", "metadata.json"],
            }

    def _determine_algorithm_compatibility(
        self, wmtp_type: str, model_info: dict[str, Any]
    ) -> list[str]:
        """알고리즘 호환성 결정"""
        training_algo = model_info.get("training_algorithm", "")

        if training_algo == "mtp" or wmtp_type == "mtp_native":
            return ["baseline-mtp", "critic-wmtp", "rho1-wmtp"]
        elif wmtp_type == "reward_model" or training_algo == "reward_modeling":
            return ["critic-wmtp", "rho1-wmtp"]
        elif wmtp_type in ["base_model", "reference_model"]:
            if training_algo == "sheared_training":
                return ["baseline-mtp", "rho1-wmtp"]
            return ["rho1-wmtp"]

        return ["rho1-wmtp"]

    def _create_mlflow_files(self, temp_path: Path, target_prefix: str):
        """MLflow 파일 생성 및 업로드"""
        # MLmodel
        mlmodel_content = {
            "artifact_path": "model",
            "flavors": {
                "python_function": {
                    "env": {
                        "conda": "conda.yaml",
                        "virtualenv": "python_env.yaml",
                    },
                    "loader_module": "mlflow.transformers",
                    "model_path": "model",
                    "python_version": "3.11.7",
                },
                "transformers": {
                    "code": None,
                    "framework": "pt",
                    "model_path": "model",
                    "task": "text-generation",
                },
            },
            "mlflow_version": "2.15.1",
            "model_size_bytes": 0,  # 나중에 채움
            "model_uuid": "12345678901234567890123456789013",
            "run_id": None,
            "signature": {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]',
            },
            "utc_time_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        }

        mlmodel_file = temp_path / "MLmodel"
        with open(mlmodel_file, "w") as f:
            yaml.dump(mlmodel_content, f, default_flow_style=False, sort_keys=False)

        # requirements.txt
        requirements_file = temp_path / "requirements.txt"
        with open(requirements_file, "w") as f:
            f.write("transformers\ntorch")

        # conda.yaml
        conda_env = {
            "name": "wmtp_env",
            "channels": ["defaults"],
            "dependencies": ["python=3.11.7", {"pip": ["transformers", "torch"]}],
        }

        conda_file = temp_path / "conda.yaml"
        with open(conda_file, "w") as f:
            yaml.dump(conda_env, f, default_flow_style=False, sort_keys=False)

        # S3에 업로드
        for file_name in ["MLmodel", "requirements.txt", "conda.yaml"]:
            local_file = temp_path / file_name
            if local_file.exists():
                self.s3_client.upload_file(
                    str(local_file),
                    self.bucket,
                    f"{target_prefix}/mlflow_model/{file_name}",
                )

    def _upload_small_files(self, local_path: Path, target_prefix: str):
        """작은 설정 파일들 업로드"""
        for file_path in local_path.glob("*"):
            if file_path.is_file() and file_path.name not in [
                "metadata.json",
                "MLmodel",
                "requirements.txt",
                "conda.yaml",
            ]:
                self.s3_client.upload_file(
                    str(file_path),
                    self.bucket,
                    f"{target_prefix}/{file_path.name}",
                )


def process_sheared_llama():
    """Sheared-LLaMA 모델 처리 (다운로드 없음)"""
    console.print(
        "[bold blue]🚀 Sheared-LLaMA S3 In-Place 표준화[/bold blue]\n"
    )

    # 환경변수 로드
    load_dotenv()

    standardizer = S3InPlaceStandardizer()

    model_info = {
        "architecture": "llama",
        "size": "2.7b",
        "training_algorithm": "sheared_training",
        "original_format": "sharded_pytorch_bins",
    }

    success = standardizer.standardize_model(
        source_prefix="models/Sheared-LLaMA-2.7B",
        target_prefix="models_v2/sheared-llama-2.7b",
        wmtp_type="base_model",
        model_info=model_info,
    )

    if success:
        console.print("\n[bold green]✨ 완료! 다운로드 없이 S3에서 처리됨[/bold green]")
    else:
        console.print("\n[red]❌ 표준화 실패[/red]")


def main():
    """메인 실행"""
    load_dotenv()

    # 처리할 모델들
    models = [
        {
            "source": "models/Sheared-LLaMA-2.7B",
            "target": "models_v2/sheared-llama-2.7b",
            "wmtp_type": "base_model",
            "info": {
                "architecture": "llama",
                "size": "2.7b",
                "training_algorithm": "sheared_training",
                "original_format": "sharded_pytorch_bins",
            },
        },
        {
            "source": "models/7b_1t_4",
            "target": "models_v2/llama-7b-mtp",
            "wmtp_type": "mtp_native",
            "info": {
                "architecture": "llama",
                "size": "7b",
                "training_algorithm": "mtp",
                "horizon": 4,
                "n_heads": 4,
                "original_format": "consolidated_pth",
            },
        },
        {
            "source": "models/Starling-RM-7B-alpha",
            "target": "models_v2/starling-rm-7b",
            "wmtp_type": "reward_model",
            "info": {
                "architecture": "llama",
                "size": "7b",
                "training_algorithm": "reward_modeling",
                "original_format": "pytorch_bin",
            },
        },
    ]

    standardizer = S3InPlaceStandardizer()

    console.print(f"[bold blue]🚀 {len(models)}개 모델 처리 (다운로드 없음)[/bold blue]\n")

    for i, model in enumerate(models, 1):
        console.print(f"\n[cyan]━━━ 모델 {i}/{len(models)} ━━━[/cyan]")

        success = standardizer.standardize_model(
            source_prefix=model["source"],
            target_prefix=model["target"],
            wmtp_type=model["wmtp_type"],
            model_info=model["info"],
        )

        if not success:
            console.print(f"[red]모델 처리 실패: {model['source']}[/red]")

    console.print("\n[bold green]🎉 모든 모델 처리 완료![/bold green]")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sheared":
        process_sheared_llama()
    else:
        main()