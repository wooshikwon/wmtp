#!/usr/bin/env python3
"""
S3 Model Pipeline - 대용량 모델 다운로드, 표준화, 업로드 통합 도구

이 스크립트는 S3에서 대용량 모델을 다운로드하고, HuggingFace 표준 구조로 변환한 뒤,
다시 S3에 업로드하는 전체 파이프라인을 제공합니다.

주요 기능:
- 10GB+ 대용량 파일 최적화 멀티파트 다운로드
- HuggingFace + MLflow 하이브리드 표준 구조로 변환
- safetensors 보안 형식 사용
- S3 업로드 with 진행률 표시

표준 출력 구조:
model_name/
├── model.safetensors           # HF 표준 (보안)
├── config.json                # HF 표준 + WMTP 확장
├── tokenizer.json             # HF 표준
├── tokenizer_config.json      # HF 표준
├── special_tokens_map.json    # HF 표준
├── metadata.json              # WMTP 전용 메타데이터
└── mlflow_model/              # MLflow 호환성
    ├── MLmodel               # MLflow 메타데이터
    ├── requirements.txt      # 의존성
    └── conda.yaml           # 환경
"""

import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import boto3
import torch
import yaml
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

console = Console()


class S3ModelPipeline:
    """S3 모델 다운로드, 표준화, 업로드 통합 파이프라인"""

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
            raise ValueError("AWS credentials not found in environment variables")

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region,
        )
        self.bucket = "wmtp"

        # 대용량 파일용 최적화 전송 설정 (속도 개선)
        self.transfer_config = TransferConfig(
            multipart_threshold=1024 * 5,  # 5MB 이상부터 멀티파트 (더 빠른 병렬화)
            max_concurrency=20,  # 최대 20개 병렬 연결 (2배 증가)
            multipart_chunksize=1024 * 10,  # 10MB 청크 (더 작은 청크로 병렬성 증가)
            use_threads=True,  # 스레드 풀 사용
            num_download_attempts=3,  # 재시도 3회
            max_io_queue=1000,  # I/O 큐 크기 10배 증가
            io_chunksize=1024 * 1024,  # 1MB I/O 청크 (처리량 증가)
        )

        console.print("[green]✅ S3 Model Pipeline 초기화 완료[/green]")

    def process_model(
        self,
        source_s3_path: str,
        target_s3_path: str,
        wmtp_type: str = "base_model",
        model_info: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        S3 모델 전체 파이프라인 실행
        1. S3에서 다운로드
        2. 표준 형식으로 변환
        3. S3에 업로드
        """
        console.print(f"[bold blue]🚀 모델 파이프라인 시작[/bold blue]")
        console.print(f"소스: {source_s3_path}")
        console.print(f"대상: {target_s3_path}")
        console.print(f"타입: {wmtp_type}")

        if model_info is None:
            model_info = {}

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                download_path = temp_path / "downloaded"
                standardized_path = temp_path / "standardized"

                # 1. S3에서 다운로드 (최적화된 멀티파트)
                console.print("\n[yellow]📥 Step 1/3: S3 다운로드 중...[/yellow]")
                if not self._download_from_s3(source_s3_path, download_path):
                    return False

                # 2. 표준 형식으로 변환
                console.print("\n[yellow]🔄 Step 2/3: 표준 형식 변환 중...[/yellow]")
                self._convert_to_standard(
                    download_path, standardized_path, wmtp_type, model_info
                )

                # 3. S3에 업로드
                console.print("\n[yellow]📤 Step 3/3: S3 업로드 중...[/yellow]")
                self._upload_to_s3(standardized_path, target_s3_path)

            console.print(
                f"\n[bold green]✅ 파이프라인 완료: {target_s3_path}[/bold green]"
            )
            return True

        except Exception as e:
            console.print(f"\n[red]❌ 파이프라인 실패: {e}[/red]")
            return False

    def _download_from_s3(self, s3_path: str, local_path: Path) -> bool:
        """S3에서 최적화된 멀티파트 다운로드"""
        local_path.mkdir(parents=True, exist_ok=True)
        prefix = s3_path.replace(f"s3://{self.bucket}/", "").rstrip("/")

        # 파일 리스트 조회
        files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                size = obj["Size"]
                # 디렉토리가 아닌 실제 파일만
                if not key.endswith("/"):
                    files.append((key, size))

        if not files:
            console.print("[red]❌ 다운로드할 파일이 없습니다[/red]")
            return False

        # 총 크기 계산
        total_size = sum(size for _, size in files)
        console.print(
            f"총 {len(files)}개 파일, {total_size / (1024**3):.2f}GB 다운로드"
        )

        # 진행률 표시와 함께 다운로드
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # 대용량 파일과 작은 파일 분리
            large_files = [(k, s) for k, s in files if s > 100 * 1024 * 1024]
            small_files = [(k, s) for k, s in files if s <= 100 * 1024 * 1024]

            # 대용량 파일 순차 다운로드 (멀티파트)
            for key, size in large_files:
                file_name = key.split("/")[-1]
                task_id = progress.add_task(f"📥 {file_name}", total=size)

                local_file = local_path / key.replace(prefix + "/", "")
                local_file.parent.mkdir(parents=True, exist_ok=True)

                def progress_callback(bytes_transferred):
                    progress.update(task_id, advance=bytes_transferred)

                try:
                    self.s3_client.download_file(
                        self.bucket,
                        key,
                        str(local_file),
                        Config=self.transfer_config,
                        Callback=progress_callback,
                    )
                    progress.update(task_id, completed=size)
                except Exception as e:
                    console.print(f"[red]❌ 다운로드 실패 {file_name}: {e}[/red]")
                    return False

            # 작은 파일 병렬 다운로드
            if small_files:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = []
                    for key, size in small_files:
                        file_name = key.split("/")[-1]
                        task_id = progress.add_task(f"📥 {file_name}", total=size)

                        local_file = local_path / key.replace(prefix + "/", "")
                        local_file.parent.mkdir(parents=True, exist_ok=True)

                        future = executor.submit(
                            self._download_single_file, key, local_file, size, progress, task_id
                        )
                        futures.append(future)

                    for future in as_completed(futures):
                        if not future.result():
                            return False

        console.print("[green]✅ 다운로드 완료[/green]")
        return True

    def _download_single_file(
        self, key: str, local_file: Path, size: int, progress: Progress, task_id: TaskID
    ) -> bool:
        """단일 파일 다운로드 (작은 파일용)"""
        try:

            def progress_callback(bytes_transferred):
                progress.update(task_id, advance=bytes_transferred)

            self.s3_client.download_file(
                self.bucket,
                key,
                str(local_file),
                Config=self.transfer_config,
                Callback=progress_callback,
            )
            progress.update(task_id, completed=size)
            return True
        except Exception as e:
            console.print(f"[red]❌ 다운로드 실패 {key}: {e}[/red]")
            return False

    def _convert_to_standard(
        self,
        source_path: Path,
        target_path: Path,
        wmtp_type: str,
        model_info: dict[str, Any],
    ):
        """모델을 HuggingFace 표준 형식으로 변환"""
        target_path.mkdir(parents=True, exist_ok=True)

        # 1. 모델 가중치를 safetensors로 변환
        self._convert_weights_to_safetensors(source_path, target_path, wmtp_type)

        # 2. 설정 파일 복사 및 표준화
        self._standardize_configs(source_path, target_path)

        # 3. WMTP 메타데이터 생성
        self._create_wmtp_metadata(target_path, wmtp_type, model_info)

        # 4. MLflow 호환성 파일 생성
        self._create_mlflow_files(target_path, wmtp_type, model_info)

        console.print("[green]✅ 표준 형식 변환 완료[/green]")

    def _convert_weights_to_safetensors(
        self, source_path: Path, target_path: Path, wmtp_type: str
    ):
        """모델 가중치를 safetensors 형식으로 변환"""
        console.print("모델 가중치 변환 중...")

        # 이미 safetensors인 경우
        if (source_path / "model.safetensors").exists():
            shutil.copy2(
                source_path / "model.safetensors", target_path / "model.safetensors"
            )
            console.print("✅ safetensors 형식 복사 완료")
            return

        # Meta consolidated.pth 변환
        if wmtp_type == "mtp_native" and (source_path / "consolidated.pth").exists():
            checkpoint = torch.load(
                source_path / "consolidated.pth", map_location="cpu", weights_only=True
            )
            save_file(checkpoint, target_path / "model.safetensors")
            console.print("✅ consolidated.pth → safetensors 변환 완료")
            return

        # 단일 pytorch_model.bin 변환
        if (source_path / "pytorch_model.bin").exists():
            try:
                # HuggingFace 모델로 로드 후 저장
                model = AutoModelForCausalLM.from_pretrained(
                    source_path, torch_dtype=torch.float32, trust_remote_code=True
                )
                model.save_pretrained(target_path, safe_serialization=True)

                # pytorch_model.bin 제거
                bin_file = target_path / "pytorch_model.bin"
                if bin_file.exists():
                    bin_file.unlink()

                console.print("✅ pytorch_model.bin → safetensors 변환 완료")
            except Exception:
                # 직접 변환
                state_dict = torch.load(
                    source_path / "pytorch_model.bin", map_location="cpu", weights_only=True
                )
                save_file(state_dict, target_path / "model.safetensors")
                console.print("✅ pytorch_model.bin → safetensors 직접 변환 완료")
            return

        # 분할된 pytorch_model-*.bin 병합 및 변환
        sharded_files = list(source_path.glob("pytorch_model-*.bin"))
        if sharded_files:
            # 인덱스 파일 읽기
            index_file = source_path / "pytorch_model.bin.index.json"
            if index_file.exists():
                with open(index_file) as f:
                    index_data = json.load(f)

                merged_state_dict = {}
                weight_map = index_data["weight_map"]

                # 모든 shard 병합
                for shard_file in set(weight_map.values()):
                    shard_path = source_path / shard_file
                    if shard_path.exists():
                        shard_dict = torch.load(
                            shard_path, map_location="cpu", weights_only=True
                        )
                        merged_state_dict.update(shard_dict)

                save_file(merged_state_dict, target_path / "model.safetensors")
                console.print("✅ 분할 모델 → safetensors 병합 변환 완료")
                return

        console.print("[yellow]⚠️ 지원되는 모델 가중치 형식을 찾을 수 없습니다[/yellow]")

    def _standardize_configs(self, source_path: Path, target_path: Path):
        """설정 파일들 복사 및 표준화"""
        config_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "vocab.json",  # GPT-2 계열
            "merges.txt",  # GPT-2 계열
            "tokenizer.model",  # LLaMA 계열
            "modeling.py",  # MTP 모델용 커스텀 모듈
        ]

        for file_name in config_files:
            source_file = source_path / file_name
            if source_file.exists():
                shutil.copy2(source_file, target_path / file_name)

        console.print("✅ 설정 파일 표준화 완료")

    def _create_wmtp_metadata(
        self, target_path: Path, wmtp_type: str, model_info: dict[str, Any]
    ):
        """WMTP 메타데이터 생성 (새로운 표준 구조)"""

        # loading_strategy 결정
        loading_strategy = self._determine_loading_strategy(
            wmtp_type, model_info, target_path
        )

        # algorithm_compatibility 결정
        algorithm_compatibility = self._determine_algorithm_compatibility(
            wmtp_type, model_info
        )

        metadata = {
            "wmtp_type": wmtp_type,
            "training_algorithm": model_info.get("training_algorithm", "unknown"),
            "horizon": model_info.get("horizon", 4),
            "n_heads": model_info.get("n_heads", 4),
            "base_architecture": model_info.get("architecture", "unknown"),
            "model_size": model_info.get("size", "unknown"),
            "storage_version": "2.0",
            "created_by": "s3_model_pipeline",
            "standardization_date": datetime.now().isoformat(),
            "original_format": model_info.get("original_format", "unknown"),
            "source_path": model_info.get("source_path", "unknown"),
            "loading_strategy": loading_strategy,
            "algorithm_compatibility": algorithm_compatibility,
        }

        with open(target_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        console.print("✅ WMTP 메타데이터 생성 완료")

    def _determine_loading_strategy(
        self, wmtp_type: str, model_info: dict[str, Any], target_path: Path
    ) -> dict[str, Any]:
        """모델 타입과 정보를 기반으로 loading_strategy 결정"""

        # MTP 모델인지 확인
        is_mtp = (
            model_info.get("training_algorithm") == "mtp"
            or wmtp_type == "mtp_native"
            or (target_path / "modeling.py").exists()
        )

        if is_mtp:
            # MTP 모델용 커스텀 로딩 전략
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
            # 일반 HuggingFace 모델 로딩 전략
            transformers_class = "AutoModelForCausalLM"
            if wmtp_type == "reward_model":
                transformers_class = "AutoModel"

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
        """모델 타입과 정보를 기반으로 algorithm_compatibility 결정"""

        training_algo = model_info.get("training_algorithm", "")

        # MTP 모델들
        if training_algo == "mtp" or wmtp_type == "mtp_native":
            return ["baseline-mtp", "critic-wmtp", "rho1-wmtp"]

        # Reward 모델
        if wmtp_type == "reward_model" or training_algo == "reward_modeling":
            return ["critic-wmtp", "rho1-wmtp"]

        # Base/Reference 모델
        if wmtp_type in ["base_model", "reference_model"]:
            # Sheared LLaMA 같은 특수 모델
            if training_algo == "sheared_training":
                return ["baseline-mtp", "rho1-wmtp"]
            # 일반 base 모델
            return ["rho1-wmtp"]

        # 기본값
        return ["rho1-wmtp"]

    def _create_mlflow_files(
        self, target_path: Path, wmtp_type: str, model_info: dict[str, Any]
    ):
        """MLflow 호환성 파일 생성"""
        mlflow_dir = target_path / "mlflow_model"
        mlflow_dir.mkdir(exist_ok=True)

        # 모델 크기 계산 (실제 safetensors 파일 크기)
        model_file = target_path / "model.safetensors"
        model_size_bytes = model_file.stat().st_size if model_file.exists() else 0

        # MLmodel 파일 (기존 tiny_models와 동일한 구조)
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
            "model_size_bytes": model_size_bytes,
            "model_uuid": "12345678901234567890123456789013",
            "run_id": None,
            "signature": {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]',
            },
            "utc_time_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        }

        with open(mlflow_dir / "MLmodel", "w") as f:
            yaml.dump(mlmodel_content, f, default_flow_style=False, sort_keys=False)

        # requirements.txt
        requirements = ["transformers", "torch"]

        with open(mlflow_dir / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))

        # conda.yaml
        conda_env = {
            "name": f"wmtp_{wmtp_type}_env",
            "channels": ["defaults"],
            "dependencies": ["python=3.11.7", {"pip": requirements}],
        }

        with open(mlflow_dir / "conda.yaml", "w") as f:
            yaml.dump(conda_env, f, default_flow_style=False, sort_keys=False)

        console.print("✅ MLflow 호환성 파일 생성 완료")

    def _upload_to_s3(self, local_path: Path, s3_path: str):
        """표준화된 모델을 S3에 업로드"""
        prefix = s3_path.replace(f"s3://{self.bucket}/", "").rstrip("/")

        # 업로드할 파일 목록 수집
        files_to_upload = []
        total_size = 0

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                files_to_upload.append((file_path, size))
                total_size += size

        console.print(
            f"총 {len(files_to_upload)}개 파일, {total_size / (1024**3):.2f}GB 업로드"
        )

        # 진행률 표시와 함께 업로드
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TransferSpeedColumn(),
            console=console,
        ) as progress:
            for file_path, size in files_to_upload:
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{prefix}/{relative_path}".replace("\\", "/")
                file_name = file_path.name

                task_id = progress.add_task(f"📤 {file_name}", total=size)

                def progress_callback(bytes_transferred):
                    progress.update(task_id, advance=bytes_transferred)

                try:
                    self.s3_client.upload_file(
                        str(file_path),
                        self.bucket,
                        s3_key,
                        Config=self.transfer_config,
                        Callback=progress_callback,
                    )
                    progress.update(task_id, completed=size)
                except Exception as e:
                    console.print(f"[red]❌ 업로드 실패 {file_name}: {e}[/red]")
                    raise

        console.print("[green]✅ S3 업로드 완료[/green]")


def test_sheared_llama_pipeline():
    """Sheared-LLaMA 2.7B 모델 파이프라인 테스트"""
    console.print(
        "[bold blue]🧪 Sheared-LLaMA 2.7B 파이프라인 테스트[/bold blue]\n"
    )

    # S3 파이프라인 초기화
    pipeline = S3ModelPipeline()

    # Sheared-LLaMA 모델 처리
    model_info = {
        "architecture": "llama",
        "size": "2.7b",
        "training_algorithm": "sheared_training",
        "original_format": "sharded_pytorch_bins",
        "source_path": "s3://wmtp/models/Sheared-LLaMA-2.7B/",
    }

    success = pipeline.process_model(
        source_s3_path="s3://wmtp/models/Sheared-LLaMA-2.7B/",
        target_s3_path="s3://wmtp/models_v2/sheared-llama-2.7b-standard/",
        wmtp_type="base_model",
        model_info=model_info,
    )

    if success:
        console.print("\n[bold green]🎉 파이프라인 테스트 성공![/bold green]")
        console.print("표준화된 모델이 S3에 업로드되었습니다.")
    else:
        console.print("\n[red]❌ 파이프라인 테스트 실패[/red]")


def main():
    """메인 실행 함수"""
    # .env 파일에서 환경변수 로드
    load_dotenv()

    # 실행할 모델 정의
    models_to_process = [
        {
            "source": "s3://wmtp/models/Sheared-LLaMA-2.7B/",
            "target": "s3://wmtp/models_v2/sheared-llama-2.7b/",
            "wmtp_type": "base_model",
            "info": {
                "architecture": "llama",
                "size": "2.7b",
                "training_algorithm": "sheared_training",
                "original_format": "sharded_pytorch_bins",
            },
        },
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
            },
        },
    ]

    # 파이프라인 실행
    pipeline = S3ModelPipeline()

    console.print(
        f"[bold blue]🚀 {len(models_to_process)}개 모델 처리 시작[/bold blue]\n"
    )

    for i, model in enumerate(models_to_process, 1):
        console.print(f"\n[cyan]━━━ 모델 {i}/{len(models_to_process)} ━━━[/cyan]")

        success = pipeline.process_model(
            source_s3_path=model["source"],
            target_s3_path=model["target"],
            wmtp_type=model["wmtp_type"],
            model_info=model["info"],
        )

        if not success:
            console.print(f"[red]모델 처리 실패: {model['source']}[/red]")

    console.print("\n[bold green]🎉 모든 모델 처리 완료![/bold green]")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 테스트 모드 실행
        test_sheared_llama_pipeline()
    else:
        # 전체 파이프라인 실행
        main()