#!/usr/bin/env python3
"""
S3 모든 모델 표준화 - 백그라운드 실행용
Sheared-LLaMA-2.7B, 7b_1t_4, Starling-RM-7B-alpha 모두 처리
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
import yaml
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


class S3ModelStandardizer:
    """S3 모델 표준화 (최적화 버전)"""

    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name="eu-north-1",
        )
        self.bucket = "wmtp"
        print(f"[{datetime.now()}] S3 Standardizer 초기화 완료", flush=True)

    def standardize_model(self, source_prefix: str, target_prefix: str, model_config: dict):
        """단일 모델 표준화"""
        print(f"\n[{datetime.now()}] ========================================", flush=True)
        print(f"모델 표준화 시작: {model_config['name']}", flush=True)
        print(f"소스: s3://{self.bucket}/{source_prefix}", flush=True)
        print(f"대상: s3://{self.bucket}/{target_prefix}", flush=True)

        try:
            # 1. S3 파일 복사
            print(f"[{datetime.now()}] S3 파일 복사 시작...", flush=True)
            self._copy_s3_files(source_prefix, target_prefix)

            # 2. 메타데이터 생성 및 업로드
            print(f"[{datetime.now()}] 메타데이터 생성...", flush=True)
            self._create_metadata(target_prefix, model_config)

            print(f"[{datetime.now()}] ✅ {model_config['name']} 표준화 완료!", flush=True)
            return True

        except Exception as e:
            print(f"[{datetime.now()}] ❌ 오류 발생: {e}", flush=True)
            return False

    def _copy_s3_files(self, source_prefix: str, target_prefix: str):
        """S3 파일 복사 (병렬 처리)"""
        paginator = self.s3_client.get_paginator("list_objects_v2")

        # 파일 목록 가져오기
        files_to_copy = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=source_prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                if not obj["Key"].endswith("/"):
                    files_to_copy.append((obj["Key"], obj["Size"]))

        total_files = len(files_to_copy)
        total_size = sum(size for _, size in files_to_copy) / (1024**3)  # GB
        print(f"[{datetime.now()}] 총 {total_files}개 파일, {total_size:.2f}GB", flush=True)

        # 파일 복사
        for i, (source_key, size) in enumerate(files_to_copy, 1):
            file_name = source_key.split("/")[-1]
            target_key = source_key.replace(source_prefix, target_prefix)

            # 진행 상황 출력
            if i % 5 == 0 or size > 100 * 1024 * 1024:  # 5개마다 또는 100MB 이상
                print(f"[{datetime.now()}] [{i}/{total_files}] {file_name} ({size/(1024**2):.1f}MB) 복사 중...", flush=True)

            try:
                # S3 서버사이드 복사
                copy_source = {"Bucket": self.bucket, "Key": source_key}

                if size > 5 * 1024**3:  # 5GB 이상
                    # 멀티파트 복사 (대용량)
                    self._multipart_copy(copy_source, target_key, size)
                else:
                    # 일반 복사
                    self.s3_client.copy_object(
                        CopySource=copy_source,
                        Bucket=self.bucket,
                        Key=target_key,
                        MetadataDirective='COPY',
                        TaggingDirective='COPY'
                    )

            except Exception as e:
                print(f"[{datetime.now()}] ⚠️ {file_name} 복사 실패: {e}", flush=True)
                # 실패해도 계속 진행

        print(f"[{datetime.now()}] S3 파일 복사 완료", flush=True)

    def _multipart_copy(self, copy_source: dict, target_key: str, size: int):
        """멀티파트 복사 (5GB 이상)"""
        # 멀티파트 업로드 시작
        mpu = self.s3_client.create_multipart_upload(Bucket=self.bucket, Key=target_key)
        mpu_id = mpu["UploadId"]

        # 1GB 청크
        chunk_size = 1024 * 1024 * 1024
        parts = []

        try:
            for i, offset in enumerate(range(0, size, chunk_size), 1):
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

            # 완료
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=target_key,
                UploadId=mpu_id,
                MultipartUpload={"Parts": parts},
            )

        except Exception as e:
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket, Key=target_key, UploadId=mpu_id
            )
            raise e

    def _create_metadata(self, target_prefix: str, model_config: dict):
        """메타데이터 생성 및 업로드"""

        # metadata.json
        metadata = {
            "wmtp_type": model_config["wmtp_type"],
            "training_algorithm": model_config["training_algorithm"],
            "horizon": model_config.get("horizon", 4),
            "n_heads": model_config.get("n_heads", 4),
            "base_architecture": model_config["architecture"],
            "model_size": model_config["size"],
            "storage_version": "2.0",
            "created_by": "s3_standardize_all",
            "standardization_date": datetime.now().isoformat(),
            "original_format": model_config["original_format"],
            "source_path": f"s3://{self.bucket}/{target_prefix}",
            "loading_strategy": self._get_loading_strategy(model_config),
            "algorithm_compatibility": self._get_algorithm_compatibility(model_config),
        }

        # JSON을 문자열로 변환
        metadata_json = json.dumps(metadata, indent=2)

        # S3에 직접 업로드
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=f"{target_prefix}/metadata.json",
            Body=metadata_json,
            ContentType="application/json"
        )

        # MLflow 파일들
        self._create_mlflow_files(target_prefix, model_config)

        print(f"[{datetime.now()}] 메타데이터 업로드 완료", flush=True)

    def _get_loading_strategy(self, config: dict) -> dict:
        """로딩 전략 결정"""
        if config["training_algorithm"] == "mtp" or config["wmtp_type"] == "mtp_native":
            return {
                "loader_type": "custom_mtp",
                "model_class_name": "GPTMTPForCausalLM",
                "custom_module_file": "modeling.py",
                "transformers_class": None,
                "state_dict_mapping": {
                    "remove_prefix": "base_model." if config["wmtp_type"] == "mtp_native" else None,
                    "add_prefix": None,
                    "key_transforms": {}
                },
                "required_files": ["config.json", "model.safetensors", "modeling.py", "metadata.json"]
            }
        else:
            transformers_class = "AutoModel" if config["wmtp_type"] == "reward_model" else "AutoModelForCausalLM"
            return {
                "loader_type": "huggingface",
                "model_class_name": None,
                "custom_module_file": None,
                "transformers_class": transformers_class,
                "state_dict_mapping": {
                    "remove_prefix": None,
                    "add_prefix": None,
                    "key_transforms": {}
                },
                "required_files": ["config.json", "metadata.json"]
            }

    def _get_algorithm_compatibility(self, config: dict) -> list:
        """알고리즘 호환성"""
        if config["training_algorithm"] == "mtp" or config["wmtp_type"] == "mtp_native":
            return ["baseline-mtp", "critic-wmtp", "rho1-wmtp"]
        elif config["wmtp_type"] == "reward_model":
            return ["critic-wmtp", "rho1-wmtp"]
        elif config["training_algorithm"] == "sheared_training":
            return ["baseline-mtp", "rho1-wmtp"]
        else:
            return ["rho1-wmtp"]

    def _create_mlflow_files(self, target_prefix: str, config: dict):
        """MLflow 파일 생성"""

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
            "model_uuid": f"{config['name']}",
            "signature": {
                "inputs": '[{"type": "string"}]',
                "outputs": '[{"type": "string"}]'
            },
            "utc_time_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        }

        # YAML 문자열로 변환
        mlmodel_yaml = yaml.dump(mlmodel, default_flow_style=False, sort_keys=False)

        # S3 업로드
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=f"{target_prefix}/mlflow_model/MLmodel",
            Body=mlmodel_yaml,
            ContentType="text/plain"
        )

        # requirements.txt
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=f"{target_prefix}/mlflow_model/requirements.txt",
            Body="transformers\ntorch",
            ContentType="text/plain"
        )

        # conda.yaml
        conda_env = {
            "name": f"wmtp_{config['name']}_env",
            "channels": ["defaults"],
            "dependencies": ["python=3.11.7", {"pip": ["transformers", "torch"]}]
        }

        conda_yaml = yaml.dump(conda_env, default_flow_style=False, sort_keys=False)

        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=f"{target_prefix}/mlflow_model/conda.yaml",
            Body=conda_yaml,
            ContentType="text/plain"
        )


def main():
    """메인 실행 함수 - 모든 모델 처리"""

    print(f"[{datetime.now()}] =====================================", flush=True)
    print(f"S3 모델 표준화 시작 (3개 모델)", flush=True)
    print(f"=====================================", flush=True)

    standardizer = S3ModelStandardizer()

    # 처리할 모델 정의
    models = [
        {
            "name": "sheared-llama-2.7b",
            "source": "models/Sheared-LLaMA-2.7B",
            "target": "models_v2/sheared-llama-2.7b",
            "wmtp_type": "base_model",
            "architecture": "llama",
            "size": "2.7b",
            "training_algorithm": "sheared_training",
            "original_format": "sharded_pytorch_bins",
        },
        {
            "name": "llama-7b-mtp",
            "source": "models/7b_1t_4",
            "target": "models_v2/llama-7b-mtp",
            "wmtp_type": "mtp_native",
            "architecture": "llama",
            "size": "7b",
            "training_algorithm": "mtp",
            "horizon": 4,
            "n_heads": 4,
            "original_format": "consolidated_pth",
        },
        {
            "name": "starling-rm-7b",
            "source": "models/Starling-RM-7B-alpha",
            "target": "models_v2/starling-rm-7b",
            "wmtp_type": "reward_model",
            "architecture": "llama",
            "size": "7b",
            "training_algorithm": "reward_modeling",
            "original_format": "pytorch_bin",
        },
    ]

    # 각 모델 처리
    success_count = 0
    for i, model in enumerate(models, 1):
        print(f"\n[{datetime.now()}] 모델 {i}/{len(models)}: {model['name']}", flush=True)

        success = standardizer.standardize_model(
            source_prefix=model["source"],
            target_prefix=model["target"],
            model_config=model
        )

        if success:
            success_count += 1
            print(f"[{datetime.now()}] ✅ {model['name']} 완료", flush=True)
        else:
            print(f"[{datetime.now()}] ❌ {model['name']} 실패", flush=True)

    # 최종 결과
    print(f"\n[{datetime.now()}] =====================================", flush=True)
    print(f"최종 결과: {success_count}/{len(models)} 모델 표준화 완료", flush=True)
    print(f"=====================================", flush=True)

    if success_count == len(models):
        print("🎉 모든 모델 표준화 성공!", flush=True)
        sys.exit(0)
    else:
        print("⚠️ 일부 모델 표준화 실패", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()