"""
StandardizedModelLoader: 표준화된 메타데이터 기반 스마트 모델 로더

WMTP Phase 2의 핵심 구현체로, metadata.json 기반의 지능적 모델 타입 감지와
config/recipe 정보를 활용한 정확한 모델 로딩을 제공합니다.

핵심 개선사항:
1. 메타데이터 기반 감지: metadata.json의 wmtp_type으로 정확한 타입 추론
2. Config/Recipe 연동: 알고리즘별 모델 경로 자동 선택
3. 표준 구조 활용: HF + WMTP 표준을 완전히 지원
4. 단순화된 흐름: 경로 → 메타데이터 → 로더 → 모델

지원하는 모델 타입:
- base_model: MTP 확장 모델 (우리의 핵심 모델, config.paths.models.base)
- reference_model: 일반 언어 모델 (Rho1-WMTP용, config.paths.models.ref)
- reward_model: RLHF용 보상 모델 (Critic-WMTP용, config.paths.models.rm)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from src.components.loader.base_loader import ModelLoader
from src.components.registry import loader_registry
from src.utils.optimized_s3_transfer import OptimizedS3Transfer
from src.utils.path_resolver import PathResolver
from src.utils.s3 import create_s3_manager

ModelType = Literal["base_model", "reference_model", "reward_model"]


@loader_registry.register(
    "standardized-model-loader",
    version="2.0.0",
    description="Metadata-based standardized model loader",
)
class StandardizedModelLoader(ModelLoader):
    """표준화된 메타데이터 기반 모델 로더

    WMTP Phase 2 핵심 설계:
    - metadata.json 필수: 모든 표준화된 모델은 메타데이터를 가져야 함
    - 타입별 특화 로딩: wmtp_type에 따른 최적화된 로드 방식
    - Config/Recipe 연동: 알고리즘 요구사항에 맞는 자동 모델 선택
    - S3/로컬 투명성: PathResolver로 스토리지 위치 추상화

    예상 메타데이터 구조:
    ```json
    {
        "wmtp_type": "mtp_model",
        "base_architecture": "gpt2",
        "model_size": "124m",
        "training_algorithm": "baseline_mtp",
        "storage_version": "2.0"
    }
    ```
    """

    def __init__(self, config: dict[str, Any]):
        """표준화된 로더 초기화

        Args:
            config: 환경 설정 (paths, devices, auth 등)
        """
        super().__init__(config)

        # 핵심 구성요소 초기화
        self.path_resolver = PathResolver()
        self.s3_manager = create_s3_manager(config)

        # S3 최적화 다운로더 초기화 (대용량 모델용)
        if self.s3_manager:
            # S3Manager에서 s3_client와 bucket 정보 추출
            s3_client = getattr(self.s3_manager, "client", None)
            bucket = getattr(self.s3_manager, "bucket", "wmtp")

            if s3_client:
                self.s3_transfer = OptimizedS3Transfer(s3_client, bucket)
            else:
                self.s3_transfer = None
        else:
            self.s3_transfer = None

        # 디바이스 및 성능 설정
        devices_config = config.get("devices", {})
        self.device = self._resolve_device(devices_config)
        self.mixed_precision = devices_config.get("mixed_precision", "fp32")
        self.use_4bit = config.get("use_4bit", False)
        self.use_8bit = config.get("use_8bit", False)

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """표준화된 모델 로드 실행

        Args:
            inputs: 입력 딕셔너리
                - model_path: 모델 경로 (로컬 또는 S3 URI)
                - force_type: (선택) 강제 타입 지정

        Returns:
            로드된 모델과 메타데이터
        """
        model_path = inputs.get("model_path")
        if not model_path:
            raise ValueError("model_path is required")

        # 1. 경로 해석
        path_type, resolved = self.path_resolver.resolve(model_path)

        # 2. 메타데이터 기반 타입 감지
        model_type = inputs.get("force_type") or self._detect_model_type_from_metadata(
            model_path, path_type, resolved
        )

        # 3. 타입별 특화 로딩
        if path_type == "s3":
            model = self._load_from_s3(resolved, model_type)
        else:
            model = self._load_from_local(resolved, model_type)

        return {
            "model": model,
            "model_type": model_type,
            "path": model_path,
            "loader": self.__class__.__name__,
        }

    def _detect_model_type_from_metadata(
        self, original_path: str, path_type: str, resolved_path: str
    ) -> ModelType:
        """메타데이터에서 정확한 모델 타입 감지

        Args:
            original_path: 원본 경로
            path_type: 경로 타입 ("s3" or "local")
            resolved_path: 해석된 경로

        Returns:
            감지된 모델 타입
        """
        try:
            if path_type == "s3":
                metadata = self._load_metadata_from_s3(resolved_path)
            else:
                metadata_path = Path(resolved_path) / "metadata.json"
                if not metadata_path.exists():
                    # 메타데이터가 없으면 폴백: 기본 HuggingFace 모델로 가정
                    return "base_model"

                with open(metadata_path) as f:
                    metadata = json.load(f)

            # wmtp_type 필드로 정확한 타입 반환
            wmtp_type = metadata.get("wmtp_type", "base_model")

            # 호환성 매핑: 기존 메타데이터의 타입을 새로운 명명 규칙으로 변환
            type_mapping = {
                "base_model": "base_model",  # MTP 모델 (변경 없음)
                "mtp_model": "base_model",  # MTP 모델 (호환성)
                "reference_model": "reference_model",  # 일반 언어 모델
                "reward_model": "reward_model",  # 보상 모델
            }

            mapped_type = type_mapping.get(wmtp_type, "base_model")
            if mapped_type not in ["base_model", "reference_model", "reward_model"]:
                raise ValueError(f"Unknown wmtp_type: {wmtp_type}")

            return mapped_type

        except Exception as e:
            # 메타데이터 로드 실패 시 폴백
            print(f"Warning: Failed to load metadata from {original_path}: {e}")
            return "base_model"

    def _load_metadata_from_s3(self, s3_path: str) -> dict[str, Any]:
        """S3에서 메타데이터 로드

        Args:
            s3_path: S3 URI

        Returns:
            메타데이터 딕셔너리
        """
        if not self.s3_manager:
            raise RuntimeError("S3 manager not available")

        # S3 경로에서 버킷과 키 추출
        bucket, key_prefix = self.path_resolver.extract_bucket_and_key(s3_path)
        metadata_key = f"{key_prefix.rstrip('/')}/metadata.json"

        # 메타데이터 스트리밍
        metadata_stream = self.s3_manager.stream_dataset(metadata_key)
        return json.load(metadata_stream)

    def _load_from_s3(self, s3_path: str, model_type: ModelType) -> Any:
        """S3에서 타입별 모델 로딩

        Args:
            s3_path: S3 URI
            model_type: 모델 타입

        Returns:
            로드된 모델
        """
        if not self.s3_manager:
            raise RuntimeError("S3 manager not available")

        bucket, key_prefix = self.path_resolver.extract_bucket_and_key(s3_path)

        if model_type == "base_model":
            return self._load_mtp_model_from_s3(s3_path)
        elif model_type == "reference_model":
            return self._load_huggingface_from_s3(s3_path)
        elif model_type == "reward_model":
            return self._load_reward_model_from_s3(s3_path)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def _load_from_local(self, local_path: str, model_type: ModelType) -> Any:
        """로컬에서 타입별 모델 로딩

        Args:
            local_path: 로컬 경로
            model_type: 모델 타입

        Returns:
            로드된 모델
        """
        if model_type == "base_model":
            return self._load_mtp_model_from_local(local_path)
        elif model_type == "reference_model":
            return self._load_huggingface_from_local(local_path)
        elif model_type == "reward_model":
            return self._load_reward_model_from_local(local_path)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def _load_huggingface_from_local(self, path: str) -> Any:
        """로컬 HuggingFace 모델 로드 (reference 모델용)

        Args:
            path: 모델 디렉토리 경로

        Returns:
            로드된 HF 모델
        """
        return self._load_huggingface_common(path, "reference_model")

    def _load_huggingface_from_s3(self, s3_path: str) -> Any:
        """S3 HuggingFace 모델 로드 - 최적화된 다운로드

        Args:
            s3_path: S3 URI

        Returns:
            로드된 HF 모델
        """
        # HuggingFace transformers는 로컬 경로만 지원하므로 임시 다운로드 필요
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "model"
            temp_path.mkdir()

            # S3에서 필요한 파일들 최적화 다운로드
            bucket, key_prefix = self.path_resolver.extract_bucket_and_key(s3_path)

            required_files = [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ]

            # OptimizedS3Transfer 사용하여 병렬 다운로드
            if self.optimized_s3_transfer:
                success, downloaded_files = (
                    self.optimized_s3_transfer.download_specific_files(
                        key_prefix, required_files, temp_path, show_progress=True
                    )
                )

                if not success or not downloaded_files:
                    raise RuntimeError(
                        f"Failed to download HuggingFace model files from {s3_path}"
                    )
            else:
                # Fallback: 기존 스트리밍 방식
                for filename in required_files:
                    try:
                        file_key = f"{key_prefix.rstrip('/')}/{filename}"
                        file_stream = self.s3_manager.stream_model(file_key)

                        with open(temp_path / filename, "wb") as f:
                            f.write(file_stream.read())
                    except Exception:
                        # 일부 파일은 선택적일 수 있음
                        continue

            return self._load_huggingface_common(str(temp_path))

    def _load_mtp_model_from_local(self, path: str) -> Any:
        """로컬 MTP 모델 로드 (custom modeling.py 활용)

        Args:
            path: 모델 디렉토리 경로

        Returns:
            로드된 MTP 모델
        """
        # modeling.py가 있는지 확인
        model_path = Path(path)
        modeling_file = model_path / "modeling.py"

        if modeling_file.exists():
            # Custom MTP 모델 클래스 동적 로드
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "custom_modeling", modeling_file
            )
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            # Config 로드
            config = AutoConfig.from_pretrained(path, trust_remote_code=True)

            # MTP 모델 클래스 찾기 (관례적으로 MTP가 포함된 클래스명)
            mtp_class = None
            for attr_name in dir(custom_module):
                attr = getattr(custom_module, attr_name)
                if (
                    isinstance(attr, type)
                    and hasattr(attr, "__bases__")
                    and "MTP" in attr_name
                ):
                    mtp_class = attr
                    break

            if mtp_class:
                # safetensors에서 직접 로드
                from safetensors.torch import load_file

                safetensors_path = model_path / "model.safetensors"
                if safetensors_path.exists():
                    model = mtp_class(config)
                    state_dict = load_file(safetensors_path)

                    # 키 매핑: base_model. 프리픽스 제거 또는 추가
                    mapped_state_dict = {}
                    for key, value in state_dict.items():
                        # base_model. 프리픽스가 있으면 제거
                        if key.startswith("base_model."):
                            mapped_key = key[11:]  # "base_model." 길이 제거
                        else:
                            mapped_key = key
                        mapped_state_dict[mapped_key] = value

                    model.load_state_dict(mapped_state_dict, strict=False)
                    return model

        # 폴백: 일반 HuggingFace로 로드
        return self._load_huggingface_common(path)

    def _load_mtp_model_from_s3(self, s3_path: str) -> Any:
        """S3 MTP 모델 로드 - 최적화된 다운로드

        Args:
            s3_path: S3 URI

        Returns:
            로드된 MTP 모델
        """
        # S3에서는 임시 다운로드 후 로컬 방식으로 처리
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "model"
            temp_path.mkdir()

            bucket, key_prefix = self.path_resolver.extract_bucket_and_key(s3_path)

            # OptimizedS3Transfer 사용하여 대용량 모델 최적화 다운로드
            if self.optimized_s3_transfer:
                # 전체 모델 디렉토리 다운로드 (대용량 모델 최적화 적용)
                success, downloaded_files = (
                    self.optimized_s3_transfer.download_model_directory(
                        key_prefix, temp_path, show_progress=True
                    )
                )

                if not success:
                    raise RuntimeError(f"Failed to download MTP model from {s3_path}")

                # 필수 파일 존재 확인
                for filename in ["config.json", "model.safetensors"]:
                    if not (temp_path / filename).exists():
                        raise RuntimeError(
                            f"Required file {filename} not found after download"
                        )
            else:
                # Fallback: 기존 스트리밍 방식
                required_files = [
                    "config.json",
                    "model.safetensors",
                    "modeling.py",
                    "metadata.json",
                ]

                for filename in required_files:
                    try:
                        file_key = f"{key_prefix.rstrip('/')}/{filename}"
                        if filename.endswith((".json", ".py")):
                            file_stream = self.s3_manager.stream_dataset(file_key)
                        else:
                            file_stream = self.s3_manager.stream_model(file_key)

                        with open(temp_path / filename, "wb") as f:
                            f.write(file_stream.read())
                    except Exception as e:
                        if filename in ["config.json", "model.safetensors"]:
                            raise RuntimeError(
                                f"Required file {filename} not found: {e}"
                            )

            return self._load_mtp_model_from_local(str(temp_path))

    def _load_reward_model_from_local(self, path: str) -> Any:
        """로컬 보상 모델 로드

        Args:
            path: 모델 디렉토리 경로

        Returns:
            로드된 보상 모델
        """
        # 보상 모델: AutoModel 사용 (sequence classification head)
        return self._load_huggingface_common(path, "reward_model")

    def _load_reward_model_from_s3(self, s3_path: str) -> Any:
        """S3 보상 모델 로드

        Args:
            s3_path: S3 URI

        Returns:
            로드된 보상 모델
        """
        return self._load_huggingface_from_s3(s3_path)

    def _load_huggingface_common(self, path: str, model_type: ModelType = None) -> Any:
        """공통 HuggingFace 모델 로드 로직 - metadata 기반 정확한 클래스 선택

        Args:
            path: 모델 경로
            model_type: metadata.json에서 추출한 모델 타입

        Returns:
            로드된 모델
        """
        # 양자화 설정
        quantization_config = None
        if self.use_4bit or self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.use_4bit,
                load_in_8bit=self.use_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16
                if self.mixed_precision == "bf16"
                else torch.float16,
            )

        # metadata 기반 정확한 모델 클래스 선택
        common_args = {
            "quantization_config": quantization_config,
            "device_map": "auto" if self.device != "cpu" else None,
            "torch_dtype": self._get_torch_dtype(),
            "trust_remote_code": True,
        }

        if model_type == "reward_model":
            # 보상 모델: AutoModel 사용 (sequence classification head)
            from transformers import AutoModel

            model = AutoModel.from_pretrained(path, **common_args)
        else:
            # reference_model 또는 디폴트: AutoModelForCausalLM 사용 (언어 모델)
            model = AutoModelForCausalLM.from_pretrained(path, **common_args)

        return model

    def _resolve_device(self, devices_config: dict) -> str:
        """디바이스 설정 해석"""
        compute_backend = devices_config.get("compute_backend", "auto")

        if compute_backend == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        return compute_backend

    def _get_torch_dtype(self):
        """Mixed precision에 따른 torch dtype 반환"""
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.mixed_precision == "fp16":
            return torch.float16
        else:
            return torch.float32

    def load_model(self, path: str, **kwargs) -> Any:
        """ModelLoader 인터페이스 구현

        Args:
            path: 모델 경로
            **kwargs: 추가 파라미터

        Returns:
            로드된 모델
        """
        inputs = {"model_path": path, **kwargs}
        result = self.run(inputs)
        return result["model"]
