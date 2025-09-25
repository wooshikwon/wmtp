"""
UnifiedModelLoader: 모든 모델 타입을 처리하는 통합 로더

Phase 2 리팩토링의 핵심 구현체로, 기존 5개 모델 로더를 하나로 통합합니다.
PathResolver를 활용하여 로컬/S3 경로를 자동 판별하고,
모델 타입을 자동 감지하여 적절한 로드 방식을 적용합니다.

통합 대상:
- hf_model_loader.py
- mtp_native_loader.py
- checkpoint_loader.py
- sheared_llama_loader.py
- starling_rm_loader.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.components.loader.base_loader import ModelLoader
from src.components.registry import loader_registry
from src.utils.path_resolver import PathResolver
from src.utils.s3 import create_s3_manager


@loader_registry.register(
    "unified-model-loader",
    version="2.0.0",
    description="Unified model loader for all model types",
)
class UnifiedModelLoader(ModelLoader):
    """모든 모델 타입을 처리하는 통합 로더

    WMTP Phase 2 리팩토링의 핵심:
    - 경로 자동 판별: PathResolver로 로컬/S3 자동 구분
    - 모델 타입 자동 감지: 파일명/경로 패턴으로 타입 추론
    - 스트리밍 지원: S3에서 직접 메모리로 로드
    - 통합 인터페이스: 모든 모델에 동일한 API 제공

    지원하는 모델 타입:
    1. MTP Native: Facebook consolidated.pth 형식
    2. HuggingFace: AutoModelForCausalLM 호환 모델
    3. Checkpoint: 학습 중단점 파일
    4. Sheared LLaMA: Princeton 경량화 모델
    5. Starling RM: Berkeley 보상 모델
    """

    def __init__(self, config: dict[str, Any]):
        """통합 로더 초기화

        Args:
            config: 환경 설정 딕셔너리
                - storage: 스토리지 설정 (mode, s3 정보)
                - devices: 디바이스 설정 (compute_backend, mixed_precision)
                - paths: 경로 설정 (models, datasets)
        """
        super().__init__(config)

        # PathResolver와 S3Manager 초기화
        self.path_resolver = PathResolver()
        self.s3_manager = create_s3_manager(config)

        # 디바이스 설정
        devices_config = config.get("devices", {})
        self.device = self._resolve_device(devices_config)
        self.mixed_precision = devices_config.get("mixed_precision", "fp32")

        # 양자화 설정
        self.use_4bit = config.get("use_4bit", False)
        self.use_8bit = config.get("use_8bit", False)

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """통합 모델 로드 실행

        Args:
            inputs: 입력 딕셔너리
                - model_path: 모델 경로 (로컬 또는 S3 URI)
                - model_type: (선택) 명시적 모델 타입
                - device: (선택) 디바이스 오버라이드

        Returns:
            로드된 모델과 메타데이터를 포함하는 딕셔너리
        """
        model_path = inputs.get("model_path")
        if not model_path:
            raise ValueError("model_path is required")

        # 경로 해석
        path_type, resolved = self.path_resolver.resolve(model_path)

        # 모델 타입 감지
        model_type = inputs.get("model_type") or self._detect_model_type(model_path)

        # 경로 타입에 따른 로드
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

    def _detect_model_type(self, path: str) -> str:
        """경로에서 모델 타입 자동 감지

        Args:
            path: 모델 경로

        Returns:
            감지된 모델 타입
        """
        path_lower = path.lower()

        # MTP Native 패턴
        if "consolidated" in path_lower or "mtp" in path_lower:
            return "mtp_native"

        # Checkpoint 패턴
        if "checkpoint" in path_lower or path_lower.endswith((".ckpt", ".pt")):
            return "checkpoint"

        # Sheared LLaMA 패턴
        if "sheared" in path_lower:
            return "sheared_llama"

        # Starling RM 패턴
        if "starling" in path_lower or "rm" in path_lower:
            return "starling_rm"

        # 기본값: HuggingFace
        return "huggingface"

    def _load_from_s3(self, s3_path: str, model_type: str) -> Any:
        """S3에서 모델 스트리밍 로드

        Args:
            s3_path: S3 URI
            model_type: 모델 타입

        Returns:
            로드된 모델
        """
        if not self.s3_manager:
            raise RuntimeError("S3 manager not available")

        # S3 경로에서 버킷과 키 추출
        bucket, key = self.path_resolver.extract_bucket_and_key(s3_path)

        # 모델 타입별 로드
        if model_type == "mtp_native":
            # MTP Native는 직접 스트리밍
            stream = self.s3_manager.stream_model(key)
            return self._load_mtp_native(stream)

        elif model_type in ["checkpoint", "sheared_llama", "starling_rm"]:
            # 체크포인트류는 스트리밍 로드
            stream = self.s3_manager.stream_model(key)
            return self._load_checkpoint(stream)

        else:
            # HuggingFace 모델은 로컬 캐시 필요 (transformers 라이브러리 제약)
            # 임시로 /tmp에 다운로드 후 로드
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                tmp_path = tmp.name
                stream = self.s3_manager.stream_model(key)
                tmp.write(stream.read())

            model = self._load_huggingface(tmp_path)
            Path(tmp_path).unlink()  # 임시 파일 삭제
            return model

    def _load_from_local(self, local_path: str, model_type: str) -> Any:
        """로컬 파일에서 모델 로드

        Args:
            local_path: 로컬 파일 경로
            model_type: 모델 타입

        Returns:
            로드된 모델
        """
        if model_type == "mtp_native":
            return self._load_mtp_native_local(local_path)
        elif model_type in ["checkpoint", "sheared_llama", "starling_rm"]:
            return self._load_checkpoint_local(local_path)
        else:
            return self._load_huggingface(local_path)

    def _load_mtp_native(self, stream: io.BytesIO) -> Any:
        """MTP Native 모델 로드 (스트리밍)

        Args:
            stream: 모델 데이터 스트림

        Returns:
            로드된 MTP 모델
        """
        # Facebook MTP native 형식 로드
        checkpoint = torch.load(stream, map_location=self.device)

        # MTP 구조 검증
        if "model" in checkpoint:
            model_state = checkpoint["model"]
        else:
            model_state = checkpoint

        # 모델 구조 생성 (실제 구현은 MTP 모델 정의 필요)
        # 여기서는 state_dict만 반환 (실제 사용시 모델 클래스와 결합)
        return model_state

    def _load_mtp_native_local(self, path: str) -> Any:
        """MTP Native 모델 로드 (로컬)

        Args:
            path: 로컬 파일 경로

        Returns:
            로드된 MTP 모델
        """
        checkpoint = torch.load(path, map_location=self.device)

        if "model" in checkpoint:
            return checkpoint["model"]
        return checkpoint

    def _load_checkpoint(self, stream: io.BytesIO) -> Any:
        """체크포인트 로드 (스트리밍)

        Args:
            stream: 체크포인트 데이터 스트림

        Returns:
            로드된 체크포인트
        """
        return torch.load(stream, map_location=self.device)

    def _load_checkpoint_local(self, path: str) -> Any:
        """체크포인트 로드 (로컬)

        Args:
            path: 로컬 파일 경로

        Returns:
            로드된 체크포인트
        """
        return torch.load(path, map_location=self.device)

    def _load_huggingface(self, path: str) -> Any:
        """HuggingFace 모델 로드

        Args:
            path: 모델 경로 또는 ID

        Returns:
            로드된 HuggingFace 모델
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

        # HuggingFace 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=quantization_config,
            device_map="auto" if self.device != "cpu" else None,
            torch_dtype=self._get_torch_dtype(),
            trust_remote_code=True,
        )

        return model

    def _resolve_device(self, devices_config: dict) -> str:
        """디바이스 설정 해석

        Args:
            devices_config: 디바이스 설정

        Returns:
            PyTorch 디바이스 문자열
        """
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
        """Mixed precision에 따른 torch dtype 반환

        Returns:
            torch dtype
        """
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
