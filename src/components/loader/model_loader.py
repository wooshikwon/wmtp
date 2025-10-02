"""
ModelLoader V2: 순차적이고 직관적인 모델 로더

4단계 순차 프로세스:
1. 메타데이터 로드
2. S3 다운로드 (필요시)
3. 로딩 전략 결정
4. 모델 로드 (양자화 및 state dict 매핑 포함)
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

from src.components.loader.base_loader import ModelLoader as BaseModelLoader
from src.components.registry import loader_registry
from src.utils import get_console_output
from src.utils.path_resolver import PathResolver
from src.utils.s3 import create_s3_manager


@loader_registry.register(
    "standardized-model-loader",
    version="4.0.0",
    description="Sequential and intuitive model loader",
)
class ModelLoader(BaseModelLoader):
    """4단계 프로세스로 모델을 로드합니다.
    각 단계는 독립적이며 순차적으로 실행됩니다.
    """

    def __init__(self, config: dict[str, Any]):
        """초기화: 필수 설정만 저장"""
        super().__init__(config)

        # 경로와 S3 설정
        self.model_path = config.get("model_path")
        self.path_resolver = PathResolver()
        self.s3_manager = create_s3_manager(config)

        # 양자화 설정
        self.use_4bit = config.get("use_4bit", False)
        self.use_8bit = config.get("use_8bit", False)
        devices_config = config.get("devices", {})
        self.mixed_precision = devices_config.get("mixed_precision", "fp32")

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """메인 실행 메서드"""
        model_path = inputs.get("model_path", self.model_path)
        if not model_path:
            raise ValueError("model_path is required")

        # 순차적 4단계 실행
        model = self.load_model(model_path)

        return {
            "model": model,
            "path": model_path,
            "loader": self.__class__.__name__,
        }

    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        모델 로딩의 전체 흐름을 관리하는 메인 메서드
        4단계를 순차적으로 실행
        """
        # Step 1: 메타데이터 로드
        metadata, local_path = self.step1_load_metadata(model_path)

        # Step 2: S3 다운로드 (필요시)
        local_path = self.step2_download_if_needed(model_path, local_path, metadata)

        # Step 3: 로딩 전략 결정
        strategy = self.step3_determine_strategy(metadata)

        # Step 4: 모델 로드 (전략에 따라)
        if strategy["loader_type"] == "custom_mtp":
            model = self.step4_load_custom_model(local_path, strategy)
        else:
            model = self.step4_load_huggingface_model(local_path, strategy)

        # Step 5: hidden_states 출력 설정 (Critic-WMTP 지원을 위해)
        from src.utils.model_utils import ensure_output_hidden_states

        ensure_output_hidden_states(model)

        return model

    # ============= STEP 1: 메타데이터 로드 =============
    def step1_load_metadata(self, model_path: str) -> tuple[dict, Path | None]:
        """Step 1: 메타데이터를 로드하고 경로 타입 확인"""
        console_out = get_console_output()
        console_out.detail("메타데이터 로드")

        path_type, resolved = self.path_resolver.resolve(model_path)
        metadata = {}
        local_path = None

        if path_type == "s3":
            # S3에서 메타데이터만 먼저 로드
            metadata = self._load_metadata_from_s3(resolved)
        else:
            # 로컬에서 메타데이터 로드
            local_path = Path(resolved)
            metadata_file = local_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

        # 메타데이터가 없으면 기본값 추론
        if not metadata.get("loading_strategy"):
            metadata["loading_strategy"] = self._infer_strategy(metadata)

        return metadata, local_path

    # ============= STEP 2: S3 다운로드 =============
    def step2_download_if_needed(
        self, model_path: str, local_path: Path | None, metadata: dict
    ) -> Path:
        """Step 2: S3 경로인 경우 필요한 파일들을 다운로드"""
        console_out = get_console_output()
        path_type, resolved = self.path_resolver.resolve(model_path)

        if path_type != "s3":
            console_out.detail("로컬 모델 사용")
            return local_path

        console_out.detail("S3에서 모델 다운로드")

        # 임시 디렉토리에 다운로드
        temp_dir = tempfile.mkdtemp()
        local_path = Path(temp_dir) / "model"
        local_path.mkdir(parents=True)

        # 필요한 파일 목록 결정
        strategy = metadata.get("loading_strategy", {})
        required_files = strategy.get(
            "required_files",
            ["config.json", "model.safetensors", "modeling.py", "metadata.json"],
        )

        # S3에서 다운로드
        _, key_prefix = self.path_resolver.extract_bucket_and_key(resolved)
        for filename in required_files:
            self._download_file_from_s3(key_prefix, filename, local_path)

        return local_path

    # ============= STEP 3: 전략 결정 =============
    def step3_determine_strategy(self, metadata: dict) -> dict:
        """Step 3: 메타데이터를 기반으로 로딩 전략 결정"""
        console_out = get_console_output()

        strategy = metadata.get("loading_strategy", {})

        # 기본값 설정
        strategy.setdefault("loader_type", "huggingface")
        strategy.setdefault("state_dict_mapping", {})

        console_out.detail(f"로딩 전략: {strategy['loader_type']}")
        return strategy

    # ============= STEP 4: 모델 로드 =============
    def step4_load_custom_model(self, local_path: Path, strategy: dict) -> Any:
        """Step 4-A: 커스텀 MTP 모델 로드"""
        console_out = get_console_output()
        console_out.detail("커스텀 MTP 모델 로드")

        # 4.1: modeling.py 동적 임포트
        module_file = strategy.get("custom_module_file", "modeling.py")
        modeling_path = local_path / module_file

        spec = importlib.util.spec_from_file_location("custom_modeling", modeling_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        # 4.2: 모델 클래스 찾기
        model_class_name = strategy.get("model_class_name")
        model_class = getattr(custom_module, model_class_name, None)

        if not model_class:
            # MTP 클래스 자동 탐색
            for name in dir(custom_module):
                if "MTP" in name and isinstance(getattr(custom_module, name), type):
                    model_class = getattr(custom_module, name)
                    break

        # 4.3: 모델 인스턴스 생성
        config = AutoConfig.from_pretrained(str(local_path), trust_remote_code=True)
        model = model_class(config)

        # 4.4: State dict 로드 및 매핑
        safetensors_path = local_path / "model.safetensors"
        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path))

            # State dict 매핑 적용
            mapping = strategy.get("state_dict_mapping", {})
            if mapping:
                state_dict = self.apply_state_dict_mapping(state_dict, mapping)

            model.load_state_dict(state_dict, strict=False)

        # 4.5: Meta MTP 모델에 HuggingFace 호환성 패치 적용
        if "llama" in str(local_path).lower() and hasattr(model, "forward"):
            self._patch_meta_mtp_forward(model)

        return model

    def step4_load_huggingface_model(self, local_path: Path, strategy: dict) -> Any:
        """Step 4-B: HuggingFace 모델 로드"""
        console_out = get_console_output()
        console_out.detail("HuggingFace 모델 로드")

        # 4.1: 모델 클래스 결정
        transformers_class = strategy.get("transformers_class", "AutoModelForCausalLM")
        model_class = (
            AutoModel if transformers_class == "AutoModel" else AutoModelForCausalLM
        )

        # 4.2: 양자화 설정
        quantization_config = None
        if self.use_4bit or self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.use_4bit,
                load_in_8bit=self.use_8bit,
                bnb_4bit_compute_dtype=self.get_compute_dtype(),
            )

        # 4.3: 모델 로드
        model = model_class.from_pretrained(
            str(local_path),
            quantization_config=quantization_config,
            device_map=None,  # Trainer가 담당
            torch_dtype=self.get_torch_dtype(),
            trust_remote_code=True,
        )

        return model

    # ============= 유틸리티 메서드 =============
    def apply_state_dict_mapping(self, state_dict: dict, mapping: dict) -> dict:
        """State dict 키 매핑 적용"""
        if not mapping:
            return state_dict

        result = {}
        remove_prefix = mapping.get("remove_prefix", "")
        add_prefix = mapping.get("add_prefix", "")

        for key, value in state_dict.items():
            new_key = key

            if remove_prefix and new_key.startswith(remove_prefix):
                new_key = new_key[len(remove_prefix) :]

            if add_prefix:
                new_key = f"{add_prefix}{new_key}"

            result[new_key] = value

        return result

    def get_torch_dtype(self):
        """Torch dtype 결정"""
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        elif self.mixed_precision == "fp16":
            return torch.float16
        return torch.float32

    def get_compute_dtype(self):
        """양자화 compute dtype 결정"""
        return torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16

    def _patch_meta_mtp_forward(self, model):
        """Meta MTP 모델에 HuggingFace 호환 forward 메서드 패치

        Meta의 Llama MTP 모델은 다른 인터페이스를 사용:
        - Meta: forward(tokens, start_pos, return_all_heads)
        - HF: forward(input_ids, attention_mask, output_hidden_states, ...)

        이 패치는 런타임에 HF 호환성을 추가하여 WMTP 시스템과 통합합니다.
        """
        original_forward = model.forward
        model._original_forward = original_forward  # 원본 보존

        def hf_compatible_forward(
            input_ids=None,
            attention_mask=None,
            tokens=None,
            start_pos=None,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
            **kwargs,
        ):
            """HuggingFace 호환 forward 인터페이스"""

            # HF 스타일 입력 처리
            if input_ids is not None:
                tokens = input_ids
                start_pos = 0 if start_pos is None else start_pos

                # Meta MTP forward 호출
                logits = original_forward(
                    tokens=tokens,
                    start_pos=start_pos,
                    return_all_heads=True,  # 모든 MTP 헤드 사용
                )

                # HF 스타일 출력 형식으로 변환
                if return_dict:
                    result = {
                        "logits": logits,
                    }

                    # hidden_states 추출 시도 (모델이 저장했다면)
                    if output_hidden_states and hasattr(model, "_last_hidden_states"):
                        result["hidden_states"] = model._last_hidden_states
                        result["last_hidden_state"] = model._last_hidden_states[-1]

                    return result  # Return dict directly for trainer compatibility
                else:
                    return logits

            # Meta 스타일 직접 호출
            elif tokens is not None:
                return original_forward(
                    tokens=tokens,
                    start_pos=start_pos if start_pos is not None else 0,
                    **kwargs,
                )
            else:
                raise ValueError("Either input_ids or tokens must be provided")

        # forward 메서드 교체
        model.forward = hf_compatible_forward

        # HF 호환 속성 추가
        if hasattr(model, "params"):
            model.config = type(
                "Config",
                (),
                {
                    "vocab_size": model.params.vocab_size,
                    "hidden_size": model.params.dim,
                    "n_layers": model.params.n_layers,
                    "n_heads": model.params.n_heads,
                    "output_hidden_states": False,
                    "output_attentions": False,
                },
            )()

        return model

    def _infer_strategy(self, metadata: dict) -> dict:
        """메타데이터가 없을 때 전략 추론"""
        wmtp_type = metadata.get("wmtp_type", "base_model")

        if wmtp_type == "base_model":
            return {
                "loader_type": "custom_mtp",
                "model_class_name": "GPTMTPForCausalLM",
                "custom_module_file": "modeling.py",
            }
        else:
            return {
                "loader_type": "huggingface",
                "transformers_class": "AutoModelForCausalLM",
            }

    def _load_metadata_from_s3(self, s3_path: str) -> dict:
        """S3에서 metadata.json만 로드"""
        if not self.s3_manager:
            return {}

        try:
            bucket, key_prefix = self.path_resolver.extract_bucket_and_key(s3_path)
            metadata_key = f"{key_prefix.rstrip('/')}/metadata.json"
            stream = self.s3_manager.stream_dataset(metadata_key)
            return json.load(stream)
        except Exception:
            return {}

    def _download_file_from_s3(self, key_prefix: str, filename: str, local_path: Path):
        """S3에서 단일 파일 다운로드"""
        if not self.s3_manager:
            return

        try:
            file_key = f"{key_prefix.rstrip('/')}/{filename}"

            # 파일 타입에 따라 적절한 스트림 메서드 사용
            if filename.endswith((".json", ".py")):
                stream = self.s3_manager.stream_dataset(file_key)
            else:
                stream = self.s3_manager.stream_model(file_key)

            with open(local_path / filename, "wb") as f:
                f.write(stream.read())
        except Exception as e:
            # 필수 파일이 아니면 경고만
            if filename in ["config.json", "model.safetensors"]:
                raise RuntimeError(f"Required file {filename} not found: {e}") from e
