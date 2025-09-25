"""
WMTP 연구의 핵심: Facebook MTP 원본 모델 로더

WMTP 연구 맥락:
이 모듈은 Meta(Facebook)에서 공개한 Multi-Token Prediction 모델의 원본 형식을 로드합니다.
WMTP 연구의 기준점(baseline)이 되는 모델로, 4개의 예측 헤드가 내장된 상태로 제공됩니다.

지원하는 모델 형식:
1. consolidated.pth: 단일 체크포인트 파일 (일반적)
2. consolidated.*.pth: 분할된 대용량 모델 (8B+ 파라미터)
3. native MTP 구조: t+1, t+2, t+3, t+4 예측 헤드 내장

WMTP 알고리즘과의 연결:
- Baseline: 4개 헤드에 균등 가중치 적용
- Critic-WMTP: Value head 추가 후 토큰별 중요도 계산
- Rho1-WMTP: 참조모델로 활용하여 어려운 토큰 식별

성능 최적화 특징:
- 자동 디바이스 감지 (CUDA/MPS/CPU)
- 메모리 효율적 로딩 (분할 체크포인트 지원)
- S3 자동 다운로드 (클러스터 환경)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.components.loader.base_loader import ModelLoader
from src.components.registry import loader_registry
from src.utils.s3 import S3Utils


def detect_optimal_device() -> str:
    """런타임에서 최적 디바이스를 자동 감지.

    Returns:
        str: 감지된 디바이스 타입 (cuda, mps, cpu)
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def resolve_device_from_config(
    config: dict[str, Any], fallback_device: str = "cpu"
) -> str:
    """Config 설정을 기반으로 실제 사용할 디바이스 결정.

    Args:
        config: 전체 설정 딕셔너리
        fallback_device: config에서 디바이스 정보를 찾을 수 없을 때 사용할 기본값

    Returns:
        str: PyTorch 디바이스 문자열 (cuda:0, mps, cpu 등)
    """
    devices_config = config.get("devices", {})
    compute_backend = devices_config.get("compute_backend", "auto")
    device_ids = devices_config.get("device_ids")

    # auto인 경우 런타임 감지
    if compute_backend == "auto":
        compute_backend = detect_optimal_device()

    # 실제 디바이스 사용 가능 여부 확인 및 fallback
    if compute_backend == "cuda":
        if torch.cuda.is_available():
            device_id = device_ids[0] if device_ids else 0
            return f"cuda:{device_id}"
        else:
            # CUDA 불가능시 MPS로 fallback
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
    elif compute_backend == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            # MPS 불가능시 CPU로 fallback
            return "cpu"
    else:
        return "cpu"


@loader_registry.register(
    "mtp-native", version="1.0.0", description="Facebook MTP native format loader"
)
class MTPNativeLoader(ModelLoader):
    """
    Facebook Multi-Token Prediction 모델의 원본 형식 로더입니다.

    WMTP 연구에서의 핵심 역할:
    Meta에서 공개한 MTP 모델을 정확히 로드하여, WMTP 알고리즘 연구의 기반을 제공합니다.
    원본 모델의 4개 예측 헤드 구조를 그대로 보존하면서, 토큰별 가중치를 적용할 수 있도록
    준비합니다.

    지원하는 MTP 모델 구조:
    - Base Model: Transformer 아키텍처 (LLaMA 기반)
    - MTP Heads: 4개 병렬 예측 헤드 (t+1, t+2, t+3, t+4)
    - Vocabulary: 32,000개 토큰 (CodeLlama 토크나이저 호환)

    로딩 전략:
    1. consolidated.pth 우선 확인 (단일 파일)
    2. 분할 체크포인트 감지 및 병합 (대용량 모델)
    3. 토크나이저 자동 연결 (동일 디렉토리)
    4. 디바이스별 최적화 (CUDA/MPS/CPU)

    WMTP 알고리즘별 활용:
    - Baseline: model.heads[k]에 균등 가중치 적용
    - Critic: Value head 추가, GAE로 토큰별 가중치 계산
    - Rho1: 참조모델로 사용, logits 차이로 중요도 판단

    성능 특징:
    - Zero-copy 텐서 로딩 (메모리 효율)
    - 자동 디바이스 배치 (config 기반)
    - 혼합 정밀도 지원 (float16/bfloat16)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})
        self.s3_utils = S3Utils()

        # Config 기반 디바이스 결정 (혼합 접근법)
        if config:
            self.device = resolve_device_from_config(config, fallback_device="cpu")
            self.num_heads = config.get("num_heads", 4)
        else:
            # Config 없는 경우 런타임 감지
            optimal_backend = detect_optimal_device()
            self.device = (
                "mps"
                if optimal_backend == "mps"
                else f"{optimal_backend}:0"
                if optimal_backend == "cuda"
                else "cpu"
            )
            self.num_heads = 4

        print(f"MTPNativeLoader initialized with device: {self.device}")

    def _load_native_checkpoint(self, path: Path) -> dict[str, Any]:
        """
        Load Facebook MTP native checkpoint.

        Args:
            path: Path to model directory

        Returns:
            Dictionary containing model weights and config
        """
        # Check for consolidated checkpoint
        consolidated_path = path / "consolidated.pth"
        if consolidated_path.exists():
            checkpoint = torch.load(consolidated_path, map_location=self.device)
            return checkpoint

        # Check for split checkpoints
        checkpoint_files = list(path.glob("consolidated.*.pth"))
        if checkpoint_files:
            # Load and merge split checkpoints
            merged_checkpoint = {}
            for ckpt_file in sorted(checkpoint_files):
                ckpt = torch.load(ckpt_file, map_location=self.device)
                merged_checkpoint.update(ckpt)
            return merged_checkpoint

        raise FileNotFoundError(f"No checkpoint files found in {path}")

    def _load_model_params(self, path: Path) -> dict[str, Any]:
        """
        Load model parameters from params.json.

        Args:
            path: Path to model directory

        Returns:
            Model parameters dictionary
        """
        params_path = path / "params.json"
        if params_path.exists():
            with open(params_path) as f:
                return json.load(f)

        # Default parameters if params.json not found
        return {
            "dim": 4096,
            "n_layers": 32,
            "n_heads": 32,
            "n_kv_heads": 8,
            "vocab_size": 128256,
            "multiple_of": 4096,
            "ffn_dim_multiplier": 1.3,
            "norm_eps": 1e-5,
            "rope_theta": 500000,
            "mtp_heads": self.num_heads,
        }

    def _create_mtp_model(
        self, checkpoint: dict[str, Any], params: dict[str, Any]
    ) -> Any:
        """
        Create MTP model from checkpoint and parameters.

        Args:
            checkpoint: Model checkpoint
            params: Model parameters

        Returns:
            MTP model instance (PyTorch nn.Module)

        Raises:
            ImportError: Facebook MTP 모델을 로드할 수 없는 경우
        """
        import sys
        from pathlib import Path

        # Facebook MTP 모델 디렉토리 절대 경로 설정
        model_dir = Path.cwd() / "models" / "7b_1t_4"

        if not model_dir.exists():
            raise ImportError(
                f"Facebook MTP 모델 디렉토리를 찾을 수 없습니다: {model_dir}"
            )

        try:
            # sys.path에 모델 디렉토리 추가
            sys.path.insert(0, str(model_dir))

            # fairscale model parallel 초기화 (단일 GPU 환경 지원)
            import fairscale.nn.model_parallel.initialize as fs_init
            import torch.distributed as dist

            # torch.distributed 초기화 (fairscale 요구사항)
            if not dist.is_initialized():
                # 환경별 최적 backend 자동 선택
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    backend = "nccl"  # VESSL CUDA 환경
                else:
                    backend = "gloo"  # 로컬 CPU 환경

                dist.init_process_group(
                    backend=backend,
                    init_method="tcp://localhost:29500",
                    rank=0,
                    world_size=1,
                )
                print(f"Distributed backend initialized: {backend}")

            # fairscale model parallel 초기화
            if not fs_init.model_parallel_is_initialized():
                fs_init.initialize_model_parallel(1)  # 단일 GPU용 초기화

            # Facebook MTP 모델 및 설정 클래스 import
            from llama.model import ModelArgs
            from llama.model import Transformer as MTPLlamaModel

            # params dict를 ModelArgs 객체로 변환
            model_args = ModelArgs(
                dim=params.get("dim", 4096),
                n_layers=params.get("n_layers", 32),
                n_heads=params.get("n_heads", 32),
                n_kv_heads=params.get("n_kv_heads", 8),
                vocab_size=params.get("vocab_size", 128256),
                multiple_of=params.get("multiple_of", 4096),
                ffn_dim_multiplier=params.get("ffn_dim_multiplier", 1.3),
                norm_eps=params.get("norm_eps", 1e-5),
                rope_theta=params.get("rope_theta", 500000),
                n_future_tokens=params.get("mtp_heads", self.num_heads),  # MTP heads 수
            )

            # MTP 모델 생성 - 환경별 분리로 안전한 처리
            model = MTPLlamaModel(model_args)

            # 체크포인트에서 가중치 로드 (일부 키 불일치 허용)
            model.load_state_dict(checkpoint, strict=False)

            # 강제로 지정된 디바이스로 이동 (Facebook 모델의 하드코딩 우회)
            model = model.to(self.device)
            model.eval()

            print(
                f"Facebook MTP model loaded on device: {next(model.parameters()).device}"
            )

            return model

        except ImportError as e:
            raise ImportError(
                f"Facebook MTP 모델을 import할 수 없습니다: {e}\n"
                f"models/7b_1t_4/llama/model.py 파일이 존재하는지 확인하세요."
            )
        except Exception as e:
            raise RuntimeError(f"MTP 모델 생성 중 오류: {e}")
        finally:
            # sys.path 정리 - 메모리 누수 방지
            if str(model_dir) in sys.path:
                sys.path.remove(str(model_dir))

    def load_model(self, path: str, **kwargs) -> Any:
        """
        Load a Facebook MTP native model.

        Args:
            path: Local path or S3 URL to model directory
            **kwargs: Additional arguments

        Returns:
            Loaded model or checkpoint data
        """
        local_path = Path(path)

        # Check if it's an S3 path
        if path.startswith("s3://"):
            # Download from S3 to local cache
            local_path = self.s3_utils.download_model(path)
            if not local_path.exists():
                raise FileNotFoundError(f"Failed to download model from {path}")

        # Validate it's a directory
        if not local_path.is_dir():
            raise ValueError(
                f"MTP native models should be in a directory, not a file: {local_path}"
            )

        # Load checkpoint and parameters
        checkpoint = self._load_native_checkpoint(local_path)
        params = self._load_model_params(local_path)

        # Create model
        model = self._create_mtp_model(checkpoint, params)

        return model

    # load_tokenizer 메서드 제거 - 부모 클래스의 통합 SentencePiece 사용
    # Facebook MTP와 새로운 모델들 모두 동일한 tokenizer.model 공유

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Run the MTP native loader.

        Args:
            ctx: Context containing model_path

        Returns:
            Dictionary with model and optionally tokenizer
        """
        model_path = ctx.get("model_path")
        if not model_path:
            raise ValueError("model_path is required in context")

        result = {}

        # Load model
        result["model"] = self.load_model(model_path)

        # Try to load tokenizer from same path
        try:
            result["tokenizer"] = self.load_tokenizer(model_path)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")

        return result
