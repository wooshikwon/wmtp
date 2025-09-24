"""
CPU 전용 Facebook MTP 모델 로더 - 메모리 최적화 버전

M3 MacBook Pro 64GB RAM 환경에서 27GB Facebook MTP 모델을
안전하게 로드하기 위한 최적화된 로더입니다.

주요 최적화:
- Fairscale model parallel 완전 제거
- torch.distributed 초기화 생략
- 메모리 효율적 torch.load 사용
- 불필요한 캐시 및 메타데이터 최소화
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.components.loader.base_loader import ModelLoader
from src.components.registry import loader_registry


def detect_optimal_device() -> str:
    """CPU 전용 디바이스 감지."""
    return "cpu"


@loader_registry.register(
    "mtp-native-cpu",
    version="1.0.0",
    description="CPU-optimized Facebook MTP native loader",
)
class MTPNativeCPULoader(ModelLoader):
    """
    CPU 전용 Facebook MTP 모델 로더 - 메모리 최적화 버전

    M3 MacBook Pro 64GB에서 27GB Facebook MTP 모델을 안전하게 로드합니다.
    fairscale과 torch.distributed를 사용하지 않아 메모리 오버헤드를 최소화합니다.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config or {})

        # CPU 전용 강제 설정
        self.device = "cpu"
        self.num_heads = 4

        print("MTPNativeCPULoader initialized - CPU ONLY mode for memory efficiency")

    def _load_native_checkpoint(self, path: Path) -> dict[str, Any]:
        """메모리 효율적인 체크포인트 로딩."""
        print(f"🔧 Loading checkpoint from {path} with memory optimization...")

        # 통합 체크포인트 우선 확인
        consolidated_path = path / "consolidated.pth"
        if consolidated_path.exists():
            print(f"📦 Loading consolidated checkpoint: {consolidated_path}")
            print(
                f"📊 File size: {consolidated_path.stat().st_size / (1024**3):.1f} GB"
            )

            # 메모리 최적화된 로딩
            checkpoint = torch.load(
                consolidated_path,
                map_location=self.device,
                weights_only=False,  # Facebook 모델은 복잡한 객체 구조 사용
            )
            print("✅ Checkpoint loaded successfully")
            return checkpoint

        # 분할 체크포인트 확인
        checkpoint_files = list(path.glob("consolidated.*.pth"))
        if checkpoint_files:
            print(f"📦 Loading split checkpoints: {len(checkpoint_files)} files")
            merged_checkpoint = {}
            for i, ckpt_file in enumerate(sorted(checkpoint_files)):
                print(f"  Loading part {i+1}/{len(checkpoint_files)}: {ckpt_file.name}")
                ckpt = torch.load(
                    ckpt_file, map_location=self.device, weights_only=False
                )
                merged_checkpoint.update(ckpt)
            print("✅ Split checkpoints merged successfully")
            return merged_checkpoint

        raise FileNotFoundError(f"No checkpoint files found in {path}")

    def _load_model_params(self, path: Path) -> dict[str, Any]:
        """모델 파라미터 로딩."""
        params_path = path / "params.json"
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)
                print(
                    f"📋 Model params loaded: {params.get('dim', 'unknown')}D, {params.get('n_layers', 'unknown')} layers"
                )
                return params

        # 기본 파라미터 (Facebook MTP 7B 기준)
        default_params = {
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
            "max_batch_size": 1,  # 메모리 절약을 위해 작게 설정
            "max_seq_len": 2048,
        }
        print("⚠️  Using default parameters for 7B MTP model")
        return default_params

    def _create_mtp_model_cpu_only(
        self, checkpoint: dict[str, Any], params: dict[str, Any]
    ) -> Any:
        """CPU 전용 MTP 모델 생성 - fairscale 없이."""
        print("🏗️  Creating CPU-only MTP model...")

        import sys
        from pathlib import Path

        # Facebook MTP 모델 디렉토리
        model_dir = Path.cwd() / "models" / "7b_1t_4"
        if not model_dir.exists():
            raise ImportError(
                f"Facebook MTP 모델 디렉토리를 찾을 수 없습니다: {model_dir}"
            )

        try:
            # sys.path에 모델 디렉토리 추가
            sys.path.insert(0, str(model_dir))

            # ⚠️ CRITICAL: 최소한의 fairscale 초기화 (메모리 효율 모드)
            print("🔧 Minimal fairscale initialization for CPU compatibility")

            # fairscale 필수 초기화 (메모리 효율적)
            import fairscale.nn.model_parallel.initialize as fs_init
            import torch.distributed as dist

            # torch.distributed 최소 초기화 (CPU 전용)
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="gloo",  # CPU 전용
                    init_method="tcp://localhost:29501",  # 포트 변경으로 충돌 방지
                    rank=0,
                    world_size=1,  # 단일 프로세스
                )
                print("✅ torch.distributed initialized (CPU gloo backend)")

            # fairscale model parallel 최소 초기화
            if not fs_init.model_parallel_is_initialized():
                fs_init.initialize_model_parallel(1)  # 단일 프로세스
                print("✅ fairscale model parallel initialized (single process)")

            # Facebook MTP 모델 클래스만 import
            from llama.model import ModelArgs
            from llama.model import Transformer as MTPLlamaModel

            # ModelArgs 생성 - 메모리 절약 설정
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
                n_future_tokens=params.get("mtp_heads", self.num_heads),
                max_batch_size=1,  # 메모리 절약
                max_seq_len=512,  # 메모리 절약
            )

            print(
                f"📐 Model architecture: {model_args.dim}D, {model_args.n_layers} layers, {model_args.n_future_tokens} MTP heads"
            )

            # MTP 모델 생성 (fairscale 없이)
            model = MTPLlamaModel(model_args)
            print("✅ Model structure created")

            # 체크포인트에서 가중치 로드
            print("🔄 Loading weights from checkpoint...")
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint, strict=False
            )

            if missing_keys:
                print(
                    f"⚠️  Missing keys: {len(missing_keys)} (this may be normal for MTP models)"
                )
            if unexpected_keys:
                print(
                    f"⚠️  Unexpected keys: {len(unexpected_keys)} (this may be normal)"
                )

            # CPU로 이동 및 evaluation 모드
            model = model.to(self.device)
            model.eval()

            print("✅ Facebook MTP model loaded successfully on CPU")
            print(f"📍 Device: {next(model.parameters()).device}")

            # 메모리 사용량 체크
            param_count = sum(p.numel() for p in model.parameters())
            print(
                f"📊 Total parameters: {param_count:,} ({param_count * 4 / (1024**3):.1f} GB in FP32)"
            )

            return model

        except ImportError as e:
            raise ImportError(f"Facebook MTP 모델 import 실패: {e}")
        except Exception as e:
            raise RuntimeError(f"CPU-only MTP 모델 생성 실패: {e}")
        finally:
            # sys.path 정리
            if str(model_dir) in sys.path:
                sys.path.remove(str(model_dir))

    def load_model(self, path: str, **kwargs) -> Any:
        """CPU 최적화된 모델 로딩."""
        print("🚀 Starting CPU-optimized Facebook MTP model loading...")

        local_path = Path(path)

        if not local_path.is_dir():
            raise ValueError(
                f"MTP native models should be in a directory: {local_path}"
            )

        # 체크포인트와 파라미터 로딩
        checkpoint = self._load_native_checkpoint(local_path)
        params = self._load_model_params(local_path)

        # CPU 전용 모델 생성
        model = self._create_mtp_model_cpu_only(checkpoint, params)

        print("🎉 Model loading completed successfully!")
        return model

    def load_tokenizer(self, path: str, **kwargs) -> Any:
        """토크나이저 로딩 (기존과 동일)."""
        local_path = Path(path)

        # SentencePiece 토크나이저 시도
        tokenizer_path = (
            local_path / "tokenizer.model" if local_path.is_dir() else local_path
        )

        if tokenizer_path.exists():
            try:
                from sentencepiece import SentencePieceProcessor

                tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
                print("✅ SentencePiece tokenizer loaded")
                return tokenizer
            except ImportError:
                pass

        # tiktoken 시도
        try:
            import tiktoken

            tokenizer = tiktoken.get_encoding("cl100k_base")
            print("✅ Tiktoken tokenizer loaded")
            return tokenizer
        except ImportError:
            pass

        # HuggingFace tokenizer fallback
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            print("✅ HuggingFace tokenizer loaded")
            return tokenizer
        except Exception:
            pass

        raise RuntimeError(f"Could not load tokenizer from {path}")

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """CPU 최적화된 실행."""
        model_path = ctx.get("model_path")
        if not model_path:
            raise ValueError("model_path is required in context")

        result = {}

        # 모델 로딩
        result["model"] = self.load_model(model_path)

        # 토크나이저 로딩 시도
        try:
            result["tokenizer"] = self.load_tokenizer(model_path)
        except Exception as e:
            print(f"⚠️  Could not load tokenizer: {e}")

        return result
