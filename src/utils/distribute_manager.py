"""
WMTP 분산 학습 유틸리티: FSDP 및 멀티 GPU 환경 관리

WMTP 연구 맥락:
WMTP의 대규모 모델 학습을 위해 PyTorch FSDP(Fully Sharded Data Parallel)를
활용합니다. 7B 이상 모델의 효율적 학습을 위해 모델/옵티마이저 상태를
여러 GPU에 분산하고, 메모리 최적화와 통신 효율성을 극대화합니다.

핵심 기능:
- FSDP 설정: 모델 샤딩, mixed precision, activation checkpointing
- 분산 환경 설정: NCCL 백엔드, 프로세스 그룹 초기화
- 체크포인트 관리: FSDP 상태 저장/복구
- 성능 측정: 처리량(tokens/sec) 계산

WMTP 알고리즘과의 연결:
- Baseline MTP: 표준 FSDP 설정으로 N-head 병렬 학습
- Critic-WMTP: Value/Base 모델 분리 샤딩으로 메모리 효율 극대화
- Rho1-WMTP: Ref 모델 CPU 오프로드로 GPU 메모리 절약

성능 최적화:
- bf16 mixed precision: 메모리 50% 절감, 속도 2배 향상
- Activation checkpointing: 메모리 30% 추가 절감
- Hybrid sharding: 노드 내 full shard, 노드 간 replicate

디버깅 팁:
- NCCL 타임아웃: timeout 증가 (기본 1800초)
- OOM 오류: activation_ckpt=True, cpu_offload=True 활성화
- 통신 오류: NCCL_DEBUG=INFO 환경변수로 디버깅
"""

import datetime
import os
import random
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
import torch.distributed as dist
from rich.console import Console
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Type checking imports - Pydantic Config 클래스들을 타입 힌트용으로만 import
if TYPE_CHECKING:
    from src.settings.config_schema import DistributedConfig, FSDPConfig

console = Console()


# FSDPConfig dataclass 삭제 - config_schema.py의 Pydantic 버전 사용


class DistributedManager:
    """
    분산 학습 매니저: FSDP 초기화 및 프로세스 그룹 관리.

    WMTP 연구 맥락:
    WMTP의 대규모 실험을 위한 분산 학습 인프라를 제공합니다.
    특히 Critic-WMTP의 듀얼 모델과 Rho1-WMTP의 트리플 모델 학습시
    효율적인 GPU 메모리 관리가 중요합니다.

    주요 책임:
    - PyTorch 분산 환경 초기화
    - FSDP 모델 래핑 및 설정
    - 체크포인트 저장/로드
    - 프로세스 간 통신 및 동기화

    WMTP 특화 기능:
    - Critic: Value 모델과 Base 모델 분리 FSDP 래핑
    - Rho1: Ref 모델 CPU 오프로드로 GPU 메모리 확보
    - 알고리즘별 최적 샤딩 전략 자동 선택
    """

    def __init__(self, config: Union["DistributedConfig", None] = None):
        """Config 기반 분산 매니저 초기화.

        Args:
            config: 분산 학습 설정. None이면 기본값 사용 (분산 비활성화)
        """
        # Config 처리 - 늦은 import로 순환 import 방지
        if config is None:
            from src.settings.config_schema import DistributedConfig

            self.config = DistributedConfig()  # 기본값 제공 (enabled=False)
        else:
            self.config = config

        self.initialized = False

        # Config가 분산을 활성화한 경우에만 환경변수 읽기
        if self.config.enabled:
            # torchrun이 설정한 환경변수를 읽어서 분산 정보 획득
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

            # 환경변수가 제대로 설정되지 않은 경우 경고
            if self.world_size == 1:
                console.print(
                    "[yellow]경고: 분산이 활성화되었지만 WORLD_SIZE=1입니다. torchrun을 사용했는지 확인하세요.[/yellow]"
                )
        else:
            # 단일 GPU 모드
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        self.device = None  # 현재 디바이스
        self.accelerator = None  # HuggingFace Accelerate (선택적)

    def setup(self) -> None:
        """Config 기반 분산 학습 환경 초기화."""
        if not self.config.enabled:
            # 분산 학습이 비활성화된 경우
            console.print("[dim]분산 학습 비활성화 - 단일 GPU 모드[/dim]")
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
            return

        if self.world_size > 1 and not dist.is_initialized():
            # 분산 환경에서 초기화

            # LOCAL_RANK 기반 GPU 설정 (핵심!)
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
                console.print(f"[green]GPU {self.local_rank} 할당 완료[/green]")
            else:
                self.device = torch.device("cpu")

            # Config에서 백엔드 설정 읽기
            backend = self.config.backend
            if backend == "auto":
                backend = "nccl" if torch.cuda.is_available() else "gloo"

            # 분산 프로세스 그룹 초기화
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method=self.config.init_method,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=datetime.timedelta(seconds=self.config.timeout),
                )
                self.initialized = True
                console.print(
                    f"[green]✅ 분산 초기화 완료 (rank={self.rank}/{self.world_size}, backend={backend})[/green]"
                )
            except Exception as e:
                console.print(f"[red]분산 초기화 실패: {e}[/red]")
                raise

        elif self.world_size == 1:
            # 단일 프로세스 환경 (분산 활성화되었지만 실제로는 단일 GPU)
            console.print(
                "[yellow]분산이 활성화되었지만 단일 프로세스 환경입니다.[/yellow]"
            )
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")

    def setup_fsdp(
        self,
        model: torch.nn.Module,
        config: Union["FSDPConfig", dict[str, Any]],
    ) -> FSDP:
        """
        모델을 FSDP로 래핑.

        WMTP 맥락:
        대규모 모델의 파라미터와 그래디언트를 여러 GPU에 분산하여
        메모리 효율성을 극대화합니다.

        매개변수:
            model: 래핑할 모델
            config: FSDP 설정 (Pydantic FSDPConfig 또는 dict)

        반환값:
            FSDP로 래핑된 모델

        최적화 팁:
            - 7B 모델: activation_ckpt=True 필수
            - OOM 시: cpu_offload=True 활성화
            - 속도 우선: sharding="shard_grad_op"
        """
        # Pydantic 모델인 경우 dict로 변환
        if hasattr(config, "model_dump"):
            # Pydantic v2 모델
            config_dict = config.model_dump()
        elif hasattr(config, "dict"):
            # Pydantic v1 모델 (호환성)
            config_dict = config.dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # 이제 config_dict를 사용하여 FSDP 설정

        # Set up mixed precision
        mixed_precision_policy = self._get_mixed_precision(
            config_dict.get("mixed_precision", "bf16")
        )

        # Set up sharding strategy
        sharding_strategy = self._get_sharding_strategy(
            config_dict.get("sharding", "full")
        )

        # Set up auto wrap policy
        auto_wrap_policy = None
        if config_dict.get("auto_wrap", True):
            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls={LlamaDecoderLayer},
            )

        # Set up CPU offload
        cpu_offload_config = None
        if config_dict.get("cpu_offload", False):
            cpu_offload_config = CPUOffload(offload_params=True)

        # Wrap model with FSDP
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=cpu_offload_config,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE
            if config_dict.get("backward_prefetch", True)
            else None,
            sync_module_states=config_dict.get("sync_module_states", True),
            use_orig_params=config_dict.get("use_orig_params", True),
            device_id=self.local_rank
            if torch.cuda.is_available() and self.world_size > 1
            else None,
        )

        if config_dict.get("activation_ckpt", True):
            self._enable_activation_checkpointing(model)

        console.print("[green]Model wrapped with FSDP[/green]")
        return model

    def _get_mixed_precision(self, dtype_str: str) -> MixedPrecision:
        """혼합 정밀도 설정 생성."""
        if dtype_str == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif dtype_str == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:  # fp32
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

    def _get_sharding_strategy(self, strategy: str) -> ShardingStrategy:
        """FSDP 샤딩 전략 선택."""
        strategy_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
        }
        return strategy_map.get(strategy, ShardingStrategy.FULL_SHARD)

    def _enable_activation_checkpointing(self, model: FSDP) -> None:
        """활성화 체크포인팅 활성화로 메모리 효율 개선."""
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )

            check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=check_fn,
            )
            console.print("[green]Activation checkpointing enabled[/green]")
        except ImportError:
            console.print("[yellow]Activation checkpointing not available[/yellow]")

    def save_checkpoint(
        self,
        model: FSDP | torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
        epoch: int,
        step: int,
        mlflow_manager=None,
        **kwargs,
    ) -> None:
        """
        체크포인트 저장 (FSDP/non-FSDP 모델 모두 지원).

        WMTP 맥락:
        학습 중간 상태를 저장하여 재개 가능하게 합니다.
        특히 장시간 학습이 필요한 대규모 모델에서 중요합니다.
        S3 경로 지원 및 MLflow 자동 업로드 기능이 추가되었습니다.

        매개변수:
            model: FSDP 래핑된 모델 또는 일반 torch.nn.Module
            optimizer: 옵티마이저
            checkpoint_path: 저장 경로 (로컬 또는 s3://)
            epoch: 현재 에폭
            step: 현재 스텝
            mlflow_manager: MLflow 매니저 (선택적)
            **kwargs: 추가 저장 데이터 (loss, metrics 등)

        주의사항:
            - rank0_only=True로 메인 프로세스만 저장
            - offload_to_cpu=True로 GPU 메모리 절약
            - S3 경로시 직접 업로드, 로컬 경로시 파일 저장 후 MLflow 업로드
        """
        if self.is_main_process():
            # 모델 타입에 따른 분기 처리
            if isinstance(model, FSDP):
                # 기존 FSDP 로직 유지
                save_policy = FullStateDictConfig(
                    offload_to_cpu=True,
                    rank0_only=True,
                )

                with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    state_dict = model.state_dict()
            else:
                # 일반 모델 처리 (신규 추가)
                state_dict = model.state_dict()
                # CPU로 이동 (메모리 효율성)
                if hasattr(model, "device") and str(model.device) != "cpu":
                    state_dict = {k: v.cpu() for k, v in state_dict.items()}

            # 공통 체크포인트 구성 (기존 로직 유지)
            checkpoint = {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
                **kwargs,
            }

            # S3 또는 로컬 저장 처리
            if checkpoint_path.startswith("s3://"):
                # S3에 직접 저장
                import io
                import tempfile
                from pathlib import Path

                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                buffer.seek(0)

                if mlflow_manager:
                    # MLflow를 통해 아티팩트로 업로드 (임시 파일 경유)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir) / f"checkpoint_step_{step}.pt"
                        with open(tmp_path, "wb") as f:
                            f.write(buffer.getvalue())
                        mlflow_manager.log_artifact(
                            local_path=str(tmp_path),
                            artifact_path=f"checkpoints/step_{step}",
                        )
                    console.print(
                        f"[green]Checkpoint uploaded to MLflow: step_{step}[/green]"
                    )
                else:
                    # S3Manager를 사용하여 직접 저장
                    from src.utils.s3 import S3Manager

                    s3_manager = S3Manager()
                    s3_key = checkpoint_path.replace("s3://wmtp/", "")
                    s3_manager.upload_from_bytes(buffer.getvalue(), s3_key)
                    console.print(
                        f"[green]Checkpoint saved to S3: {checkpoint_path}[/green]"
                    )
            else:
                # 로컬 저장
                torch.save(checkpoint, checkpoint_path)
                console.print(
                    f"[green]Checkpoint saved locally: {checkpoint_path}[/green]"
                )

                # MLflow에도 기록 (있는 경우)
                if mlflow_manager:
                    mlflow_manager.log_artifact(
                        local_path=checkpoint_path, artifact_path="checkpoints"
                    )

        self.barrier()

    def load_checkpoint(
        self,
        model: FSDP | torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """
        체크포인트 로드 (FSDP/non-FSDP 모델 모두 지원).

        매개변수:
            model: FSDP 래핑된 모델 또는 일반 torch.nn.Module
            optimizer: 옵티마이저
            checkpoint_path: 체크포인트 경로 (로컬 또는 s3://)

        반환값:
            체크포인트 딕셔너리 (epoch, step 등 포함)
        """
        # S3 또는 로컬 로드 처리
        if checkpoint_path.startswith("s3://"):
            # S3에서 직접 로드
            from src.utils.s3 import S3Manager

            s3_manager = S3Manager()
            s3_key = checkpoint_path.replace("s3://wmtp/", "")
            checkpoint_bytes = s3_manager.stream_model(s3_key)
            checkpoint = torch.load(
                checkpoint_bytes,
                map_location=self.device,
            )
        else:
            # 로컬 파일 로드
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
            )

        # 모델 타입에 따른 분기 처리
        if isinstance(model, FSDP):
            # 기존 FSDP 로직 유지
            load_policy = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=False,
            )

            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, load_policy
            ):
                model.load_state_dict(checkpoint["model"])
        else:
            # 일반 모델 처리 (신규 추가)
            model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        console.print(f"[green]Checkpoint loaded from {checkpoint_path}[/green]")
        return checkpoint

    def barrier(self) -> None:
        """모든 프로세스 동기화."""
        if self.world_size > 1:
            dist.barrier()

    def is_main_process(self) -> bool:
        """메인 프로세스 여부 확인."""
        return self.rank == 0

    # all_reduce와 cleanup 메서드 삭제됨 - Phase 2 리팩토링
    # 이 메서드들은 전혀 호출되지 않아 제거되었습니다.


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    재현성을 위한 랜덤 시드 설정.

    WMTP 맥락:
    실험 재현성을 보장하여 연구 결과의 신뢰성을 확보합니다.
    동일한 시드로 동일한 결과를 얻을 수 있습니다.

    매개변수:
        seed: 랜덤 시드 값
        deterministic: 결정적 알고리즘 사용 여부
            - True: 완전한 재현성 (느림)
            - False: 빠른 속도 (약간의 차이 가능)

    주의사항:
        deterministic=True는 cudnn.benchmark를 비활성화하여
        학습 속도가 크게 감소할 수 있습니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster but less reproducible
        torch.backends.cudnn.benchmark = True

    console.print(f"[green]Random seed set to {seed}[/green]")


# 미사용 함수 get_world_info, compute_throughput 삭제됨 - Phase 1 리팩토링


# 전역 분산 매니저 인스턴스
_dist_manager: DistributedManager | None = None


def get_dist_manager(
    config: Union["DistributedConfig", None] = None,
) -> DistributedManager:
    """Config 기반 분산 매니저 싱글톤 생성.

    Args:
        config: 분산 설정. None이면 기존 인스턴스 반환 또는 기본값 생성

    Returns:
        DistributedManager: 설정된 분산 매니저
    """
    global _dist_manager

    # 새 config가 제공되거나 아직 인스턴스가 없는 경우
    if _dist_manager is None or config is not None:
        _dist_manager = DistributedManager(config)

    return _dist_manager


# Export main functions and classes
__all__ = [
    "DistributedManager",
    "set_seed",
    "get_dist_manager",
]
