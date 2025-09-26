"""
Checkpoint Loader: 훈련 재개를 위한 체크포인트 로딩 전용 컴포넌트

WMTP Phase 2 리팩토링의 일환으로 training_pipeline.py에 하드코딩된
체크포인트 로딩 로직을 독립적인 컴포넌트로 분리합니다.

기능:
1. S3/로컬 체크포인트 통합 로딩
2. 메타데이터 추출 (epoch, step, mlflow_run_id)
3. Rich Console 기반 진행상황 표시
4. 오류 처리 및 로그

ComponentFactory 통합:
- ComponentFactory.create_checkpoint_loader() 지원
- 기존 UnifiedModelLoader와 분리된 전용 인터페이스
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from src.components.loader.base_loader import BaseLoader
from src.components.registry import loader_registry
from src.utils.path_resolver import PathResolver
from src.utils.s3 import create_s3_manager


@loader_registry.register(
    "checkpoint-loader",
    version="1.0.0",
    description="Specialized loader for training checkpoints with metadata extraction",
)
class CheckpointLoader(BaseLoader):
    """훈련 체크포인트 전용 로더

    UnifiedModelLoader와 차별화된 전용 기능:
    1. 메타데이터 자동 추출: epoch, step, mlflow_run_id
    2. 재개 전용 인터페이스: 훈련 재개에 필요한 모든 정보 제공
    3. 상세한 로깅: Rich Console을 통한 시각적 피드백
    4. 견고한 오류 처리: 체크포인트 손상/부재 시 명확한 메시지
    """

    def __init__(self, config: dict[str, Any]):
        """체크포인트 로더 초기화

        Args:
            config: 환경 설정
                - storage: S3 설정
                - paths: 로컬 경로 설정
                - devices: GPU/CPU 설정
        """
        super().__init__(config)

        # 의존성 초기화
        self.path_resolver = PathResolver()
        self.s3_manager = create_s3_manager(config)
        self.console = Console()

        # 디바이스 설정
        devices_config = config.get("devices", {})
        self.device = self._resolve_device(devices_config)

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """체크포인트 로딩 실행

        Args:
            inputs: 입력 딕셔너리
                - model_path: 체크포인트 경로 (로컬/S3 URI/문자열)
                - load_metadata: 메타데이터 추출 여부 (기본: True)

        Returns:
            체크포인트 데이터와 메타데이터
                - checkpoint_data: 원본 체크포인트 데이터
                - epoch: 훈련 에포크 (기본: 0)
                - step: 훈련 스텝 (기본: 0)
                - mlflow_run_id: MLflow 실행 ID (선택적)
                - path: 원본 경로
                - loader: 로더 클래스명
        """
        checkpoint_path = inputs.get("model_path")
        if not checkpoint_path:
            raise ValueError("model_path is required for checkpoint loading")

        load_metadata = inputs.get("load_metadata", True)

        try:
            # 경로 타입별 로딩
            checkpoint_data = self._load_checkpoint(checkpoint_path)

            if checkpoint_data is None:
                self.console.print(
                    f"[yellow]Checkpoint not found: {checkpoint_path}[/yellow]"
                )
                return self._create_empty_result(checkpoint_path)

            # 메타데이터 추출
            metadata = self._extract_metadata(checkpoint_data) if load_metadata else {}

            # 성공 로그
            if metadata:
                self.console.print(
                    f"[green]Resuming from epoch {metadata.get('epoch', 0)}, "
                    f"step {metadata.get('step', 0)}[/green]"
                )

            return {
                "checkpoint_data": checkpoint_data,
                **metadata,
                "path": checkpoint_path,
                "loader": self.__class__.__name__,
            }

        except Exception as e:
            self.console.print(f"[red]Failed to load checkpoint: {e}[/red]")
            return self._create_empty_result(checkpoint_path, error=str(e))

    def _load_checkpoint(self, checkpoint_path: Any) -> dict[str, Any] | None:
        """경로 타입에 따른 체크포인트 로딩

        Args:
            checkpoint_path: 체크포인트 경로 (다양한 타입 지원)

        Returns:
            로드된 체크포인트 데이터 또는 None
        """
        # S3 URI 처리
        if isinstance(checkpoint_path, str) and checkpoint_path.startswith("s3://"):
            return self._load_from_s3(checkpoint_path)

        # Path 객체 처리
        elif hasattr(checkpoint_path, "exists"):
            if checkpoint_path.exists():
                return self._load_from_local(checkpoint_path)
            return None

        # 문자열 경로 처리
        elif isinstance(checkpoint_path, str):
            local_path = Path(checkpoint_path)
            if local_path.exists():
                return self._load_from_local(local_path)
            return None

        else:
            raise ValueError(
                f"Unsupported checkpoint path type: {type(checkpoint_path)}"
            )

    def _load_from_s3(self, s3_uri: str) -> dict[str, Any] | None:
        """S3에서 체크포인트 스트리밍 로드

        Args:
            s3_uri: S3 URI (s3://bucket/key)

        Returns:
            로드된 체크포인트 또는 None
        """
        if not self.s3_manager or not self.s3_manager.connected:
            raise RuntimeError("S3 manager not available for checkpoint loading")

        try:
            # S3 키 추출
            s3_key = s3_uri.replace("s3://wmtp/", "")

            # 스트리밍 로드
            self.console.print(f"[blue]Streaming checkpoint from S3: {s3_uri}[/blue]")
            checkpoint_bytes = self.s3_manager.stream_model(s3_key)
            checkpoint_data = torch.load(checkpoint_bytes, map_location=self.device)

            self.console.print("[green]Successfully loaded checkpoint from S3[/green]")
            return checkpoint_data

        except Exception as e:
            self.console.print(f"[red]S3 checkpoint loading failed: {e}[/red]")
            return None

    def _load_from_local(self, local_path: Path) -> dict[str, Any]:
        """로컬 파일에서 체크포인트 로드

        Args:
            local_path: 로컬 파일 경로

        Returns:
            로드된 체크포인트
        """
        self.console.print(f"[blue]Loading checkpoint from local: {local_path}[/blue]")
        checkpoint_data = torch.load(local_path, map_location=self.device)
        self.console.print("[green]Successfully loaded local checkpoint[/green]")
        return checkpoint_data

    def _extract_metadata(self, checkpoint_data: dict[str, Any]) -> dict[str, Any]:
        """체크포인트에서 훈련 메타데이터 추출

        Args:
            checkpoint_data: 로드된 체크포인트

        Returns:
            추출된 메타데이터
        """
        metadata = {}

        # 표준 훈련 메타데이터 추출
        if isinstance(checkpoint_data, dict):
            metadata["epoch"] = checkpoint_data.get("epoch", 0)
            metadata["step"] = checkpoint_data.get("step", 0)
            metadata["mlflow_run_id"] = checkpoint_data.get("mlflow_run_id")

            # 선택적 메타데이터
            if "loss" in checkpoint_data:
                metadata["last_loss"] = checkpoint_data["loss"]
            if "lr" in checkpoint_data:
                metadata["last_lr"] = checkpoint_data["lr"]

        return metadata

    def _create_empty_result(self, path: Any, error: str = None) -> dict[str, Any]:
        """빈 결과 생성 (체크포인트 없음)

        Args:
            path: 원래 요청된 경로
            error: 오류 메시지 (선택적)

        Returns:
            빈 결과 딕셔너리
        """
        return {
            "checkpoint_data": None,
            "epoch": 0,
            "step": 0,
            "mlflow_run_id": None,
            "path": path,
            "loader": self.__class__.__name__,
            "error": error,
        }

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

    def load(self, path: str, **kwargs) -> dict[str, Any]:
        """BaseLoader 인터페이스 구현

        Args:
            path: 체크포인트 경로
            **kwargs: 추가 파라미터

        Returns:
            체크포인트와 메타데이터
        """
        inputs = {"model_path": path, **kwargs}
        return self.run(inputs)
