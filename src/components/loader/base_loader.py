"""
WMTP 로더 시스템의 기반: 로컬 우선 S3 스트리밍 정책

WMTP 연구 맥락:
이 모듈은 WMTP 실험의 핵심 인프라입니다. 연구자들이 세 가지 알고리즘
(baseline/critic/rho1)을 실험할 때, 동일한 모델과 데이터셋을 안정적으로
로드할 수 있도록 보장합니다.

핵심 철학:
"로컬 파일 → S3 직접 스트리밍" 순서로 효율적인 데이터 접근 (캐시 제거)

지원하는 실험 환경:
- 개발자 로컬 환경: 로컬 파일 직접 사용
- VESSL 클러스터: S3에서 메모리로 직접 스트리밍
- 오프라인 환경: 로컬 파일 사용

WMTP에서의 역할:
1. Facebook Native MTP 모델 안정적 로딩 (consolidated.pth)
2. 코딩 평가 데이터셋 일관된 전처리 (MBPP/CodeContests/HumanEval)
3. 실험 재현성 보장 (동일한 S3 경로 → 동일한 데이터)
4. 메모리 효율성 극대화 (디스크 I/O 제거)
"""

from pathlib import Path
from typing import Any

from rich.console import Console

from ...components.base import BaseComponent
from ...utils.s3 import create_s3_manager

console = Console()


class BaseLoader(BaseComponent):
    """
    모든 WMTP 로더의 추상 기본 클래스입니다 (Cache-Free).

    연구 맥락:
    WMTP 실험에서는 다양한 환경(로컬/클러스터)에서 동일한 데이터에 접근해야 합니다.
    이 클래스는 "로컬 우선 → S3 스트리밍" 정책으로 효율적이고 안정적인 데이터 로딩을 보장합니다.

    동작 원리 (캐시 제거):
    1단계: 로컬 경로 확인 (가장 빠름)
    2단계: S3에서 메모리로 직접 스트리밍 (디스크 캐시 없음)
    3단계: 오류 발생 (모든 경로 실패시)

    WMTP 실험 시나리오:
    - 개발: configs/config.local.yaml + 로컬 파일
    - 클러스터: configs/config.vessl.yaml + S3 직접 스트리밍
    - CI/CD: 로컬 파일 사용

    상속 구조:
    BaseLoader
    ├── DatasetLoader (MBPP, CodeContests, HumanEval)
    └── ModelLoader (Facebook MTP, HuggingFace)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize base loader (cache-free).

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Initialize S3 manager if configured
        self.s3_manager = create_s3_manager(config)

        # Phase 2: storage 필드 완전 제거 - PathResolver가 프로토콜 기반으로 처리

        # Extract path configurations
        self.paths_config = config.get("paths", {})

    def get_local_path(self, path_spec: str) -> Path | None:
        """
        Get local path from configuration.

        Args:
            path_spec: Path specification key

        Returns:
            Path object if exists, None otherwise
        """
        # Check if path is in config
        if path_spec in self.paths_config:
            local_path = Path(self.paths_config[path_spec])
            if local_path.exists():
                return local_path

        # Check if it's a direct path
        direct_path = Path(path_spec)
        if direct_path.exists():
            return direct_path

        return None

    def load_with_streaming(
        self,
        local_path: str | Path | None,
        s3_key: str,
        loader_fn: Any,
        **kwargs,
    ) -> Any:
        """
        Load data with local-first S3 streaming policy (cache-free).

        Args:
            local_path: Local path to check first
            s3_key: S3 key for streaming fallback
            loader_fn: Function to load the data
            **kwargs: Additional arguments for loader_fn

        Returns:
            Loaded data
        """
        # 1. Check local path first
        if local_path:
            local_path = Path(local_path)
            if local_path.exists():
                console.print(f"[green]Loading from local: {local_path}[/green]")
                return loader_fn(local_path, **kwargs)

        # 2. Stream from S3 if available
        if self.s3_manager and self.s3_manager.connected:
            console.print("[yellow]Local not found, streaming from S3...[/yellow]")
            try:
                # Stream directly from S3 to memory
                if s3_key.endswith((".pth", ".pt", ".safetensors")):
                    # Model files - use stream_model
                    stream = self.s3_manager.stream_model(s3_key)
                    console.print(f"[green]Streaming model from S3: {s3_key}[/green]")
                    return loader_fn(stream, **kwargs)
                else:
                    # Dataset files - use stream_dataset
                    stream = self.s3_manager.stream_dataset(s3_key)
                    console.print(f"[green]Streaming dataset from S3: {s3_key}[/green]")
                    return loader_fn(stream, **kwargs)
            except Exception as e:
                console.print(f"[red]S3 streaming failed: {e}[/red]")

        # 3. Raise error if nothing found
        raise FileNotFoundError(
            f"Could not find data at local path '{local_path}' or S3 key '{s3_key}'"
        )

    def load(self, path: str, **kwargs) -> Any:
        """
        Load data from path - 기본 구현 제공.

        구현체는 이 메서드를 오버라이드하거나,
        run() 메서드로 대체 구현할 수 있습니다.

        Args:
            path: Path to data
            **kwargs: Additional loading parameters

        Returns:
            Loaded data
        """
        # 기본 구현: NotImplementedError
        # 하위 클래스는 이 메서드를 오버라이드하거나
        # run() 메서드로 대체 구현 가능
        raise NotImplementedError(
            "Data loading must be implemented by subclasses. "
            "Override load() or implement run() method."
        )

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Execute loading operation within component framework.

        Args:
            ctx: Context dictionary

        Returns:
            Dictionary with loaded data
        """
        self.validate_initialized()

        # Extract loading parameters from context
        path = ctx.get("path") or ctx.get("data_path")
        if not path:
            raise ValueError("No path specified in context")

        # Load data
        data = self.load(path, **ctx)

        return {
            "data": data,
            "path": path,
            "loader": self.__class__.__name__,
        }


class DatasetLoader(BaseLoader):
    """
    Base class for dataset loaders with split support.

    현재 구현체(DataLoader)는 run() 메서드로 실행되며,
    load()와 preprocess()는 호환성을 위한 인터페이스로 제공됩니다.
    """

    def create_splits(
        self,
        data: Any,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Create train/val/test splits from data.

        Args:
            data: Dataset to split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # This is a placeholder - subclasses should implement actual splitting
        return {
            "train": data,
            "val": data,
            "test": data,
        }

    def preprocess(self, data: Any, **kwargs) -> Any:
        """
        Preprocess dataset - 기본 구현 제공.

        구현체는 필요시 오버라이드하거나,
        내부적으로 다른 전처리 메서드를 사용할 수 있습니다.

        Args:
            data: Raw dataset
            **kwargs: Preprocessing parameters

        Returns:
            Preprocessed data
        """
        # 기본 구현: 데이터를 그대로 반환
        return data


class ModelLoader(BaseLoader):
    """
    Base class for model loaders.

    현재 구현체(model_loader.py의 ModelLoader)는 run() 메서드로 실행되며,
    load_model()은 실제 모델 로딩 로직을 구현합니다.
    load()는 호환성을 위한 래퍼 메서드입니다.
    """

    def load_model(self, path: str, **kwargs) -> Any:
        """
        Load model from path - 기본 구현 제공.

        구현체는 이 메서드를 오버라이드하거나,
        내부적으로 다른 로딩 메서드를 사용할 수 있습니다.

        Args:
            path: Model path
            **kwargs: Additional parameters

        Returns:
            Loaded model
        """
        # 기본 구현: NotImplementedError
        # 하위 클래스는 이 메서드를 오버라이드하거나
        # run() 메서드로 대체 구현 가능
        raise NotImplementedError(
            "Model loading must be implemented by subclasses. "
            "Override load_model() or implement run() method."
        )

    def load(self, path: str, **kwargs) -> dict[str, Any]:
        """
        Load model only - tokenizer 제거됨.

        토크나이저는 ComponentFactory.create_tokenizer()를 통해
        별도로 생성하도록 변경되었습니다. 이로써 중복이 제거되고
        토크나이저 생성 경로가 단일화됩니다.

        Args:
            path: Model path
            **kwargs: Additional parameters

        Returns:
            Dictionary with model only
        """
        model = self.load_model(path, **kwargs)

        return {
            "model": model,
        }
