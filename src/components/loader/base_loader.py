"""
WMTP 로더 시스템의 기반: 로컬 우선 S3 미러링 정책

WMTP 연구 맥락:
이 모듈은 WMTP 실험의 핵심 인프라입니다. 연구자들이 세 가지 알고리즘
(baseline/critic/rho1)을 실험할 때, 동일한 모델과 데이터셋을 안정적으로
로드할 수 있도록 보장합니다.

핵심 철학:
"로컬 파일 → 캐시 → S3 다운로드" 순서로 효율적인 데이터 접근

지원하는 실험 환경:
- 개발자 로컬 환경: 로컬 파일 직접 사용
- VESSL 클러스터: S3에서 자동 다운로드 및 캐싱
- 오프라인 환경: 미리 다운로드된 캐시 활용

WMTP에서의 역할:
1. Facebook Native MTP 모델 안정적 로딩 (consolidated.pth)
2. 코딩 평가 데이터셋 일관된 전처리 (MBPP/CodeContests/HumanEval)
3. 실험 재현성 보장 (동일한 캐시 키 → 동일한 데이터)
4. GPU 클러스터 비용 절약 (중복 다운로드 방지)
"""

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rich.console import Console

from ...components.base import BaseComponent
from ...utils.s3 import create_s3_manager

console = Console()


class BaseLoader(BaseComponent, ABC):
    """
    Abstract base class for all loaders.

    Implements local-first policy: check local path first,
    fall back to S3 with caching if not found.
    """

    def __init__(
        self,
        config: dict[str, Any],
        cache_dir: str | Path = ".cache",
    ):
        """
        Initialize base loader.

        Args:
            config: Configuration dictionary
            cache_dir: Local cache directory for S3 downloads
        """
        super().__init__(config)
        # Prefer config.paths.cache if provided
        cfg_cache = (
            Path(config.get("paths", {}).get("cache"))
            if config.get("paths", {}).get("cache")
            else Path(cache_dir)
        )
        self.cache_dir = cfg_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 manager if configured
        self.s3_manager = create_s3_manager(config)
        self.storage_mode = config.get("storage", {}).get("mode", "local")

        # Extract path configurations
        self.paths_config = config.get("paths", {})

    def compute_cache_key(
        self,
        data_id: str,
        version: str = "latest",
        preprocessing_config: dict[str, Any] | None = None,
        split_seed: int = 42,
    ) -> str:
        """
        Compute deterministic cache key for data.

        Args:
            data_id: Dataset or model identifier
            version: Version string
            preprocessing_config: Preprocessing parameters
            split_seed: Random seed for splits

        Returns:
            Hex digest cache key
        """
        # Combine all parameters into a unique string
        key_parts = [
            data_id,
            version,
            str(split_seed),
        ]

        if preprocessing_config:
            # Sort keys for deterministic ordering
            config_str = json.dumps(preprocessing_config, sort_keys=True)
            key_parts.append(config_str)

        key_string = "|".join(key_parts)

        # Create MD5 hash for compact cache key
        hasher = hashlib.md5()
        hasher.update(key_string.encode("utf-8"))

        return hasher.hexdigest()

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

    def get_cached_path(
        self,
        s3_key: str,
        cache_key: str,
    ) -> Path:
        """
        Get path in cache directory.

        Args:
            s3_key: S3 object key
            cache_key: Computed cache key

        Returns:
            Path to cached file/directory
        """
        # Use cache key as subdirectory to organize cached data
        cache_subdir = self.cache_dir / cache_key
        cache_subdir.mkdir(parents=True, exist_ok=True)

        # Extract filename from S3 key
        filename = Path(s3_key).name

        return cache_subdir / filename

    def load_with_cache(
        self,
        local_path: str | Path | None,
        s3_key: str,
        cache_key: str,
        loader_fn: Any,
        **kwargs,
    ) -> Any:
        """
        Load data with local-first S3 fallback policy.

        Args:
            local_path: Local path to check first
            s3_key: S3 key for fallback
            cache_key: Cache key for organizing cached data
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

        # 2. Check cache
        cached_path = self.get_cached_path(s3_key, cache_key)
        if cached_path.exists():
            console.print(f"[cyan]Loading from cache: {cached_path}[/cyan]")
            return loader_fn(cached_path, **kwargs)

        # 3. Download from S3 if available
        if self.s3_manager and self.s3_manager.connected:
            console.print("[yellow]Local not found, downloading from S3...[/yellow]")

            # Download to cache
            downloaded_path = self.s3_manager.download_if_missing(
                s3_key,
                cached_path,
                force=False,
            )

            if downloaded_path and downloaded_path.exists():
                console.print(
                    f"[green]Loading from S3 cache: {downloaded_path}[/green]"
                )
                return loader_fn(downloaded_path, **kwargs)

        # 4. Raise error if nothing found
        raise FileNotFoundError(
            f"Could not find data at local path '{local_path}' " f"or S3 key '{s3_key}'"
        )

    def sync_directory_with_cache(
        self,
        local_dir: str | Path | None,
        s3_prefix: str,
        cache_key: str,
    ) -> Path:
        """
        Sync directory with local-first S3 fallback.

        Args:
            local_dir: Local directory to check first
            s3_prefix: S3 prefix for directory
            cache_key: Cache key for organizing cached data

        Returns:
            Path to directory (local or cached)
        """
        # 1. Check local directory first
        if local_dir:
            local_dir = Path(local_dir)
            if local_dir.exists() and local_dir.is_dir():
                console.print(f"[green]Using local directory: {local_dir}[/green]")
                return local_dir

        # 2. Check/create cache directory
        cached_dir = self.cache_dir / cache_key / s3_prefix.replace("/", "_")

        if cached_dir.exists() and any(cached_dir.iterdir()):
            console.print(f"[cyan]Using cached directory: {cached_dir}[/cyan]")
            return cached_dir

        # 3. Sync from S3 if available
        if self.s3_manager and self.s3_manager.connected:
            console.print("[yellow]Syncing directory from S3...[/yellow]")

            cached_dir.mkdir(parents=True, exist_ok=True)
            self.s3_manager.sync_directory(
                cached_dir,
                s3_prefix,
                direction="download",
            )

            if cached_dir.exists():
                console.print(f"[green]Synced to cache: {cached_dir}[/green]")
                return cached_dir

        # 4. Raise error if nothing found
        raise FileNotFoundError(
            f"Could not find directory at local path '{local_dir}' "
            f"or S3 prefix '{s3_prefix}'"
        )

    @abstractmethod
    def load(self, path: str, **kwargs) -> Any:
        """
        Load data from path (must be implemented by subclasses).

        Args:
            path: Path to data
            **kwargs: Additional loading parameters

        Returns:
            Loaded data
        """
        pass

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
    """Base class for dataset loaders with split support."""

    def create_splits(
        self,
        data: Any,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
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

    @abstractmethod
    def preprocess(self, data: Any, **kwargs) -> Any:
        """
        Preprocess dataset (must be implemented by subclasses).

        Args:
            data: Raw dataset
            **kwargs: Preprocessing parameters

        Returns:
            Preprocessed data
        """
        pass


class ModelLoader(BaseLoader):
    """Base class for model loaders."""

    @abstractmethod
    def load_model(self, path: str, **kwargs) -> Any:
        """
        Load model from path (must be implemented by subclasses).

        Args:
            path: Model path
            **kwargs: Additional parameters

        Returns:
            Loaded model
        """
        pass

    @abstractmethod
    def load_tokenizer(self, path: str, **kwargs) -> Any:
        """
        Load tokenizer from path (must be implemented by subclasses).

        Args:
            path: Tokenizer path
            **kwargs: Additional parameters

        Returns:
            Loaded tokenizer
        """
        pass

    def load(self, path: str, **kwargs) -> dict[str, Any]:
        """
        Load both model and tokenizer.

        Args:
            path: Model path
            **kwargs: Additional parameters

        Returns:
            Dictionary with model and tokenizer
        """
        model = self.load_model(path, **kwargs)
        tokenizer = self.load_tokenizer(path, **kwargs)

        return {
            "model": model,
            "tokenizer": tokenizer,
        }
