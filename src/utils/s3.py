"""
S3 utility functions for WMTP framework.

This module centralizes all S3 operations to prevent direct boto3 usage
outside of utils. Implements local-first policy with S3 mirroring.
"""

import hashlib
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class S3Utils:
    """
    Simplified S3 utility wrapper for loader modules.

    Provides common S3 operations with a simpler interface.
    """

    def __init__(self, bucket: str = "wmtp-models", region: str = "ap-northeast-2"):
        """Initialize S3 utilities with default WMTP bucket."""
        self.manager = S3Manager(bucket=bucket, region=region)

    def download_model(self, s3_path: str, local_dir: Path | None = None) -> Path:
        """Download model from S3 path like s3://bucket/path/to/model."""
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")

        # Parse S3 path
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Update manager if different bucket
        if bucket != self.manager.bucket:
            self.manager = S3Manager(bucket=bucket, region=self.manager.region)

        # Determine local path
        if local_dir is None:
            local_dir = self.manager.cache_dir / "models"
        local_path = local_dir / key.split("/")[-1]

        return self.manager.download_if_missing(key, local_path)

    def download_dataset(self, s3_path: str, local_dir: Path | None = None) -> Path:
        """Download dataset from S3 path."""
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")

        # Parse S3 path
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Update manager if different bucket
        if bucket != self.manager.bucket:
            self.manager = S3Manager(bucket=bucket, region=self.manager.region)

        # Determine local path
        if local_dir is None:
            local_dir = self.manager.cache_dir / "datasets"
        local_path = local_dir / key.split("/")[-1]

        return self.manager.download_if_missing(key, local_path)

    def download_checkpoint(self, s3_path: str, local_dir: Path | None = None) -> Path:
        """Download checkpoint from S3 path."""
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")

        # Parse S3 path
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Update manager if different bucket
        if bucket != self.manager.bucket:
            self.manager = S3Manager(bucket=bucket, region=self.manager.region)

        # Determine local path
        if local_dir is None:
            local_dir = self.manager.cache_dir / "checkpoints"
        local_path = local_dir / key.split("/")[-1]

        return self.manager.download_if_missing(key, local_path)

    def upload_checkpoint(self, local_path: Path, s3_path: str) -> None:
        """Upload checkpoint to S3."""
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")

        # Parse S3 path
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        # Update manager if different bucket
        if bucket != self.manager.bucket:
            self.manager = S3Manager(bucket=bucket, region=self.manager.region)

        self.manager.upload_file(str(local_path), key)


class S3Manager:
    """
    Manager for S3 operations with local caching.

    Implements the local-first policy: use local if exists,
    otherwise download from S3 and cache.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "ap-northeast-2",
        prefix: str = "",
        cache_dir: str | Path = ".cache",
    ):
        """
        Initialize S3 manager.

        Args:
            bucket: S3 bucket name
            region: AWS region
            prefix: S3 key prefix
            cache_dir: Local cache directory
        """
        self.bucket = bucket
        self.region = region
        self.prefix = prefix
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.s3_client = boto3.client("s3", region_name=region)
            self._test_connection()
            self.connected = True
        except (NoCredentialsError, ClientError) as e:
            console.print(f"[yellow]Warning: S3 not available: {e}[/yellow]")
            self.connected = False
            self.s3_client = None

    def _test_connection(self) -> None:
        """Test S3 connection by listing bucket."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise ValueError(f"Bucket '{self.bucket}' not found")
            else:
                raise

    def download_if_missing(
        self,
        s3_key: str,
        local_path: str | Path,
        force: bool = False,
    ) -> Path:
        """
        Download from S3 if local file doesn't exist.

        Args:
            s3_key: S3 object key
            local_path: Local file path
            force: Force download even if exists

        Returns:
            Path to local file
        """
        local_path = Path(local_path)

        # Check local first (unless forced)
        if local_path.exists() and not force:
            console.print(f"[green]Using local file: {local_path}[/green]")
            return local_path

        if not self.connected:
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Local file not found and S3 not available: {local_path}"
                )
            return local_path

        # Download from S3
        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Downloading {s3_key} from S3...",
                    total=None,
                )

                self.s3_client.download_file(
                    self.bucket,
                    full_key,
                    str(local_path),
                )

                progress.update(task, completed=True)

            console.print(f"[green]Downloaded to: {local_path}[/green]")
            return local_path

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise FileNotFoundError(
                    f"S3 object not found: s3://{self.bucket}/{full_key}"
                )
            else:
                raise

    def upload_artifact(
        self,
        local_path: str | Path,
        s3_key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload artifact to S3.

        Args:
            local_path: Local file path
            s3_key: S3 object key
            metadata: Optional metadata

        Returns:
            S3 URI
        """
        if not self.connected:
            raise RuntimeError("S3 not connected. Cannot upload.")

        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Uploading {local_path.name} to S3...",
                    total=None,
                )

                self.s3_client.upload_file(
                    str(local_path),
                    self.bucket,
                    full_key,
                    ExtraArgs=extra_args,
                )

                progress.update(task, completed=True)

            s3_uri = f"s3://{self.bucket}/{full_key}"
            console.print(f"[green]Uploaded to: {s3_uri}[/green]")
            return s3_uri

        except ClientError as e:
            raise RuntimeError(f"Failed to upload to S3: {e}")

    def exists(self, s3_key: str) -> bool:
        """
        Check if object exists in S3.

        Args:
            s3_key: S3 object key

        Returns:
            True if exists
        """
        if not self.connected:
            return False

        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError:
            return False

    def list_objects(
        self,
        prefix: str = "",
        recursive: bool = True,
    ) -> list[str]:
        """
        List objects in S3.

        Args:
            prefix: Key prefix to filter
            recursive: List recursively

        Returns:
            List of S3 keys
        """
        if not self.connected:
            return []

        full_prefix = f"{self.prefix}/{prefix}" if self.prefix else prefix

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.bucket,
                Prefix=full_prefix,
            )

            keys = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Remove the base prefix if present
                        if self.prefix and key.startswith(self.prefix + "/"):
                            key = key[len(self.prefix) + 1 :]
                        keys.append(key)

            if not recursive:
                # Filter to only immediate children
                keys = [k for k in keys if "/" not in k[len(prefix) :].lstrip("/")]

            return keys

        except ClientError as e:
            console.print(f"[red]Failed to list S3 objects: {e}[/red]")
            return []

    def get_etag(self, s3_key: str) -> str | None:
        """
        Get ETag for S3 object (for version checking).

        Args:
            s3_key: S3 object key

        Returns:
            ETag string or None
        """
        if not self.connected:
            return None

        full_key = f"{self.prefix}/{s3_key}" if self.prefix else s3_key

        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=full_key)
            return response.get("ETag", "").strip('"')
        except ClientError:
            return None

    def sync_directory(
        self,
        local_dir: str | Path,
        s3_prefix: str,
        direction: str = "download",
    ) -> None:
        """
        Sync entire directory with S3.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 key prefix
            direction: 'download' or 'upload'
        """
        if not self.connected:
            raise RuntimeError("S3 not connected. Cannot sync.")

        local_dir = Path(local_dir)

        if direction == "download":
            # List all objects with prefix
            objects = self.list_objects(s3_prefix)

            for s3_key in objects:
                # Reconstruct local path
                relative_path = s3_key[len(s3_prefix) :].lstrip("/")
                local_path = local_dir / relative_path

                # Download if missing
                self.download_if_missing(s3_key, local_path)

        elif direction == "upload":
            # Walk local directory
            for local_path in local_dir.rglob("*"):
                if local_path.is_file():
                    # Construct S3 key
                    relative_path = local_path.relative_to(local_dir)
                    s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")

                    # Upload file
                    self.upload_artifact(local_path, s3_key)

        else:
            raise ValueError(
                f"Invalid direction: {direction}. Use 'download' or 'upload'"
            )

    def get_cache_path(self, s3_key: str, version: str | None = None) -> Path:
        """
        Get local cache path for S3 object.

        Args:
            s3_key: S3 object key
            version: Optional version identifier

        Returns:
            Cache file path
        """
        # Create hash-based cache key
        cache_key = s3_key.replace("/", "_")
        if version:
            cache_key = f"{cache_key}_{version}"

        return self.cache_dir / cache_key


def create_s3_manager(config: dict[str, Any]) -> S3Manager | None:
    """
    Create S3 manager from configuration.

    Args:
        config: Configuration dictionary with S3 settings

    Returns:
        S3Manager instance or None if not configured
    """
    storage_mode = config.get("storage", {}).get("mode")

    if storage_mode != "s3":
        return None

    s3_config = config.get("storage", {}).get("s3", {})
    if not s3_config:
        console.print(
            "[yellow]Warning: S3 mode selected but no S3 config found[/yellow]"
        )
        return None

    return S3Manager(
        bucket=s3_config.get("bucket"),
        region=s3_config.get("region", "ap-northeast-2"),
        prefix=s3_config.get("prefix", ""),
        cache_dir=config.get("paths", {}).get("cache", ".cache"),
    )


def compute_file_hash(file_path: str | Path, algorithm: str = "md5") -> str:
    """
    Compute hash of file for cache validation.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5' or 'sha256')

    Returns:
        Hex digest string
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.md5() if algorithm == "md5" else hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def ensure_s3_uri(path_or_uri: str, bucket: str, prefix: str = "") -> str:
    """
    Ensure path is formatted as S3 URI.

    Args:
        path_or_uri: Path or S3 URI
        bucket: S3 bucket name
        prefix: S3 prefix

    Returns:
        S3 URI string
    """
    if path_or_uri.startswith("s3://"):
        return path_or_uri

    # Construct S3 URI
    key = f"{prefix}/{path_or_uri}" if prefix else path_or_uri
    return f"s3://{bucket}/{key}"


# Export main functions
__all__ = [
    "S3Manager",
    "create_s3_manager",
    "compute_file_hash",
    "ensure_s3_uri",
]
