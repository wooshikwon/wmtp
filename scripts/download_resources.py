#!/usr/bin/env python3
"""
Production S3 resource downloader for WMTP Docker containers.

Downloads models and datasets from S3 to local /app directories.
Designed for cloud-agnostic deployment with minimal dependencies.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.distributed_s3_transfer import DistributedS3Transfer  # noqa: E402


def check_aws_credentials() -> None:
    """Verify AWS credentials are available."""
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("❌ AWS credentials required!")
        print("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        sys.exit(1)


def download_resources() -> None:
    """Download all required models and datasets from S3."""
    print("📦 WMTP Resource Downloader")
    print("=" * 50)

    # Check credentials
    check_aws_credentials()

    # Initialize S3 transfer
    bucket = os.getenv("S3_BUCKET_NAME", "wmtp")
    print(f"S3 Bucket: {bucket}")

    transfer = DistributedS3Transfer(
        bucket=bucket,
        max_workers=16,
        use_multiprocess=True,
        chunk_size_mb=100,
        enable_acceleration=True,
    )

    # Define resources to download
    models = [
        ("models/llama-7b-mtp", "/app/models/llama-7b-mtp"),
        ("models/starling-rm-7b", "/app/models/starling-rm-7b"),
        ("models/sheared-llama-2.7b", "/app/models/sheared-llama-2.7b"),
    ]

    datasets = [
        ("datasets/mbpp", "/app/datasets/mbpp"),
        ("datasets/contest", "/app/datasets/contest"),
    ]

    # Download models
    print("\n📥 Downloading models...")
    for s3_prefix, local_path in models:
        local = Path(local_path)

        # Skip if already exists
        if local.exists() and list(local.glob("*")):
            print(f"✓ {local_path} already exists, skipping")
            continue

        print(f"\n→ Downloading {s3_prefix}...")
        success, _ = transfer.download_directory_distributed(
            s3_prefix, local, show_progress=True
        )

        if not success:
            print(f"❌ Failed: {s3_prefix}")
            sys.exit(1)

    # Download datasets
    print("\n📥 Downloading datasets...")
    for s3_prefix, local_path in datasets:
        local = Path(local_path)

        if local.exists() and list(local.glob("*")):
            print(f"✓ {local_path} already exists, skipping")
            continue

        print(f"\n→ Downloading {s3_prefix}...")
        success, _ = transfer.download_directory_distributed(
            s3_prefix, local, show_progress=True
        )

        if not success:
            print(f"⚠️  Dataset download failed (non-critical): {s3_prefix}")

    print("\n" + "=" * 50)
    print("✅ Resource download complete!")
    print("✅ Ready for training!")


if __name__ == "__main__":
    download_resources()
