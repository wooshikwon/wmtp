#!/usr/bin/env python3
"""
Unified script to download all required models and datasets for WMTP.
Based on BLUEPRINT.md specifications.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from transformers import AutoConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ModelDatasetSetup:
    """Setup all models and datasets according to BLUEPRINT.md"""

    def __init__(self):
        """Initialize with environment variables"""
        load_dotenv()

        # HuggingFace authentication
        self.hf_token = os.getenv("HF_TOKEN")
        if self.hf_token:
            login(token=self.hf_token, add_to_git_credential=False)
            logger.info("✅ HuggingFace authentication successful")

        # Define paths based on BLUEPRINT.md
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.dataset_dir = self.project_root / "dataset"

        # Model specifications - Updated for unified tokenizer
        self.model_specs = {
            "7b_1t_4": {
                "hub_id": "facebook/multi-token-prediction",  # Actual MTP model!
                "description": "MTP Base Model (n_heads=4, 1T tokens)",
                "download_weights": True,  # Download actual weights
                "subfolder": "7B_1T_4",  # Specific model variant
                "required": True,
            },
            "Starling-RM-7B-alpha": {
                "hub_id": "berkeley-nest/Starling-RM-7B-alpha",
                "description": "Reward Model for Critic-WMTP (Llama-2 based)",
                "download_weights": True,  # Full download
                "required": True,
            },
            "Sheared-LLaMA-2.7B": {
                "hub_id": "princeton-nlp/Sheared-LLaMA-2.7B",
                "description": "2.7B Reference Model for Rho1-WMTP (efficient)",
                "download_weights": True,  # Full download
                "required": True,
            },
        }

        # Dataset specifications from BLUEPRINT.md
        self.dataset_specs = {
            "mbpp": {
                "hub_id": "google-research-datasets/mbpp",
                "description": "MBPP: Mostly Basic Python Problems",
                "required": True,
            },
            "contest": {
                "hub_id": "deepmind/code_contests",
                "description": "CodeContests dataset",
                "required": True,
            },
        }

    def clean_up_old_models(self):
        """Remove models not in current specs"""
        logger.info("\n" + "=" * 60)
        logger.info("CLEANUP: Removing unnecessary models")
        logger.info("=" * 60)

        # List of old models to remove (no longer needed with unified tokenizer)
        old_models = ["opt-1.3b", "Llama_3_8B_RM", "codellama_7b_python"]

        for model_name in old_models:
            model_path = self.models_dir / model_name
            if model_path.exists():
                logger.info(
                    f"🗑️  Removing {model_path} (replaced with unified tokenizer models)"
                )
                shutil.rmtree(model_path)

        # Check for any other unexpected models
        if self.models_dir.exists():
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir() and model_dir.name not in self.model_specs:
                    logger.warning(f"⚠️  Found unexpected model: {model_dir.name}")

    def download_model(self, name: str, spec: dict[str, Any]) -> bool:
        """Download a single model"""
        model_path = self.models_dir / name
        hub_id = spec["hub_id"]

        logger.info(f"\n📦 Processing: {name}")
        logger.info(f"   Description: {spec['description']}")
        logger.info(f"   HuggingFace ID: {hub_id}")

        try:
            if spec["download_weights"]:
                # Check if this is the MTP model with subfolder
                if "subfolder" in spec and name == "7b_1t_4":
                    # Download MTP model from specific subfolder
                    logger.info(f"   Downloading MTP model from {spec['subfolder']}...")
                    from huggingface_hub import hf_hub_download

                    # Download the specific MTP variant files
                    files = ["consolidated.pth", "params.json"]
                    for file in files:
                        file_path = f"{spec['subfolder']}/{file}"
                        logger.info(f"   Downloading {file_path}...")
                        hf_hub_download(
                            repo_id=hub_id,
                            filename=file_path,
                            local_dir=model_path,
                            token=self.hf_token,
                        )

                    # Also download tokenizer.model from root
                    hf_hub_download(
                        repo_id=hub_id,
                        filename="tokenizer.model",
                        local_dir=model_path,
                        token=self.hf_token,
                    )
                else:
                    # Standard full model download
                    logger.info("   Downloading full model weights...")
                    snapshot_download(
                        repo_id=hub_id,
                        local_dir=model_path,
                        token=self.hf_token,
                        ignore_patterns=["*.md", "*.txt"]
                        if name == "Llama_3_8B_RM"
                        else None,
                    )

                # Check download size
                total_size = sum(
                    f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                )
                logger.info(f"   ✅ Downloaded: {total_size / 1e9:.2f} GB")
            else:
                # Config only
                logger.info("   Downloading config only (placeholder)...")
                model_path.mkdir(parents=True, exist_ok=True)

                # Download config file
                config = AutoConfig.from_pretrained(hub_id, token=self.hf_token)
                config.save_pretrained(model_path)

                # Add note about placeholder
                note_file = model_path / "PLACEHOLDER_NOTE.txt"
                note_file.write_text(
                    f"This is a placeholder for the actual MTP model.\n"
                    f"Using {hub_id} config for development.\n"
                    f"Replace with actual facebook/multi-token-prediction model when available.\n"
                )
                logger.info("   ✅ Config downloaded (placeholder)")

            return True

        except Exception as e:
            logger.error(f"   ❌ Failed to download {name}: {e}")
            return False

    def download_dataset(self, name: str, spec: dict[str, Any]) -> bool:
        """Download a single dataset"""
        dataset_path = self.dataset_dir / name
        hub_id = spec["hub_id"]

        logger.info(f"\n📊 Processing: {name}")
        logger.info(f"   Description: {spec['description']}")
        logger.info(f"   HuggingFace ID: {hub_id}")

        try:
            dataset_path.mkdir(parents=True, exist_ok=True)

            if name == "mbpp":
                # Load MBPP dataset
                dataset = load_dataset(hub_id, trust_remote_code=False)

                # Save splits
                for split in ["train", "test", "validation"]:
                    if split in dataset:
                        data = dataset[split]
                        output_file = dataset_path / f"{split}.json"

                        # Convert to JSON format
                        records = []
                        for item in data:
                            records.append(
                                {
                                    "task_id": item.get("task_id", ""),
                                    "text": item.get("text", ""),
                                    "code": item.get("code", ""),
                                    "test_list": item.get("test_list", []),
                                    "test_setup_code": item.get("test_setup_code", ""),
                                    "challenge_test_list": item.get(
                                        "challenge_test_list", []
                                    ),
                                }
                            )

                        with open(output_file, "w") as f:
                            json.dump(records, f, indent=2)

                        logger.info(f"   ✅ Saved {split}: {len(records)} samples")

            elif name == "contest":
                # For CodeContests, create placeholder for now
                logger.info(
                    "   Creating placeholder structure (actual download pending)"
                )

                for split in ["train", "test", "validation"]:
                    output_file = dataset_path / f"{split}.json"

                    # Create minimal placeholder
                    placeholder = [
                        {
                            "name": f"contest_problem_{i}",
                            "description": f"Placeholder problem {i}",
                            "public_tests": [],
                            "private_tests": [],
                            "generated_tests": [],
                            "source": "placeholder",
                            "difficulty": "unknown",
                            "solutions": [],
                        }
                        for i in range(10 if split == "train" else 5)
                    ]

                    with open(output_file, "w") as f:
                        json.dump(placeholder, f, indent=2)

                logger.info(
                    "   ⚠️  Created placeholder (actual CodeContests download TODO)"
                )

            return True

        except Exception as e:
            logger.error(f"   ❌ Failed to download {name}: {e}")
            return False

    def verify_setup(self):
        """Verify all models and datasets are properly set up"""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION: Checking setup completeness")
        logger.info("=" * 60)

        all_good = True

        # Check models
        logger.info("\n📦 Models:")
        for name, spec in self.model_specs.items():
            model_path = self.models_dir / name
            if model_path.exists():
                # Check for actual weights or config (including .pth for MTP)
                has_weights = (
                    any(model_path.glob("*.bin"))
                    or any(model_path.glob("*.safetensors"))
                    or any(model_path.glob("**/*.pth"))
                )
                has_config = (model_path / "config.json").exists() or any(
                    model_path.glob("**/params.json")
                )

                if spec["download_weights"] and not has_weights:
                    logger.warning(f"   ⚠️  {name}: Config only (weights missing)")
                elif has_weights:
                    size = sum(
                        f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                    )
                    logger.info(f"   ✅ {name}: {size / 1e9:.2f} GB")
                elif has_config:
                    logger.info(f"   ✅ {name}: Config only (placeholder)")
                else:
                    logger.error(f"   ❌ {name}: Invalid state")
                    all_good = False
            else:
                logger.error(f"   ❌ {name}: Not found")
                all_good = False

        # Check datasets
        logger.info("\n📊 Datasets:")
        for name, spec in self.dataset_specs.items():
            dataset_path = self.dataset_dir / name
            if dataset_path.exists():
                files = list(dataset_path.glob("*.json"))
                if files:
                    total_size = sum(f.stat().st_size for f in files)
                    logger.info(
                        f"   ✅ {name}: {len(files)} files, {total_size / 1e6:.2f} MB"
                    )
                else:
                    logger.error(f"   ❌ {name}: No data files")
                    all_good = False
            else:
                logger.error(f"   ❌ {name}: Not found")
                all_good = False

        return all_good

    def upload_to_s3(self) -> None:
        """Upload models to S3"""
        logger.info("\n" + "=" * 60)
        logger.info("S3: Uploading models to S3")
        logger.info("=" * 60)

        # Import S3 utilities
        try:
            from src.utils.s3 import create_s3_manager

            s3_manager = create_s3_manager({"storage": {"s3": {"enabled": True}}})

            if not s3_manager or not s3_manager.connected:
                logger.error("❌ S3 connection failed. Check AWS credentials.")
                return

            # Upload each model
            for name in self.model_specs.keys():
                model_path = self.models_dir / name
                if model_path.exists():
                    s3_key = f"models/{name}"
                    logger.info(f"📤 Uploading {name} to s3://wmtp/{s3_key}")
                    s3_manager.sync_directory(
                        local_dir=model_path, s3_prefix=s3_key, direction="upload"
                    )
                    logger.info(f"✅ {name} uploaded successfully")
                else:
                    logger.warning(f"⚠️  {name} not found locally, skipping upload")

        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")

    def run(self, force_redownload: bool = False, upload_s3: bool = False):
        """Main execution flow"""
        logger.info("\n" + "=" * 60)
        logger.info("WMTP Model & Dataset Setup")
        logger.info("Based on BLUEPRINT.md specifications")
        logger.info("=" * 60)

        # Step 1: Clean up old/unnecessary models
        self.clean_up_old_models()

        # Step 2: Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.dataset_dir.mkdir(exist_ok=True)

        # Step 3: Download models
        logger.info("\n" + "=" * 60)
        logger.info("MODELS: Downloading required models")
        logger.info("=" * 60)

        for name, spec in self.model_specs.items():
            if not spec["required"]:
                continue

            model_path = self.models_dir / name
            if model_path.exists() and not force_redownload:
                logger.info(f"\n📦 {name}: Already exists, skipping...")
                continue

            self.download_model(name, spec)

        # Step 4: Download datasets
        logger.info("\n" + "=" * 60)
        logger.info("DATASETS: Downloading required datasets")
        logger.info("=" * 60)

        for name, spec in self.dataset_specs.items():
            if not spec["required"]:
                continue

            dataset_path = self.dataset_dir / name
            if dataset_path.exists() and not force_redownload:
                logger.info(f"\n📊 {name}: Already exists, skipping...")
                continue

            self.download_dataset(name, spec)

        # Step 5: Upload to S3 if requested
        if upload_s3:
            self.upload_to_s3()

        # Step 6: Verify setup
        if self.verify_setup():
            logger.info("\n✨ Setup completed successfully!")
            logger.info("\nYou can now run training with:")
            logger.info(
                "  uv run python -m src.cli.train --config configs/config.local.yaml --recipe configs/recipe.rho1.yaml"
            )
        else:
            logger.error("\n❌ Setup incomplete. Please check errors above.")
            sys.exit(1)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup WMTP models and datasets")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--upload-s3", action="store_true", help="Upload models to S3 after download"
    )
    args = parser.parse_args()

    setup = ModelDatasetSetup()
    setup.run(force_redownload=args.force, upload_s3=args.upload_s3)


if __name__ == "__main__":
    main()
