"""
Unit tests for data and model loaders.

Tests the local-first S3 fallback policy and caching behavior.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.components.loader import (
    CodeContestsDatasetLoader,
    HFLocalS3Loader,
    MBPPDatasetLoader,
)
from src.components.registry import loader_registry


class TestLoaderRegistry:
    """Test loader registration."""

    def test_loaders_registered(self):
        """Check that all loaders are properly registered."""
        assert loader_registry.exists("hf-local-s3-loader")
        assert loader_registry.exists("dataset-mbpp-loader")
        assert loader_registry.exists("dataset-contest-loader")

    def test_loader_metadata(self):
        """Check loader metadata."""
        hf_meta = loader_registry.get_metadata("hf-local-s3-loader")
        assert hf_meta["category"] == "loader"
        assert hf_meta["version"] == "v1"

        mbpp_meta = loader_registry.get_metadata("dataset-mbpp-loader")
        assert mbpp_meta["category"] == "loader"
        assert mbpp_meta["version"] == "v1"

    def test_create_loader_from_registry(self):
        """Test creating loader instances from registry."""
        config = {
            "storage_mode": "local",
            "cache_dir": ".cache",
            "model_paths": {},
        }

        loader = loader_registry.create("hf-local-s3-loader", config)
        assert isinstance(loader, HFLocalS3Loader)


class TestBaseLoaderCaching:
    """Test base loader caching logic."""

    def test_compute_cache_key(self):
        """Test deterministic cache key computation."""
        config = {"cache_dir": tempfile.mkdtemp()}
        loader = MBPPDatasetLoader(config)

        # Same inputs should produce same key
        key1 = loader.compute_cache_key("test_data", "v1", {"max_len": 100}, 42)
        key2 = loader.compute_cache_key("test_data", "v1", {"max_len": 100}, 42)
        assert key1 == key2

        # Different inputs should produce different keys
        key3 = loader.compute_cache_key("test_data", "v2", {"max_len": 100}, 42)
        assert key1 != key3

        key4 = loader.compute_cache_key("test_data", "v1", {"max_len": 200}, 42)
        assert key1 != key4

        key5 = loader.compute_cache_key("test_data", "v1", {"max_len": 100}, 123)
        assert key1 != key5

    def test_local_path_priority(self):
        """Test that local paths are used first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a local file
            local_path = Path(tmpdir) / "test_data.json"
            test_data = {"data": [{"text": "test"}]}
            local_path.write_text(json.dumps(test_data))

            # Configure loader with local path
            config = {
                "cache_dir": tmpdir,
                "local_path": str(local_path),
            }

            loader = MBPPDatasetLoader(config)

            # Check get_local_path works
            found_path = loader.get_local_path(str(local_path))
            assert found_path == local_path

            # Mock S3 manager to verify it's not called
            loader.s3_manager = MagicMock()

            # Load with local path - should not call S3
            with patch.object(loader, "_load_from_json") as mock_load:
                mock_load.return_value = MagicMock()
                loader.load()

                # Verify S3 was not accessed
                assert not loader.s3_manager.download_if_missing.called


class TestMBPPDatasetLoader:
    """Test MBPP dataset loader."""

    def test_preprocess_mbpp(self):
        """Test MBPP preprocessing."""
        config = {"cache_dir": tempfile.mkdtemp()}
        loader = MBPPDatasetLoader(config)

        # Create mock dataset
        from datasets import Dataset

        raw_data = [
            {
                "text": "Write a function to add two numbers",
                "test_list": [
                    "assert add(1, 2) == 3",
                    "assert add(5, 3) == 8",
                ],
                "code": "def add(a, b):\n    return a + b",
                "task_id": "mbpp_001",
            }
        ]

        dataset = Dataset.from_list(raw_data)

        # Preprocess
        processed = loader.preprocess(dataset, add_solution=True)

        # Check result
        assert len(processed) == 1
        example = processed[0]

        assert "# Problem:" in example["prompt"]
        assert "Write a function to add two numbers" in example["prompt"]
        assert "# Test Cases:" in example["prompt"]
        assert "assert add(1, 2) == 3" in example["prompt"]
        assert example["solution"] == "def add(a, b):\n    return a + b"
        assert example["task_id"] == "mbpp_001"

    def test_create_splits(self):
        """Test dataset splitting."""
        config = {"cache_dir": tempfile.mkdtemp()}
        loader = MBPPDatasetLoader(config)

        # Create test data
        from datasets import Dataset

        data = [{"id": i, "text": f"example_{i}"} for i in range(100)]
        dataset = Dataset.from_list(data)

        # Create splits
        splits = loader.create_splits(
            dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        # Verify split sizes
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

        # Verify no overlap
        train_ids = set(splits["train"]["id"])
        val_ids = set(splits["val"]["id"])
        test_ids = set(splits["test"]["id"])

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


class TestCodeContestsLoader:
    """Test CodeContests dataset loader."""

    def test_preprocess_codecontests(self):
        """Test CodeContests preprocessing."""
        config = {"cache_dir": tempfile.mkdtemp()}
        loader = CodeContestsDatasetLoader(config)

        # Create mock dataset
        from datasets import Dataset

        raw_data = [
            {
                "description": "Given two numbers, find their sum",
                "input_specification": "Two integers a and b",
                "output_specification": "The sum of a and b",
                "public_tests": [
                    {"input": "1 2", "output": "3"},
                    {"input": "5 3", "output": "8"},
                ],
                "solutions": {
                    "python3": "a, b = map(int, input().split())\nprint(a + b)",
                },
                "difficulty": "easy",
                "name": "sum_problem",
            }
        ]

        dataset = Dataset.from_list(raw_data)

        # Preprocess
        processed = loader.preprocess(dataset, language="python3", add_solution=True)

        # Check result
        assert len(processed) == 1
        example = processed[0]

        assert "# Problem:" in example["prompt"]
        assert "Given two numbers, find their sum" in example["prompt"]
        assert "# Input:" in example["prompt"]
        assert "# Output:" in example["prompt"]
        assert "# Examples:" in example["prompt"]
        assert "Input: 1 2" in example["prompt"]
        assert "Output: 3" in example["prompt"]
        assert example["solution"] == "a, b = map(int, input().split())\nprint(a + b)"
        assert example["difficulty"] == "easy"
        assert example["language"] == "python3"

    def test_difficulty_filtering(self):
        """Test filtering by difficulty."""
        config = {"cache_dir": tempfile.mkdtemp()}
        loader = CodeContestsDatasetLoader(config)

        # Create mock dataset with different difficulties
        from datasets import Dataset

        raw_data = [
            {"description": f"Problem {i}", "difficulty": diff}
            for i, diff in enumerate(["easy", "medium", "hard", "easy", "medium"])
        ]

        dataset = Dataset.from_list(raw_data)

        # Preprocess with difficulty filter
        processed = loader.preprocess(dataset, difficulty="easy", add_solution=False)

        # Check only easy problems remain
        assert len(processed) == 2
        for example in processed:
            assert example["difficulty"] == "easy"


class TestHFModelLoader:
    """Test HuggingFace model loader."""

    def test_model_type_mapping(self):
        """Test model type to path mapping."""
        config = {
            "cache_dir": tempfile.mkdtemp(),
            "model_paths": {
                "base": "/path/to/base",
                "rm": "/path/to/rm",
                "ref": "/path/to/ref",
            },
        }

        loader = HFLocalS3Loader(config)

        # Check paths are properly extracted
        assert loader.model_paths["base"] == "/path/to/base"
        assert loader.model_paths["rm"] == "/path/to/rm"
        assert loader.model_paths["ref"] == "/path/to/ref"

    def test_default_model_ids(self):
        """Test default model IDs are set correctly."""
        config = {"cache_dir": tempfile.mkdtemp()}
        loader = HFLocalS3Loader(config)

        # Check default IDs
        assert loader.default_model_ids["base"] == "facebook/multi-token-prediction-7b"
        assert loader.default_model_ids["rm"] == "sfair/Llama-3-8B-RM-Reward-Model"
        assert loader.default_model_ids["ref"] == "princeton-nlp/Sheared-LLaMA-1.3B"

    @patch("src.components.loader.hf_local_s3_loader.AutoModelForCausalLM")
    @patch("src.components.loader.hf_local_s3_loader.AutoTokenizer")
    def test_load_from_local(self, mock_tokenizer, mock_model):
        """Test loading from local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock model directory
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")

            config = {
                "cache_dir": tmpdir,
                "model_paths": {"base": str(model_dir)},
            }

            loader = HFLocalS3Loader(config)

            # Mock return values
            mock_model.from_pretrained.return_value = "mock_model"
            mock_tokenizer.from_pretrained.return_value = MagicMock(
                pad_token=None,
                eos_token="</s>",
            )

            # Load model
            result = loader.load("dummy", model_type="base")

            # Verify local loading was used
            assert mock_model.from_pretrained.called
            call_args = mock_model.from_pretrained.call_args
            assert str(model_dir) in str(call_args)
            assert call_args[1].get("local_files_only") is True


class TestLocalS3Fallback:
    """Test the local-first S3 fallback policy."""

    @patch("src.utils.s3.boto3")
    def test_s3_fallback_when_local_missing(self, mock_boto3):
        """Test S3 is used when local path doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock S3
            mock_s3_client = MagicMock()
            mock_boto3.client.return_value = mock_s3_client
            mock_s3_client.head_bucket.return_value = True

            config = {
                "cache_dir": tmpdir,
                "local_path": "/nonexistent/path",
                "storage_mode": "s3",
                "s3_config": {
                    "bucket": "test-bucket",
                    "region": "us-east-1",
                },
            }

            loader = MBPPDatasetLoader(config)

            # Mock S3 download
            mock_data = {"data": [{"text": "test"}]}
            cached_file = Path(tmpdir) / "test_cache" / "data.json"
            cached_file.parent.mkdir(parents=True)
            cached_file.write_text(json.dumps(mock_data))

            with patch.object(
                loader.s3_manager, "download_if_missing"
            ) as mock_download:
                mock_download.return_value = cached_file

                # Try to load - should fall back to S3
                with patch.object(loader, "_load_from_json") as mock_load:
                    mock_load.return_value = MagicMock()
                    with patch.object(loader, "_process_dataset") as mock_process:
                        mock_process.return_value = MagicMock()

                        # This should trigger S3 fallback
                        loader.load()

                        # Verify S3 download was attempted
                        assert mock_download.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
