"""
MBPP dataset loader with local-first S3 fallback support.

Implements loading and preprocessing for the MBPP (Mostly Basic Python Problems)
dataset used for evaluating code generation capabilities.
"""

import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from rich.console import Console

from ...components.registry import loader_registry
from .base_loader import DatasetLoader

console = Console()


@loader_registry.register("dataset-mbpp-loader", category="loader", version="v1")
class MBPPDatasetLoader(DatasetLoader):
    """
    MBPP dataset loader with local-first S3 fallback.

    The MBPP dataset contains Python programming problems with:
    - Problem descriptions
    - Test cases
    - Canonical solutions
    """

    DATASET_NAME = "mbpp"
    HUGGINGFACE_DATASET_ID = "google-research-datasets/mbpp"

    def __init__(self, config: dict[str, Any]):
        """
        Initialize MBPP loader.

        Args:
            config: Configuration with paths and S3 settings
        """
        super().__init__(config)

        # Extract local path for MBPP
        self.local_path = config.get("local_path")
        if self.local_path:
            self.local_path = Path(self.local_path)

        # Default split ratios
        self.split_ratios = {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        }

    def load(
        self,
        path: str | None = None,
        split: str | None = None,
        **kwargs,
    ) -> DatasetDict | Dataset:
        """
        Load MBPP dataset with local-first S3 fallback.

        Args:
            path: Dataset path (optional, uses config if not provided)
            split: Specific split to load ('train', 'val', 'test', or None for all)
            **kwargs: Additional loading arguments

        Returns:
            Loaded dataset or dataset dictionary
        """
        # Compute cache key for this dataset configuration
        cache_key = self.compute_cache_key(
            data_id=self.DATASET_NAME,
            version=kwargs.get("version", "latest"),
            preprocessing_config={
                "split": split,
                "max_length": kwargs.get("max_length", 2048),
            },
            split_seed=42,
        )

        # 1. Try local path first
        local_path = self.local_path or (Path(path) if path else None)
        if local_path and local_path.exists():
            console.print(f"[green]Loading MBPP from local: {local_path}[/green]")
            dataset = self._load_from_local(local_path)
            return self._process_dataset(dataset, split, **kwargs)

        # 2. Check cache
        cached_path = self.cache_dir / cache_key / "mbpp_processed.json"
        if cached_path.exists():
            console.print(f"[cyan]Loading MBPP from cache: {cached_path}[/cyan]")
            dataset = self._load_from_json(cached_path)
            return self._process_dataset(dataset, split, **kwargs)

        # 3. Try S3 with caching
        if self.s3_manager and self.s3_manager.connected:
            s3_key = "datasets/mbpp/mbpp_full.json"

            try:
                # Create cache directory
                cache_dir = self.cache_dir / cache_key
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Download from S3
                downloaded_path = self.s3_manager.download_if_missing(
                    s3_key,
                    cache_dir / "mbpp_full.json",
                    force=False,
                )

                if downloaded_path and downloaded_path.exists():
                    console.print(
                        f"[green]Loading MBPP from S3 cache: {downloaded_path}[/green]"
                    )
                    dataset = self._load_from_json(downloaded_path)

                    # Process and cache
                    processed = self._process_dataset(dataset, split, **kwargs)
                    self._save_to_cache(processed, cached_path)
                    return processed

            except Exception as e:
                console.print(f"[yellow]Could not load from S3: {e}[/yellow]")

        # 4. Fall back to HuggingFace datasets
        console.print(
            f"[cyan]Loading MBPP from HuggingFace: {self.HUGGINGFACE_DATASET_ID}[/cyan]"
        )

        try:
            # Load from HuggingFace
            dataset = load_dataset(
                self.HUGGINGFACE_DATASET_ID,
                cache_dir=str(self.cache_dir / "huggingface"),
            )

            # Process and cache
            processed = self._process_dataset(dataset, split, **kwargs)

            # Save to cache for next time
            cache_dir = self.cache_dir / cache_key
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._save_to_cache(processed, cache_dir / "mbpp_processed.json")

            return processed

        except Exception as e:
            raise RuntimeError(f"Failed to load MBPP dataset from any source: {e}")

    def preprocess(
        self,
        data: Dataset | DatasetDict,
        max_length: int = 2048,
        add_solution: bool = True,
        **kwargs,
    ) -> Dataset | DatasetDict:
        """
        Preprocess MBPP dataset.

        Args:
            data: Raw dataset
            max_length: Maximum sequence length
            add_solution: Whether to include canonical solution
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed dataset
        """

        def format_example(example):
            """Format a single MBPP example."""
            # Build prompt
            prompt_parts = [
                "# Problem:",
                example.get("text", example.get("prompt", "")),
                "",
                "# Test Cases:",
            ]

            # Add test cases if available
            if "test_list" in example:
                for test in example["test_list"][:3]:  # Show first 3 test cases
                    prompt_parts.append(f"# {test}")
            elif "test" in example:
                prompt_parts.append(f"# {example['test']}")

            prompt_parts.extend(["", "# Solution:"])
            prompt = "\n".join(prompt_parts)

            # Get solution
            solution = ""
            if add_solution:
                if "code" in example:
                    solution = example["code"]
                elif "solution" in example:
                    solution = example["solution"]

            return {
                "prompt": prompt,
                "solution": solution,
                "full_text": prompt + "\n" + solution if solution else prompt,
                "task_id": example.get("task_id", example.get("id", 0)),
            }

        # Apply preprocessing
        if isinstance(data, DatasetDict):
            processed = {}
            for split_name, split_data in data.items():
                processed[split_name] = split_data.map(
                    format_example,
                    remove_columns=split_data.column_names,
                    desc=f"Preprocessing {split_name}",
                )
            return DatasetDict(processed)
        else:
            return data.map(
                format_example,
                remove_columns=data.column_names,
                desc="Preprocessing dataset",
            )

    def create_splits(
        self,
        data: Dataset | list[dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Create train/val/test splits from data.

        Args:
            data: Dataset to split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility

        Returns:
            DatasetDict with splits
        """
        # Set random seed for reproducibility
        random.seed(seed)

        # Convert to list if needed
        if isinstance(data, Dataset):
            examples = [data[i] for i in range(len(data))]
        else:
            examples = data

        # Shuffle
        random.shuffle(examples)

        # Calculate split sizes
        n_total = len(examples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split data
        train_data = examples[:n_train]
        val_data = examples[n_train : n_train + n_val]
        test_data = examples[n_train + n_val :]

        # Create datasets
        splits = {
            "train": Dataset.from_list(train_data),
            "val": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data),
        }

        console.print(
            f"[green]Created splits - Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}[/green]"
        )

        return DatasetDict(splits)

    def _load_from_local(self, path: Path) -> Dataset | DatasetDict:
        """Load dataset from local path."""
        if path.is_file():
            # Load single file
            if path.suffix == ".json":
                return self._load_from_json(path)
            elif path.suffix == ".jsonl":
                return self._load_from_jsonl(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        elif path.is_dir():
            # Load from directory structure
            splits = {}
            for split_file in path.glob("*.json"):
                split_name = split_file.stem
                splits[split_name] = self._load_from_json(split_file)

            if not splits:
                # Try JSONL files
                for split_file in path.glob("*.jsonl"):
                    split_name = split_file.stem
                    splits[split_name] = self._load_from_jsonl(split_file)

            if splits:
                return DatasetDict(splits)
            else:
                raise ValueError(f"No dataset files found in {path}")
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

    def _load_from_json(self, path: Path) -> Dataset:
        """Load dataset from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict):
            if "data" in data:
                data = data["data"]
            elif "examples" in data:
                data = data["examples"]

        return Dataset.from_list(data)

    def _load_from_jsonl(self, path: Path) -> Dataset:
        """Load dataset from JSONL file."""
        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return Dataset.from_list(examples)

    def _save_to_cache(
        self,
        dataset: Dataset | DatasetDict,
        cache_path: Path,
    ) -> None:
        """Save processed dataset to cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(dataset, DatasetDict):
            # Save each split
            cache_data = {}
            for split_name, split_data in dataset.items():
                cache_data[split_name] = [split_data[i] for i in range(len(split_data))]

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        else:
            # Save single dataset
            data = [dataset[i] for i in range(len(dataset))]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def _process_dataset(
        self,
        dataset: Dataset | DatasetDict,
        split: str | None = None,
        **kwargs,
    ) -> Dataset | DatasetDict:
        """Process and optionally filter dataset by split."""
        # Preprocess
        processed = self.preprocess(dataset, **kwargs)

        # If specific split requested and we have a DatasetDict
        if split and isinstance(processed, DatasetDict):
            if split in processed:
                return processed[split]
            else:
                console.print(
                    f"[yellow]Split '{split}' not found. "
                    f"Available splits: {list(processed.keys())}[/yellow]"
                )
                return processed

        # If no splits exist, create them
        if not isinstance(processed, DatasetDict):
            processed = self.create_splits(processed, seed=42)

        return processed

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Execute loading operation within component framework.

        Args:
            ctx: Context dictionary

        Returns:
            Dictionary with loaded dataset
        """
        self.validate_initialized()

        # Extract parameters from context
        path = ctx.get("path") or ctx.get("dataset_path")
        split = ctx.get("split")
        max_length = ctx.get("max_length", 2048)
        add_solution = ctx.get("add_solution", True)

        # Load dataset
        dataset = self.load(
            path=path,
            split=split,
            max_length=max_length,
            add_solution=add_solution,
        )

        return {
            "dataset": dataset,
            "dataset_name": self.DATASET_NAME,
            "split": split,
            "loader": self.__class__.__name__,
        }
