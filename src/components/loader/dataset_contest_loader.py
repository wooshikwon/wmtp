"""
CodeContests dataset loader with local-first S3 fallback support.

Implements loading and preprocessing for the CodeContests dataset
used for evaluating competitive programming capabilities.
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


@loader_registry.register("dataset-contest-loader", category="loader", version="v1")
class CodeContestsDatasetLoader(DatasetLoader):
    """
    CodeContests dataset loader with local-first S3 fallback.

    The CodeContests dataset contains competitive programming problems with:
    - Problem descriptions
    - Input/output examples
    - Test cases
    - Multiple solutions in various languages
    """

    DATASET_NAME = "codecontests"
    HUGGINGFACE_DATASET_ID = "deepmind/code_contests"

    def __init__(self, config: dict[str, Any]):
        """
        Initialize CodeContests loader.

        Args:
            config: Configuration with paths and S3 settings
        """
        super().__init__(config)

        # Extract local path for CodeContests
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
        language: str = "python3",
        **kwargs,
    ) -> DatasetDict | Dataset:
        """
        Load CodeContests dataset with local-first S3 fallback.

        Args:
            path: Dataset path (optional, uses config if not provided)
            split: Specific split to load ('train', 'val', 'test', or None for all)
            language: Programming language for solutions
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
                "language": language,
                "max_length": kwargs.get("max_length", 4096),
                "difficulty": kwargs.get("difficulty"),
            },
            split_seed=42,
        )

        # 1. Try local path first
        local_path = self.local_path or (Path(path) if path else None)
        if local_path and local_path.exists():
            console.print(
                f"[green]Loading CodeContests from local: {local_path}[/green]"
            )
            dataset = self._load_from_local(local_path)
            return self._process_dataset(dataset, split, language=language, **kwargs)

        # 2. Check cache
        cached_path = self.cache_dir / cache_key / "codecontests_processed.json"
        if cached_path.exists():
            console.print(
                f"[cyan]Loading CodeContests from cache: {cached_path}[/cyan]"
            )
            dataset = self._load_from_json(cached_path)
            return self._process_dataset(dataset, split, language=language, **kwargs)

        # 3. Try S3 with caching
        if self.s3_manager and self.s3_manager.connected:
            s3_key = "datasets/codecontests/codecontests_full.json"

            try:
                # Create cache directory
                cache_dir = self.cache_dir / cache_key
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Download from S3
                downloaded_path = self.s3_manager.download_if_missing(
                    s3_key,
                    cache_dir / "codecontests_full.json",
                    force=False,
                )

                if downloaded_path and downloaded_path.exists():
                    console.print(
                        f"[green]Loading CodeContests from S3 cache: {downloaded_path}[/green]"
                    )
                    dataset = self._load_from_json(downloaded_path)

                    # Process and cache
                    processed = self._process_dataset(
                        dataset, split, language=language, **kwargs
                    )
                    self._save_to_cache(processed, cached_path)
                    return processed

            except Exception as e:
                console.print(f"[yellow]Could not load from S3: {e}[/yellow]")

        # 4. Fall back to HuggingFace datasets
        console.print(
            f"[cyan]Loading CodeContests from HuggingFace: {self.HUGGINGFACE_DATASET_ID}[/cyan]"
        )

        try:
            # Load from HuggingFace
            dataset = load_dataset(
                self.HUGGINGFACE_DATASET_ID,
                cache_dir=str(self.cache_dir / "huggingface"),
            )

            # Process and cache
            processed = self._process_dataset(
                dataset, split, language=language, **kwargs
            )

            # Save to cache for next time
            cache_dir = self.cache_dir / cache_key
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._save_to_cache(processed, cache_dir / "codecontests_processed.json")

            return processed

        except Exception as e:
            raise RuntimeError(
                f"Failed to load CodeContests dataset from any source: {e}"
            )

    def preprocess(
        self,
        data: Dataset | DatasetDict,
        language: str = "python3",
        max_length: int = 4096,
        add_solution: bool = True,
        difficulty: str | None = None,
        **kwargs,
    ) -> Dataset | DatasetDict:
        """
        Preprocess CodeContests dataset.

        Args:
            data: Raw dataset
            language: Programming language for solutions
            max_length: Maximum sequence length
            add_solution: Whether to include solution
            difficulty: Filter by difficulty level
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed dataset
        """

        def format_example(example):
            """Format a single CodeContests example."""
            # Build problem statement
            prompt_parts = ["# Problem:"]

            # Add problem description
            if "description" in example:
                prompt_parts.append(example["description"])
            elif "problem_description" in example:
                prompt_parts.append(example["problem_description"])

            # Add input/output format
            if "input_specification" in example:
                prompt_parts.extend(["", "# Input:", example["input_specification"]])
            if "output_specification" in example:
                prompt_parts.extend(["", "# Output:", example["output_specification"]])

            # Add examples
            if "public_tests" in example and example["public_tests"]:
                prompt_parts.extend(["", "# Examples:"])
                tests = example["public_tests"]

                # Limit to first 3 examples
                for i, test in enumerate(tests[:3], 1):
                    if isinstance(test, dict):
                        input_str = test.get("input", "")
                        output_str = test.get("output", "")
                    else:
                        # Handle different data structures
                        input_str = str(test[0]) if len(test) > 0 else ""
                        output_str = str(test[1]) if len(test) > 1 else ""

                    prompt_parts.append(f"## Example {i}:")
                    prompt_parts.append(f"Input: {input_str}")
                    prompt_parts.append(f"Output: {output_str}")
                    prompt_parts.append("")

            # Add solution prompt
            prompt_parts.append("# Solution:")
            prompt = "\n".join(prompt_parts)

            # Get solution for the specified language
            solution = ""
            if add_solution:
                if "solutions" in example:
                    # Find solution in the requested language
                    solutions = example["solutions"]
                    if isinstance(solutions, dict):
                        if language in solutions:
                            solution = solutions[language]
                        elif "python" in solutions:
                            solution = solutions["python"]
                        elif "python3" in solutions:
                            solution = solutions["python3"]
                    elif isinstance(solutions, list):
                        # Look for solution in the specified language
                        for sol in solutions:
                            if isinstance(sol, dict):
                                if sol.get("language") == language:
                                    solution = sol.get("solution", "")
                                    break
                        # Fallback to first solution
                        if not solution and solutions:
                            if isinstance(solutions[0], str):
                                solution = solutions[0]
                            elif isinstance(solutions[0], dict):
                                solution = solutions[0].get("solution", "")
                elif "solution" in example:
                    solution = example["solution"]

            # Extract difficulty if available
            difficulty_level = example.get(
                "difficulty", example.get("rating", "unknown")
            )

            return {
                "prompt": prompt,
                "solution": solution,
                "full_text": prompt + "\n" + solution if solution else prompt,
                "task_id": example.get("name", example.get("id", 0)),
                "difficulty": difficulty_level,
                "language": language,
            }

        # Filter by difficulty if specified
        def filter_difficulty(example):
            if difficulty is None:
                return True
            return str(example.get("difficulty", "")).lower() == difficulty.lower()

        # Apply preprocessing
        if isinstance(data, DatasetDict):
            processed = {}
            for split_name, split_data in data.items():
                # Filter first if needed
                if difficulty:
                    split_data = split_data.filter(filter_difficulty)

                processed[split_name] = split_data.map(
                    format_example,
                    remove_columns=split_data.column_names,
                    desc=f"Preprocessing {split_name}",
                )
            return DatasetDict(processed)
        else:
            # Filter first if needed
            if difficulty:
                data = data.filter(filter_difficulty)

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

        # Group by difficulty if available
        by_difficulty = {}
        for ex in examples:
            diff = ex.get("difficulty", "unknown")
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(ex)

        # Stratified sampling by difficulty
        train_data = []
        val_data = []
        test_data = []

        for diff, diff_examples in by_difficulty.items():
            # Shuffle examples within difficulty
            random.shuffle(diff_examples)

            # Calculate split sizes for this difficulty
            n_total = len(diff_examples)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            # Split
            train_data.extend(diff_examples[:n_train])
            val_data.extend(diff_examples[n_train : n_train + n_val])
            test_data.extend(diff_examples[n_train + n_val :])

        # Final shuffle
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

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
            elif "problems" in data:
                data = data["problems"]
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
        language = ctx.get("language", "python3")
        max_length = ctx.get("max_length", 4096)
        add_solution = ctx.get("add_solution", True)
        difficulty = ctx.get("difficulty")

        # Load dataset
        dataset = self.load(
            path=path,
            split=split,
            language=language,
            max_length=max_length,
            add_solution=add_solution,
            difficulty=difficulty,
        )

        return {
            "dataset": dataset,
            "dataset_name": self.DATASET_NAME,
            "split": split,
            "language": language,
            "loader": self.__class__.__name__,
        }
