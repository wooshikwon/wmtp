"""
File I/O utilities for WMTP framework.

This module provides functions for reading and writing datasets from various
file formats including JSON, disk-based formats, and HuggingFace datasets.
"""

import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from rich.console import Console

console = Console()


def load_dataset_from_json(path: str | Path) -> Dataset:
    """
    Load dataset from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dataset loaded from JSON

    Raises:
        FileNotFoundError: If file doesn't exist
        JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Handle both list and dict with 'data' key
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError(f"Expected list or dict with 'data' key, got {type(data)}")

    return Dataset.from_list(data)


def load_dataset_from_disk(
    path: str | Path,
    split: str | None = None,
) -> Dataset:
    """
    Load dataset from disk in HuggingFace format.

    Args:
        path: Path to dataset directory
        split: Optional split to load

    Returns:
        Loaded dataset or specific split

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If split not found in dataset
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    dataset = load_from_disk(str(path))

    # Handle DatasetDict
    if isinstance(dataset, DatasetDict):
        if split:
            if split in dataset:
                return dataset[split]
            else:
                raise ValueError(
                    f"Split '{split}' not found. Available: {list(dataset.keys())}"
                )
        else:
            # Return first split if no split specified
            console.print(
                f"[yellow]No split specified, available: {list(dataset.keys())}[/yellow]"
            )
            first_split = list(dataset.keys())[0]
            console.print(f"[yellow]Returning split: {first_split}[/yellow]")
            return dataset[first_split]

    return dataset


def save_dataset_to_json(
    dataset: Dataset,
    path: str | Path,
    indent: int = 2,
) -> None:
    """
    Save dataset to JSON file.

    Args:
        dataset: Dataset to save
        path: Output path for JSON file
        indent: JSON indentation (None for compact)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataset to list of dicts
    data = dataset.to_list()

    with open(path, "w") as f:
        json.dump({"data": data}, f, indent=indent)

    console.print(f"[green]Saved {len(data)} examples to {path}[/green]")


def save_dataset_to_disk(
    dataset: Dataset | DatasetDict,
    path: str | Path,
) -> None:
    """
    Save dataset to disk in HuggingFace format.

    Args:
        dataset: Dataset or DatasetDict to save
        path: Output directory path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(path))
    console.print(f"[green]Saved dataset to {path}[/green]")


def read_jsonl(path: str | Path) -> list[dict]:
    """
    Read JSONL (JSON Lines) file.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries from JSONL file
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    data = []
    with open(path) as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    return data


def write_jsonl(
    data: list[dict],
    path: str | Path,
) -> None:
    """
    Write data to JSONL (JSON Lines) file.

    Args:
        data: List of dictionaries to write
        path: Output path for JSONL file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    console.print(f"[green]Wrote {len(data)} lines to {path}[/green]")


def ensure_path_exists(path: str | Path, is_dir: bool = False) -> Path:
    """
    Ensure a path exists, creating parent directories if needed.

    Args:
        path: Path to check/create
        is_dir: Whether path is a directory (True) or file (False)

    Returns:
        Path object

    Raises:
        ValueError: If path exists but is wrong type (file vs dir)
    """
    path = Path(path)

    if path.exists():
        if is_dir and not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")
        elif not is_dir and path.is_dir():
            raise ValueError(f"Path exists but is a directory: {path}")
        return path

    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def get_file_size(path: str | Path, human_readable: bool = True) -> str | int:
    """
    Get file size in bytes or human-readable format.

    Args:
        path: Path to file
        human_readable: Return human-readable string if True

    Returns:
        File size in bytes (int) or human-readable string
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size_bytes = path.stat().st_size

    if not human_readable:
        return size_bytes

    # Convert to human-readable format
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} PB"
