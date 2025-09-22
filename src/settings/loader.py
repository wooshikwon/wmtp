"""
YAML configuration loader with Pydantic validation.

This module provides functions to load and validate configuration files
with detailed error messages for debugging.
"""

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .config_schema import Config
from .recipe_schema import Recipe

console = Console()
T = TypeVar("T", bound=BaseModel)


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""

    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and return as dictionary.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary containing YAML contents

    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    path = Path(path)

    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    if not path.is_file():
        raise ConfigurationError(f"Path is not a file: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ConfigurationError(f"Empty configuration file: {path}")

        return data

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file {path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration from {path}: {e}")


def validate_schema(data: dict[str, Any], schema: type[T], name: str = "config") -> T:
    """
    Validate data against a Pydantic schema with detailed error reporting.

    Args:
        data: Dictionary to validate
        schema: Pydantic model class
        name: Configuration name for error messages

    Returns:
        Validated Pydantic model instance

    Raises:
        ConfigurationError: If validation fails
    """
    try:
        return schema(**data)
    except ValidationError as e:
        # Format detailed error message
        error_messages = []
        for error in e.errors():
            loc = " → ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            error_messages.append(f"  • {loc}: {msg}")

        error_detail = "\n".join(error_messages)
        raise ConfigurationError(
            f"Validation failed for {name}:\n{error_detail}\n\n"
            f"Please check your configuration against the schema requirements."
        )
    except Exception as e:
        raise ConfigurationError(f"Unexpected error validating {name}: {e}")


def load_config(path: str | Path, verbose: bool = False) -> Config:
    """
    Load and validate environment configuration.

    Args:
        path: Path to config.yaml file
        verbose: Whether to print loaded configuration

    Returns:
        Validated Config object

    Raises:
        ConfigurationError: If loading or validation fails
    """
    path = Path(path)

    if verbose:
        console.print(f"[cyan]Loading configuration from {path}...[/cyan]")

    # Load YAML
    data = load_yaml(path)

    # Validate against schema
    config = validate_schema(data, Config, "config.yaml")

    if verbose:
        # Pretty print configuration
        yaml_str = yaml.dump(data, default_flow_style=False)
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
        panel = Panel(syntax, title="[green]Configuration Loaded[/green]", expand=False)
        console.print(panel)

    return config


def load_recipe(path: str | Path, verbose: bool = False) -> Recipe:
    """
    Load and validate recipe configuration.

    Args:
        path: Path to recipe.yaml file
        verbose: Whether to print loaded configuration

    Returns:
        Validated Recipe object

    Raises:
        ConfigurationError: If loading or validation fails
    """
    path = Path(path)

    if verbose:
        console.print(f"[cyan]Loading recipe from {path}...[/cyan]")

    # Load YAML
    data = load_yaml(path)

    # Validate against schema
    recipe = validate_schema(data, Recipe, "recipe.yaml")

    if verbose:
        # Pretty print key settings
        console.print("[green]Recipe loaded:[/green]")
        console.print(f"  • Run: {recipe.run.name}")
        console.print(f"  • Algorithm: {recipe.train.algo}")
        console.print(f"  • Model: {recipe.model.base_id}")
        console.print(f"  • Learning Rate: {recipe.optim.lr}")
        console.print(f"  • Batch Tokens: {recipe.batching.global_batch_tokens:,}")

    return recipe


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration with overrides applied
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Config | Recipe, path: str | Path) -> None:
    """
    Save a configuration object to YAML file.

    Args:
        config: Configuration object to save
        path: Path to save to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(exclude_none=True, exclude_unset=False)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Configuration saved to {path}[/green]")


# Export main functions
__all__ = [
    "load_config",
    "load_recipe",
    "load_yaml",
    "validate_schema",
    "merge_configs",
    "save_config",
    "ConfigurationError",
]
