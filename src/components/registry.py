"""
Component registry for plugin architecture.

This module implements a decorator-based registry pattern that allows
components to be registered with kebab-case keys and retrieved dynamically.
"""

from collections.abc import Callable
from typing import Any, TypeVar

from .base import Component

T = TypeVar("T", bound=Component)


class ComponentRegistry:
    """
    Registry for managing component classes.

    Provides decorator-based registration and dynamic retrieval
    of components by string keys.
    """

    def __init__(self, name: str = "default"):
        """
        Initialize registry.

        Args:
            name: Registry name for identification
        """
        self.name = name
        self._registry: dict[str, type[Component]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        key: str,
        *,
        category: str | None = None,
        version: str | None = None,
        description: str | None = None,
        **metadata: Any,
    ) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a component class.

        Args:
            key: Registry key (kebab-case)
            category: Component category (e.g., 'scorer', 'trainer')
            version: Component version
            description: Component description
            **metadata: Additional metadata

        Returns:
            Decorator function

        Example:
            @registry.register("critic-delta-v1", category="scorer")
            class CriticDeltaScorer:
                ...
        """

        def decorator(cls: type[T]) -> type[T]:
            # Validate key format (kebab-case)
            if not self._is_valid_key(key):
                raise ValueError(
                    f"Invalid registry key '{key}'. "
                    "Keys must be lowercase with hyphens (kebab-case)"
                )

            # Check for duplicate registration
            if key in self._registry:
                raise ValueError(
                    f"Component with key '{key}' is already registered "
                    f"with class {self._registry[key].__name__}"
                )

            # Register the component
            self._registry[key] = cls

            # Store metadata
            self._metadata[key] = {
                "category": category,
                "version": version,
                "description": description or cls.__doc__,
                "class_name": cls.__name__,
                "module": cls.__module__,
                **metadata,
            }

            # Add registry info to class
            cls._registry_key = key
            cls._registry = self

            return cls

        return decorator

    def get(self, key: str) -> type[Component]:
        """
        Retrieve a component class by key.

        Args:
            key: Registry key

        Returns:
            Component class

        Raises:
            KeyError: If key not found
        """
        if key not in self._registry:
            available = self.list_keys()
            raise KeyError(
                f"Component with key '{key}' not found in registry '{self.name}'. "
                f"Available keys: {available}"
            )
        return self._registry[key]

    def create(self, key: str, config: dict[str, Any] | None = None) -> Component:
        """
        Create a component instance by key.

        Args:
            key: Registry key
            config: Component configuration

        Returns:
            Component instance
        """
        cls = self.get(key)
        return cls(config) if config else cls()

    def exists(self, key: str) -> bool:
        """Check if a key exists in the registry."""
        return key in self._registry

    def list_keys(self, category: str | None = None) -> list[str]:
        """
        List all registered keys.

        Args:
            category: Filter by category

        Returns:
            List of registry keys
        """
        if category:
            return [
                k
                for k, meta in self._metadata.items()
                if meta.get("category") == category
            ]
        return list(self._registry.keys())

    def get_metadata(self, key: str) -> dict[str, Any]:
        """Get metadata for a registered component."""
        if key not in self._metadata:
            raise KeyError(f"No metadata found for key '{key}'")
        return self._metadata[key]

    def list_components(self, category: str | None = None) -> dict[str, dict[str, Any]]:
        """
        List all components with metadata.

        Args:
            category: Filter by category

        Returns:
            Dictionary of components and their metadata
        """
        result = {}
        for key in self.list_keys(category):
            result[key] = self._metadata[key]
        return result

    def clear(self) -> None:
        """Clear all registrations."""
        self._registry.clear()
        self._metadata.clear()

    @staticmethod
    def _is_valid_key(key: str) -> bool:
        """
        Validate registry key format.

        Keys should be lowercase kebab-case.
        """
        if not key:
            return False

        # Check for valid characters (lowercase letters, numbers, hyphens)
        import re

        pattern = r"^[a-z0-9]+(-[a-z0-9]+)*$"
        return bool(re.match(pattern, key))

    def __contains__(self, key: str) -> bool:
        """Check if key exists using 'in' operator."""
        return self.exists(key)

    def __len__(self) -> int:
        """Get number of registered components."""
        return len(self._registry)

    def __repr__(self) -> str:
        """String representation."""
        return f"ComponentRegistry(name='{self.name}', components={len(self)})"


# Create global registries for each component type
loader_registry = ComponentRegistry("loader")
scorer_registry = ComponentRegistry("scorer")
trainer_registry = ComponentRegistry("trainer")
optimizer_registry = ComponentRegistry("optimizer")
evaluator_registry = ComponentRegistry("evaluator")

# Main registry that can hold any component type
main_registry = ComponentRegistry("main")


def register(key: str, **kwargs) -> Callable:
    """
    Convenience decorator to register with main registry.

    Args:
        key: Registry key
        **kwargs: Additional metadata

    Returns:
        Decorator function
    """
    return main_registry.register(key, **kwargs)


def get(key: str) -> type[Component]:
    """
    Convenience function to get from main registry.

    Args:
        key: Registry key

    Returns:
        Component class
    """
    return main_registry.get(key)


def create(key: str, config: dict[str, Any] | None = None) -> Component:
    """
    Convenience function to create from main registry.

    Args:
        key: Registry key
        config: Component configuration

    Returns:
        Component instance
    """
    return main_registry.create(key, config)


def list_all() -> dict[str, ComponentRegistry]:
    """
    List all available registries.

    Returns:
        Dictionary of registry name to registry object
    """
    return {
        "main": main_registry,
        "loader": loader_registry,
        "scorer": scorer_registry,
        "trainer": trainer_registry,
        "optimizer": optimizer_registry,
        "evaluator": evaluator_registry,
    }


class Registry:
    """Convenience class for unified registry access."""

    def register(self, key: str, category: str | None = None, **kwargs) -> Callable:
        """Register a component with the appropriate registry."""
        if category == "scorer":
            return scorer_registry.register(key, category=category, **kwargs)
        elif category == "trainer":
            return trainer_registry.register(key, category=category, **kwargs)
        elif category == "optimizer":
            return optimizer_registry.register(key, category=category, **kwargs)
        elif category == "loader":
            return loader_registry.register(key, category=category, **kwargs)
        elif category == "evaluator":
            return evaluator_registry.register(key, category=category, **kwargs)
        else:
            return main_registry.register(key, category=category, **kwargs)

    def get(self, key: str, category: str | None = None) -> type[Component]:
        """Get a component from the appropriate registry."""
        if category == "scorer":
            return scorer_registry.get(key)
        elif category == "trainer":
            return trainer_registry.get(key)
        elif category == "optimizer":
            return optimizer_registry.get(key)
        elif category == "loader":
            return loader_registry.get(key)
        elif category == "evaluator":
            return evaluator_registry.get(key)
        else:
            return main_registry.get(key)

    def create(
        self,
        key: str,
        config: dict[str, Any] | None = None,
        category: str | None = None,
    ) -> Component:
        """Create a component from the appropriate registry."""
        if category == "scorer":
            return scorer_registry.create(key, config)
        elif category == "trainer":
            return trainer_registry.create(key, config)
        elif category == "optimizer":
            return optimizer_registry.create(key, config)
        elif category == "loader":
            return loader_registry.create(key, config)
        elif category == "evaluator":
            return evaluator_registry.create(key, config)
        else:
            return main_registry.create(key, config)


# Create unified registry interface
registry = Registry()


# Export main functions and registries
__all__ = [
    "ComponentRegistry",
    "Registry",
    "registry",
    "register",
    "get",
    "create",
    "list_all",
    "main_registry",
    "loader_registry",
    "scorer_registry",
    "trainer_registry",
    "optimizer_registry",
    "evaluator_registry",
]
