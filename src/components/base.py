"""
Base interfaces for all components using Protocol.

This module defines the common interfaces that all components must implement,
enabling a plugin architecture where components can be easily swapped.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Component(Protocol):
    """
    Common interface for all components in the WMTP framework.

    All components must implement setup and run methods to participate
    in the registry/factory pattern.
    """

    def setup(self, ctx: dict[str, Any]) -> None:
        """
        Initialize the component with context.

        Args:
            ctx: Context dictionary containing configuration and shared resources
        """
        ...

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the component's main logic.

        Args:
            ctx: Input context containing data and configuration

        Returns:
            Output dictionary containing results
        """
        ...


class BaseComponent(ABC):
    """
    Abstract base class providing common functionality for components.

    Concrete components can inherit from this class to get default
    implementations and shared utilities.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize base component.

        Args:
            config: Component-specific configuration
        """
        self.config = config or {}
        self.initialized = False

    def setup(self, ctx: dict[str, Any]) -> None:
        """
        Default setup implementation.

        Args:
            ctx: Context dictionary
        """
        self.ctx = ctx
        self.initialized = True

    @abstractmethod
    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the component logic (must be implemented by subclasses).

        Args:
            ctx: Input context

        Returns:
            Output dictionary
        """
        pass

    def validate_initialized(self) -> None:
        """Check if component has been initialized."""
        if not self.initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} must be initialized with setup() before use"
            )


@runtime_checkable
class Loader(Protocol):
    """Interface for data and model loading components."""

    def load(self, path: str, **kwargs) -> Any:
        """Load data or model from path."""
        ...

    def setup(self, ctx: dict[str, Any]) -> None:
        """Initialize loader."""
        ...

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute loading operation."""
        ...


@runtime_checkable
class Scorer(Protocol):
    """Interface for token importance scoring components."""

    def score(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Compute token importance scores."""
        ...

    def setup(self, ctx: dict[str, Any]) -> None:
        """Initialize scorer."""
        ...

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute scoring operation."""
        ...


@runtime_checkable
class Trainer(Protocol):
    """Interface for model training components."""

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Execute single training step."""
        ...

    def setup(self, ctx: dict[str, Any]) -> None:
        """Initialize trainer."""
        ...

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute training loop."""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Interface for optimizer components."""

    def step(self) -> None:
        """Execute optimization step."""
        ...

    def zero_grad(self) -> None:
        """Zero gradients."""
        ...

    def setup(self, ctx: dict[str, Any]) -> None:
        """Initialize optimizer."""
        ...

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute optimization."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Interface for evaluation components."""

    def evaluate(self, data: Any, **kwargs) -> dict[str, Any]:
        """Evaluate model on data."""
        ...

    def setup(self, ctx: dict[str, Any]) -> None:
        """Initialize evaluator."""
        ...

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Execute evaluation."""
        ...


class ComponentContext:
    """
    Context manager for component execution.

    Provides a structured way to pass data and configuration
    between components in a pipeline.
    """

    def __init__(self, initial: dict[str, Any] | None = None):
        """
        Initialize context.

        Args:
            initial: Initial context values
        """
        self._data = initial or {}
        self._history: list[tuple[str, Any]] = []

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in context."""
        self._data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        """Update context with dictionary."""
        self._data.update(data)

    def checkpoint(self, name: str) -> None:
        """Save current context state."""
        self._history.append((name, self._data.copy()))

    @property
    def data(self) -> dict[str, Any]:
        """Get full context data."""
        return self._data

    def __contains__(self, key: str) -> bool:
        """Check if key exists in context."""
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        """Get item from context."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item in context."""
        self._data[key] = value
