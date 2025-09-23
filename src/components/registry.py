"""
Unified registry for plugin architecture (single-class core).

Provides a single class `UnifiedRegistry` that manages all categories,
while exposing per-category adapter objects (e.g., `loader_registry`)
that keep the existing decorator and factory usage compatible.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, TypeVar

from .base import Component

T = TypeVar("T", bound=Component)


class UnifiedRegistry:
    """
    단일 레지스트리로 모든 카테고리의 컴포넌트를 관리한다.

    - 내부 구조: category -> { key -> class }
    - 메타데이터는 (category, key) 단위로 저장
    - kebab-case 키 검증, 중복 등록 방지, 클래스에 레지스트리 속성 부여 보장
    """

    def __init__(self, name: str = "unified"):
        self.name = name
        self._by_category: dict[str, dict[str, type[Component]]] = {}
        self._metadata: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _is_valid_key(key: str) -> bool:
        if not key:
            return False
        import re

        pattern = r"^[a-z0-9]+(-[a-z0-9]+)*$"
        return bool(re.match(pattern, key))

    def _ensure_category(self, category: str) -> None:
        if category not in self._by_category:
            self._by_category[category] = {}

    def register(
        self,
        key: str,
        *,
        category: str,
        version: str | None = None,
        description: str | None = None,
        **metadata: Any,
    ) -> Callable[[type[T]], type[T]]:
        """데코레이터를 반환하여 클래스 등록."""

        def decorator(cls: type[T]) -> type[T]:
            if not self._is_valid_key(key):
                raise ValueError(
                    f"Invalid registry key '{key}'. Keys must be lowercase with hyphens (kebab-case)"
                )
            self._ensure_category(category)
            category_map = self._by_category[category]
            if key in category_map:
                raise ValueError(
                    f"Component with key '{key}' is already registered in category '{category}' with class {category_map[key].__name__}"
                )
            category_map[key] = cls
            self._metadata[(category, key)] = {
                "category": category,
                "version": version,
                "description": description or cls.__doc__,
                "class_name": cls.__name__,
                "module": cls.__module__,
                **metadata,
            }
            cls._registry_key = key
            cls._registry_category = category
            cls._registry = self
            return cls

        return decorator

    def get(self, key: str, *, category: str) -> type[Component]:
        if category not in self._by_category or key not in self._by_category[category]:
            available = self.list_keys(category)
            raise KeyError(
                f"Component with key '{key}' not found in category '{category}' of registry '{self.name}'. Available keys: {available}"
            )
        return self._by_category[category][key]

    def create(
        self,
        key: str,
        *,
        category: str,
        config: dict[str, Any] | None = None,
    ) -> Component:
        cls = self.get(key, category=category)
        return cls(config) if config else cls()

    def exists(self, key: str, *, category: str) -> bool:
        return category in self._by_category and key in self._by_category[category]

    def list_keys(self, category: str | None = None) -> list[str]:
        if category is None:
            keys: list[str] = []
            for cat, mapping in self._by_category.items():
                keys.extend([f"{cat}:{k}" for k in mapping.keys()])
            return keys
        if category not in self._by_category:
            return []
        return list(self._by_category[category].keys())

    def get_metadata(self, key: str, *, category: str) -> dict[str, Any]:
        meta_key = (category, key)
        if meta_key not in self._metadata:
            raise KeyError(
                f"No metadata found for key '{key}' in category '{category}'"
            )
        return self._metadata[meta_key]

    def list_components(self, category: str | None = None) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        if category is None:
            for (cat, key), meta in self._metadata.items():
                result[f"{cat}:{key}"] = meta
            return result
        for key in self.list_keys(category):
            result[key] = self._metadata[(category, key)]
        return result

    def clear(self) -> None:
        self._by_category.clear()
        self._metadata.clear()

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_category.values())

    def __repr__(self) -> str:
        return (
            f"UnifiedRegistry(name='{self.name}', "
            f"categories={len(self._by_category)}, components={len(self)})"
        )


# Per-category adapter (no extra classes; only function-bound objects)


def _category_api(root: UnifiedRegistry, category: str):
    def register(
        key: str,
        *,
        category: str | None = None,
        version: str | None = None,
        description: str | None = None,
        **metadata: Any,
    ) -> Callable[[type[T]], type[T]]:
        return root.register(
            key,
            category=category or category_name,
            version=version,
            description=description,
            **metadata,
        )

    def get(key: str) -> type[Component]:
        return root.get(key, category=category_name)

    def create(key: str, config: dict[str, Any] | None = None) -> Component:
        return root.create(key, category=category_name, config=config)

    def exists(key: str) -> bool:
        return root.exists(key, category=category_name)

    def list_keys() -> list[str]:
        return root.list_keys(category_name)

    def get_metadata(key: str) -> dict[str, Any]:
        return root.get_metadata(key, category=category_name)

    category_name = category
    return SimpleNamespace(
        register=register,
        get=get,
        create=create,
        exists=exists,
        list_keys=list_keys,
        get_metadata=get_metadata,
    )


# Single unified registry and compatibility adapters
unified_registry = UnifiedRegistry("main")
loader_registry = _category_api(unified_registry, "loader")
scorer_registry = _category_api(unified_registry, "scorer")
trainer_registry = _category_api(unified_registry, "trainer")
optimizer_registry = _category_api(unified_registry, "optimizer")
evaluator_registry = _category_api(unified_registry, "evaluator")
pretrainer_registry = _category_api(unified_registry, "pretrainer")


def list_all() -> dict[str, Any]:
    return {
        "main": unified_registry,
        "loader": loader_registry,
        "scorer": scorer_registry,
        "trainer": trainer_registry,
        "optimizer": optimizer_registry,
        "evaluator": evaluator_registry,
        "pretrainer": pretrainer_registry,
    }


__all__ = [
    "UnifiedRegistry",
    "unified_registry",
    "list_all",
    "loader_registry",
    "scorer_registry",
    "trainer_registry",
    "optimizer_registry",
    "evaluator_registry",
    "pretrainer_registry",
]
