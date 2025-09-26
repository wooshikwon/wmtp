"""
Unified registry for plugin architecture (single-class core).

Provides a single class `UnifiedRegistry` that manages all categories,
while exposing per-category adapter objects (e.g., `loader_registry`)
that keep the existing decorator and factory usage compatible.
"""

from __future__ import annotations

from collections.abc import Callable
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

    def _get(self, key: str, *, category: str) -> type[Component]:
        """내부용: create()에서만 사용하는 클래스 조회 메서드"""
        if category not in self._by_category or key not in self._by_category[category]:
            available = list(self._by_category.get(category, {}).keys())
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
        cls = self._get(key, category=category)
        return cls(config) if config else cls()

    # 사용되지 않는 메서드들 제거됨 (exists, list_keys, get_metadata, list_components, clear)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_category.values())

    def __repr__(self) -> str:
        return (
            f"UnifiedRegistry(name='{self.name}', "
            f"categories={len(self._by_category)}, components={len(self)})"
        )


# 단일 통합 레지스트리 - 모든 컴포넌트를 여기서 관리
registry = UnifiedRegistry("main")


# 하위 호환성을 위한 기존 인터페이스 유지 (내부적으로 registry 사용)
class _CompatibilityAdapter:
    """기존 *_registry.register() 패턴 지원을 위한 어댑터"""

    def __init__(self, category: str):
        self.category = category

    def register(self, key: str, **kwargs):
        # category 파라미터 제거 (중복)
        kwargs.pop("category", None)
        return registry.register(key, category=self.category, **kwargs)

    def create(self, key: str, config=None):
        return registry.create(key, category=self.category, config=config)

    def list_keys(self, category: str = None):
        """List all keys in this category."""
        # If category is provided and matches, or if not provided, use self.category
        if category is None or category == self.category:
            return list(registry._by_category.get(self.category, {}).keys())
        return []


# 기존 인터페이스 유지 - 코드 변경 없이 작동
loader_registry = _CompatibilityAdapter("loader")
trainer_registry = _CompatibilityAdapter("trainer")
optimizer_registry = _CompatibilityAdapter("optimizer")
evaluator_registry = _CompatibilityAdapter("evaluator")
pretrainer_registry = _CompatibilityAdapter("pretrainer")
tokenizer_registry = _CompatibilityAdapter("tokenizer")


# list_all 함수 제거됨 (사용되지 않음)


__all__ = [
    "UnifiedRegistry",
    "registry",
    "loader_registry",
    "trainer_registry",
    "optimizer_registry",
    "evaluator_registry",
    "pretrainer_registry",
    "tokenizer_registry",
]
