"""
Response caching for LLM judges in RAG evaluation.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for cache backends."""

    def get(self, key: str) -> str | None:
        """Get a value from the cache."""
        ...

    def set(self, key: str, value: str) -> None:
        """Set a value in the cache."""
        ...


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the cache backend."""

    type: str = "memory"  # "memory" or "file"
    directory: str | None = None  # Used for file backend
    max_size: int = 1000  # Maximum number of entries


class BaseCacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Get a value from the cache."""

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Set a value in the cache."""


class InMemoryCacheBackend(BaseCacheBackend):
    """Simple in-memory cache backend."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Get a value from the cache."""
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        """Set a value in the cache."""
        self._store[key] = value


class FileCacheBackend(BaseCacheBackend):
    """File-based cache backend."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use MD5 hash to create a safe filename
        hash_hash = hashlib.md5(key.encode()).hexdigest()
        return self.directory / f"{hash_hash}.json"

    def get(self, key: str) -> str | None:
        """Get a value from the cache."""
        file_path = self._get_file_path(key)
        if file_path.is_file():
            try:
                return file_path.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    def set(self, key: str, value: str) -> None:
        """Set a value in the cache."""
        file_path = self._get_file_path(key)
        try:
            file_path.write_text(value, encoding="utf-8")
        except OSError:
            pass  # Silently fail if we can't write to disk


class JudgeCache:
    """Cache for LLM judge responses to avoid duplicate API calls."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._backend: CacheBackend = self._get_backend(self.config)

    def _get_backend(self, config: CacheConfig) -> CacheBackend:
        """Get the appropriate cache backend based on config."""
        if config.type == "file":
            if not config.directory:
                raise ValueError("directory must be specified for file cache")
            return FileCacheBackend(config.directory)
        # Default to memory backend
        return InMemoryCacheBackend()

    def get(self, key: str) -> str | None:
        """Get a cached value by key."""
        return self._backend.get(key)

    def set(self, key: str, value: str) -> None:
        """Cache a value by key."""
        self._backend.set(key, value)


def get_cache_backend(config: CacheConfig | None = None) -> CacheBackend:
    """Get a cache backend instance based on configuration.

    Args:
        config: Cache configuration. If None, uses default CacheConfig.

    Returns:
        A cache backend instance.
    """
    config = config or CacheConfig()
    if config.type == "file":
        if not config.directory:
            raise ValueError("directory must be specified for file cache")
        return FileCacheBackend(config.directory)
    return InMemoryCacheBackend()


__all__ = [
    "CacheConfig",
    "FileCacheBackend",
    "InMemoryCacheBackend",
    "JudgeCache",
    "get_cache_backend",
]