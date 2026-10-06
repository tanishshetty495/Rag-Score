"""
Judge response caching for ragmark.

Provides a cache keyed on the actual judge inputs (metric name, system prompt,
user prompt, and judge's identifying info) to avoid re-paying for identical
LLM judge calls when re-running evals on unchanged test cases.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rag_score.judges.base import JudgeVerdict


class JudgeCache(Protocol):
    """Async cache for judge verdicts."""

    async def get(self, key: str) -> JudgeVerdict | None:
        """Return cached verdict if present and not expired, else None."""
        ...

    async def set(self, key: str, verdict: JudgeVerdict) -> None:
        """Store a verdict in the cache."""
        ...

    def stats(self) -> tuple[int, int]:
        """Return (hits, misses) for this cache."""
        ...


@dataclass
class CacheConfig:
    """Configuration for the judge cache."""

    backend: str = "memory"  # "memory" or "file"
    path: str = ".ragmark_cache/judge_cache.db"  # for file backend
    max_age_seconds: int | None = None  # None means no expiry
    ignore_cache: bool = False  # escape hatch for forcing a fresh run


@dataclass
class _CacheEntry:
    """Internal representation of a cached verdict."""

    verdict: JudgeVerdict
    cached_at: float = field(default_factory=time.time)

    def is_expired(self, max_age_seconds: int | None) -> bool:
        if max_age_seconds is None:
            return False
        return (time.time() - self.cached_at) > max_age_seconds


class InMemoryCacheBackend:
    """Simple in-memory cache backend (plain dict). Safe for single-process use."""

    def __init__(self) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> JudgeVerdict | None:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        # Note: InMemoryCacheBackend doesn't handle expiry; that's handled by the caller
        # using max_age_seconds via the CacheConfig. We'll check expiry in the judge.
        self.hits += 1
        return entry.verdict

    async def set(self, key: str, verdict: JudgeVerdict) -> None:
        self._store[key] = _CacheEntry(verdict=verdict)

    def stats(self) -> tuple[int, int]:
        return self.hits, self.misses


class FileCacheBackend:
    """SQLite-based file cache backend with WAL mode for concurrent safety."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.hits = 0
        self.misses = 0
        self._conn: sqlite3.Connection | None = None
        print(f"FileCacheBackend initialized with path: {path}")

    async def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # Ensure the directory exists
            import os

            os.makedirs(os.path.dirname(self.path), exist_ok=True)

            self._conn = sqlite3.connect(self.path, timeout=10)
            # Enable WAL mode for better concurrent read/write
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    key TEXT PRIMARY KEY,
                    verdict_json TEXT NOT NULL,
                    cached_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()
        return self._conn

    async def get(self, key: str) -> JudgeVerdict | None:
        conn = await self._get_conn()
        cursor = conn.execute(
            "SELECT verdict_json, cached_at FROM judge_cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            self.misses += 1
            return None
        verdict_json, _ = row
        # Check expiry (handled by caller via max_age_seconds, but we can do it here too)
        # We'll let the caller handle expiry based on CacheConfig.max_age_seconds
        from rag_score.judges.base import JudgeVerdict

        self.hits += 1
        verdict = JudgeVerdict.model_validate_json(verdict_json)
        return verdict

    async def set(self, key: str, verdict: JudgeVerdict) -> None:
        print(f"FileCacheBackend.set called with key: {key}")
        conn = await self._get_conn()
        verdict_json = verdict.model_dump_json()
        conn.execute(
            """
            INSERT OR REPLACE INTO judge_cache (key, verdict_json, cached_at)
            VALUES (?, ?, ?)
            """,
            (key, verdict_json, time.time()),
        )
        conn.commit()
        print(f"FileCacheBackend.set committed for key: {key}")

    def stats(self) -> tuple[int, int]:
        return self.hits, self.misses


def _make_cache_key(
    metric_name: str,
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model_name: str,
) -> str:
    """Create a deterministic hash of the judge inputs.

    The key is a SHA256 hash of a JSON-serialized representation of the inputs,
    with sort_keys=True for stability.
    """
    payload = {
        "metric_name": metric_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "provider": provider,
        "model_name": model_name,
    }
    # Use separators=(',', ':') to eliminate whitespace and ensure compactness
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def get_cache_backend(config: CacheConfig) -> JudgeCache:
    """Factory function to get the appropriate cache backend."""
    if config.backend == "memory":
        return InMemoryCacheBackend()
    elif config.backend == "file":
        return FileCacheBackend(config.path)
    else:
        raise ValueError(f"Unknown cache backend: {config.backend}")
