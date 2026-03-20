"""
Prefix Cache — stores computed KV states for sequence prefixes.

When a sequence is first submitted:
  1. Run the full prompt through the model → get KV cache state
  2. Store that state keyed by a hash of the prompt tokens

On subsequent steps (or new requests with same prefix):
  1. Look up the cached KV state
  2. Only run the new tokens through the model
  3. This is O(1) per token instead of O(n)

This is what makes continuous batching fast and output coherent.
"""

import hashlib
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def hash_prompt(prompt: str) -> str:
    """Stable hash of a prompt string."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


@dataclass
class CachedState:
    """Cached KV state for a prompt prefix."""
    prompt_hash:   str
    prompt:        str
    prompt_tokens: int
    kv_state:      Any        # mlx prompt_cache object
    created_at:    float = field(default_factory=time.time)
    last_access:   float = field(default_factory=time.time)
    hit_count:     int = 0

    def touch(self):
        self.last_access = time.time()
        self.hit_count += 1


class PrefixCache:
    """
    LRU cache of prompt KV states.
    Thread-safe for concurrent sequence access.
    """

    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._cache: Dict[str, CachedState] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, prompt: str) -> Optional[CachedState]:
        key = hash_prompt(prompt)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.touch()
                self._hits += 1
                logger.debug(f"Prefix cache HIT [{key}] — {entry.prompt_tokens} tokens")
                return entry
            self._misses += 1
            return None

    def put(self, prompt: str, prompt_tokens: int, kv_state: Any):
        key = hash_prompt(prompt)
        with self._lock:
            if len(self._cache) >= self.max_entries:
                self._evict_lru()
            self._cache[key] = CachedState(
                prompt_hash=key,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                kv_state=kv_state,
            )
            logger.debug(f"Prefix cache STORE [{key}] — {prompt_tokens} tokens")

    def _evict_lru(self):
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k].last_access)
        del self._cache[lru_key]
        self._evictions += 1

    def invalidate(self, prompt: str):
        key = hash_prompt(prompt)
        with self._lock:
            self._cache.pop(key, None)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return round(self._hits / total, 3)

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries":       len(self._cache),
                "max_entries":   self.max_entries,
                "hits":          self._hits,
                "misses":        self._misses,
                "evictions":     self._evictions,
                "hit_rate":      self.hit_rate(),
                "hit_rate_pct":  round(self.hit_rate() * 100, 1),
            }

    def clear(self):
        with self._lock:
            self._cache.clear()


# Global singleton
prefix_cache = PrefixCache(max_entries=64)
