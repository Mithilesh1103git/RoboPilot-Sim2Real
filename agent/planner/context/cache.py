"""Small in-memory TTL cache for normalized robot evidence."""

import time
from typing import Any


class ContextCache:
    def __init__(self, ttl_seconds: int = 86400):
        self.ttl_seconds = max(0, int(ttl_seconds))
        self._items: dict[str, tuple[float, Any]] = {}

    def get(self, key: str):
        item = self._items.get(key)
        if item is None:
            return None
        created, value = item
        if self.ttl_seconds == 0 or time.monotonic() - created > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._items.clear()
