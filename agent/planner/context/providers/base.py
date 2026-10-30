"""Provider interface for external and local planner context sources."""

from abc import ABC, abstractmethod
from typing import Any


class ContextProvider(ABC):
    name = "base"

    @abstractmethod
    def retrieve(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Return normalized evidence records."""
