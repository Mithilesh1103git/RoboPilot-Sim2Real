"""Optional Tavily Python SDK provider."""

import os


class TavilyProvider:
    name = "tavily"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float | None = None,
        max_results: int | None = None,
    ):
        self.api_key = (
            api_key if api_key is not None else os.getenv("TAVILY_API_KEY", "")
        ).strip()
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("TAVILY_TIMEOUT_SECONDS", "15"))
        )
        self.max_results = (
            max_results
            if max_results is not None
            else int(os.getenv("TAVILY_MAX_RESULTS", "5"))
        )
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise RuntimeError(
                "TAVILY_ENABLED is true but TAVILY_API_KEY is not configured."
            )
        if self._client is None:
            try:
                from tavily import TavilyClient
            except ImportError as exc:
                raise RuntimeError(
                    "Tavily is enabled but tavily-python is not installed. "
                    "Install it with: pip install tavily-python"
                ) from exc
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def retrieve(
        self,
        query: str,
        preferred_domains: list[str] | None = None,
        search_depth: str = "advanced",
    ) -> list[dict]:
        kwargs = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max(1, min(int(self.max_results), 10)),
            "include_answer": False,
            "include_raw_content": False,
            "timeout": self.timeout,
        }
        if preferred_domains:
            kwargs["include_domains"] = preferred_domains

        response = self._get_client().search(**kwargs)
        return [
            {**result, "source": self.name} for result in response.get("results", [])
        ]
