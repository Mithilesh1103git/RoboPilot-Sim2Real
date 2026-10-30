"""Build a compact, structured prompt payload from planner context."""

from typing import Any


def build_context_payload(context: dict[str, Any]) -> dict[str, Any]:
    # Keep evidence snippets bounded. The planner receives traceable excerpts,
    # not full scraped pages or arbitrary raw HTML.
    result = dict(context)
    result["evidence"] = [
        {
            **item,
            "snippet": str(item.get("snippet", ""))[:1800],
        }
        for item in context.get("evidence", [])
    ]
    return result
