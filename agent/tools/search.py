from __future__ import annotations

import asyncio
import logging
from typing import Any

from exa_py import Exa

logger = logging.getLogger(__name__)


def _exa_search_sync(client: Exa, query: str, max_results: int) -> Any:
    return client.search(
        query,
        num_results=max_results,
        contents={
            "text": {"max_characters": 8000},
            "highlights": {"max_characters": 2000, "query": query},
        },
    )


def _result_content(item: Any) -> str:
    text = getattr(item, "text", None) or ""
    if isinstance(text, str) and text.strip():
        return text.strip()
    highlights = getattr(item, "highlights", None) or []
    if isinstance(highlights, list) and highlights:
        return "\n".join(str(h) for h in highlights if h).strip()
    summary = getattr(item, "summary", None) or ""
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return ""


class WebSearchTool:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = Exa(api_key=api_key)

    async def search(self, query: str, max_results: int = 3) -> list[dict[str, Any]]:
        if not self.api_key.strip():
            logger.warning("Exa search skipped: EXA_API_KEY is not set.")
            return []
        try:
            response = await asyncio.to_thread(
                _exa_search_sync,
                self.client,
                query,
                max_results,
            )
        except Exception:
            logger.exception("Exa search failed for query: %s", query)
            return []

        raw = getattr(response, "results", None) or []
        out: list[dict[str, Any]] = []
        for item in raw:
            content = _result_content(item)
            if not content:
                continue
            out.append(
                {
                    "url": str(getattr(item, "url", "") or ""),
                    "title": str(getattr(item, "title", "") or ""),
                    "content": content,
                    "score": float(getattr(item, "score", 0.0) or 0.0),
                }
            )
        return out

    async def batch_search(self, queries: list[str]) -> list[list[dict[str, Any]]]:
        return await asyncio.gather(*(self.search(query) for query in queries))
