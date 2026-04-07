from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class WebhookMemoryBridge:
    """
    Calls n8n / Dify (HTTP Request or workflow trigger) for long-term or platform memory.

    Request body always includes action + session_id so a single webhook can branch (Switch).
    """

    def __init__(
        self,
        webhook_url: str | None,
        store_webhook_url: str | None,
        session_id: str,
        memory_scope: str | None = None,
        *,
        timeout_seconds: float = 45.0,
        bearer_token: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.webhook_url = (webhook_url or "").strip() or None
        self.store_webhook_url = (store_webhook_url or "").strip() or None
        self.session_id = session_id
        self.memory_scope = (memory_scope or "").strip() or None
        self.timeout_seconds = timeout_seconds
        self._headers: dict[str, str] = dict(extra_headers or {})
        if bearer_token:
            self._headers["Authorization"] = f"Bearer {bearer_token}"

    @classmethod
    def from_env(
        cls, session_id: str, memory_scope: str | None = None
    ) -> WebhookMemoryBridge:
        base = os.getenv("EXTERNAL_MEMORY_WEBHOOK_URL", "").strip() or None
        store = os.getenv("EXTERNAL_MEMORY_STORE_WEBHOOK_URL", "").strip() or None
        token = os.getenv("EXTERNAL_MEMORY_WEBHOOK_BEARER", "").strip() or None
        timeout = float(os.getenv("EXTERNAL_MEMORY_WEBHOOK_TIMEOUT", "45"))
        return cls(
            webhook_url=base,
            store_webhook_url=store or None,
            session_id=session_id,
            memory_scope=memory_scope,
            timeout_seconds=timeout,
            bearer_token=token,
        )

    def is_configured(self) -> bool:
        return self.webhook_url is not None

    def _headers_out(self) -> dict[str, str]:
        return {**self._headers, "Content-Type": "application/json"}

    async def retrieve_similar(self, query: str) -> list[dict[str, Any]]:
        """Merge with Chroma hits: each item uses text, metadata, distance (optional)."""
        if not self.webhook_url:
            return []
        payload: dict[str, Any] = {
            "action": "retrieve_similar",
            "query": query,
            "session_id": self.session_id,
        }
        if self.memory_scope:
            payload["memory_scope"] = self.memory_scope
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers=self._headers_out(),
                )
                response.raise_for_status()
                data = _maybe_json(response)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("External memory retrieve_similar failed: %s", exc)
            return []

        return _parse_hits(data)

    async def retrieve_documents(self, query: str) -> list[dict[str, Any]]:
        """Same shape as web search results for memory_only routing (url, title, content)."""
        if not self.webhook_url:
            return []
        payload: dict[str, Any] = {
            "action": "retrieve_documents",
            "query": query,
            "session_id": self.session_id,
        }
        if self.memory_scope:
            payload["memory_scope"] = self.memory_scope
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers=self._headers_out(),
                )
                response.raise_for_status()
                data = _maybe_json(response)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("External memory retrieve_documents failed: %s", exc)
            return []

        return _parse_documents(data)

    async def store(self, text: str, metadata: dict[str, Any]) -> None:
        url = self.store_webhook_url or self.webhook_url
        if not url:
            return
        payload: dict[str, Any] = {
            "action": "store",
            "session_id": self.session_id,
            "text": text,
            "metadata": metadata,
        }
        if self.memory_scope:
            payload["memory_scope"] = self.memory_scope
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self._headers_out(),
                )
                response.raise_for_status()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("External memory store failed: %s", exc)


def _maybe_json(response: httpx.Response) -> Any:
    text = response.text.strip()
    if not text:
        return {}
    return response.json()


def _parse_hits(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    hits = data.get("hits") or data.get("chunks") or data.get("records") or []
    if not isinstance(hits, list):
        return []
    out: list[dict[str, Any]] = []
    for item in hits:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append({"text": text, "metadata": {}, "distance": 0.12})
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content") or ""
        if isinstance(text, str):
            text = text.strip()
        else:
            text = ""
        if not text:
            continue
        meta = item.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        dist = item.get("distance")
        try:
            distance = float(dist) if dist is not None else 0.12
        except (TypeError, ValueError):
            distance = 0.12
        out.append({"text": text, "metadata": meta, "distance": distance})
    return out


def _parse_documents(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    docs = data.get("results") or data.get("documents") or data.get("chunks") or []
    if not isinstance(docs, list):
        return []
    out: list[dict[str, Any]] = []
    for item in docs:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("text") or ""
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content:
            continue
        out.append(
            {
                "url": str(item.get("url", "") or ""),
                "title": str(item.get("title", "") or ""),
                "content": content,
                "score": float(item.get("score", 0.0) or 0.0),
            }
        )
    return out
