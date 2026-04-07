from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

VALID_ROUTES = frozenset({"deep_research", "memory_only", "blocked"})


@dataclass(frozen=True)
class RouteResult:
    route: str
    effective_query: str
    constraint_patch: dict[str, Any]


class QueryRouter:
    """POSTs to an n8n / Dify (HTTP trigger) workflow to classify and optionally rewrite the query."""

    def __init__(
        self,
        webhook_url: str | None,
        *,
        timeout_seconds: float = 30.0,
        bearer_token: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.webhook_url = (webhook_url or "").strip() or None
        self.timeout_seconds = timeout_seconds
        self._headers: dict[str, str] = dict(extra_headers or {})
        if bearer_token:
            self._headers["Authorization"] = f"Bearer {bearer_token}"

    @classmethod
    def from_env(cls) -> QueryRouter:
        url = os.getenv("QUERY_ROUTER_WEBHOOK_URL", "").strip() or None
        token = os.getenv("QUERY_ROUTER_WEBHOOK_BEARER", "").strip() or None
        timeout = float(os.getenv("QUERY_ROUTER_WEBHOOK_TIMEOUT", "30"))
        return cls(webhook_url=url, timeout_seconds=timeout, bearer_token=token)

    def is_configured(self) -> bool:
        return self.webhook_url is not None

    async def resolve(self, query: str, session_id: str) -> RouteResult:
        if not self.webhook_url:
            return RouteResult(
                route="deep_research",
                effective_query=query,
                constraint_patch={},
            )

        payload = {"query": query, "session_id": session_id}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={**self._headers, "Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("Query router webhook failed (%s); using deep_research.", exc)
            return RouteResult(
                route="deep_research",
                effective_query=query,
                constraint_patch={},
            )

        if not isinstance(data, dict):
            return RouteResult(
                route="deep_research",
                effective_query=query,
                constraint_patch={},
            )

        route = str(data.get("route", "deep_research")).strip()
        if route not in VALID_ROUTES:
            route = "deep_research"

        effective = data.get("query")
        if isinstance(effective, str) and effective.strip():
            effective_query = effective.strip()
        else:
            effective_query = query

        raw_constraints = data.get("constraints")
        constraint_patch: dict[str, Any] = {}
        if isinstance(raw_constraints, dict):
            for key in ("max_tokens", "max_cost_usd", "max_sub_questions"):
                if key in raw_constraints and raw_constraints[key] is not None:
                    constraint_patch[key] = raw_constraints[key]

        return RouteResult(
            route=route,
            effective_query=effective_query,
            constraint_patch=constraint_patch,
        )
