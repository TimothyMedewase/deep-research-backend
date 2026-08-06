from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models.schemas import ResearchConstraints


@dataclass
class ThreadState:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    last_answer: str = ""
    constraints: ResearchConstraints = field(default_factory=ResearchConstraints)
    token_count: int = 0
    cost_usd: float = 0.0
    context_chunks: list[str] = field(default_factory=list)
    memory_scope: str | None = None

    def to_stats(self) -> dict[str, float | int]:
        return {
            "used": self.token_count,
            "budget": self.constraints.max_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cost_budget": self.constraints.max_cost_usd,
        }


class ThreadStore:
    def __init__(self) -> None:
        self._threads: dict[str, ThreadState] = {}

    def get(self, session_id: str) -> ThreadState | None:
        return self._threads.get(session_id)

    def get_or_create(
        self,
        session_id: str,
        constraints: ResearchConstraints,
        memory_scope: str | None = None,
    ) -> ThreadState:
        existing = self._threads.get(session_id)
        if existing is not None:
            existing.constraints = constraints
            if memory_scope:
                existing.memory_scope = memory_scope
            return existing
        thread = ThreadState(
            session_id=session_id,
            constraints=constraints,
            memory_scope=memory_scope,
        )
        self._threads[session_id] = thread
        return thread

    def delete(self, session_id: str) -> bool:
        return self._threads.pop(session_id, None) is not None

    def update_from_tracker(
        self,
        thread: ThreadState,
        *,
        token_count: int,
        cost_usd: float,
        context_chunks: list[str],
    ) -> None:
        thread.token_count = token_count
        thread.cost_usd = cost_usd
        thread.context_chunks = list(context_chunks)

    def append_turn(
        self,
        thread: ThreadState,
        *,
        user_query: str,
        assistant_answer: str,
    ) -> None:
        thread.messages.append({"role": "user", "content": user_query})
        if assistant_answer:
            thread.messages.append({"role": "assistant", "content": assistant_answer})
            thread.last_answer = assistant_answer


THREAD_STORE = ThreadStore()
