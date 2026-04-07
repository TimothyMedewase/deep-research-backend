from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.constraints import ConstraintTracker
from agent.memory.summarizer import Summarizer
from agent.memory.vector_store import VectorStore

if TYPE_CHECKING:
    from agent.integrations.webhook_memory import WebhookMemoryBridge


class MemoryManager:
    def __init__(
        self,
        session_id: str,
        vector_store: VectorStore,
        summarizer: Summarizer,
        constraint_tracker: ConstraintTracker,
        compression_threshold: int,
        external_memory: WebhookMemoryBridge | None = None,
    ) -> None:
        self.session_id = session_id
        self.vector_store = vector_store
        self.summarizer = summarizer
        self.constraint_tracker = constraint_tracker
        self.compression_threshold = compression_threshold
        self.external_memory = external_memory
        self.context_chunks: list[str] = []

    async def _compress_oldest_n(self, n_to_compress: int) -> dict[str, Any]:
        n_to_compress = max(1, min(n_to_compress, len(self.context_chunks)))
        if len(self.context_chunks) < n_to_compress:
            return {"compressed": False, "saved_tokens": 0}

        original_chunks = self.context_chunks[:n_to_compress]
        original_tokens = sum(
            self.constraint_tracker.estimate_tokens(chunk) for chunk in original_chunks
        )

        summary, remaining_chunks, input_tokens, output_tokens = (
            await self.summarizer.compress_oldest(
                self.context_chunks,
                n_to_compress=n_to_compress,
            )
        )
        self.constraint_tracker.estimate_and_add_cost(
            "gpt-4o-mini",
            input_tokens,
            output_tokens,
        )

        summary_text = summary or "\n\n".join(original_chunks)
        summary_tokens = self.constraint_tracker.estimate_tokens(summary_text)
        self.context_chunks = [summary_text, *remaining_chunks]
        self.constraint_tracker.token_count = max(
            self.constraint_tracker.token_count - original_tokens + summary_tokens,
            0,
        )

        return {
            "compressed": True,
            "saved_tokens": max(original_tokens - summary_tokens, 0),
        }

    async def add_chunk(self, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        chunk_text = self._format_chunk(text, metadata)
        new_tokens = self.constraint_tracker.estimate_tokens(chunk_text)

        while (
            self.constraint_tracker.token_count + new_tokens
            > self.constraint_tracker.max_tokens
            and len(self.context_chunks) >= 2
        ):
            prev_count = self.constraint_tracker.token_count
            upd = await self._compress_oldest_n(
                min(2, len(self.context_chunks)),
            )
            if not upd.get("compressed"):
                break
            if self.constraint_tracker.token_count >= prev_count:
                break

        if self.constraint_tracker.token_count + new_tokens > self.constraint_tracker.max_tokens:
            return {
                "compressed": False,
                "skipped": True,
                "reason": "session_token_budget_exceeded",
            }

        self.context_chunks.append(chunk_text)
        self.constraint_tracker.add_tokens(new_tokens)

        embedding_tokens = await self.vector_store.add(chunk_text, metadata)
        self.constraint_tracker.estimate_and_add_cost(
            "text-embedding-3-small",
            embedding_tokens,
        )
        await self._store_external(chunk_text, metadata)

        if not self.constraint_tracker.is_over_compression_threshold(
            self.compression_threshold
        ):
            return {"compressed": False}

        return await self._compress_oldest_n(min(2, len(self.context_chunks)))

    async def retrieve_relevant(self, query: str) -> list[dict[str, Any]]:
        local = await self.vector_store.query(query, n_results=3)
        if not self.external_memory or not self.external_memory.is_configured():
            return local
        external = await self.external_memory.retrieve_similar(query)
        return [*external, *local]

    async def _store_external(self, chunk_text: str, metadata: dict[str, Any]) -> None:
        if self.external_memory and self.external_memory.is_configured():
            await self.external_memory.store(chunk_text, metadata)

    def get_context_window(self) -> str:
        return "\n\n---\n\n".join(self.context_chunks)

    def get_token_count(self) -> int:
        return self.constraint_tracker.token_count

    @staticmethod
    def _format_chunk(text: str, metadata: dict[str, Any]) -> str:
        title = metadata.get("title")
        url = metadata.get("url")
        if not title and not url:
            return text

        lines = []
        if title:
            lines.append(f"Title: {title}")
        if url:
            lines.append(f"URL: {url}")
        lines.append("Content:")
        lines.append(text)
        return "\n".join(lines)
