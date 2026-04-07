from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from agent.constraints import ConstraintTracker
from agent.decomposer import QueryDecomposer
from agent.memory.manager import MemoryManager
from agent.research_tool_loop import run_subquestion_tool_loop
from agent.tools.search import WebSearchTool

if TYPE_CHECKING:
    from agent.integrations.webhook_memory import WebhookMemoryBridge


class ResearchOrchestrator:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        search_tool: WebSearchTool,
        memory_manager: MemoryManager,
        constraint_tracker: ConstraintTracker,
        session_id: str,
        decomposer: QueryDecomposer | None = None,
        external_memory: WebhookMemoryBridge | None = None,
        skip_web_search: bool = False,
        max_tool_rounds: int = 8,
    ) -> None:
        self.openai_client = openai_client
        self.search_tool = search_tool
        self.memory_manager = memory_manager
        self.constraint_tracker = constraint_tracker
        self.session_id = session_id
        self.external_memory = external_memory
        self.skip_web_search = skip_web_search
        self.max_tool_rounds = max_tool_rounds
        self.decomposer = decomposer or QueryDecomposer(
            openai_client=openai_client,
            max_sub_questions=getattr(constraint_tracker, "max_sub_questions", 4),
        )

    async def run(self, query: str) -> AsyncGenerator[dict[str, Any], None]:
        yield {"type": "thinking", "content": "Breaking down your research question..."}

        sub_questions = await self.decomposer.decompose(query)
        yield {
            "type": "decompose",
            "content": f"Identified {len(sub_questions)} sub-questions",
            "data": {"sub_questions": sub_questions},
        }
        self.constraint_tracker.estimate_and_add_cost(
            "gpt-4o",
            self.decomposer.last_input_tokens,
            self.decomposer.last_output_tokens,
        )

        for sub_question in sub_questions:
            yield {
                "type": "thinking",
                "content": f"Researching (tool use): {sub_question}",
            }

            if self.skip_web_search and not (
                self.external_memory and self.external_memory.is_configured()
            ):
                yield {
                    "type": "thinking",
                    "content": (
                        "Memory-only route: no external memory webhook; "
                        "agent can only use query_session_memory."
                    ),
                }

            async for ev in run_subquestion_tool_loop(
                openai_client=self.openai_client,
                constraint_tracker=self.constraint_tracker,
                memory_manager=self.memory_manager,
                search_tool=self.search_tool,
                external_memory=self.external_memory,
                sub_question=sub_question,
                skip_web_search=self.skip_web_search,
                max_rounds=self.max_tool_rounds,
            ):
                yield ev

            yield {"type": "token_update", "data": self.constraint_tracker.to_dict()}
            if self.constraint_tracker.is_over_cost_limit():
                yield {
                    "type": "thinking",
                    "content": "Cost limit reached. Synthesizing from available context.",
                }
                break

        yield {"type": "thinking", "content": "Synthesizing findings..."}

        system_prompt = (
            "You are a research analyst. Answer the user's original query comprehensively "
            "using ONLY the provided context.\n"
            "For every factual claim, cite the source URL in the format [Source: <url>].\n"
            "Be structured, precise, and thorough."
        )
        context_window = self.memory_manager.get_context_window()
        user_prompt = (
            f"Original query:\n{query}\n\n"
            f"Context:\n{context_window or 'No research context was collected.'}"
        )

        estimated_input_tokens = self.constraint_tracker.estimate_tokens(system_prompt) + (
            self.constraint_tracker.estimate_tokens(user_prompt)
        )
        emitted_chunks: list[str] = []
        observed_output_tokens = 0

        stream = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                estimated_input_tokens = getattr(
                    chunk.usage, "prompt_tokens", estimated_input_tokens
                )
                observed_output_tokens = getattr(
                    chunk.usage,
                    "completion_tokens",
                    observed_output_tokens,
                )

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            emitted_chunks.append(delta)
            yield {"type": "answer", "content": delta}

        if observed_output_tokens == 0:
            observed_output_tokens = self.constraint_tracker.estimate_tokens(
                "".join(emitted_chunks)
            )

        self.constraint_tracker.estimate_and_add_cost(
            "gpt-4o",
            estimated_input_tokens,
            observed_output_tokens,
        )
        yield {"type": "done"}
