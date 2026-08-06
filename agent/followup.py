from __future__ import annotations

import json
import logging
from typing import Any, Literal

from openai import AsyncOpenAI

from agent.constraints import ConstraintTracker

logger = logging.getLogger(__name__)

FollowUpDecision = Literal["answer_from_context", "needs_research"]


async def classify_followup(
    *,
    openai_client: AsyncOpenAI,
    model: str,
    query: str,
    chat_history: list[dict[str, str]],
    memory_hits: list[dict[str, Any]],
    last_answer: str,
    constraint_tracker: ConstraintTracker,
) -> FollowUpDecision:
    """Decide whether a follow-up can be answered from context or needs more research."""
    history_preview = "\n".join(
        f"{m.get('role', 'user')}: {(m.get('content') or '')[:500]}"
        for m in chat_history[-6:]
    )
    memory_preview = "\n---\n".join(
        (hit.get("text") or "")[:600] for hit in memory_hits[:5]
    )
    system_prompt = (
        "You classify research follow-up questions.\n"
        "Return ONLY JSON: {\"decision\": \"answer_from_context\"} or "
        "{\"decision\": \"needs_research\"}.\n"
        "Use answer_from_context when prior answer + memory clearly suffice.\n"
        "Use needs_research when new facts, sources, or web search are required."
    )
    user_prompt = (
        f"Follow-up query:\n{query}\n\n"
        f"Last answer (truncated):\n{(last_answer or '')[:2000]}\n\n"
        f"Recent chat:\n{history_preview or '(none)'}\n\n"
        f"Retrieved memory:\n{memory_preview or '(none)'}"
    )

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        usage = response.usage
        constraint_tracker.estimate_and_add_cost(
            model,
            getattr(usage, "prompt_tokens", ConstraintTracker.estimate_tokens(user_prompt)),
            getattr(
                usage,
                "completion_tokens",
                ConstraintTracker.estimate_tokens(content),
            ),
        )
        parsed = json.loads(content)
        decision = parsed.get("decision")
        if decision in ("answer_from_context", "needs_research"):
            return decision  # type: ignore[return-value]
    except Exception:
        logger.exception("Follow-up classification failed; defaulting to needs_research.")

    return "needs_research"
