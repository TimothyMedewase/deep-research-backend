from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from agent.constraints import ConstraintTracker
from agent.integrations.webhook_memory import WebhookMemoryBridge
from agent.json_utils import make_json_safe
from agent.memory.manager import MemoryManager
from agent.tools.search import WebSearchTool

RESEARCH_MODEL = "gpt-4o"

SYSTEM_PROMPT = (
    "You are a research agent gathering evidence for one sub-question of a larger query.\n"
    "You have tools: use query_session_memory to see what is already in session context, "
    "and search_web or fetch_external_documents to pull new sources.\n"
    "Call tools as needed. When you have enough material, respond with a single short sentence "
    "summary (no tool calls)."
)


def _tools_for_mode(
    skip_web_search: bool, external_configured: bool
) -> list[dict[str, Any]]:
    memory_tool = {
        "type": "function",
        "function": {
            "name": "query_session_memory",
            "description": (
                "Semantic search over the current session's stored research chunks "
                "(vector + any external memory hits merged in)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query aligned with what to recall from session memory.",
                    },
                },
                "required": ["query"],
            },
        },
    }
    if skip_web_search:
        tools: list[dict[str, Any]] = [memory_tool]
        if external_configured:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_external_documents",
                        "description": (
                            "Load document snippets from the external memory / workflow store "
                            "(n8n or Dify-backed webhook)."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "What to retrieve from external long-term memory.",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            )
        return tools

    return [
        memory_tool,
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the public web for sources (Exa).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Focused web search query.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def _json_safe_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        if hasattr(tc, "model_dump"):
            dumped = tc.model_dump()
            if isinstance(dumped, dict):
                out.append(make_json_safe(dumped))
                continue
        out.append(
            make_json_safe(
                {
                    "id": getattr(tc, "id", ""),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", ""),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                    },
                }
            )
        )
    return out


def _json_dumps(value: Any) -> str:
    return json.dumps(make_json_safe(value), ensure_ascii=False)


async def run_subquestion_tool_loop(
    *,
    openai_client: AsyncOpenAI,
    constraint_tracker: ConstraintTracker,
    memory_manager: MemoryManager,
    search_tool: WebSearchTool,
    external_memory: WebhookMemoryBridge | None,
    sub_question: str,
    skip_web_search: bool,
    max_rounds: int,
) -> AsyncGenerator[dict[str, Any], None]:
    """LLM-driven research via OpenAI tool calls; ingests web/external hits into MemoryManager."""
    external_ok = bool(
        external_memory and external_memory.is_configured()
    )
    tools = _tools_for_mode(skip_web_search, external_ok)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Sub-question:\n{sub_question}\n\n"
                "Use tools to gather evidence. Finish with one brief sentence when done."
            ),
        },
    ]

    for round_i in range(max_rounds):
        if constraint_tracker.is_over_cost_limit():
            yield {
                "type": "error",
                "content": "Cost limit reached; stopping tool research for this sub-question.",
            }
            break

        if constraint_tracker.is_over_token_limit():
            yield {
                "type": "error",
                "content": "Session token budget full; stopping tool research for this sub-question.",
            }
            break

        response = await openai_client.chat.completions.create(
            model=RESEARCH_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        usage = response.usage
        in_tok = getattr(
            usage, "prompt_tokens", constraint_tracker.estimate_tokens(json.dumps(messages))
        )
        out_tok = getattr(
            usage,
            "completion_tokens",
            constraint_tracker.estimate_tokens(
                (msg.content or "") + _json_dumps(_json_safe_tool_calls(msg.tool_calls))
            ),
        )
        constraint_tracker.estimate_and_add_cost(RESEARCH_MODEL, in_tok, out_tok)

        if not msg.tool_calls:
            if msg.content:
                yield {
                    "type": "thinking",
                    "content": f"Sub-question wrap-up: {msg.content.strip()}",
                }
            break

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            yield {
                "type": "tool_call",
                "content": name,
                "data": {"name": name, "arguments": args, "round": round_i + 1},
            }

            tool_text: str
            if name == "query_session_memory":
                q = str(args.get("query", "")).strip() or sub_question
                hits = await memory_manager.retrieve_relevant(q)
                close = [h for h in hits if h.get("distance", 1.0) < 0.4]
                if close:
                    yield {
                        "type": "memory_hit",
                        "content": "Session memory (tool)",
                        "data": {"chunks": close},
                    }
                tool_text = _json_dumps(
                    {
                        "hits": [
                            {
                                "text": (h.get("text") or "")[:2000],
                                "distance": h.get("distance"),
                                "metadata": h.get("metadata") or {},
                            }
                            for h in hits[:8]
                        ]
                    }
                )

            elif name == "search_web" and not skip_web_search:
                q = str(args.get("query", "")).strip() or sub_question
                results = await search_tool.search(q)
                yield {
                    "type": "search",
                    "content": f"search_web: {q}",
                    "data": {"results_count": len(results), "source": "exa"},
                }
                ingested = 0
                for result in results:
                    if not result.get("content"):
                        continue
                    upd = await memory_manager.add_chunk(
                        result["content"],
                        {
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "sub_question": sub_question,
                        },
                    )
                    if upd.get("skipped"):
                        yield {
                            "type": "error",
                            "content": "Session token budget full; skipping further ingest.",
                            "data": upd,
                        }
                        break
                    if upd.get("compressed"):
                        yield {
                            "type": "compress",
                            "content": (
                                "Context compressed to stay within "
                                f"{constraint_tracker.max_tokens} token budget"
                            ),
                            "data": {"saved_tokens": upd.get("saved_tokens", 0)},
                        }
                    ingested += 1
                tool_text = _json_dumps(
                    {"ingested_count": ingested, "results_preview": results[:5]}
                )

            elif name == "fetch_external_documents" and skip_web_search and external_ok:
                q = str(args.get("query", "")).strip() or sub_question
                assert external_memory is not None
                results = await external_memory.retrieve_documents(q)
                yield {
                    "type": "search",
                    "content": f"fetch_external_documents: {q}",
                    "data": {
                        "results_count": len(results),
                        "source": "external_memory",
                    },
                }
                ingested = 0
                for result in results:
                    if not result.get("content"):
                        continue
                    upd = await memory_manager.add_chunk(
                        result["content"],
                        {
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "sub_question": sub_question,
                        },
                    )
                    if upd.get("skipped"):
                        yield {
                            "type": "error",
                            "content": "Session token budget full; skipping further ingest.",
                            "data": upd,
                        }
                        break
                    if upd.get("compressed"):
                        yield {
                            "type": "compress",
                            "content": (
                                "Context compressed to stay within "
                                f"{constraint_tracker.max_tokens} token budget"
                            ),
                            "data": {"saved_tokens": upd.get("saved_tokens", 0)},
                        }
                    ingested += 1
                tool_text = _json_dumps(
                    {"ingested_count": ingested, "results_preview": results[:5]}
                )

            else:
                tool_text = _json_dumps({"error": f"unknown or unavailable tool: {name}"})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_text,
                }
            )

        if constraint_tracker.is_over_cost_limit():
            break
