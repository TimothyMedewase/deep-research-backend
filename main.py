from __future__ import annotations

import json
import logging
import os
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from sse_starlette.sse import EventSourceResponse

from agent.constraints import ConstraintTracker
from agent.decomposer import QueryDecomposer
from agent.followup import classify_followup
from agent.integrations.query_router import QueryRouter
from agent.integrations.webhook_memory import WebhookMemoryBridge
from agent.json_utils import make_json_safe
from agent.memory.manager import MemoryManager
from agent.memory.summarizer import Summarizer
from agent.memory.vector_store import VectorStore, VectorStoreError
from agent.orchestrator import ResearchOrchestrator, TurnMode
from agent.thread_store import THREAD_STORE
from agent.tools.search import WebSearchTool
from models.schemas import ResearchConstraints, ResearchRequest

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _n8n_integration_required() -> bool:
    return os.getenv("N8N_INTEGRATION_REQUIRED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _router_constraints_enabled() -> bool:
    return os.getenv("QUERY_ROUTER_APPLY_CONSTRAINTS", "false").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _json_default(value: object) -> object:
    return make_json_safe(value)


def _event_json(event: object) -> str:
    return json.dumps(event, default=_json_default)


app = FastAPI(title="Deep Research Agent")
ACTIVE_SESSIONS: dict[str, ConstraintTracker] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_constraint_config(
    request_config: ResearchConstraints | None,
) -> ResearchConstraints:
    env_defaults = ResearchConstraints(
        max_tokens=int(os.getenv("MAX_TOKENS_PER_SESSION", "8000")),
        max_cost_usd=float(os.getenv("MAX_COST_PER_SESSION", "0.10")),
        max_sub_questions=int(os.getenv("MAX_SUB_QUESTIONS", "3")),
        compression_threshold=int(os.getenv("COMPRESSION_THRESHOLD", "6000")),
        max_tool_rounds=int(os.getenv("RESEARCH_TOOL_MAX_ROUNDS", "8")),
    )
    if request_config is None:
        return env_defaults
    return env_defaults.model_copy(update=request_config.model_dump(exclude_unset=True))


def _resolve_chroma_config() -> dict[str, str | None]:
    chroma_url = os.getenv("CHROMA_URL", "").strip() or None
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db").strip() or None
    return {
        "chroma_url": chroma_url,
        "persist_dir": persist_dir,
        "chroma_host": os.getenv("CHROMA_HOST", "").strip() or None,
        "chroma_api_key": os.getenv("CHROMA_API_KEY", "").strip() or None,
        "chroma_tenant": os.getenv("CHROMA_TENANT", "").strip() or None,
        "chroma_database": os.getenv("CHROMA_DATABASE", "").strip() or None,
        "collection_name": os.getenv("CHROMA_COLLECTION_NAME", "").strip() or None,
        "collection_prefix": os.getenv("CHROMA_COLLECTION_PREFIX", "research-memory").strip()
        or "research-memory",
        "default_scope": os.getenv("CHROMA_DEFAULT_SCOPE", "").strip() or None,
    }


def _apply_route_patch(
    tracker: ConstraintTracker, config: ResearchConstraints, patch: dict
) -> dict:
    applied: dict[str, float | int] = {}
    if not patch:
        return applied
    if not _router_constraints_enabled():
        return applied
    if "max_tokens" in patch:
        try:
            tracker.max_tokens = int(patch["max_tokens"])
            config.max_tokens = tracker.max_tokens
            applied["max_tokens"] = tracker.max_tokens
        except (TypeError, ValueError):
            pass
    if "max_cost_usd" in patch:
        try:
            tracker.max_cost_usd = float(patch["max_cost_usd"])
            config.max_cost_usd = tracker.max_cost_usd
            applied["max_cost_usd"] = tracker.max_cost_usd
        except (TypeError, ValueError):
            pass
    if "max_sub_questions" in patch:
        try:
            n = int(patch["max_sub_questions"])
            tracker.max_sub_questions = n
            config.max_sub_questions = n
            applied["max_sub_questions"] = n
        except (TypeError, ValueError):
            pass
    return applied


def _build_vector_store(
    session_id: str,
    memory_scope: str,
    chroma_config: dict[str, str | None],
) -> VectorStore:
    return VectorStore(
        session_id=session_id,
        persist_dir=chroma_config["persist_dir"],
        chroma_url=chroma_config["chroma_url"],
        chroma_host=chroma_config["chroma_host"],
        chroma_api_key=chroma_config["chroma_api_key"],
        chroma_tenant=chroma_config["chroma_tenant"],
        chroma_database=chroma_config["chroma_database"],
        collection_name=chroma_config["collection_name"],
        collection_prefix=str(chroma_config["collection_prefix"]),
        memory_scope=memory_scope,
        openai_client=app.state.openai_client,
    )


async def _cleanup_session(session_id: str, memory_scope: str | None = None) -> None:
    THREAD_STORE.delete(session_id)
    ACTIVE_SESSIONS.pop(session_id, None)
    chroma_config = _resolve_chroma_config()
    scope = memory_scope or chroma_config["default_scope"] or session_id
    try:
        store = _build_vector_store(session_id, scope, chroma_config)
        store.delete_collection(force=True)
    except Exception:
        logger.exception("Failed to delete vector store for session %s.", session_id)


@app.on_event("startup")
async def startup_event() -> None:
    if _n8n_integration_required():
        missing: list[str] = []
        if not os.getenv("QUERY_ROUTER_WEBHOOK_URL", "").strip():
            missing.append("QUERY_ROUTER_WEBHOOK_URL")
        if not os.getenv("EXTERNAL_MEMORY_WEBHOOK_URL", "").strip():
            missing.append("EXTERNAL_MEMORY_WEBHOOK_URL")
        if missing:
            msg = (
                "n8n/Dify webhooks are required (N8N_INTEGRATION_REQUIRED is true by default) "
                f"but these are unset: {', '.join(missing)}. "
                "Configure them or set N8N_INTEGRATION_REQUIRED=false for local dev without workflows."
            )
            logger.error(msg)
            raise RuntimeError(msg)
    app.state.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("G3 backend server ready.")


@app.post("/research")
async def research(request: ResearchRequest) -> EventSourceResponse:
    session_id = request.session_id or str(uuid4())
    config = _build_constraint_config(request.constraints)

    if request.reset:
        await _cleanup_session(session_id, request.memory_scope)

    existing_thread = THREAD_STORE.get(session_id)
    is_follow_up = bool(existing_thread and existing_thread.messages and not request.reset)
    if request.mode == "research":
        is_follow_up = False
    elif request.mode == "follow_up":
        is_follow_up = True

    thread = THREAD_STORE.get_or_create(
        session_id,
        config,
        memory_scope=request.memory_scope,
    )

    constraint_tracker = ConstraintTracker(
        max_tokens=config.max_tokens,
        max_cost_usd=config.max_cost_usd,
    )
    constraint_tracker.max_sub_questions = config.max_sub_questions
    constraint_tracker.token_count = thread.token_count
    constraint_tracker.cost_usd = thread.cost_usd
    ACTIVE_SESSIONS[session_id] = constraint_tracker

    async def event_generator():
        vector_store: VectorStore | None = None
        answer_text = ""
        try:
            budget_exhausted = (
                constraint_tracker.is_over_cost_limit()
                or constraint_tracker.is_over_token_limit()
            )
            # First turn with no prior context cannot proceed if already over budget.
            # Follow-ups soft-fail into synthesize-only so prior research remains usable.
            if budget_exhausted and not is_follow_up and not thread.messages:
                yield {
                    "data": _event_json(
                        {
                            "type": "error",
                            "content": (
                                "Thread budget already exhausted. "
                                "Start a new research session or raise limits."
                            ),
                        }
                    )
                }
                yield {"data": _event_json({"type": "done"})}
                return

            router = QueryRouter.from_env()
            route = await router.resolve(request.query, session_id)
            applied_constraint_patch = _apply_route_patch(
                constraint_tracker, config, route.constraint_patch
            )

            if router.is_configured():
                yield {
                    "data": _event_json(
                        {
                            "type": "route",
                            "content": f"Route: {route.route}",
                            "data": {
                                "route": route.route,
                                "effective_query": route.effective_query,
                                "constraints_applied": applied_constraint_patch,
                            },
                        }
                    ),
                }

            if route.route == "blocked":
                yield {
                    "data": _event_json(
                        {
                            "type": "error",
                            "content": "Query blocked by router workflow.",
                        }
                    )
                }
                yield {"data": _event_json({"type": "done"})}
                return

            skip_web = route.route == "memory_only"
            chroma_config = _resolve_chroma_config()
            memory_scope = (
                request.memory_scope
                or thread.memory_scope
                or chroma_config["default_scope"]
                or session_id
            )
            thread.memory_scope = memory_scope
            external_memory = WebhookMemoryBridge.from_env(session_id, memory_scope)

            vector_store = _build_vector_store(session_id, memory_scope, chroma_config)
            summarizer = Summarizer(
                app.state.openai_client,
                model=config.compression_model,
            )
            memory_manager = MemoryManager(
                session_id=session_id,
                vector_store=vector_store,
                summarizer=summarizer,
                constraint_tracker=constraint_tracker,
                compression_threshold=config.compression_threshold,
                external_memory=external_memory
                if external_memory.is_configured()
                else None,
                memory_top_k=config.memory_top_k,
                context_chunks=thread.context_chunks,
            )
            search_tool = WebSearchTool(api_key=os.getenv("EXA_API_KEY", ""))
            decomposer = QueryDecomposer(
                openai_client=app.state.openai_client,
                max_sub_questions=config.max_sub_questions,
                model=config.research_model,
            )

            turn_mode: TurnMode = "deep"
            if is_follow_up:
                memory_hits = await memory_manager.retrieve_relevant(request.query)
                if memory_hits:
                    yield {
                        "data": _event_json(
                            {
                                "type": "memory_hit",
                                "content": "Retrieved prior session memory for follow-up",
                                "data": {"chunks": memory_hits[: config.memory_top_k]},
                            }
                        )
                    }

                if budget_exhausted:
                    turn_mode = "synthesize_only"
                    yield {
                        "data": _event_json(
                            {
                                "type": "route",
                                "content": (
                                    "Follow-up: budget exhausted; answering from "
                                    "existing context only"
                                ),
                                "data": {
                                    "route": "follow_up:budget_limited",
                                    "turn_mode": turn_mode,
                                    "token_state": constraint_tracker.to_dict(),
                                },
                            }
                        )
                    }
                    yield {
                        "data": _event_json(
                            {
                                "type": "thinking",
                                "content": (
                                    "Thread budget reached after prior research. "
                                    "Answering from saved context (no new tool calls). "
                                    "Raise max cost/tokens or start a new research for "
                                    "fresh web search."
                                ),
                            }
                        )
                    }
                else:
                    decision = await classify_followup(
                        openai_client=app.state.openai_client,
                        model=config.compression_model,
                        query=request.query,
                        chat_history=thread.messages,
                        memory_hits=memory_hits,
                        last_answer=thread.last_answer,
                        constraint_tracker=constraint_tracker,
                    )
                    turn_mode = (
                        "synthesize_only"
                        if decision == "answer_from_context"
                        else "light"
                    )
                    yield {
                        "data": _event_json(
                            {
                                "type": "route",
                                "content": f"Follow-up: {decision}",
                                "data": {
                                    "route": f"follow_up:{decision}",
                                    "turn_mode": turn_mode,
                                },
                            }
                        )
                    }
            else:
                yield {
                    "data": _event_json(
                        {
                            "type": "route",
                            "content": "Deep research turn",
                            "data": {"route": "deep", "turn_mode": "deep"},
                        }
                    )
                }

            orchestrator = ResearchOrchestrator(
                openai_client=app.state.openai_client,
                search_tool=search_tool,
                memory_manager=memory_manager,
                constraint_tracker=constraint_tracker,
                session_id=session_id,
                decomposer=decomposer,
                external_memory=external_memory
                if external_memory.is_configured()
                else None,
                skip_web_search=skip_web,
                max_tool_rounds=config.max_tool_rounds,
                research_model=config.research_model,
                search_max_results=config.search_max_results,
                max_sub_questions=config.max_sub_questions,
            )

            async for event in orchestrator.run(
                route.effective_query,
                turn_mode=turn_mode,
                chat_history=thread.messages,
            ):
                if event.get("type") == "answer" and isinstance(event.get("content"), str):
                    answer_text += event["content"]
                yield {"data": _event_json(event)}

            try:
                if answer_text:
                    await memory_manager.add_chunk(
                        answer_text,
                        {
                            "title": "Prior assistant answer",
                            "document_kind": "prior_answer",
                            "session_id": session_id,
                        },
                    )
            except Exception:
                logger.exception(
                    "Failed to persist prior answer into memory for session %s.",
                    session_id,
                )

            THREAD_STORE.update_from_tracker(
                thread,
                token_count=constraint_tracker.token_count,
                cost_usd=constraint_tracker.cost_usd,
                context_chunks=memory_manager.context_chunks,
            )
            THREAD_STORE.append_turn(
                thread,
                user_query=request.query,
                assistant_answer=answer_text or orchestrator.last_answer,
            )
        except Exception as exc:
            logger.exception("Research session %s failed.", session_id)
            THREAD_STORE.update_from_tracker(
                thread,
                token_count=constraint_tracker.token_count,
                cost_usd=constraint_tracker.cost_usd,
                context_chunks=thread.context_chunks,
            )
            yield {"data": _event_json({"type": "error", "content": str(exc)})}
            yield {"data": _event_json({"type": "done"})}
        finally:
            ACTIVE_SESSIONS[session_id] = constraint_tracker

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/session/{session_id}/stats")
async def session_stats(session_id: str) -> dict[str, float | int]:
    thread = THREAD_STORE.get(session_id)
    if thread is not None:
        return thread.to_stats()
    tracker = ACTIVE_SESSIONS.get(session_id)
    if tracker is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return tracker.to_dict()


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    thread = THREAD_STORE.get(session_id)
    memory_scope = thread.memory_scope if thread else None
    await _cleanup_session(session_id, memory_scope)
    return {"status": "deleted", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
