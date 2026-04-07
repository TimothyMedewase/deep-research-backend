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
from agent.integrations.query_router import QueryRouter
from agent.integrations.webhook_memory import WebhookMemoryBridge
from agent.json_utils import make_json_safe
from agent.memory.manager import MemoryManager
from agent.memory.summarizer import Summarizer
from agent.memory.vector_store import VectorStore, VectorStoreError
from agent.orchestrator import ResearchOrchestrator
from agent.tools.search import WebSearchTool
from models.schemas import ConstraintConfig, ResearchRequest

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


def _build_constraint_config(request_config: ConstraintConfig | None) -> ConstraintConfig:
    env_defaults = ConstraintConfig(
        max_tokens=int(os.getenv("MAX_TOKENS_PER_SESSION", "8000")),
        max_cost_usd=float(os.getenv("MAX_COST_PER_SESSION", "0.10")),
        max_sub_questions=int(os.getenv("MAX_SUB_QUESTIONS", "4")),
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
    tracker: ConstraintTracker, config: ConstraintConfig, patch: dict
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

    constraint_tracker = ConstraintTracker(
        max_tokens=config.max_tokens,
        max_cost_usd=config.max_cost_usd,
    )
    constraint_tracker.max_sub_questions = config.max_sub_questions
    ACTIVE_SESSIONS[session_id] = constraint_tracker

    async def event_generator():
        vector_store: VectorStore | None = None
        try:
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
            memory_scope = request.memory_scope or chroma_config["default_scope"] or session_id
            external_memory = WebhookMemoryBridge.from_env(session_id, memory_scope)

            vector_store = VectorStore(
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
            summarizer = Summarizer(app.state.openai_client)
            memory_manager = MemoryManager(
                session_id=session_id,
                vector_store=vector_store,
                summarizer=summarizer,
                constraint_tracker=constraint_tracker,
                compression_threshold=int(os.getenv("COMPRESSION_THRESHOLD", "6000")),
                external_memory=external_memory
                if external_memory.is_configured()
                else None,
            )
            search_tool = WebSearchTool(api_key=os.getenv("EXA_API_KEY", ""))
            decomposer = QueryDecomposer(
                openai_client=app.state.openai_client,
                max_sub_questions=config.max_sub_questions,
            )
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
                max_tool_rounds=int(os.getenv("RESEARCH_TOOL_MAX_ROUNDS", "8")),
            )
            async for event in orchestrator.run(route.effective_query):
                yield {"data": _event_json(event)}
        except Exception as exc:
            logger.exception("Research session %s failed.", session_id)
            yield {"data": _event_json({"type": "error", "content": str(exc)})}
            yield {"data": _event_json({"type": "done"})}
        finally:
            if vector_store is not None:
                try:
                    vector_store.delete_collection()
                except VectorStoreError:
                    logger.exception("Failed to clean up vector store for session %s.", session_id)
            ACTIVE_SESSIONS.pop(session_id, None)

    return EventSourceResponse(event_generator())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/session/{session_id}/stats")
async def session_stats(session_id: str) -> dict[str, float | int]:
    tracker = ACTIVE_SESSIONS.get(session_id)
    if tracker is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return tracker.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
