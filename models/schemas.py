from __future__ import annotations

import os
from typing import Any, Literal, Optional
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

EventType = Literal[
    "thinking",
    "decompose",
    "search",
    "memory_hit",
    "compress",
    "token_update",
    "answer",
    "citation",
    "error",
    "done",
    "route",
    "tool_call",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class ConstraintConfig(BaseModel):
    max_tokens: int = Field(default=_env_int("MAX_TOKENS_PER_SESSION", 8000))
    max_cost_usd: float = Field(default=_env_float("MAX_COST_PER_SESSION", 0.10))
    max_sub_questions: int = Field(default=_env_int("MAX_SUB_QUESTIONS", 4))


class ResearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    memory_scope: Optional[str] = None
    constraints: Optional[ConstraintConfig] = None

    @field_validator("session_id", mode="before")
    @classmethod
    def ensure_session_id(cls, value: Optional[str]) -> str:
        return value or str(uuid4())


class SSEEvent(BaseModel):
    type: EventType
    content: Optional[str] = None
    data: Optional[dict[str, Any]] = None


class Citation(BaseModel):
    url: str
    title: str
    snippet: str


class TokenUpdate(BaseModel):
    used: int
    budget: int
    cost_usd: float
    cost_budget: float
