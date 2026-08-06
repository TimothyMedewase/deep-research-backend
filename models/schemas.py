from __future__ import annotations

import os
from typing import Any, Literal, Optional
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

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

ALLOWED_MODELS = frozenset({"gpt-4o", "gpt-4o-mini"})


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


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ResearchConstraints(BaseModel):
    max_tokens: int = Field(default=_env_int("MAX_TOKENS_PER_SESSION", 8000))
    max_cost_usd: float = Field(default=_env_float("MAX_COST_PER_SESSION", 0.10))
    max_sub_questions: int = Field(default=_env_int("MAX_SUB_QUESTIONS", 3))
    compression_threshold: int = Field(default=_env_int("COMPRESSION_THRESHOLD", 6000))
    max_tool_rounds: int = Field(default=_env_int("RESEARCH_TOOL_MAX_ROUNDS", 8))
    memory_top_k: int = Field(default=3)
    search_max_results: int = Field(default=3)
    research_model: str = Field(default="gpt-4o")
    compression_model: str = Field(default="gpt-4o-mini")

    @model_validator(mode="after")
    def clamp_and_allowlist(self) -> ResearchConstraints:
        self.max_tokens = _clamp_int(self.max_tokens, 2000, 20000)
        self.max_cost_usd = _clamp_float(self.max_cost_usd, 0.02, 1.0)
        self.max_sub_questions = _clamp_int(self.max_sub_questions, 1, 8)
        self.max_tool_rounds = _clamp_int(self.max_tool_rounds, 1, 12)
        self.memory_top_k = _clamp_int(self.memory_top_k, 1, 10)
        self.search_max_results = _clamp_int(self.search_max_results, 1, 10)
        self.compression_threshold = _clamp_int(self.compression_threshold, 1000, 18000)
        if self.compression_threshold >= self.max_tokens:
            self.compression_threshold = max(1000, self.max_tokens - 500)
        if self.research_model not in ALLOWED_MODELS:
            self.research_model = "gpt-4o"
        if self.compression_model not in ALLOWED_MODELS:
            self.compression_model = "gpt-4o-mini"
        return self


# Backward-compatible alias used by older imports.
ConstraintConfig = ResearchConstraints


class ResearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    memory_scope: Optional[str] = None
    mode: Literal["auto", "research", "follow_up"] = "auto"
    reset: bool = False
    constraints: Optional[ResearchConstraints] = None

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
