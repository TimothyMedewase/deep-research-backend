from __future__ import annotations

from functools import lru_cache

import tiktoken

COST_PER_1K = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002},
}


class ConstraintTracker:
    def __init__(self, max_tokens: int, max_cost_usd: float) -> None:
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.max_sub_questions = 4
        self.token_count = 0
        self.cost_usd = 0.0

    def add_tokens(self, n: int) -> None:
        self.token_count += max(n, 0)

    def add_cost(self, usd: float) -> None:
        self.cost_usd += max(usd, 0.0)

    def is_over_compression_threshold(self, threshold: int) -> bool:
        return self.token_count > threshold

    def is_over_cost_limit(self) -> bool:
        return self.cost_usd > self.max_cost_usd

    def is_over_token_limit(self) -> bool:
        return self.token_count > self.max_tokens

    def to_dict(self) -> dict[str, float | int]:
        return {
            "used": self.token_count,
            "budget": self.max_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "cost_budget": self.max_cost_usd,
        }

    def estimate_and_add_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> float:
        rates = COST_PER_1K.get(model)
        if not rates:
            return 0.0

        input_cost = rates["input"] * (max(input_tokens, 0) / 1000)
        output_cost = rates.get("output", 0.0) * (max(output_tokens, 0) / 1000)
        total_cost = input_cost + output_cost
        self.add_cost(total_cost)
        return total_cost

    @staticmethod
    @lru_cache(maxsize=1)
    def _encoding() -> tiktoken.Encoding:
        return tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return len(ConstraintTracker._encoding().encode(text))
