from __future__ import annotations

import json
from openai import AsyncOpenAI

from agent.constraints import ConstraintTracker


class QueryDecomposer:
    def __init__(self, openai_client: AsyncOpenAI, max_sub_questions: int) -> None:
        self.openai_client = openai_client
        self.max_sub_questions = max_sub_questions
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    async def decompose(self, query: str) -> list[str]:
        system_prompt = (
            "You are a research query decomposer. Break the user's query into the minimum "
            "number of specific, answerable sub-questions needed to fully answer it.\n"
            "Return ONLY a JSON array of strings, no preamble or explanation.\n"
            f"Maximum {self.max_sub_questions} sub-questions.\n"
            'Example output: ["What is X?", "How does Y affect Z?", "When did W occur?"]'
        )

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        usage = response.usage
        self.last_input_tokens = getattr(
            usage,
            "prompt_tokens",
            ConstraintTracker.estimate_tokens(f"{system_prompt}\n{query}"),
        )
        self.last_output_tokens = getattr(
            usage,
            "completion_tokens",
            ConstraintTracker.estimate_tokens(content),
        )

        try:
            parsed = json.loads(content)
            if not isinstance(parsed, list) or len(parsed) > self.max_sub_questions:
                return [query]
            cleaned = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
            if not cleaned:
                return [query]
            return cleaned
        except (TypeError, ValueError, json.JSONDecodeError):
            return [query]
