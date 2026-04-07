from __future__ import annotations

from typing import List

from openai import AsyncOpenAI

from agent.constraints import ConstraintTracker

SYSTEM_PROMPT = (
    "You are a compression engine. Summarize the following research context, "
    "preserving all key facts, figures, URLs, and conclusions. Be dense and precise. "
    "Output only the summary, no preamble."
)


class Summarizer:
    def __init__(self, openai_client: AsyncOpenAI) -> None:
        self.openai_client = openai_client

    async def compress(
        self,
        text: str,
        max_output_tokens: int = 400,
    ) -> tuple[str, int, int]:
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=max_output_tokens,
        )

        summary_text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        input_tokens = getattr(
            usage,
            "prompt_tokens",
            ConstraintTracker.estimate_tokens(f"{SYSTEM_PROMPT}\n{text}"),
        )
        output_tokens = getattr(
            usage,
            "completion_tokens",
            ConstraintTracker.estimate_tokens(summary_text),
        )
        return summary_text, input_tokens, output_tokens

    async def compress_oldest(
        self,
        context_chunks: List[str],
        n_to_compress: int = 2,
    ) -> tuple[str, List[str], int, int]:
        if not context_chunks:
            return "", [], 0, 0

        chunk_count = max(1, min(n_to_compress, len(context_chunks)))
        oldest_chunks = context_chunks[:chunk_count]
        remaining_chunks = context_chunks[chunk_count:]
        summary, input_tokens, output_tokens = await self.compress(
            "\n\n".join(oldest_chunks)
        )
        return summary, remaining_chunks, input_tokens, output_tokens
