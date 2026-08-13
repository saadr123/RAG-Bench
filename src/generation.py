"""
Answer generation, with token usage and modeled cost tracked per call.

Note on cost: this project runs on a free API tier, so no real money is spent.
Token counts are real; the dollar figures are those token counts priced against
a published rate card (set in config/experiments.yaml). That's a MODELED cost,
not a measured bill - stated here and in the README so the metric isn't
mistaken for actual spend.
"""

from __future__ import annotations
from dataclasses import dataclass

from src.llm import make_client, chat


@dataclass
class GenerationResult:
    answer: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    estimated_cost_usd: float


class Generator:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        input_price_per_million: float = 1.00,
        output_price_per_million: float = 5.00,
    ):
        self.client = make_client()
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million

    def generate(self, question: str, context_chunks: list[str]) -> GenerationResult:
        context_block = "\n\n---\n\n".join(context_chunks)
        user_message = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the context above."
        )

        result = chat(
            self.client,
            model=self.model,
            user_message=user_message,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        cost = (
            (result.input_tokens / 1_000_000) * self.input_price
            + (result.output_tokens / 1_000_000) * self.output_price
        )

        return GenerationResult(
            answer=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_seconds=result.latency_seconds,
            estimated_cost_usd=cost,
        )
