"""
Evaluation metrics for RAG-Bench.

Two kinds of metrics here:

1. Retrieval quality - computed directly with plain arithmetic, no API call,
   since the eval set labels which doc_ids are actually relevant.

2. Answer quality - scored by a second LLM call ("LLM-as-judge"). The judge
   compares the generated answer against the reference answer, so what it
   measures is CORRECTNESS, not faithfulness. (Faithfulness would mean
   "is the answer grounded in the retrieved context" - a different question
   that would need the context passed to the judge.) Naming this precisely
   matters: the two get conflated constantly in RAG writeups.

Judge tokens are tracked but kept OUT of the headline cost-per-query metric,
because the judge is an evaluation harness cost, not something a production
RAG system would pay. Both numbers are recorded so the split is visible.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass

from src.llm import make_client, chat


@dataclass
class RetrievalMetrics:
    hit_at_k: bool          # was ANY relevant doc retrieved in top-k
    precision_at_k: float   # fraction of retrieved chunks from a relevant doc
    recall_at_k: float      # fraction of relevant docs represented in retrieved set


def compute_retrieval_metrics(
    retrieved_chunks: list[dict], relevant_doc_ids: list[str]
) -> RetrievalMetrics:
    retrieved_doc_ids = [c["doc_id"] for c in retrieved_chunks]
    relevant_set = set(relevant_doc_ids)

    hits = [d for d in retrieved_doc_ids if d in relevant_set]
    hit_at_k = len(hits) > 0
    precision = len(hits) / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0

    retrieved_relevant_docs = set(hits)
    recall = len(retrieved_relevant_docs) / len(relevant_set) if relevant_set else 0.0

    return RetrievalMetrics(hit_at_k=hit_at_k, precision_at_k=precision, recall_at_k=recall)


JUDGE_PROMPT_TEMPLATE = """You are grading the quality of an AI-generated answer against a
reference answer, for a financial question-answering task.

Question: {question}

Reference answer: {reference}

Generated answer: {generated}

Score the generated answer 1-5 for CORRECTNESS relative to the reference answer:
1 = contradicts the reference or is completely wrong
2 = mostly wrong, missing key facts
3 = partially correct, missing some important details or numbers
4 = correct with minor omissions
5 = fully correct and matches the key facts/figures in the reference

Respond with ONLY a JSON object in this exact format, nothing else:
{{"score": <int 1-5>, "reasoning": "<one sentence explanation>"}}
"""


def _extract_json(text: str) -> dict | None:
    """Free/open-weights models often wrap JSON in prose or markdown fences.
    Try a clean parse first, then pull the first {...} block out of the text.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class LLMJudge:
    def __init__(
        self,
        model: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
        input_price_per_million: float = 1.00,
        output_price_per_million: float = 5.00,
    ):
        self.client = make_client()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million

    def score(self, question: str, reference: str, generated: str) -> dict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, reference=reference, generated=generated
        )
        result = chat(
            self.client,
            model=self.model,
            user_message=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        judge_cost = (
            (result.input_tokens / 1_000_000) * self.input_price
            + (result.output_tokens / 1_000_000) * self.output_price
        )

        parsed = _extract_json(result.text)
        if parsed is not None and "score" in parsed:
            try:
                score = int(parsed["score"])
                reasoning = str(parsed.get("reasoning", ""))
            except (TypeError, ValueError):
                score, reasoning = None, f"UNPARSEABLE_SCORE: {result.text[:200]}"
        else:
            # Record the failure rather than silently dropping it - the
            # dashboard reports how many of these happened per config, since
            # a config with 3 unparseable judges out of 6 isn't comparable.
            score, reasoning = None, f"UNPARSEABLE_JUDGE_OUTPUT: {result.text[:200]}"

        return {
            "score": score,
            "reasoning": reasoning,
            "judge_input_tokens": result.input_tokens,
            "judge_output_tokens": result.output_tokens,
            "judge_cost_usd": judge_cost,
        }
