"""
A RAGPipeline builds one end-to-end config: chunking strategy -> vector index
-> retrieval at a given top-k -> generation. The experiment runner (runner.py)
instantiates one of these per configuration in the sweep.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

from src.chunking import build_chunker
from src.vectorstore import VectorStore
from src.generation import Generator


@dataclass
class QueryResult:
    question_id: str
    question: str
    answer: str
    retrieved_chunks: list[dict]
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    estimated_cost_usd: float


class RAGPipeline:
    def __init__(
        self,
        config_name: str,
        chunking_strategy: str,
        chunking_params: dict,
        top_k: int,
        generation_cfg: dict,
        embedding_model: str,
        pricing: dict,
    ):
        self.config_name = config_name
        self.top_k = top_k
        self.chunker = build_chunker(chunking_strategy, chunking_params)
        self.vectorstore = VectorStore(
            collection_name=config_name, embedding_model=embedding_model
        )
        self.generator = Generator(
            model=generation_cfg["model"],
            system_prompt=generation_cfg["system_prompt"],
            max_tokens=generation_cfg["max_tokens"],
            temperature=generation_cfg["temperature"],
            input_price_per_million=pricing["input_per_million"],
            output_price_per_million=pricing["output_per_million"],
        )

    def index_documents(self, documents: list[dict]) -> None:
        """documents: list of {doc_id, text}"""
        for doc in documents:
            chunks = self.chunker.chunk(doc["doc_id"], doc["text"])
            self.vectorstore.add_chunks(chunks)

    def answer_question(self, question_id: str, question: str) -> QueryResult:
        retrieved = self.vectorstore.query(question, top_k=self.top_k)
        context_texts = [c["text"] for c in retrieved]
        gen = self.generator.generate(question, context_texts)

        return QueryResult(
            question_id=question_id,
            question=question,
            answer=gen.answer,
            retrieved_chunks=retrieved,
            input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens,
            latency_seconds=gen.latency_seconds,
            estimated_cost_usd=gen.estimated_cost_usd,
        )
