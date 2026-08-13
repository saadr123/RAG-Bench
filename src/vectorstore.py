"""
Thin wrapper around ChromaDB for RAG-Bench.

Embeddings run locally via sentence-transformers so building and querying
the index costs nothing in API tokens - only generation (Phase 3) and the
LLM-judge (Phase 4) call the Anthropic API.
"""

from __future__ import annotations
import chromadb
from chromadb.utils import embedding_functions

from src.chunking import Chunk


class VectorStore:
    def __init__(self, collection_name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.EphemeralClient()  # in-memory, rebuilt per config run
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        # Fresh collection per configuration - each chunking strategy produces
        # a different chunk set, so indices should never be shared across configs.
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id} for c in chunks],
        )

    def query(self, query_text: str, top_k: int) -> list[dict]:
        """Returns list of {chunk_id, doc_id, text, distance} sorted by relevance."""
        result = self.collection.query(query_texts=[query_text], n_results=top_k)
        out = []
        for i in range(len(result["ids"][0])):
            out.append(
                {
                    "chunk_id": result["ids"][0][i],
                    "doc_id": result["metadatas"][0][i]["doc_id"],
                    "text": result["documents"][0][i],
                    "distance": result["distances"][0][i],
                }
            )
        return out
