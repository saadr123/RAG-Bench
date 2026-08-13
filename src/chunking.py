"""
Chunking strategies for RAG-Bench.

Each strategy is a small class with a `.chunk(doc_id, text) -> list[Chunk]`
method, so the sweep runner can swap them in and out via config without
touching any other part of the pipeline.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no heavy NLP dependency needed)."""
    # Split on ., !, ? followed by whitespace and a capital letter/end of string.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z]|$)", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class FixedSizeChunker:
    """Naive fixed-size character chunking with overlap.

    This is the baseline every other strategy should be compared against -
    it ignores sentence/paragraph boundaries entirely, which is exactly
    why it's useful as a control.
    """

    name = "fixed_size"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        chunks = []
        start = 0
        idx = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_fixed_{idx}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                )
            )
            idx += 1
            start += step
        return chunks


class RecursiveChunker:
    """Paragraph-aware chunking: tries to split on paragraph boundaries first,
    falling back to sentence boundaries, only splitting mid-sentence as a
    last resort. Approximates LangChain's RecursiveCharacterTextSplitter
    behavior without requiring the dependency to be installed to read this code.
    """

    name = "recursive"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        separators = ["\n\n", "\n", ". ", " "]
        pieces = self._split_recursive(text, separators)

        # Greedily pack pieces into chunks up to chunk_size
        chunks = []
        current = ""
        idx = 0
        cursor = 0
        doc_pos = 0
        for piece in pieces:
            if len(current) + len(piece) <= self.chunk_size or not current:
                current += piece
            else:
                start = doc_pos - len(current)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_rec_{idx}",
                        doc_id=doc_id,
                        text=current.strip(),
                        start_char=max(0, start),
                        end_char=doc_pos,
                    )
                )
                idx += 1
                # carry overlap forward
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current = overlap_text + piece
            doc_pos += len(piece)
        if current.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_rec_{idx}",
                    doc_id=doc_id,
                    text=current.strip(),
                    start_char=max(0, doc_pos - len(current)),
                    end_char=doc_pos,
                )
            )
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Split text into pieces no longer than chunk_size, preferring the
        earliest separator in the list.

        The recursion is the whole point: try paragraph breaks first, and only
        for pieces that are STILL too long, fall through to the next separator.
        Anything still oversized after every separator is hard-split as a last
        resort, so no piece ever exceeds chunk_size.
        """
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # last resort: no separators left, cut on character count
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep, rest = separators[0], separators[1:]
        if sep not in text:
            return self._split_recursive(text, rest)

        parts = text.split(sep)
        # keep the separator attached so reconstruction preserves spacing
        parts = [p + sep for p in parts[:-1]] + [parts[-1]]

        pieces: list[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) > self.chunk_size:
                pieces.extend(self._split_recursive(part, rest))
            else:
                pieces.append(part)
        return pieces


class SentenceWindowChunker:
    """Sliding window over sentences. Each chunk is `window_size` sentences,
    advancing by `stride` sentences each step - so windows can overlap,
    which tends to help recall on questions that span a sentence boundary.
    """

    name = "sentence_window"

    def __init__(self, window_size: int = 3, stride: int = 2):
        self.window_size = window_size
        self.stride = stride

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        sentences = _split_sentences(text)
        chunks = []
        idx = 0
        i = 0
        while i < len(sentences):
            window = sentences[i : i + self.window_size]
            if not window:
                break
            chunk_text = " ".join(window)
            start_char = text.find(window[0])
            end_char = start_char + len(chunk_text) if start_char >= 0 else -1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_sw_{idx}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start_char=max(0, start_char),
                    end_char=max(0, end_char),
                )
            )
            idx += 1
            i += self.stride
        return chunks


CHUNKERS = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "sentence_window": SentenceWindowChunker,
}


def build_chunker(name: str, params: dict):
    if name not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy '{name}'. Options: {list(CHUNKERS)}")
    return CHUNKERS[name](**params)
