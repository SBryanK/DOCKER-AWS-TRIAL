from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .documents import Document

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text with stable provenance."""

    chunk_id: str
    source: str
    page: int
    index: int
    text: str


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """Split a run-on sentence that is longer than a whole chunk."""
    return [text[start : start + size] for start in range(0, len(text), size)]


def chunk_document(
    document: Document,
    *,
    size: int = 900,
    overlap: int = 150,
    min_chars: int = 80,
) -> list[Chunk]:
    """Split a document into sentence-aligned, overlapping chunks.

    Chunks always start and end on a sentence boundary (except for run-on
    sentences, which are hard-split). Overlap is carried as whole sentences from
    the end of the previous chunk, up to `overlap` characters. Character-based
    chunking keeps the demo dependency-free; see README trade-offs.
    """
    text = " ".join(document.text.split())
    if not text:
        return []

    sentences = _split_sentences(text) or [text]
    pieces: list[str] = []
    current: list[str] = []

    def joined(sentences_in_chunk: list[str]) -> str:
        return " ".join(sentences_in_chunk)

    def current_length() -> int:
        return sum(len(sentence) + 1 for sentence in current) - (1 if current else 0)

    for sentence in sentences:
        if len(sentence) > size:
            if current:
                pieces.append(joined(current))
                current = []
            pieces.extend(_hard_split(sentence, size))
            continue

        if current and current_length() + 1 + len(sentence) > size:
            pieces.append(joined(current))
            tail: list[str] = []
            tail_length = 0
            for previous in reversed(current):
                added = len(previous) + (1 if tail_length else 0)
                if tail_length + added > overlap:
                    break
                tail.insert(0, previous)
                tail_length += added
            current = tail

        current.append(sentence)

    if current:
        pieces.append(joined(current))

    chunks: list[Chunk] = []
    for index, piece in enumerate(piece for piece in pieces if len(piece) >= min_chars):
        digest = hashlib.sha1(
            f"{document.source}|{document.page}|{index}|{piece}".encode()
        ).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=digest[:12],
                source=document.source,
                page=document.page,
                index=index,
                text=piece.strip(),
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    *,
    size: int = 900,
    overlap: int = 150,
    min_chars: int = 80,
) -> list[Chunk]:
    """Chunk many documents, de-duplicating identical chunk ids."""
    unique: dict[str, Chunk] = {}
    for document in documents:
        for chunk in chunk_document(document, size=size, overlap=overlap, min_chars=min_chars):
            unique.setdefault(chunk.chunk_id, chunk)
    return list(unique.values())
