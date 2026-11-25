from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from .chunking import Chunk, chunk_documents
from .config import Settings
from .documents import load_document
from .embeddings import EmbeddingError, build_embedder
from .llm import GenerationError, build_chat
from .retrieval import hybrid_rerank
from .store import (
    NumpyStore,
    QdrantStore,
    SearchHit,
    StoreError,
    VectorRecord,
    build_store,
)

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")
NOT_COVERED = "The ingested documents do not cover this."


def _snippet(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " […]"


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = build_embedder(settings)
        self.chat = build_chat(settings)
        self.store: NumpyStore | QdrantStore = build_store(settings)

    # ------------------------------------------------------------------ ingest
    def ingest_files(self, paths: list[Path], *, reset: bool = False) -> dict[str, object]:
        if not paths:
            raise ValueError("No files to ingest.")

        started = time.perf_counter()
        documents = []
        for path in paths:
            documents.extend(load_document(path))

        chunks: list[Chunk] = chunk_documents(
            documents,
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
            min_chars=self.settings.min_chunk_chars,
        )
        if not chunks:
            raise ValueError("No text chunks produced; check that the PDFs contain text.")

        batch = self.embedder.embed([chunk.text for chunk in chunks], kind="document")
        if batch.vectors.shape[0] != len(chunks):
            raise EmbeddingError("Embedding count does not match chunk count.")

        records = [
            VectorRecord(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page=chunk.page,
                chunk_index=chunk.index,
                text=chunk.text,
                vector=batch.vectors[position],
            )
            for position, chunk in enumerate(chunks)
        ]
        if reset:
            self.store.reset()
        self.store.add(records)

        elapsed = time.perf_counter() - started
        store_stats = self.store.stats()
        summary: dict[str, object] = {
            "documents": len({document.source for document in documents}),
            "pages": len(documents),
            "chunks": len(chunks),
            "total_chunks": store_stats["count"],
            "tokens": batch.tokens,
            "dim": self.embedder.dim,
            "embedding_model": self.embedder.model_id,
            "backend": self.store.backend,
            "replaced": reset,
            "elapsed_s": round(elapsed, 2),
        }
        log.info(
            "Ingested %s pages / %s chunks in %.2fs (model=%s)",
            summary["pages"],
            summary["chunks"],
            elapsed,
            self.embedder.model_id,
        )
        return summary

    # --------------------------------------------------------------------- ask
    def ask(self, question: str, *, top_k: int | None = None) -> dict[str, object]:
        started = time.perf_counter()
        k = top_k or self.settings.top_k
        batch = self.embedder.embed([question], kind="query")
        # NumpyStore is brute force, so re-rank the whole corpus with BM25: the
        # offline hashed embedder is lexical, and BM25's corpus-wide IDF recovers
        # rare-term matches that dense-ish hashing misses. Qdrant stays a wider
        # vector candidate set because fetching the whole collection per query
        # does not scale.
        if isinstance(self.store, NumpyStore):
            candidate_count = max(int(self.store.stats()["count"]), 1)
        else:
            candidate_count = max(k * 10, 50)
        candidates = self.store.search(batch.vectors[0], k=candidate_count, min_score=0.0)
        vector_weight = 0.25 if self.embedder.name == "local" else 0.55
        hits = hybrid_rerank(question, candidates, top_k=k, vector_weight=vector_weight)

        if not hits:
            latency_ms = (time.perf_counter() - started) * 1000
            return {
                "answer": NOT_COVERED,
                "citations": [],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "embedding_tokens": batch.tokens,
                },
                "latency_ms": round(latency_ms, 1),
                "mode": self.chat.name,
                "model": self.chat.model_id,
                "grounded": False,
            }

        context_blocks = [
            self._context_block(position, hit) for position, hit in enumerate(hits, 1)
        ]
        generation = self.chat.generate(question, context_blocks)
        used = {
            int(match)
            for match in _CITATION.findall(generation.text)
            if 1 <= int(match) <= len(hits)
        }
        covered = generation.text.strip() != NOT_COVERED
        citations = (
            [
                {
                    "chunk_id": hit.chunk_id,
                    "source": hit.source,
                    "page": hit.page,
                    "score": hit.score,
                    "snippet": _snippet(hit.text),
                    "used": position in used,
                }
                for position, hit in enumerate(hits, 1)
            ]
            if covered
            else []
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "answer": generation.text,
            "citations": citations,
            "usage": {
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "embedding_tokens": batch.tokens,
            },
            "latency_ms": round(latency_ms, 1),
            "mode": generation.mode,
            "model": generation.model_id,
            "grounded": bool(used) and covered,
        }

    @staticmethod
    def _context_block(position: int, hit: SearchHit) -> str:
        return f"[{position}] Source: {hit.source}, page {hit.page}. {hit.text}"

    # -------------------------------------------------------------------- stats
    def stats(self) -> dict[str, object]:
        store_stats = self.store.stats()
        return {
            "version": self.settings.app_version,
            "backend": store_stats["backend"],
            "chunks": store_stats["count"],
            "dim": store_stats["dim"],
            "sources": store_stats["sources"],
            "embedder": self.embedder.name,
            "embedding_model": self.embedder.model_id,
            "chat": self.chat.name,
            "llm_model": self.chat.model_id,
            "region": self.settings.aws_region,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "top_k": self.settings.top_k,
        }

    def health(self) -> dict[str, object]:
        try:
            store_stats = self.store.stats()
        except StoreError:
            raise
        except Exception as exc:  # surface any unexpected store failure as 503
            raise StoreError(str(exc)) from exc
        return {
            "status": "ok",
            "chunks": store_stats["count"],
            "backend": store_stats["backend"],
            "embedder": self.embedder.name,
            "chat": self.chat.name,
        }


__all__ = ["RAGService", "NOT_COVERED", "GenerationError", "EmbeddingError", "StoreError"]
