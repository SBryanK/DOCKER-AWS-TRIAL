from __future__ import annotations

import math
import re
from dataclasses import replace

from .store import SearchHit

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "within",
}


def _tokenize(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]


def _bm25_scores(
    query_tokens: list[str], documents: list[list[str]], *, k1: float = 1.2, b: float = 0.75
) -> list[float]:
    total = len(documents)
    average_length = sum(len(document) for document in documents) / max(total, 1)
    document_frequency: dict[str, int] = {}
    for document in documents:
        for token in set(document):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    scores: list[float] = []
    for document in documents:
        frequencies: dict[str, int] = {}
        for token in document:
            frequencies[token] = frequencies.get(token, 0) + 1
        length_normalisation = k1 * (1 - b + b * len(document) / max(average_length, 1))
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if frequency == 0:
                continue
            df = document_frequency[token]
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            score += idf * (frequency * (k1 + 1)) / (frequency + length_normalisation)
        scores.append(score)
    return scores


def hybrid_rerank(
    question: str, hits: list[SearchHit], *, top_k: int, vector_weight: float = 0.55
) -> list[SearchHit]:
    """Blend vector similarity with BM25 lexical relevance (hybrid retrieval).

    Dense vectors capture meaning; BM25 catches exact identifiers and rare terms
    that small embedding models often miss. Re-ranking a wider candidate set is
    cheap and improves precision without touching the index.
    """
    if len(hits) <= 1:
        return hits[:top_k]

    query_tokens = _tokenize(question)
    if not query_tokens:
        return hits[:top_k]

    documents = [_tokenize(hit.text) for hit in hits]
    lexical = _bm25_scores(query_tokens, documents)
    max_lexical = max(lexical) or 1.0
    max_vector = max(hit.score for hit in hits) or 1.0

    ranked: list[tuple[float, SearchHit]] = []
    for hit, lexical_score in zip(hits, lexical, strict=True):
        combined = vector_weight * (hit.score / max_vector) + (1 - vector_weight) * (
            lexical_score / max_lexical
        )
        ranked.append((combined, replace(hit, score=round(combined, 4))))

    ranked.sort(key=lambda item: -item[0])
    return [hit for _, hit in ranked[:top_k]]
