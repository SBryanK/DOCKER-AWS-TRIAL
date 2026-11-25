from __future__ import annotations

from app.chunking import chunk_document
from app.documents import Document


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {index} about AI risk management." for index in range(80))
    chunks = chunk_document(Document("policy.pdf", 1, text), size=150, overlap=30, min_chars=40)

    assert chunks
    assert all(len(chunk.text) <= 150 for chunk in chunks)
    assert all(chunk.source == "policy.pdf" and chunk.page == 1 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunk_ids_are_deterministic():
    document = Document("policy.pdf", 2, "AI systems must be tested. " * 30)
    first = chunk_document(document, size=100, overlap=20, min_chars=30)
    second = chunk_document(document, size=100, overlap=20, min_chars=30)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_overlap_between_consecutive_chunks():
    document = Document(
        "policy.pdf", 1, " ".join(f"Risk rule {index} applies here." for index in range(60))
    )
    chunks = chunk_document(document, size=120, overlap=40, min_chars=30)

    assert len(chunks) >= 2
    last_sentence = chunks[0].text.split(". ")[-1]
    assert last_sentence in chunks[1].text
    assert chunks[1].text[0].isupper()


def test_run_on_sentence_is_hard_split():
    document = Document("policy.pdf", 1, "x" * 400)
    chunks = chunk_document(document, size=100, overlap=20, min_chars=30)
    assert len(chunks) == 4
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_empty_document_returns_nothing():
    assert chunk_document(Document("empty.pdf", 1, "")) == []


def test_short_pieces_are_dropped():
    document = Document("policy.pdf", 1, "Too short.")
    assert chunk_document(document, size=100, overlap=10, min_chars=50) == []
