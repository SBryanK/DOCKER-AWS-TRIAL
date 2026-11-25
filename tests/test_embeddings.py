from __future__ import annotations

import numpy as np

from app.config import Settings
from app.embeddings import LocalEmbedder


def _embedder() -> LocalEmbedder:
    return LocalEmbedder(Settings(embedding_dim=256, embeddings_provider="local"))


def test_local_embeddings_are_deterministic():
    embedder = _embedder()
    first = embedder.embed(["AI risk management"], kind="query").vectors[0]
    second = embedder.embed(["AI risk management"], kind="query").vectors[0]
    assert np.allclose(first, second)


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = _embedder()
    vectors = embedder.embed(
        [
            "The AI governance board reviews high-risk use cases.",
            "Bananas are a good source of potassium.",
        ]
    ).vectors
    query = embedder.embed(["AI governance risk"], kind="query").vectors[0]

    relevant = float(vectors[0] @ query)
    unrelated = float(vectors[1] @ query)
    assert relevant > unrelated
