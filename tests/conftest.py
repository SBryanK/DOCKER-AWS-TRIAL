from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        embeddings_provider="local",
        llm_provider="local",
        vector_backend="numpy",
        embedding_dim=256,
        chunk_size=120,
        chunk_overlap=25,
        min_chunk_chars=20,
        top_k=2,
        min_score=0.0,
        log_level="WARNING",
    )


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def policy_text() -> str:
    return (
        "AI Risk Management Policy\n"
        "Every team must complete an AI risk assessment before deploying a model. "
        "The assessment covers data privacy, model bias, and human oversight.\n"
        "Data retention is limited to 24 months. Personal data must be encrypted "
        "at rest and in transit.\n"
        "The AI governance board reviews high-risk use cases every quarter. "
        "Incidents must be reported within 72 hours.\n"
        "Employees complete AI awareness training annually. Vendors must provide "
        "model documentation before procurement.\n"
    )
