from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def _ingest_text(
    client: TestClient, text: str, *, filename: str = "policy.txt", reset: bool = False
):
    return client.post(
        "/ingest",
        files={"files": (filename, text.encode("utf-8"), "text/plain")},
        data={"reset": str(reset).lower()},
    )


def test_health_before_ingest(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 0
    assert body["backend"] == "numpy"


def test_ingest_then_ask_with_citations(client, policy_text):
    ingest = _ingest_text(client, policy_text)
    assert ingest.status_code == 201, ingest.text
    summary = ingest.json()
    assert summary["documents"] == 1
    assert summary["chunks"] >= 2
    assert summary["total_chunks"] == summary["chunks"]
    assert summary["embedding_model"].startswith("local")

    stats = client.get("/stats").json()
    assert stats["backend"] == "numpy"
    assert stats["sources"] == ["policy.txt"]
    assert stats["chunks"] == summary["chunks"]

    answer = client.post("/ask", json={"question": "How long is data retention?", "top_k": 2})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["citations"]
    assert body["grounded"] is True
    assert "24 months" in body["answer"]
    assert body["citations"][0]["source"] == "policy.txt"
    assert body["citations"][0]["page"] == 1
    assert any(citation["used"] for citation in body["citations"])
    assert body["latency_ms"] >= 0
    assert body["mode"] == "local-extractive"


def test_unknown_question_is_not_hallucinated(client, policy_text):
    _ingest_text(client, policy_text)
    response = client.post("/ask", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The ingested documents do not cover this."
    assert body["citations"] == []
    assert body["grounded"] is False


def test_reingest_with_reset_replaces_index(client, policy_text):
    first = _ingest_text(client, policy_text).json()
    second = _ingest_text(client, policy_text, filename="policy-v2.txt", reset=True).json()
    assert second["replaced"] is True
    assert second["total_chunks"] == second["chunks"]
    stats = client.get("/stats").json()
    assert stats["sources"] == ["policy-v2.txt"]
    assert stats["chunks"] >= first["chunks"]


def test_unsupported_file_type_is_rejected(client):
    response = client.post(
        "/ingest",
        files={"files": ("notes.exe", b"binary", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_validation_errors(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_index_survives_restart(settings, policy_text):
    app = create_app(settings)
    with TestClient(app) as first_client:
        _ingest_text(first_client, policy_text)
        chunks = first_client.get("/stats").json()["chunks"]

    app = create_app(settings)
    with TestClient(app) as second_client:
        stats = second_client.get("/stats").json()
        assert stats["chunks"] == chunks
        answer = second_client.post("/ask", json={"question": "When must incidents be reported?"})
        assert answer.status_code == 200
        assert "72 hours" in answer.json()["answer"]


def test_sample_ingest_endpoint(settings):
    policy_dir = settings.data_dir / "policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "sample.txt").write_text(
        "Vendors must provide model documentation. The AI governance board meets quarterly.",
        encoding="utf-8",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post("/ingest/sample")
        assert response.status_code == 200
        assert response.json()["documents"] == 1


def test_homepage_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "PolicyLens" in response.text


def test_stats_contract(client):
    body = client.get("/stats").json()
    expected = {
        "version",
        "backend",
        "chunks",
        "dim",
        "sources",
        "embedder",
        "embedding_model",
        "chat",
        "llm_model",
        "region",
        "chunk_size",
        "chunk_overlap",
        "top_k",
    }
    assert set(body) == expected
