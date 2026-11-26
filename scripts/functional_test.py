"""End-to-end functional tests against a running API instance.

Covers every public endpoint, validation errors, upload ingestion, sample
ingestion, reset semantics and a Qdrant/numpy backend round-trip.

    docker compose up -d --build
    python -m scripts.functional_test
    python -m scripts.functional_test --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

RESULTS: list[dict[str, object]] = []
STATE: dict[str, object] = {}


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    RESULTS.append({"check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("eval/functional_results.json"))
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    sample_text = (
        "AI Risk Management Policy. Every team must complete an AI risk assessment. "
        "Data retention is limited to 24 months. Incidents must be reported within 72 hours. "
        "The AI governance board reviews high-risk use cases every quarter."
    )

    try:
        client = httpx.Client(base_url=args.base_url, timeout=300.0)
        health = client.get("/health")
    except httpx.ConnectError:
        raise SystemExit(
            f"Cannot reach {args.base_url}. Start it first: docker compose up -d --build"
        ) from None

    # --- 1. health ---------------------------------------------------------
    body = health.json()
    check("GET /health returns 200", health.status_code == 200)
    check(
        "GET /health contract",
        body.get("status") == "ok"
        and isinstance(body.get("chunks"), int)
        and body.get("backend") in {"numpy", "qdrant"},
        json.dumps(body),
    )
    backend = body.get("backend")

    # --- 2. stats ----------------------------------------------------------
    stats_response = client.get("/stats")
    stats = stats_response.json()
    check("GET /stats returns 200", stats_response.status_code == 200)
    expected_stats = {
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
    check("GET /stats contract", set(stats) == expected_stats)
    check("GET /stats backend matches health", stats.get("backend") == backend)

    # --- 3. static UI + docs ----------------------------------------------
    homepage = client.get("/")
    check("GET / serves the demo UI", homepage.status_code == 200 and "PolicyLens" in homepage.text)
    docs = client.get("/docs")
    check("GET /docs serves Swagger UI", docs.status_code == 200 and "swagger" in docs.text.lower())
    openapi = client.get("/openapi.json")
    schema = openapi.json()
    paths = schema.get("paths", {})
    check(
        "GET /openapi.json documents all routes",
        openapi.status_code == 200
        and all(
            path in paths for path in ["/ask", "/ingest", "/ingest/sample", "/health", "/stats"]
        ),
        ",".join(sorted(paths)),
    )

    # --- 4. sample ingestion ----------------------------------------------
    started = time.perf_counter()
    sample = client.post("/ingest/sample", params={"reset": "true"})
    sample_elapsed = time.perf_counter() - started
    sample_body = sample.json()
    check("POST /ingest/sample returns 200", sample.status_code == 200, sample.text[:160])
    if sample.status_code == 200:
        check(
            "sample corpus ingested (3 docs, 205 pages, 719 chunks)",
            sample_body.get("documents") == 3
            and sample_body.get("pages") == 205
            and sample_body.get("chunks") == 719
            and sample_body.get("total_chunks") == 719,
            json.dumps({key: sample_body.get(key) for key in ["documents", "pages", "chunks"]}),
        )
        check("sample ingestion stays under 10 s", sample_elapsed < 10, f"{sample_elapsed:.2f}s")

    # --- 5. ask: grounded answer ------------------------------------------
    started = time.perf_counter()
    ask = client.post(
        "/ask",
        json={
            "question": (
                "What are the four core functions of the NIST AI Risk Management Framework?"
            ),
            "top_k": args.top_k,
        },
    )
    ask_elapsed = time.perf_counter() - started
    answer = ask.json()
    check("POST /ask returns 200", ask.status_code == 200, ask.text[:160])
    if ask.status_code == 200:
        check(
            "answer is grounded with valid citations",
            bool(answer.get("answer"))
            and len(answer.get("citations", [])) > 0
            and all(
                citation.get("source") and citation.get("page", 0) >= 1
                for citation in answer["citations"]
            )
            and isinstance(answer.get("grounded"), bool),
            f"{len(answer.get('citations', []))} citations",
        )
        check(
            "expected NIST source appears in citations",
            any(
                citation["source"] == "nist-ai-rmf-1.0.pdf"
                for citation in answer.get("citations", [])
            ),
        )
        check("ask latency under 5 s", ask_elapsed < 5, f"{ask_elapsed:.2f}s")

    # --- 6. ask: honest refusal -------------------------------------------
    refusal = client.post("/ask", json={"question": "What is the capital of France?"})
    refusal_body = refusal.json()
    check("unknown question returns 200", refusal.status_code == 200)
    check(
        "unknown question is refused, not hallucinated",
        refusal_body.get("answer") == "The ingested documents do not cover this."
        and refusal_body.get("citations") == []
        and refusal_body.get("grounded") is False,
        refusal_body.get("answer", "")[:120],
    )

    # --- 7. validation errors ---------------------------------------------
    check(
        "short question is rejected (422)",
        client.post("/ask", json={"question": "x"}).status_code == 422,
    )
    check(
        "top_k out of range is rejected (422)",
        client.post("/ask", json={"question": "valid question", "top_k": 99}).status_code == 422,
    )
    check("missing question is rejected (422)", client.post("/ask", json={}).status_code == 422)
    check("unknown route returns 404", client.get("/does-not-exist").status_code == 404)

    # --- 8. upload validation ---------------------------------------------
    unsupported = client.post(
        "/ingest",
        files={"files": ("notes.exe", b"binary", "application/octet-stream")},
    )
    check("unsupported upload type is rejected (400)", unsupported.status_code == 400)
    empty = client.post("/ingest", files={"files": ("empty.txt", b"", "text/plain")})
    check("empty upload is rejected (400)", empty.status_code == 400)

    # --- 9. upload ingestion + reset + ask --------------------------------
    upload = client.post(
        "/ingest",
        files={"files": ("policy.txt", sample_text.encode("utf-8"), "text/plain")},
        data={"reset": "true"},
    )
    upload_body = upload.json()
    check("POST /ingest accepts a text upload (201)", upload.status_code == 201, upload.text[:160])
    check(
        "upload reset leaves only the uploaded document",
        upload_body.get("replaced") is True
        and upload_body.get("total_chunks") == upload_body.get("chunks"),
        json.dumps(upload_body)[:160],
    )
    stats_after = client.get("/stats").json()
    check("uploaded document is the only source", stats_after.get("sources") == ["policy.txt"])
    upload_ask = client.post("/ask", json={"question": "How long is data retention?"})
    upload_answer = upload_ask.json()
    check(
        "uploaded corpus can be queried with citations",
        upload_ask.status_code == 200
        and bool(upload_answer.get("citations"))
        and "24 months" in upload_answer.get("answer", ""),
        upload_answer.get("answer", "")[:140],
    )

    # --- 10. restore the sample corpus ------------------------------------
    restore = client.post("/ingest/sample", params={"reset": "true"})
    restore_body = restore.json()
    check(
        "sample corpus restored after upload test",
        restore.status_code == 200 and restore_body.get("chunks") == 719,
    )
    final_health = client.get("/health").json()
    check(
        "final health reports 719 chunks",
        final_health.get("chunks") == 719,
        json.dumps(final_health),
    )

    # --- summary ----------------------------------------------------------
    passed = sum(1 for result in RESULTS if result["status"] == "PASS")
    failed = len(RESULTS) - passed
    summary = {
        "base_url": args.base_url,
        "backend": backend,
        "checks": RESULTS,
        "passed": passed,
        "failed": failed,
        "total": len(RESULTS),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(RESULTS)} checks passed · report: {args.out}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
