"""Run the 10-question evaluation against a live API and write JSON + Markdown.

Start the API first:
    uvicorn app.main:app --port 8000
Then:
    python -m scripts.run_eval
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
DEFAULT_OUT = Path("eval/results.json")
DEFAULT_REPORT = Path("eval/report.md")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _load_questions(path: Path) -> list[dict]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    if not questions:
        raise SystemExit(f"No questions found in {path}")
    return questions


def _score(question: dict, response: dict) -> dict:
    citations = response.get("citations", [])
    cited_sources = {citation["source"] for citation in citations if citation.get("used")}
    retrieved_sources = {citation["source"] for citation in citations}
    answer = response.get("answer", "").lower()
    terms = [term.lower() for term in question.get("expected_terms", [])]
    matched_terms = [term for term in terms if term in answer]
    coverage = len(matched_terms) / len(terms) if terms else 1.0
    expected_source = question["expected_source"]
    return {
        "source_retrieved": expected_source in retrieved_sources,
        "source_cited": expected_source in cited_sources,
        "term_coverage": round(coverage, 3),
        "matched_terms": matched_terms,
        "citation_count": len(citations),
        "used_citation_count": sum(1 for citation in citations if citation.get("used")),
        "grounded": bool(response.get("grounded")),
    }


def _write_report(
    path: Path,
    *,
    results: list[dict],
    aggregate: dict,
    stats: dict,
    base_url: str,
) -> None:
    lines = [
        "# Evaluation report",
        "",
        f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- Target: `{base_url}`",
        f"- Embedder: `{stats.get('embedding_model')}` · Generator: `{stats.get('llm_model')}` "
        f"({stats.get('chat')}) · Vector backend: `{stats.get('backend')}`",
        f"- Corpus: {stats.get('chunks')} chunks from "
        f"{len(stats.get('sources', []))} documents, "
        f"chunk_size={stats.get('chunk_size')} chars, "
        f"overlap={stats.get('chunk_overlap')}, top_k={stats.get('top_k')}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions | {aggregate['questions']} |",
        f"| Latency mean | {aggregate['latency_mean_ms']:.0f} ms |",
        f"| Latency p50 | {aggregate['latency_p50_ms']:.0f} ms |",
        f"| Latency p95 | {aggregate['latency_p95_ms']:.0f} ms |",
        f"| Expected source retrieved (top-k) | {aggregate['retrieval_hit_rate']:.0%} |",
        f"| Expected source cited in answer | {aggregate['citation_hit_rate']:.0%} |",
        f"| Answers with >=1 used citation | {aggregate['citation_rate']:.0%} |",
        f"| Grounded flag true | {aggregate['grounded_rate']:.0%} |",
        f"| Expected-term coverage in answer | {aggregate['mean_term_coverage']:.0%} |",
        "",
        "> Latency is measured client-side per request "
        "(HTTP + embedding + retrieval + generation).",
        "> Term coverage is a deterministic lexical proxy for answer quality, not a human rating.",
        "",
        "## Per question",
        "",
        "| # | Question | Latency | Cited source | Term coverage | Grounded |",
        "|---|---|---|---|---|---|",
    ]
    for index, result in enumerate(results, start=1):
        question = result["question"].replace("|", "\\|")
        if len(question) > 78:
            question = question[:75] + "..."
        cited = (
            "yes"
            if result["source_cited"]
            else ("retrieved" if result["source_retrieved"] else "no")
        )
        lines.append(
            f"| {index} | {question} | {result['latency_ms']:.0f} ms | {cited} | "
            f"{result['term_coverage']:.0%} | {'yes' if result['grounded'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Method notes",
            "",
            "- Corpus: 3 public policy PDFs (NIST AI RMF 1.0, NIST Generative AI Profile, "
            "UK AI regulation white paper), page-aware chunks with overlap.",
            "- `source_cited` counts the expected document only when the generated answer "
            "actually contains its citation marker; it penalises retrieved-but-unused context.",
            "- `local-extractive` mode copies sentences instead of paraphrasing. Bedrock mode "
            "uses the same retrieval and scoring, so the numbers are directly comparable.",
            "- Re-run in Bedrock mode after configuring credentials: `LLM_PROVIDER=bedrock "
            "EMBEDDINGS_PROVIDER=bedrock python -m scripts.run_eval`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    payload_base: dict[str, object] = {}
    if args.top_k:
        payload_base["top_k"] = args.top_k

    try:
        with httpx.Client(base_url=args.base_url, timeout=180.0) as client:
            stats = client.get("/stats").json()
            results = []
            for question in questions:
                started = time.perf_counter()
                response = client.post(
                    "/ask", json={"question": question["question"], **payload_base}
                )
                latency_ms = (time.perf_counter() - started) * 1000
                if response.status_code != 200:
                    raise SystemExit(
                        f"API error for {question['id']}: {response.status_code} {response.text}"
                    )
                body = response.json()
                results.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        "expected_source": question["expected_source"],
                        "expected_terms": question["expected_terms"],
                        "latency_ms": round(latency_ms, 1),
                        "service_latency_ms": body.get("latency_ms"),
                        "answer": body.get("answer", ""),
                        "mode": body.get("mode"),
                        "model": body.get("model"),
                        "usage": body.get("usage"),
                        **_score(question, body),
                    }
                )
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"Cannot reach {args.base_url}. Start the API with "
            "`uvicorn app.main:app --port 8000` and try again."
        ) from exc

    latencies = [result["latency_ms"] for result in results]
    aggregate = {
        "questions": len(results),
        "latency_mean_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "latency_p50_ms": round(_percentile(latencies, 0.50), 1),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
        "retrieval_hit_rate": round(
            sum(1 for result in results if result["source_retrieved"]) / len(results), 3
        ),
        "citation_hit_rate": round(
            sum(1 for result in results if result["source_cited"]) / len(results), 3
        ),
        "citation_rate": round(
            sum(1 for result in results if result["used_citation_count"] > 0) / len(results), 3
        ),
        "grounded_rate": round(
            sum(1 for result in results if result["grounded"]) / len(results), 3
        ),
        "mean_term_coverage": round(
            statistics.fmean(result["term_coverage"] for result in results), 3
        ),
    }

    output = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "stats": stats,
        "aggregate": aggregate,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(
        args.report, results=results, aggregate=aggregate, stats=stats, base_url=args.base_url
    )

    print(json.dumps(aggregate, indent=2))
    print(f"Wrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
