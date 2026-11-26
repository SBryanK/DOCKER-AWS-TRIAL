# Evaluation report

- Generated: 2026-11-26T03:00:48+00:00
- Target: `http://127.0.0.1:8000`
- Embedder: `local-hashed-bow/1` · Generator: `local-extractive/1` (local) · Vector backend: `qdrant`
- Corpus: 719 chunks from 3 documents, chunk_size=900 chars, overlap=150, top_k=4

## Aggregate

| Metric | Value |
|---|---|
| Questions | 10 |
| Latency mean | 5 ms |
| Latency p50 | 5 ms |
| Latency p95 | 8 ms |
| Expected source retrieved (top-k) | 100% |
| Expected source cited in answer | 100% |
| Answers with >=1 used citation | 100% |
| Grounded flag true | 100% |
| Expected-term coverage in answer | 58% |

> Latency is measured client-side per request (HTTP + embedding + retrieval + generation).
> Term coverage is a deterministic lexical proxy for answer quality, not a human rating.

## Per question

| # | Question | Latency | Cited source | Term coverage | Grounded |
|---|---|---|---|---|---|
| 1 | What are the four core functions of the NIST AI Risk Management Framework? | 8 ms | yes | 100% | yes |
| 2 | Which characteristic is described as a necessary condition for trustworthy AI? | 6 ms | yes | 100% | yes |
| 3 | What is the NIST AI RMF Playbook and what is it used for? | 5 ms | yes | 50% | yes |
| 4 | Is the NIST AI Risk Management Framework voluntary or mandatory? | 5 ms | yes | 100% | yes |
| 5 | What are the twelve risks listed in the NIST Generative AI Profile? | 5 ms | yes | 0% | yes |
| 6 | How does the NIST Generative AI Profile define confabulation? | 5 ms | yes | 0% | yes |
| 7 | What does CBRN information or capabilities mean in the Generative AI Profile? | 5 ms | yes | 0% | yes |
| 8 | What five principles underpin the UK's AI regulation framework? | 4 ms | yes | 100% | yes |
| 9 | Will the UK put its AI principles on a statutory footing initially? | 5 ms | yes | 100% | yes |
| 10 | How will the UK's AI principles be implemented in practice? | 5 ms | yes | 33% | yes |

## Method notes

- Corpus: 3 public policy PDFs (NIST AI RMF 1.0, NIST Generative AI Profile, UK AI regulation white paper), page-aware chunks with overlap.
- `source_cited` counts the expected document only when the generated answer actually contains its citation marker; it penalises retrieved-but-unused context.
- `local-extractive` mode copies sentences instead of paraphrasing. Bedrock mode uses the same retrieval and scoring, so the numbers are directly comparable.
- Re-run in Bedrock mode after configuring credentials: `LLM_PROVIDER=bedrock EMBEDDINGS_PROVIDER=bedrock python -m scripts.run_eval`.
