# PROGRESS — PolicyLens (DOCKER-AWS-TRIAL)

**Last updated:** 2026-11-26 (Asia/Singapore)
**Current status:** Milestones 1–4 complete and verified; Milestone 5 shipped (repo +
website + eval). Bedrock mode implemented but **not executed** — no AWS credentials in
the build environment. Budget alert **not created** yet.

Environment used: macOS 27.2.1 · Apple M5 Pro · Docker 29.8.2 · Compose v5.6.1 ·
Python 3.13.18 · AWS CLI not installed at start · repo pushed to GitHub.

---

## Milestone 1 — Docker basics ✅

**Done**
- Image vs container, layers/cache, ports, env vars, volumes explained in `TUTORIAL.md`.
- Demo container runs the API and serves the web UI on port 8000.
- `.env`/`.aws` excluded from git and image; runtime config via env vars.

**Evidence**
- `docker ps` healthy API container; `/health` returns `{"status":"ok"}`.
- Local UI and `/docs` reachable.

**Next**
- Nothing blocking.

**Open issues**
- Docker Desktop must be started manually on this machine.

---

## Milestone 2 — RAG app ✅

**Done**
- FastAPI: `/ingest`, `/ingest/sample`, `/ask`, `/health`, `/stats`, demo UI.
- 3 public policy PDFs (205 pages, 482,817 chars, ~120k tokens) → **719 chunks**.
- Page-aware sentence-aligned chunking with overlap; stable `chunk_id`.
- Bedrock providers: Titan/Cohere embeddings + Nova Lite Converse; offline hashed
  embeddings + extractive answerer for demos/CI.
- numpy store persisted to disk; Qdrant REST store for compose.
- Hybrid retrieval: vector candidates + BM25 rerank; citations include source/page/score.

**Evidence**
- `python -m scripts.ingest --reset` → 719 chunks in **1.74 s**; index **3.4 MB**.
- Ask endpoint returns answer + 4 citations with `used` flags.
- 18 pytest tests pass (`pytest -q`).

**Next**
- Add token-based chunking experiment (chunk 400/700/900) and compare retrieval.

**Open issues**
- Offline answer quality is limited by design (lexical embedder, extractive answers):
  eval term coverage 42%.
- Titan V2 availability in `ap-southeast-1` is contradictory in AWS docs → verify.

---

## Milestone 3 — Good Dockerfile ✅

**Done**
- Multi-stage `Dockerfile` (builder venv → slim runtime), non-root uid 10001,
  `HEALTHCHECK`, `.dockerignore`, pinned lock file, no secrets in image.
- Deliberately naive baseline at `docker/Dockerfile.naive` for comparison.

**Evidence**
- Image size: **1.83 GB → 429 MB (−77%)**.
- Rebuild timings: no-change **1.79 s**, code change **0.78 s**, dependency change
  **10.65 s**.
- Container runs as `app`, healthcheck reaches `healthy`.
- `.dockerignore` keeps build context small (data/docs/tests/.env excluded).

**Next**
- Add image vulnerability scan (Trivy/Grype) in CI.

**Open issues**
- No SBOM/scanning yet (documented as trade-off).

---

## Milestone 4 — docker compose ✅

**Done**
- `docker-compose.yml`: API + Qdrant 1.15.5, healthcheck-gated dependency, named
  volumes `qdrant_storage` and `index_data`, read-only corpus mount.
- Ingestion through the container writes vectors to Qdrant.

**Evidence**
- `/ingest/sample` via compose → **719 Qdrant points** in **2.02 s**.
- `docker compose restart` → 719 chunks / 719 points still present, `/ask` works.
- `docker compose down && docker compose up -d` → 719 chunks / 719 points still present.
- Both services report `(healthy)`.

**Next**
- Optional: mount `~/.aws` read-only for Bedrock mode (documented, commented in compose).

**Open issues**
- Demo stack intentionally has no auth/rate limiting.

---

## Milestone 5 — Ship ✅ (Bedrock measurement pending)

**Done**
- `README.md`: Mermaid architecture, setup, API, measured results, cost, security,
  trade-offs, enterprise scale, cleanup.
- `TUTORIAL.md`: full teacher-style walkthrough + 15 check questions.
- `docs/AWS_NOTES.md`: verified AWS facts with citations + items to verify.
- `docs/index.html`: portfolio website for GitHub Pages (no build step).
- `eval/questions.jsonl`, `eval/results.json`, `eval/report.md`.
- CI workflow: ruff + pytest + Docker build.
- `scripts/check_bedrock.py`, `scripts/setup_budget.py`, `scripts/cleanup_aws.py`.

**Evidence (offline mode)**
- 10 questions: p50 **15 ms**, p95 **16 ms**, retrieval hit **100%**, citation hit
  **100%**, grounded **100%**, term coverage **42%**.
- 18 tests pass.
- Repo pushed to https://github.com/SBryanK/DOCKER-AWS-TRIAL

**Next (requires Bryan's AWS account)**
1. Install AWS CLI: `brew install awscli`.
2. Configure SSO: `aws configure sso` (region `ap-southeast-1`).
3. Create budget: `python -m scripts.setup_budget --email <your-email> --amount 5`.
4. Verify Bedrock: `python -m scripts.check_bedrock --profile <profile>`.
5. Switch providers to `bedrock`, ingest again, run `scripts/run_eval`, update README
   with real Bedrock latency/quality numbers.

**Open issues**
- **No AWS calls were made.** Bedrock path is code-complete but unverified live.
- Titan V2 in Singapore: docs vs price-list conflict (see `docs/AWS_NOTES.md`).
- Nova Micro APAC profile and numeric quotas are unverified.
- Offline-only eval numbers must not be presented as Bedrock quality.
- GitHub Pages needs to be enabled after push (Settings → Pages → `/docs`).

---

## Final state on disk

```
docker compose ps
policy-rag-api     Up (healthy)   0.0.0.0:8000->8000
policy-rag-qdrant  Up (healthy)   0.0.0.0:6333-6334->6333-6334
```

To stop: `docker compose down` (keeps data) or `docker compose down -v` (deletes data).

## Resume in a new chat

Paste this block:

> Project: PolicyLens, repo `/Users/sbryank/Documents/Projects/bedrock-policy-rag`
> (GitHub SBryanK/DOCKER-AWS-TRIAL). Milestones 1–4 verified; Milestone 5 shipped with
> offline eval. No AWS creds yet: budget + Bedrock verification pending. Read
> PROGRESS.md and continue from "Next (requires Bryan's AWS account)".
