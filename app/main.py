from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import Settings, get_settings
from .documents import SUPPORTED_SUFFIXES
from .embeddings import EmbeddingError
from .llm import GenerationError
from .rag import RAGService
from .schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    StatsResponse,
)
from .store import StoreError

log = logging.getLogger("policy_rag")

STATIC_HTML = Path(__file__).parent / "static" / "index.html"

DESCRIPTION = """
A retrieval-augmented question answering API over public AI-policy PDFs.
Answers are grounded in retrieved chunks and returned with page-level citations.
Use `POST /ingest` (or `POST /ingest/sample`) first, then `POST /ask`.
"""


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        service = RAGService(settings)
        app.state.rag = service
        log.info(
            "Ready: embedder=%s chat=%s store=%s chunks=%s region=%s",
            service.embedder.model_id,
            service.chat.model_id,
            service.store.backend,
            service.store.stats()["count"],
            settings.aws_region,
        )
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings

    def service(request: Request) -> RAGService:
        return request.app.state.rag

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(_: Request, exc: EmbeddingError) -> JSONResponse:
        log.error("Embedding backend error: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(GenerationError)
    async def generation_error_handler(_: Request, exc: GenerationError) -> JSONResponse:
        log.error("Generation backend error: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(StoreError)
    async def store_error_handler(_: Request, exc: StoreError) -> JSONResponse:
        log.error("Vector store error: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_HTML, media_type="text/html")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        api = service(request)
        try:
            return HealthResponse.model_validate(api.health())
        except StoreError as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "chunks": 0,
                    "backend": api.store.backend,
                    "embedder": api.embedder.name,
                    "chat": api.chat.name,
                    "detail": str(exc),
                },
            )

    @app.get("/stats", response_model=StatsResponse)
    def stats(request: Request) -> StatsResponse:
        return StatsResponse.model_validate(service(request).stats())

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        return AskResponse.model_validate(
            service(request).ask(payload.question, top_k=payload.top_k)
        )

    @app.post(
        "/ingest",
        response_model=IngestResponse,
        status_code=201,
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    )
    async def ingest(
        request: Request,
        files: Annotated[list[UploadFile], File(description="PDF, TXT or MD files")],
        reset: Annotated[bool, Form(description="Replace the existing index")] = False,
    ) -> IngestResponse:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        current = request.app.state.settings
        if not current.data_dir.exists():
            current.data_dir.mkdir(parents=True, exist_ok=True)

        upload_root = current.data_dir / "uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="ingest-", dir=upload_root))
        paths: list[Path] = []
        max_bytes = current.max_upload_mb * 1024 * 1024

        try:
            for position, upload in enumerate(files):
                name = Path(upload.filename or f"upload-{position}").name
                if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {name}. Use PDF, TXT or MD.",
                    )
                content = await upload.read()
                if not content:
                    raise HTTPException(status_code=400, detail=f"Empty file: {name}")
                if len(content) > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{name} exceeds the {current.max_upload_mb} MB upload limit.",
                    )
                # Keep the original filename so citations stay readable.
                target = workdir / f"{position:02d}" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                paths.append(target)

            try:
                summary = await run_in_threadpool(
                    lambda: service(request).ingest_files(paths, reset=reset)
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return IngestResponse.model_validate(summary)

    @app.post("/ingest/sample", response_model=IngestResponse)
    def ingest_sample(request: Request, reset: bool = False) -> IngestResponse:
        current = request.app.state.settings
        files = sorted(
            path
            for path in current.policy_dir.glob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not files:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No sample policy files found in {current.policy_dir}. "
                    "Run from the repo root, mount ./data:/data, or upload files via POST /ingest."
                ),
            )
        summary = service(request).ingest_files(files, reset=reset)
        return IngestResponse.model_validate(summary)

    return app


app = create_app()
