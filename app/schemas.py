from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "What are the four core functions of the NIST AI RMF?",
                    "top_k": 4,
                }
            ]
        }
    )

    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Natural-language question about the ingested policy corpus.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Override the number of retrieved chunks."
    )


class Citation(BaseModel):
    chunk_id: str
    source: str
    page: int
    score: float
    snippet: str
    used: bool = Field(description="True when the answer references this citation number.")


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    usage: Usage
    latency_ms: float
    mode: str
    model: str
    grounded: bool


class IngestResponse(BaseModel):
    documents: int
    pages: int
    chunks: int
    total_chunks: int
    tokens: int
    dim: int
    embedding_model: str
    backend: str
    replaced: bool
    elapsed_s: float


class HealthResponse(BaseModel):
    status: str
    chunks: int
    backend: str
    embedder: str
    chat: str


class StatsResponse(BaseModel):
    version: str
    backend: str
    chunks: int
    dim: int
    sources: list[str]
    embedder: str
    embedding_model: str
    chat: str
    llm_model: str
    region: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


class ErrorResponse(BaseModel):
    detail: str
