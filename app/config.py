from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Policy Q&A RAG API"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # --- AWS / Bedrock -----------------------------------------------------
    aws_region: str = "ap-southeast-1"
    aws_profile: str | None = None
    embeddings_provider: Literal["bedrock", "local"] = "local"
    llm_provider: Literal["bedrock", "local"] = "local"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_style: Literal["auto", "titan", "cohere"] = "auto"
    embedding_dim: int = 1024
    llm_model_id: str = "apac.amazon.nova-lite-v1:0"
    bedrock_max_tokens: int = 400
    bedrock_temperature: float = 0.0
    bedrock_timeout_s: int = 60
    embed_workers: int = 4

    # --- Vector store ------------------------------------------------------
    vector_backend: Literal["numpy", "qdrant"] = "numpy"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "policy_chunks"
    qdrant_timeout_s: float = 10.0

    # --- Retrieval / chunking ---------------------------------------------
    data_dir: Path = Path("data")
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 80
    top_k: int = 4
    min_score: float = 0.10
    max_upload_mb: int = 20

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.embedding_dim not in (256, 512, 1024):
            raise ValueError("EMBEDDING_DIM must be one of 256, 512, 1024")
        if self.top_k < 1 or self.top_k > 10:
            raise ValueError("TOP_K must be between 1 and 10")
        if self.max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be >= 1")
        return self

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def policy_dir(self) -> Path:
        return self.data_dir / "policies"


@lru_cache
def get_settings() -> Settings:
    return Settings()
