from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Protocol

import boto3
import numpy as np
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
)

from .config import Settings

log = logging.getLogger(__name__)

EmbeddingKind = Literal["document", "query"]


class EmbeddingError(RuntimeError):
    """Embedding backend failed; the message contains an actionable hint."""


@dataclass(slots=True)
class EmbeddingBatch:
    vectors: np.ndarray
    tokens: int


class Embedder(Protocol):
    name: str
    model_id: str
    dim: int

    def embed(self, texts: list[str], *, kind: EmbeddingKind = "document") -> EmbeddingBatch: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class BedrockEmbedder:
    """Amazon Bedrock embeddings (Titan Text Embeddings V2 or Cohere Embed).

    The request format differs per model family, so the style is detected from
    the model ID (or set explicitly with EMBEDDING_STYLE).
    """

    name = "bedrock"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dim = settings.embedding_dim
        self.model_id = settings.embedding_model_id
        style = settings.embedding_style
        self.style = (
            style
            if style != "auto"
            else ("cohere" if "cohere" in self.model_id.lower() else "titan")
        )
        if self.style == "cohere" and self.dim != 1024:
            raise ValueError(
                "cohere.embed-multilingual-v3 returns 1024 dims; set EMBEDDING_DIM=1024"
            )

        session_kwargs: dict[str, object] = {"region_name": settings.aws_region}
        if settings.aws_profile:
            session_kwargs["profile_name"] = settings.aws_profile
        session = boto3.Session(**session_kwargs)
        self._client = session.client(
            "bedrock-runtime",
            config=Config(
                connect_timeout=10,
                read_timeout=settings.bedrock_timeout_s,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    def _embed_one(self, text: str, kind: EmbeddingKind) -> tuple[list[float], int]:
        if self.style == "cohere":
            body = {
                "texts": [text],
                "input_type": "search_query" if kind == "query" else "search_document",
                "truncate": "END",
            }
        else:
            body = {"inputText": text, "dimensions": self.dim, "normalize": True}

        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )
        except (
            ClientError,
            NoCredentialsError,
            PartialCredentialsError,
            EndpointConnectionError,
        ) as exc:
            raise EmbeddingError(self._hint(exc)) from exc

        payload = json.loads(response["body"].read())
        if self.style == "cohere":
            embeddings = payload.get("embeddings") or []
            if not embeddings:
                raise EmbeddingError(f"Empty Cohere response: {payload}")
            vector = embeddings[0]
            billed = payload.get("meta", {}).get("billed_units", {})
            tokens = int(billed.get("input_tokens", 0) or 0)
        else:
            vector = payload.get("embedding")
            tokens = int(payload.get("inputTextTokenCount", 0) or 0)
            if vector is None:
                raise EmbeddingError(f"Empty Titan response: {payload}")

        if len(vector) != self.dim:
            raise EmbeddingError(
                f"Model returned {len(vector)} dims but EMBEDDING_DIM={self.dim}. "
                "Align the setting with the model (Titan: 256/512/1024, Cohere: 1024)."
            )
        return [float(value) for value in vector], tokens

    def _hint(self, exc: Exception) -> str:
        code = ""
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
        base = f"Bedrock embedding call failed ({code or type(exc).__name__}). "
        if code in {"AccessDeniedException", "UnauthorizedException"}:
            hint = (
                "Check IAM (bedrock:InvokeModel), confirm the model is enabled for this "
                "account, and verify the model is offered in the configured region."
            )
        elif code in {"ResourceNotFoundException", "ValidationException"}:
            hint = (
                f"Model '{self.model_id}' is not usable in '{self.settings.aws_region}'. "
                "Run `python -m scripts.check_bedrock` to list what the account can see."
            )
        elif code == "ThrottlingException":
            hint = "Bedrock throttled the request; retry or request a quota increase."
        elif isinstance(exc, (NoCredentialsError, PartialCredentialsError)):
            hint = (
                "No AWS credentials found. Configure a local profile (aws configure sso) "
                "and pass AWS_PROFILE; never bake keys into the image."
            )
        elif isinstance(exc, EndpointConnectionError):
            hint = f"Cannot reach the Bedrock endpoint in {self.settings.aws_region}."
        else:
            hint = "See the Bedrock troubleshooting guide; run scripts/check_bedrock.py."
        return base + hint

    def embed(self, texts: list[str], *, kind: EmbeddingKind = "document") -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(np.zeros((0, self.dim), dtype=np.float32), 0)

        with ThreadPoolExecutor(max_workers=self.settings.embed_workers) as pool:
            results = list(pool.map(lambda text: self._embed_one(text, kind), texts))

        vectors = np.asarray([vector for vector, _ in results], dtype=np.float32)
        tokens = sum(token_count for _, token_count in results)
        return EmbeddingBatch(l2_normalize(vectors), tokens)


class LocalEmbedder:
    """Deterministic offline stand-in for an embedding model.

    It hashes word and bigram features into a fixed-size vector (the same trick
    as scikit-learn's HashingVectorizer). It captures lexical overlap, not
    semantics — perfect for demos/CI, not for production retrieval quality.
    """

    name = "local"
    model_id = "local-hashed-bow/1"

    _WORD = re.compile(r"[a-z0-9]+")

    def __init__(self, settings: Settings) -> None:
        self.dim = settings.embedding_dim

    def _features(self, text: str) -> dict[str, float]:
        words = [word for word in self._WORD.findall(text.lower()) if len(word) > 1]
        features: dict[str, float] = {}
        for word in words:
            features[word] = features.get(word, 0.0) + 1.0
        for left, right in zip(words, words[1:], strict=False):
            bigram = f"{left}_{right}"
            features[bigram] = features.get(bigram, 0.0) + 0.5
        return features

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature, count in self._features(text).items():
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        return vector

    def embed(self, texts: list[str], *, kind: EmbeddingKind = "document") -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(np.zeros((0, self.dim), dtype=np.float32), 0)
        vectors = np.asarray([self._embed_one(text) for text in texts], dtype=np.float32)
        tokens = sum(len(self._WORD.findall(text)) for text in texts)
        return EmbeddingBatch(l2_normalize(vectors), tokens)


def build_embedder(settings: Settings) -> BedrockEmbedder | LocalEmbedder:
    if settings.embeddings_provider == "bedrock":
        log.info("Embeddings: Bedrock model=%s", settings.embedding_model_id)
        return BedrockEmbedder(settings)
    log.info("Embeddings: local hashed vectors (offline mode)")
    return LocalEmbedder(settings)
