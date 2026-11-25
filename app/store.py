from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

_ID_NAMESPACE = uuid.UUID("6f0f8f4c-1c1c-4f2b-9a8e-3d5b7c9e2a10")


class StoreError(RuntimeError):
    """Vector store is unavailable or misconfigured."""


@dataclass(slots=True)
class VectorRecord:
    chunk_id: str
    source: str
    page: int
    chunk_index: int
    text: str
    vector: np.ndarray


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    source: str
    page: int
    chunk_index: int
    text: str
    score: float


class VectorStore(Protocol):
    backend: str

    def add(self, records: list[VectorRecord]) -> int: ...
    def search(self, query_vector: np.ndarray, *, k: int, min_score: float) -> list[SearchHit]: ...
    def stats(self) -> dict[str, object]: ...
    def reset(self) -> None: ...


class NumpyStore:
    """Brute-force cosine store persisted as a `.npy` matrix + JSON metadata.

    For a few thousand chunks this is faster than a network round trip and has
    zero infrastructure. It is the default for local dev.
    """

    backend = "numpy"

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self._vectors_path = index_dir / "vectors.npy"
        self._records_path = index_dir / "records.json"
        self._records: list[VectorRecord] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not (self._vectors_path.exists() and self._records_path.exists()):
            return
        try:
            matrix = np.load(self._vectors_path).astype(np.float32)
            records = json.loads(self._records_path.read_text(encoding="utf-8"))
            if matrix.shape[0] != len(records):
                raise ValueError("matrix/metadata row mismatch")
        except Exception as exc:  # corrupt index should not crash the API
            log.warning("Ignoring unreadable index at %s: %s", self.index_dir, exc)
            return
        self._matrix = matrix
        self._records = [
            VectorRecord(
                chunk_id=item["chunk_id"],
                source=item["source"],
                page=int(item["page"]),
                chunk_index=int(item["chunk_index"]),
                text=item["text"],
                vector=matrix[position],
            )
            for position, item in enumerate(records)
        ]

    def add(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        by_id = {record.chunk_id: record for record in self._records}
        for record in records:
            by_id[record.chunk_id] = record
        self._records = list(by_id.values())
        self._matrix = (
            np.vstack([record.vector for record in self._records])
            .astype(np.float32)
            .reshape(-1, records[0].vector.shape[0])
        )
        self._save()
        return len(records)

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectors_tmp = self._vectors_path.with_suffix(".npy.tmp")
        records_tmp = self._records_path.with_suffix(".json.tmp")
        # np.save appends ".npy" to a path that does not end in .npy, so write
        # through a file handle to keep the temporary filename predictable.
        with vectors_tmp.open("wb") as handle:
            np.save(handle, self._matrix)
        payload = [
            {
                "chunk_id": record.chunk_id,
                "source": record.source,
                "page": record.page,
                "chunk_index": record.chunk_index,
                "text": record.text,
            }
            for record in self._records
        ]
        records_tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        vectors_tmp.replace(self._vectors_path)
        records_tmp.replace(self._records_path)

    def search(self, query_vector: np.ndarray, *, k: int, min_score: float) -> list[SearchHit]:
        if not self._records:
            return []
        scores = self._matrix @ query_vector.astype(np.float32)
        order = np.argsort(-scores)[:k]
        hits: list[SearchHit] = []
        for position in order:
            score = float(scores[position])
            if score < min_score:
                continue
            record = self._records[int(position)]
            hits.append(
                SearchHit(
                    chunk_id=record.chunk_id,
                    source=record.source,
                    page=record.page,
                    chunk_index=record.chunk_index,
                    text=record.text,
                    score=round(score, 4),
                )
            )
        return hits

    def stats(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "count": len(self._records),
            "dim": int(self._matrix.shape[1]) if self._matrix.size else 0,
            "sources": sorted({record.source for record in self._records}),
        }

    def reset(self) -> None:
        self._records = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._vectors_path.unlink(missing_ok=True)
        self._records_path.unlink(missing_ok=True)


class QdrantStore:
    """Qdrant over its REST API (no extra client dependency)."""

    backend = "qdrant"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.url = settings.qdrant_url.rstrip("/")
        self.collection = settings.qdrant_collection
        self.dim = settings.embedding_dim
        self._client = httpx.Client(base_url=self.url, timeout=settings.qdrant_timeout_s)
        try:
            self._ensure_collection()
        except httpx.HTTPError as exc:
            raise StoreError(
                f"Cannot reach Qdrant at {self.url} ({exc}). Start it with "
                "`docker compose up -d qdrant`, or set VECTOR_BACKEND=numpy."
            ) from exc

    def _ensure_collection(self) -> None:
        response = self._client.get(f"/collections/{self.collection}")
        if response.status_code == 200:
            return
        if response.status_code != 404:
            raise StoreError(f"Qdrant error {response.status_code}: {response.text[:200]}")
        created = self._client.put(
            f"/collections/{self.collection}",
            json={"vectors": {"size": self.dim, "distance": "Cosine"}},
        )
        created.raise_for_status()
        log.info("Created Qdrant collection %s (dim=%s)", self.collection, self.dim)

    def add(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        points = [
            {
                "id": str(uuid.uuid5(_ID_NAMESPACE, record.chunk_id)),
                "vector": record.vector.tolist(),
                "payload": {
                    "chunk_id": record.chunk_id,
                    "source": record.source,
                    "page": record.page,
                    "chunk_index": record.chunk_index,
                    "text": record.text,
                },
            }
            for record in records
        ]
        response = self._client.put(
            f"/collections/{self.collection}/points?wait=true", json={"points": points}
        )
        if response.status_code >= 400:
            raise StoreError(f"Qdrant upsert failed: {response.status_code} {response.text[:200]}")
        return len(points)

    def _hits_from_points(self, points: list[dict[str, object]]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for point in points:
            payload = point.get("payload") or {}
            hits.append(
                SearchHit(
                    chunk_id=str(payload.get("chunk_id", "")),
                    source=str(payload.get("source", "unknown")),
                    page=int(payload.get("page", 0) or 0),
                    chunk_index=int(payload.get("chunk_index", 0) or 0),
                    text=str(payload.get("text", "")),
                    score=round(float(point.get("score", 0.0) or 0.0), 4),
                )
            )
        return hits

    def search(self, query_vector: np.ndarray, *, k: int, min_score: float) -> list[SearchHit]:
        body: dict[str, object] = {
            "query": query_vector.tolist(),
            "limit": k,
            "with_payload": True,
            "score_threshold": min_score,
        }
        response = self._client.post(f"/collections/{self.collection}/points/query", json=body)
        if response.status_code == 200:
            return self._hits_from_points(response.json().get("result", {}).get("points", []))

        # Older Qdrant versions expose the legacy search endpoint only.
        legacy = {
            "vector": query_vector.tolist(),
            "limit": k,
            "with_payload": True,
            "score_threshold": min_score,
        }
        response = self._client.post(f"/collections/{self.collection}/points/search", json=legacy)
        if response.status_code >= 400:
            raise StoreError(f"Qdrant search failed: {response.status_code} {response.text[:200]}")
        return self._hits_from_points(response.json().get("result", []))

    def stats(self) -> dict[str, object]:
        info = self._client.get(f"/collections/{self.collection}")
        info.raise_for_status()
        result = info.json().get("result", {})
        count = int(result.get("points_count", 0) or 0)
        sources: set[str] = set()
        try:
            scroll = self._client.post(
                f"/collections/{self.collection}/points/scroll",
                json={"limit": 1000, "with_payload": ["source"], "with_vector": False},
            )
            if scroll.status_code == 200:
                for point in scroll.json().get("result", {}).get("points", []):
                    sources.add(str((point.get("payload") or {}).get("source", "")))
        except httpx.HTTPError:
            pass
        vectors = result.get("config", {}).get("params", {}).get("vectors", {})
        dim = int(vectors.get("size", self.dim)) if isinstance(vectors, dict) else self.dim
        return {
            "backend": self.backend,
            "count": count,
            "dim": dim,
            "sources": sorted(source for source in sources if source),
        }

    def reset(self) -> None:
        self._client.delete(f"/collections/{self.collection}")
        self._ensure_collection()


def build_store(settings: Settings) -> NumpyStore | QdrantStore:
    if settings.vector_backend == "qdrant":
        log.info(
            "Vector store: Qdrant at %s (collection=%s)",
            settings.qdrant_url,
            settings.qdrant_collection,
        )
        return QdrantStore(settings)
    log.info("Vector store: numpy index at %s", settings.index_dir)
    return NumpyStore(settings.index_dir)
