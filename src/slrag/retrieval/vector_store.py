"""Dense retrieval — a persistent Chroma collection over the incident windows.

Design notes
------------
**We embed the window ``text``, not the raw log lines.** A window is already a coherent
incident (same source, principal and IP, no gap over 15 minutes) rendered as a compact
self-describing block, so one vector covers one incident. Embedding individual log lines
instead would retrieve fragments of an attack chain, which is the failure mode the chunking
stage exists to prevent.

**We compute the vectors ourselves** (via :mod:`slrag.providers.embeddings`) and hand them to
Chroma with ``embeddings=``. That keeps the embedding backend swappable with a single ``.env``
line — fastembed / sentence-transformers / OpenAI — without Chroma ever owning the model, and
it lets us batch exactly the way ``fastembed`` wants.

**Cosine space.** Collections are created with ``hnsw:space=cosine``, and Chroma returns a
*distance*, which is converted back to a similarity (``1 - distance``) so higher is always
better for the rest of the pipeline.

Metadata is deliberately scalar-only (Chroma stores it in SQLite): the fields an analyst
filters or displays on — window id, time range, source, principal, source IP, size, failure
count and risk score. The window text itself lives in the SQLite store, so the pipeline
re-reads it from there instead of duplicating ~190 MB of text into the vector database.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..ingest.chunker import Chunk
from ..logging_setup import get_logger

logger = get_logger(__name__)

#: Written next to the Chroma database so a stale index is detectable without loading a model.
INDEX_META_FILE = "index_meta.json"


def chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    """The scalar metadata Chroma stores for one window (no lists, no ``None``)."""
    return {
        "chunk_id": chunk.chunk_id,
        "ts_start": chunk.ts_start,
        "ts_end": chunk.ts_end,
        "source": chunk.source or "",
        "principal": chunk.principal or "",
        "src_ip": chunk.src_ip or "",
        "n_events": int(chunk.n_events),
        "n_failures": int(chunk.n_failures),
        "risk_score": int(chunk.risk_score),
    }


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One dense hit: window id, cosine similarity in [-1, 1], 1-based rank, metadata."""

    chunk_id: str
    score: float
    rank: int
    metadata: dict[str, Any]


class VectorStore:
    """Thin wrapper around a persistent Chroma collection.

    ``chromadb`` is imported inside ``__init__`` so that importing this module (and running
    the fast unit tests) never pays for it.
    """

    def __init__(
        self,
        persist_dir: str | Path,
        *,
        collection_name: str | None = None,
        settings: Settings | None = None,
        embeddings: Any = None,
    ) -> None:
        import chromadb

        self.settings = settings or get_settings()
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or self.settings.chroma_collection
        self._embeddings = embeddings

        try:
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        except Exception:  # pragma: no cover - older/newer chromadb without that Settings shape
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        self._collection = self._get_or_create(self.collection_name)
        logger.info(
            "vector_store.open",
            extra={"path": str(self.persist_dir), "collection": self.collection_name},
        )

    # ------------------------------------------------------------ collection
    def _get_or_create(self, name: str):
        """Fetch the collection, creating it with cosine space only when it does not exist.

        Passing metadata to ``get_or_create_collection`` on an *existing* collection is a
        no-op at best and a conflict at worst, so the two paths are kept separate.
        """
        try:
            return self._client.get_collection(name)
        except Exception:
            return self._client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    @property
    def embeddings(self):
        """The embedding model, built on first use (this is what downloads the ONNX model)."""
        if self._embeddings is None:
            from ..providers.embeddings import get_embedding_model

            self._embeddings = get_embedding_model(self.settings)
        return self._embeddings

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        """Number of windows currently indexed (0 if the collection is unreadable)."""
        try:
            return int(self._collection.count())
        except Exception:  # pragma: no cover - defensive
            return 0

    def existing_ids(self) -> set[str]:
        """Every window id already in the collection (one round trip, no embedding).

        The indexer uses this to make a run resumable. Reading ids is O(n) SQLite reads;
        re-embedding the same text is O(n) transformer passes. Fetching the whole id set
        once therefore beats both probing per batch and re-doing the embeddings.
        """
        try:
            payload = self._collection.get(include=[])
            return {str(value) for value in (payload.get("ids") or [])}
        except Exception:  # pragma: no cover - defensive
            return set()

    def reset(self) -> None:
        """Delete and recreate the collection — the ``slrag index --reset`` path."""
        with contextlib.suppress(Exception):  # the collection may not exist yet
            self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create(self.collection_name)

    # ------------------------------------------------------------------ write
    def add_chunks(
        self,
        chunks: Iterable[Chunk],
        *,
        batch_size: int | None = None,
        on_batch: Callable[[int], None] | None = None,
        skip_existing: bool = False,
    ) -> int:
        """Embed and store windows in batches. Returns the number written this call.

        ``skip_existing`` makes the run resumable: ids already in the collection are dropped
        *before* embedding. Chroma's ``add`` silently ignores duplicate ids anyway, but only
        after the embedding has been paid for — and the embedding is the expensive half.

        Streaming in batches keeps peak memory flat regardless of corpus size — important
        because the full corpus is ~38k windows (~190 MB of text).
        """
        size = int(batch_size or self.settings.index_batch_size)
        known = self.existing_ids() if skip_existing else set()
        written = 0
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            if known and chunk.chunk_id in known:
                continue
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(chunk_metadata(chunk))
            if len(ids) >= size:
                written += self._add_batch(ids, documents, metadatas)
                ids, documents, metadatas = [], [], []
                if on_batch is not None:
                    on_batch(written)

        if ids:
            written += self._add_batch(ids, documents, metadatas)
            if on_batch is not None:
                on_batch(written)
        return written

    def _add_batch(
        self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]
    ) -> int:
        vectors = self.embeddings.embed_documents(documents)
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=[[float(value) for value in vector] for vector in vectors],
        )
        return len(ids)

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 50, *, where: dict[str, Any] | None = None) -> list[VectorHit]:
        """Nearest windows to ``query``, best first (``score`` is cosine similarity)."""
        if k <= 0 or not query.strip():
            return []
        total = self.count()
        if total == 0:
            return []

        vector = self.embeddings.embed_query(query)
        result = self._collection.query(
            query_embeddings=[[float(value) for value in vector]],
            n_results=min(int(k), total),
            where=where or None,
            include=["metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        hits: list[VectorHit] = []
        for position, chunk_id in enumerate(ids):
            distance = float(distances[position]) if position < len(distances) else 1.0
            metadata = metadatas[position] if position < len(metadatas) else None
            hits.append(
                VectorHit(
                    chunk_id=str(chunk_id),
                    score=1.0 - distance,  # cosine distance -> similarity
                    rank=position + 1,
                    metadata=dict(metadata or {}),
                )
            )
        return hits


# --------------------------------------------------------------------------- #
# index metadata
# --------------------------------------------------------------------------- #


def write_index_meta(persist_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Persist build facts next to the Chroma database (model, dimension, count, time)."""
    target = Path(persist_dir) / INDEX_META_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def read_index_meta(persist_dir: str | Path) -> dict[str, Any]:
    """Read the build facts, or ``{}`` when the index was never built."""
    target = Path(persist_dir) / INDEX_META_FILE
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - corrupt file
        return {}


def utc_now_iso() -> str:
    """Second-resolution UTC timestamp, matching the corpus convention."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "INDEX_META_FILE",
    "VectorHit",
    "VectorStore",
    "chunk_metadata",
    "read_index_meta",
    "utc_now_iso",
    "write_index_meta",
]
