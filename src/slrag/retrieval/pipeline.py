"""The hybrid retrieval pipeline.

    question
       |
       +-- dense  : Chroma over the window text          -> top candidate_pool (50)
       +-- lexical: BM25 over the window search_text     -> top candidate_pool (50)
       |
       +-- RRF (k=60)  fuse by rank                      -> one ranking
       |
       +-- fetch the winning windows' text from SQLite   (one query, not N)
       |
       +-- FlashRank cross-encoder over the top-50       -> final top-k (5)

Two deliberate choices worth defending in an interview:

**Windows come back from SQLite, not from Chroma.** The vector database stores only ids,
metadata and vectors; the ~190 MB of window text stays in one place. That keeps the two views
from drifting and means citations (the raw records behind a window) are always one join away.

**The engine is a long-lived object.** Building the BM25 index tokenises every window, which
is the expensive part of startup; a query then costs milliseconds. The CLI builds one per
process, and the Phase 6 API keeps one warm for the whole server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..ingest.chunker import Chunk
from ..ingest.store import EventStore
from ..logging_setup import get_logger
from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion
from .rerank import NullReranker, get_reranker
from .vector_store import VectorStore

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedWindow:
    """One result: the window, its final score, and how each retriever ranked it."""

    chunk_id: str
    rank: int
    score: float
    chunk: Chunk
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None

    @property
    def found_by(self) -> tuple[str, ...]:
        """Which retrievers surfaced this window — ``('bm25', 'dense')``, sorted."""
        found = []
        if self.dense_rank is not None:
            found.append("dense")
        if self.bm25_rank is not None:
            found.append("bm25")
        return tuple(sorted(found))

    def as_dict(self, *, text_chars: int = 0) -> dict[str, Any]:
        """JSON-safe view (used by ``slrag search --json`` and the eval harness)."""
        chunk = self.chunk
        text = chunk.text
        if text_chars and len(text) > text_chars:
            text = text[:text_chars] + "...[truncated]"
        return {
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "score": round(self.score, 6),
            "fusion_score": round(self.fusion_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "found_by": list(self.found_by),
            "ts_start": chunk.ts_start,
            "ts_end": chunk.ts_end,
            "source": chunk.source,
            "principal": chunk.principal,
            "src_ip": chunk.src_ip,
            "n_events": chunk.n_events,
            "n_failures": chunk.n_failures,
            "risk_score": chunk.risk_score,
            "event_names": list(chunk.event_names),
            "event_ids": list(chunk.event_ids),
            "text": text,
        }


@dataclass(slots=True)
class RetrievalResult:
    """Everything one ``retrieve()`` call produced, plus the diagnostics to explain it."""

    question: str
    hits: list[RetrievedWindow] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def chunk_ids(self) -> list[str]:
        return [hit.chunk_id for hit in self.hits]

    def render(self, *, max_chars_per_window: int = 0, separator: str = "\n\n") -> str:
        """The prompt block handed to the LLM in Phase 4 (each window keeps its own header)."""
        blocks: list[str] = []
        for hit in self.hits:
            text = hit.chunk.text
            if max_chars_per_window and len(text) > max_chars_per_window:
                text = text[:max_chars_per_window].rstrip() + "\n...[window truncated]"
            blocks.append(text)
        return separator.join(blocks)

    def as_dict(self, *, text_chars: int = 600) -> dict[str, Any]:
        return {
            "question": self.question,
            "hits": [hit.as_dict(text_chars=text_chars) for hit in self.hits],
            "diagnostics": self.diagnostics,
        }


class RetrievalEngine:
    """Holds the three retrieval components and answers questions against them."""

    def __init__(
        self,
        *,
        store: EventStore,
        vector_store: VectorStore,
        bm25: BM25Index,
        reranker: Any = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker if reranker is not None else get_reranker(self.settings)

    # ------------------------------------------------------------- factories
    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        db_path: str | Path | None = None,
        embeddings: Any = None,
        vector_store: VectorStore | None = None,
        reranker: Any = None,
        bm25_index: BM25Index | None = None,
        build_bm25: bool = True,
        verbose: bool = False,
    ) -> RetrievalEngine:
        """Wire the engine from settings.

        ``bm25_index`` and ``vector_store`` can be injected, which is what the unit tests do
        so they never need Chroma or a downloaded model.
        """
        settings = settings or get_settings()

        store = EventStore(db_path or settings.sqlite_path, read_only=True)
        if not Path(store.path).exists():
            raise FileNotFoundError(
                f"{store.path} not found — run `slrag ingest` before searching"
            )

        store_vectors = vector_store or VectorStore(
            settings.chroma_dir, settings=settings, embeddings=embeddings
        )

        if bm25_index is not None:
            lexical = bm25_index
        elif build_bm25:
            start = time.perf_counter()
            lexical = BM25Index.from_pairs(
                ((chunk.chunk_id, chunk.search_text or chunk.text) for chunk in store.iter_chunks()),
                k1=settings.bm25_k1,
                b=settings.bm25_b,
                max_tokens=settings.bm25_max_tokens,
            )
            if verbose:
                print(
                    f"  bm25 index built: {len(lexical):,} windows, "
                    f"{lexical.vocabulary_size:,} terms, "
                    f"{time.perf_counter() - start:.1f}s"
                )
        else:
            lexical = BM25Index()

        return cls(
            store=store,
            vector_store=store_vectors,
            bm25=lexical,
            reranker=reranker,
            settings=settings,
        )

    # --------------------------------------------------------------- retrieve
    def retrieve(
        self,
        question: str,
        k: int | None = None,
        *,
        rerank: bool | None = None,
        pool: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Answer one question with the top ``k`` incident windows.

        Args:
            question: natural-language question (also used verbatim as the retrieval query).
            k: windows to return (default ``RETRIEVE_TOP_K``).
            rerank: force the cross-encoder on/off for this call; ``None`` uses the setting.
            pool: per-retriever candidate depth (default ``CANDIDATE_POOL``).
            where: optional Chroma metadata filter for the dense side.
        """
        question = (question or "").strip()
        top_k = int(k or self.settings.retrieve_top_k)
        depth = int(pool or self.settings.candidate_pool)
        timings: dict[str, float] = {}
        started = time.perf_counter()

        # ---- stage 1: independent candidate generation ----------------------
        mark = time.perf_counter()
        dense_hits = self.vector_store.search(question, k=depth, where=where) if question else []
        timings["dense_ms"] = round((time.perf_counter() - mark) * 1000, 1)

        mark = time.perf_counter()
        lexical_hits = self.bm25.search(question, k=depth) if question else []
        timings["bm25_ms"] = round((time.perf_counter() - mark) * 1000, 1)

        # ---- stage 2: fuse by rank ------------------------------------------
        mark = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"dense": [hit.chunk_id for hit in dense_hits], "bm25": [hit.chunk_id for hit in lexical_hits]},
            k=self.settings.rrf_k,
            limit=depth,
        )
        timings["fusion_ms"] = round((time.perf_counter() - mark) * 1000, 1)

        dense_ranks = {hit.chunk_id: hit.rank for hit in dense_hits}
        bm25_ranks = {hit.chunk_id: hit.rank for hit in lexical_hits}

        # ---- stage 3: rehydrate the window text (one SQL query) --------------
        mark = time.perf_counter()
        chunks = self.store.chunks_by_ids([hit.chunk_id for hit in fused]) if fused else {}
        timings["fetch_ms"] = round((time.perf_counter() - mark) * 1000, 1)

        ordered: list[tuple[Any, Chunk]] = [
            (hit, chunks[hit.chunk_id]) for hit in fused if hit.chunk_id in chunks
        ]
        missing = len(fused) - len(ordered)

        # ---- stage 4: cross-encoder rerank ----------------------------------
        want_rerank = self.settings.rerank_enabled if rerank is None else bool(rerank)
        reranker = self.reranker if want_rerank else None
        if isinstance(reranker, NullReranker):
            reranker = None

        rerank_scores: dict[str, float] = {}
        timings["rerank_ms"] = 0.0
        if reranker is not None and ordered:
            mark = time.perf_counter()
            rerank_depth = min(int(self.settings.rerank_top_n), len(ordered))
            candidates = [(hit.chunk_id, chunk.text) for hit, chunk in ordered[:rerank_depth]]
            reranked = reranker.rerank(question, candidates, top_n=rerank_depth)

            by_id = {hit.chunk_id: (hit, chunk) for hit, chunk in ordered}
            head = [by_id[hit.chunk_id] for hit in reranked if hit.chunk_id in by_id]
            # Anything the reranker did not return keeps its fused position, after the head.
            seen = {hit.chunk_id for hit in reranked}
            tail = [
                (hit, chunk)
                for hit, chunk in ordered[rerank_depth:]
                if hit.chunk_id not in seen
            ]
            ordered = head + tail
            rerank_scores = {hit.chunk_id: hit.score for hit in reranked}
            timings["rerank_ms"] = round((time.perf_counter() - mark) * 1000, 1)

        # ---- stage 5: assemble ----------------------------------------------
        hits: list[RetrievedWindow] = []
        for position, (fused_hit, chunk) in enumerate(ordered[: max(0, top_k)], start=1):
            rerank_score = rerank_scores.get(fused_hit.chunk_id)
            hits.append(
                RetrievedWindow(
                    chunk_id=fused_hit.chunk_id,
                    rank=position,
                    score=rerank_score if rerank_score is not None else fused_hit.score,
                    chunk=chunk,
                    dense_rank=dense_ranks.get(fused_hit.chunk_id),
                    bm25_rank=bm25_ranks.get(fused_hit.chunk_id),
                    fusion_score=fused_hit.score,
                    rerank_score=rerank_score,
                )
            )

        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        diagnostics: dict[str, Any] = {
            "question": question,
            "k": top_k,
            "candidate_pool": depth,
            "rrf_k": self.settings.rrf_k,
            "dense_candidates": len(dense_hits),
            "bm25_candidates": len(lexical_hits),
            "fused_candidates": len(fused),
            "missing_from_store": missing,
            "indexed_windows": self.vector_store.count(),
            "bm25_windows": len(self.bm25),
            "reranker": getattr(reranker, "name", "none"),
            "rerank_applied": reranker is not None,
            "timings_ms": timings,
        }
        logger.info(
            "retrieve.done",
            extra={
                "hits": len(hits),
                "dense": len(dense_hits),
                "bm25": len(lexical_hits),
                "total_ms": timings["total_ms"],
            },
        )
        return RetrievalResult(question=question, hits=hits, diagnostics=diagnostics)

    # --------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> RetrievalEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def retrieve(
    question: str, k: int | None = None, *, settings: Settings | None = None, **kwargs: Any
) -> RetrievalResult:
    """One-shot convenience wrapper — builds an engine, queries, closes it.

    Use :class:`RetrievalEngine` directly when you will ask more than one question.
    """
    engine = RetrievalEngine.from_settings(settings)
    try:
        return engine.retrieve(question, k, **kwargs)
    finally:
        engine.close()


__all__ = [
    "RetrievalEngine",
    "RetrievalResult",
    "RetrievedWindow",
    "retrieve",
]
