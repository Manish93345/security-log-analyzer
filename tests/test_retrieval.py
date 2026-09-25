"""Retrieval pipeline tests.

Everything here runs with **no Chroma, no downloaded model and no network**: the dense
retriever is a fake, BM25 is real (it is pure Python) and the windows come from a real SQLite
file. The one test that does need Chroma uses a deterministic fake embedder, so it never
downloads anything; the genuine fastembed round-trip is marked ``integration`` and excluded
from the default run.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from slrag.ingest.chunker import Chunk
from slrag.ingest.store import EventStore
from slrag.retrieval.bm25 import BM25Index
from slrag.retrieval.pipeline import RetrievalEngine, RetrievedWindow
from slrag.retrieval.rerank import NullReranker, RerankedHit
from slrag.retrieval.vector_store import VectorHit

# --------------------------------------------------------------------------- #
# helpers / fakes
# --------------------------------------------------------------------------- #


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    search_text: str | None = None,
    source: str = "auth.log",
    principal: str = "alice",
    src_ip: str = "10.0.1.23",
    risk: int = 0,
    n_events: int = 3,
    n_failures: int = 1,
    event_ids: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        ts_start="2026-09-14T02:11:00Z",
        ts_end="2026-09-14T02:19:00Z",
        source=source,
        principal=principal,
        src_ip=src_ip,
        n_events=n_events,
        n_failures=n_failures,
        event_names=["FailedPassword", "sudo"],
        event_ids=list(event_ids or [f"sl-{chunk_id}"]),
        risk_score=risk,
        text=text,
        search_text=text if search_text is None else search_text,
        accounts=[],
    )


CHUNKS = [
    make_chunk(
        "w-1",
        "[WINDOW w-1] auth.log\nFailed password for invalid user admin from 203.0.113.44\n"
        "sudo COMMAND=/usr/bin/systemctl restart nginx",
        risk=62,
        n_failures=12,
    ),
    make_chunk(
        "w-2",
        "[WINDOW w-2] auth.log\nAccepted password for alice from 10.0.1.23\nConsoleLogin MFAUsed=No",
    ),
    make_chunk(
        "w-3",
        "[WINDOW w-3] cloudtrail\nCreateAccessKey arn:aws:iam::123456789012:user/bob",
        source="cloudtrail",
        principal="arn:aws:iam::123456789012:user/bob",
        src_ip="198.51.100.7",
        risk=30,
    ),
]


class FakeVectorStore:
    """Dense retriever stub returning a fixed ranking, recording the calls it received."""

    def __init__(self, ranking: list[str]) -> None:
        self.ranking = list(ranking)
        self.calls: list[dict] = []

    def search(self, query: str, k: int = 50, *, where=None) -> list[VectorHit]:
        self.calls.append({"query": query, "k": k, "where": where})
        return [
            VectorHit(chunk_id=chunk_id, score=1.0 - index * 0.01, rank=index + 1, metadata={})
            for index, chunk_id in enumerate(self.ranking[:k])
        ]

    def count(self) -> int:
        return len(self.ranking)


class FakeReranker:
    """Cross-encoder stub that imposes a caller-chosen order."""

    name = "fake"
    available = True

    def __init__(self, order: list[str]) -> None:
        self.order = list(order)
        self.seen: list[tuple[str, str]] = []

    def rerank(self, query: str, candidates, *, top_n: int = 50) -> list[RerankedHit]:
        self.seen = list(candidates)
        present = {chunk_id for chunk_id, _ in candidates}
        ordered = [chunk_id for chunk_id in self.order if chunk_id in present]
        ordered += [chunk_id for chunk_id, _ in candidates if chunk_id not in ordered]
        return [
            RerankedHit(chunk_id=chunk_id, score=1.0 - index * 0.1, rank=index + 1)
            for index, chunk_id in enumerate(ordered[:top_n])
        ]


@pytest.fixture
def db_path(tmp_path):
    """A real SQLite store holding the three fixture windows."""
    path = tmp_path / "events.sqlite"
    with EventStore(path) as store:
        store.create_schema()
        store.insert_chunks(CHUNKS)
    return path


def build_engine(db_path, dense_ranking, *, reranker=None, k1: float = 1.5):
    return RetrievalEngine.from_settings(
        db_path=db_path,
        vector_store=FakeVectorStore(dense_ranking),
        bm25_index=BM25Index.from_pairs(
            [(chunk.chunk_id, chunk.search_text) for chunk in CHUNKS], k1=k1
        ),
        reranker=reranker if reranker is not None else NullReranker(),
    )


# --------------------------------------------------------------------------- #
# store support
# --------------------------------------------------------------------------- #


def test_chunks_by_ids_returns_a_mapping(db_path):
    with EventStore(db_path, read_only=True) as store:
        found = store.chunks_by_ids(["w-1", "w-3"])
    assert set(found) == {"w-1", "w-3"}
    assert found["w-1"].risk_score == 62
    assert found["w-1"].text.startswith("[WINDOW w-1]")


def test_chunks_by_ids_ignores_unknown_ids_and_duplicates(db_path):
    with EventStore(db_path, read_only=True) as store:
        found = store.chunks_by_ids(["w-2", "w-2", "w-does-not-exist", ""])
    assert list(found) == ["w-2"]


def test_chunks_by_ids_handles_empty_input(db_path):
    with EventStore(db_path, read_only=True) as store:
        assert store.chunks_by_ids([]) == {}


# --------------------------------------------------------------------------- #
# fusion behaviour through the engine
# --------------------------------------------------------------------------- #


def test_hybrid_fusion_prefers_the_window_both_retrievers_found(db_path):
    # dense likes w-2, BM25 will like w-1 (it literally contains "sudo"/"Failed password")
    engine = build_engine(db_path, ["w-2", "w-1", "w-3"])
    try:
        result = engine.retrieve("failed sudo commands on the web server", k=3)
    finally:
        engine.close()

    assert result.hits[0].chunk_id == "w-1"
    assert result.hits[0].bm25_rank == 1
    assert result.hits[0].dense_rank == 2
    assert result.hits[0].found_by == ("bm25", "dense")


def test_dense_only_window_still_appears(db_path):
    engine = build_engine(db_path, ["w-3", "w-1", "w-2"])
    try:
        result = engine.retrieve("kubernetes helm rollout", k=3)
    finally:
        engine.close()

    assert "w-3" in result.chunk_ids()
    hit = next(hit for hit in result.hits if hit.chunk_id == "w-3")
    assert hit.bm25_rank is None
    assert hit.found_by == ("dense",)


def test_diagnostics_report_every_stage(db_path):
    engine = build_engine(db_path, ["w-1", "w-2", "w-3"])
    try:
        result = engine.retrieve("failed password", k=2)
    finally:
        engine.close()

    diag = result.diagnostics
    assert diag["k"] == 2
    assert diag["dense_candidates"] == 3
    assert diag["bm25_candidates"] >= 1
    assert diag["fused_candidates"] >= 1
    assert diag["missing_from_store"] == 0
    assert diag["rerank_applied"] is False
    assert diag["reranker"] == "none"
    assert set(diag["timings_ms"]) >= {"dense_ms", "bm25_ms", "fusion_ms", "total_ms"}


def test_k_limits_the_returned_windows(db_path):
    engine = build_engine(db_path, ["w-1", "w-2", "w-3"])
    try:
        assert len(engine.retrieve("failed password", k=1).hits) == 1
        assert len(engine.retrieve("failed password", k=3).hits) == 3
    finally:
        engine.close()


def test_empty_question_returns_no_hits_and_never_calls_a_retriever(db_path):
    engine = build_engine(db_path, ["w-1"])
    try:
        result = engine.retrieve("   ", k=3)
    finally:
        engine.close()

    assert result.hits == []
    assert engine.vector_store.calls == []


def test_windows_missing_from_sqlite_are_reported_not_crashed(db_path):
    """The vector index can drift from the store; retrieval must degrade, not explode."""
    engine = build_engine(db_path, ["w-ghost", "w-1"])
    try:
        result = engine.retrieve("failed password", k=3)
    finally:
        engine.close()

    assert "w-ghost" not in result.chunk_ids()
    assert result.diagnostics["missing_from_store"] >= 1


def test_dense_filter_is_passed_through(db_path):
    engine = build_engine(db_path, ["w-1", "w-2"])
    try:
        engine.retrieve("failed password", k=2, where={"source": "auth.log"})
    finally:
        engine.close()

    assert engine.vector_store.calls[0]["where"] == {"source": "auth.log"}


def test_candidate_pool_override_reaches_the_retrievers(db_path):
    engine = build_engine(db_path, ["w-1", "w-2", "w-3"])
    try:
        result = engine.retrieve("failed password", k=2, pool=7)
    finally:
        engine.close()

    assert engine.vector_store.calls[0]["k"] == 7
    assert result.diagnostics["candidate_pool"] == 7


# --------------------------------------------------------------------------- #
# reranking
# --------------------------------------------------------------------------- #


def test_reranker_reorders_and_its_score_becomes_the_hit_score(db_path):
    reranker = FakeReranker(["w-3", "w-1", "w-2"])
    engine = build_engine(db_path, ["w-1", "w-2", "w-3"], reranker=reranker)
    try:
        result = engine.retrieve("failed password", k=3)
    finally:
        engine.close()

    assert result.chunk_ids() == ["w-3", "w-1", "w-2"]
    assert result.hits[0].rerank_score == pytest.approx(1.0)
    assert result.hits[0].score == pytest.approx(1.0)
    assert result.diagnostics["rerank_applied"] is True
    assert result.diagnostics["reranker"] == "fake"


def test_reranker_receives_the_window_text_not_the_id(db_path):
    reranker = FakeReranker(["w-1"])
    engine = build_engine(db_path, ["w-1"], reranker=reranker)
    try:
        engine.retrieve("failed password", k=1)
    finally:
        engine.close()

    chunk_id, text = reranker.seen[0]
    assert chunk_id == "w-1"
    assert "Failed password" in text


def test_rerank_false_keeps_the_fused_order(db_path):
    reranker = FakeReranker(["w-3", "w-1", "w-2"])
    engine = build_engine(db_path, ["w-1", "w-2", "w-3"], reranker=reranker)
    try:
        result = engine.retrieve("failed password", k=3, rerank=False)
    finally:
        engine.close()

    assert result.diagnostics["rerank_applied"] is False
    assert result.hits[0].rerank_score is None
    assert reranker.seen == []


def test_null_reranker_is_treated_as_no_reranking(db_path):
    engine = build_engine(db_path, ["w-1", "w-2"], reranker=NullReranker())
    try:
        result = engine.retrieve("failed password", k=2)
    finally:
        engine.close()

    assert result.diagnostics["rerank_applied"] is False
    assert all(hit.rerank_score is None for hit in result.hits)


# --------------------------------------------------------------------------- #
# rendering / serialisation
# --------------------------------------------------------------------------- #


def test_render_joins_window_text_and_can_truncate(db_path):
    engine = build_engine(db_path, ["w-1", "w-2"])
    try:
        result = engine.retrieve("failed password", k=2)
    finally:
        engine.close()

    rendered = result.render()
    assert "[WINDOW w-1]" in rendered
    assert rendered.count("[WINDOW") == len(result.hits)

    truncated = result.render(max_chars_per_window=40)
    assert "[window truncated]" in truncated


def test_as_dict_is_json_serialisable(db_path):
    engine = build_engine(db_path, ["w-1", "w-2"])
    try:
        payload = engine.retrieve("failed password", k=2).as_dict()
    finally:
        engine.close()

    text = json.dumps(payload)  # must not raise
    assert json.loads(text)["hits"][0]["chunk_id"] == payload["hits"][0]["chunk_id"]
    assert payload["hits"][0]["found_by"]


def test_retrieved_window_exposes_both_provenance_ranks():
    hit = RetrievedWindow(
        chunk_id="w-1", rank=1, score=0.5, chunk=CHUNKS[0], dense_rank=4, bm25_rank=1
    )
    assert hit.found_by == ("bm25", "dense")


def test_engine_requires_an_ingested_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        RetrievalEngine.from_settings(db_path=tmp_path / "missing.sqlite")


# --------------------------------------------------------------------------- #
# indexer (Chroma) — deterministic fake embedder, no model download
# --------------------------------------------------------------------------- #


class FakeEmbeddings:
    """Hashed bag-of-words vectors: deterministic, dependency-free, no download."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in (text or "").lower().split():
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dim] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def chunks_jsonl(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps(chunk.to_dict()) for chunk in CHUNKS), encoding="utf-8"
    )
    return path


def test_build_index_and_search_roundtrip(tmp_path, chunks_jsonl):
    chromadb = pytest.importorskip("chromadb", reason="chromadb is not installed")
    del chromadb

    from slrag.retrieval.indexer import build_index
    from slrag.retrieval.vector_store import VectorStore, read_index_meta

    chroma_dir = tmp_path / "chroma"
    result = build_index(
        chunks_jsonl,
        settings=_settings_for(tmp_path, chroma_dir),
        embeddings=FakeEmbeddings(),
        verbose=False,
    )
    assert result.chunks_indexed == 3
    assert result.batches == 1

    meta = read_index_meta(chroma_dir)
    assert meta["chunks_indexed"] == 3
    assert meta["total_in_collection"] == 3
    assert meta["collection"] == result.collection

    store = VectorStore(
        chroma_dir, settings=_settings_for(tmp_path, chroma_dir), embeddings=FakeEmbeddings()
    )
    assert store.count() == 3

    hits = store.search("failed password sudo nginx", k=3)
    assert len(hits) == 3
    assert hits[0].chunk_id == "w-1"
    assert hits[0].rank == 1
    assert -1.0001 <= hits[0].score <= 1.0001
    assert hits[0].metadata["risk_score"] == 62

    # reset must empty the collection without deleting the directory
    store.reset()
    assert store.count() == 0


def test_index_reports_a_missing_chunks_file(tmp_path):
    from slrag.retrieval.indexer import build_index

    with pytest.raises(FileNotFoundError):
        build_index(
            tmp_path / "nope.jsonl",
            settings=_settings_for(tmp_path, tmp_path / "chroma"),
            embeddings=FakeEmbeddings(),
            verbose=False,
        )


def _settings_for(tmp_path, chroma_dir):
    """Settings pointed at a temporary directory (keeps the real data/ untouched)."""
    from slrag.config import Settings

    return Settings(
        chroma_dir=chroma_dir,
        sqlite_path=tmp_path / "events.sqlite",
        data_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        cache_dir=tmp_path / "cache",
        reports_dir=tmp_path / "reports",
        rerank_enabled=False,
    )
