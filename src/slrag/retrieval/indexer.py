"""Build the dense index from ``chunks.jsonl``.

``slrag ingest`` writes ``data/processed/chunks.jsonl`` (one window per line) as the hand-off
artefact to this stage. Indexing streams that file, embeds in batches and writes into the
persistent Chroma collection, then records the build facts in ``index_meta.json`` so a stale
index (different embedding model, different dimension, fewer windows than the corpus) is
detectable without loading a model.

Indexing the full corpus (~38k windows) is a one-time cost; queries afterwards are
milliseconds.

Two properties matter at this corpus size and are easy to get wrong:

**Resumable.** Each batch is committed to Chroma as soon as it is embedded, so a run
interrupted mid-way (Ctrl-C because the laptop was needed for something else) leaves
everything it finished on disk. ``INDEX_RESUME=true`` — the default — reads the ids already
present and skips them *before* embedding, so a re-run costs only the remainder instead of
all 38k windows again. ``resume=False`` forces a full pass.

**Self-reporting.** The progress line carries a percentage, a live windows/s rate and an ETA.
Without them a run that is 99% finished looks identical to one that has barely started —
which is how a finished index gets killed by mistake.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..ingest.store import iter_chunks_jsonl
from ..logging_setup import get_logger
from ..providers.embeddings import get_embedding_dimension
from .vector_store import INDEX_META_FILE, VectorStore, utc_now_iso, write_index_meta

logger = get_logger(__name__)


@dataclass(slots=True)
class IndexResult:
    """What ``slrag index`` did — enough to print a report and to diff two runs."""

    chunks_indexed: int = 0
    batches: int = 0
    elapsed_s: float = 0.0
    persist_dir: str = ""
    collection: str = ""
    embedding_model: str = ""
    dimension: int = 0
    chunks_path: str = ""
    limit: int | None = None
    windows_per_second: float = 0.0
    total_windows: int = 0
    already_indexed: int = 0
    resumed: bool = False
    device: str = "auto"

    @property
    def skipped_existing(self) -> int:
        """Windows that were already in the collection and cost nothing this run."""
        return max(self.total_windows - self.chunks_indexed, 0) if self.resumed else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunks_indexed": self.chunks_indexed,
            "batches": self.batches,
            "elapsed_s": round(self.elapsed_s, 2),
            "windows_per_second": round(self.windows_per_second, 1),
            "total_windows": self.total_windows,
            "already_indexed": self.already_indexed,
            "skipped_existing": self.skipped_existing,
            "resumed": self.resumed,
            "device": self.device,
            "persist_dir": self.persist_dir,
            "collection": self.collection,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "chunks_path": self.chunks_path,
            "limit": self.limit,
        }


def _resolve_dimension(settings: Settings) -> int:
    """Best-effort embedding dimension; ``0`` for a model we have no table entry for."""
    try:
        return get_embedding_dimension(settings.embedding_model)
    except ValueError:
        return 0


def _count_lines(path: Path) -> int:
    """Line count of ``chunks.jsonl`` — the denominator for the progress percentage."""
    total = 0
    with path.open("rb") as handle:
        for _ in handle:
            total += 1
    return total


def _format_eta(seconds: float) -> str:
    """Compact ETA: ``12s`` / ``3m 40s`` / ``1h 05m``."""
    if seconds <= 0 or seconds != seconds:  # <= 0, and NaN (seconds != seconds)
        return "--"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {total % 3600 // 60:02d}m"


def build_index(
    chunks_path: str | Path,
    *,
    settings: Settings | None = None,
    limit: int | None = None,
    reset: bool = False,
    resume: bool = True,
    embeddings: Any = None,
    verbose: bool = True,
) -> IndexResult:
    """Embed every window in ``chunks_path`` into the persistent Chroma collection.

    Args:
        chunks_path: ``data/processed/chunks.jsonl`` from the ingest stage.
        limit: index only the first N windows (smoke test / quick sanity check).
        reset: drop the existing collection first (otherwise windows are added by id).
        resume: skip windows already in the collection. ``True`` makes an interrupted run
            cheap to finish; ``False`` re-embeds everything.
        embeddings: inject an embedding model (used by tests to avoid a model download).
        verbose: print progress lines.
    """
    settings = settings or get_settings()
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `slrag ingest` first (it writes chunks.jsonl)"
        )

    store = VectorStore(settings.chroma_dir, settings=settings, embeddings=embeddings)
    if reset:
        if verbose:
            print(f"  resetting collection '{store.collection_name}'", flush=True)
        store.reset()

    total = _count_lines(path)
    if limit:
        total = min(total, int(limit))
    already = 0 if reset else store.count()
    resuming = bool(resume and not reset and already)

    result = IndexResult(
        persist_dir=str(settings.chroma_dir),
        collection=store.collection_name,
        embedding_model=settings.embedding_model,
        dimension=_resolve_dimension(settings),
        chunks_path=str(path),
        limit=limit,
        total_windows=total,
        already_indexed=already,
        resumed=resuming,
        device=settings.embedding_device,
    )

    if verbose:
        print(f"  corpus      {total:,} windows in {path.name}")
        print(f"  collection  {already:,} windows already in '{store.collection_name}'")
        print(f"  device      {settings.embedding_device}")
        if resuming:
            print(f"  resume      on — embedding only the ~{max(total - already, 0):,} missing")
        else:
            print("  resume      off — embedding every window")
        print()

    started = time.perf_counter()

    def on_batch(written: int) -> None:
        result.batches += 1
        if not verbose:
            return
        elapsed = time.perf_counter() - started
        rate = written / elapsed if elapsed > 0 else 0.0
        done = written + (already if resuming else 0)
        pct = 100.0 * done / total if total else 100.0
        eta = (total - done) / rate if rate > 0 else 0.0
        print(
            f"  {done:>7,}/{total:,} ({pct:5.1f}%)  {rate:6.1f}/s  eta {_format_eta(eta)}",
            flush=True,
        )

    chunks = iter_chunks_jsonl(path)
    if limit:
        chunks = _take(chunks, int(limit))
    result.chunks_indexed = store.add_chunks(
        chunks,
        on_batch=on_batch,
        skip_existing=bool(resume and not reset),
    )
    result.elapsed_s = time.perf_counter() - started
    if result.elapsed_s > 0:
        result.windows_per_second = result.chunks_indexed / result.elapsed_s

    # ``chunks_indexed`` records the size of the *index*, not of this run, so the stale-index
    # check compares like with like after a resumed build.
    in_collection = store.count()
    write_index_meta(
        settings.chroma_dir,
        {
            "built_at": utc_now_iso(),
            "collection": result.collection,
            "embedding_model": result.embedding_model,
            "dimension": result.dimension,
            "embedding_device": result.device,
            "chunks_indexed": in_collection,
            "embedded_this_run": result.chunks_indexed,
            "total_in_collection": in_collection,
            "chunks_path": result.chunks_path,
            "limit": result.limit,
            "elapsed_s": round(result.elapsed_s, 2),
            "windows_per_second": round(result.windows_per_second, 1),
            "resumed": resuming,
        },
    )

    logger.info(
        "index.done",
        extra={
            "chunks_indexed": result.chunks_indexed,
            "already_indexed": result.already_indexed,
            "elapsed_s": round(result.elapsed_s, 2),
        },
    )
    return result


def _take(iterator, count: int):
    """``itertools.islice`` with a readable name (kept local to avoid the import noise)."""
    from itertools import islice

    return islice(iterator, count)


def read_index_meta(persist_dir: str | Path) -> dict[str, Any]:
    """Re-exported from :mod:`slrag.retrieval.vector_store` for convenience."""
    from .vector_store import read_index_meta as _read

    return _read(persist_dir)


__all__ = ["INDEX_META_FILE", "IndexResult", "build_index", "read_index_meta"]
