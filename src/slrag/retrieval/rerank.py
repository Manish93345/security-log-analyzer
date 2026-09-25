"""Cross-encoder reranking.

Why rerank at all? Dense and lexical retrieval are *bi-encoders*: the query and the document
are embedded separately, so the two never actually read each other. That is what makes them
fast enough to scan 38k windows, and also why their ordering is approximate. A cross-encoder
takes ``(query, window)`` as a single input and scores it with full attention — far more
accurate, far too slow to run over the whole corpus. So we run it over the fused top-50 only.

Three interchangeable backends, all CPU-only and torch-free:

``fastembed`` (default)
    ``TextCrossEncoder`` over ``Xenova/ms-marco-MiniLM-L-6-v2`` — the same MiniLM-L6
    cross-encoder, fetched through fastembed's HuggingFace downloader. It reuses a dependency
    we already have, and it goes through the exact download path that also serves our
    embedding model, so there is one code path to trust rather than two.
``flashrank``
    Faster on paper, but FlashRank 0.2.10 hardcodes
    ``https://huggingface.co/prithivida/flashrank/resolve/main/<model>.zip`` and its
    historical default ``ms-marco-MiniLM-L-6-v2.zip`` returns **404** upstream (verified
    2026-09-24), so that model simply cannot be downloaded. ``ms-marco-TinyBERT-L-2-v2.zip``
    does return 200 and is the working choice for this backend.
``none``
    :class:`NullReranker` — keeps the RRF order.

A missing library, a failed model download or ``RERANK_ENABLED=false`` all degrade to
:class:`NullReranker` with a warning instead of failing the query. A retrieval system that
cannot start is worse than one that ranks slightly less well.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..config import Settings, get_settings
from ..logging_setup import get_logger

logger = get_logger(__name__)

RERANK_BACKEND_FASTEMBED = "fastembed"
RERANK_BACKEND_FLASHRANK = "flashrank"
RERANK_BACKEND_NONE = "none"

#: The fastembed repository id of the MiniLM-L6 cross-encoder (same model FlashRank used).
DEFAULT_FASTEMBED_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

#: FlashRank's historical default is a dead 404 upstream; this zip is actually present.
FLASHRANK_DEFAULT_MODEL = "ms-marco-TinyBERT-L-2-v2"

#: FlashRank truncates at this many tokens; ms-marco was trained at 512.
DEFAULT_MAX_LENGTH = 512

#: Characters per passage handed to the cross-encoder. A window can be 4800 chars; capping
#: here bounds the tokeniser input without changing the ranking of a well-formed window.
MAX_PASSAGE_CHARS = 4000

#: Short names (what people and older configs write) -> fastembed repository ids.
FASTEMBED_MODEL_ALIASES: dict[str, str] = {
    "ms-marco-minilm-l-6-v2": DEFAULT_FASTEMBED_MODEL,
    "ms-marco-minilm-l-12-v2": "Xenova/ms-marco-MiniLM-L-12-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "jina-reranker-v1-tiny-en": "jinaai/jina-reranker-v1-tiny-en",
    "jina-reranker-v1-turbo-en": "jinaai/jina-reranker-v1-turbo-en",
    "jina-reranker-v2-base-multilingual": "jinaai/jina-reranker-v2-base-multilingual",
}


def resolve_fastembed_model(model_name: str | None) -> str:
    """Map a bare model name onto fastembed's repository id (idempotent, never raises).

    ``RERANK_MODEL`` may hold either the short name we document
    (``ms-marco-MiniLM-L-6-v2``) or a full repo id (``Xenova/...``); both resolve correctly.
    """
    key = (model_name or "").strip()
    if not key:
        return DEFAULT_FASTEMBED_MODEL
    if "/" in key:
        return key
    return FASTEMBED_MODEL_ALIASES.get(key.lower(), key)


@dataclass(frozen=True, slots=True)
class RerankedHit:
    """A window re-scored by the cross-encoder, best first."""

    chunk_id: str
    score: float
    rank: int


class NullReranker:
    """Pass-through reranker: keeps the incoming order and reports no scores."""

    name = "none"
    available = False

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]], *, top_n: int = 50
    ) -> list[RerankedHit]:
        del query
        return [
            RerankedHit(chunk_id=chunk_id, score=0.0, rank=position + 1)
            for position, (chunk_id, _text) in enumerate(candidates[: max(0, top_n)])
        ]


class FastEmbedReranker:
    """Cross-encoder reranker backed by fastembed's ONNX ``TextCrossEncoder`` (default)."""

    name = RERANK_BACKEND_FASTEMBED
    available = True

    def __init__(self, model_name: str | None = None, *, cache_dir: str | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = resolve_fastembed_model(model_name)
        logger.info("rerank.load", extra={"backend": self.name, "model": self.model_name})
        # Downloads (and caches) the ~23 MB ONNX model on first construction.
        self._encoder = TextCrossEncoder(model_name=self.model_name, cache_dir=cache_dir)

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]], *, top_n: int = 50
    ) -> list[RerankedHit]:
        """Re-score ``candidates`` (``(chunk_id, text)`` pairs) against ``query``."""
        if not candidates or top_n <= 0:
            return []

        documents = [(text or "")[:MAX_PASSAGE_CHARS] for _chunk_id, text in candidates]
        # fastembed yields one float per document, in the order the documents were given.
        scores = [float(score) for score in self._encoder.rerank(query, documents)]

        scored = [
            (score, str(chunk_id))
            for (chunk_id, _text), score in zip(candidates, scores, strict=False)
        ]
        # Sort defensively rather than trusting the yielded order; ties break by id.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RerankedHit(chunk_id=chunk_id, score=score, rank=position + 1)
            for position, (score, chunk_id) in enumerate(scored[:top_n])
        ]


class FlashRankReranker:
    """Cross-encoder reranker backed by FlashRank (ONNX, CPU, no torch).

    Secondary backend — see the module docstring for the dead-URL caveat.
    """

    name = RERANK_BACKEND_FLASHRANK
    available = True

    def __init__(self, model_name: str | None = None, *, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        from flashrank import Ranker

        self.model_name = model_name or FLASHRANK_DEFAULT_MODEL
        self.max_length = int(max_length)
        logger.info("rerank.load", extra={"backend": self.name, "model": self.model_name})
        self._ranker = Ranker(model_name=self.model_name, max_length=self.max_length)

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]], *, top_n: int = 50
    ) -> list[RerankedHit]:
        from flashrank import RerankRequest

        if not candidates or top_n <= 0:
            return []

        passages = [
            {"id": str(chunk_id), "text": (text or "")[:MAX_PASSAGE_CHARS]}
            for chunk_id, text in candidates
        ]
        results = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        ordered = sorted(
            results, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("id")))
        )
        return [
            RerankedHit(
                chunk_id=str(item.get("id")),
                score=float(item.get("score") or 0.0),
                rank=position + 1,
            )
            for position, item in enumerate(ordered[:top_n])
        ]


_BACKENDS = {
    RERANK_BACKEND_FASTEMBED: FastEmbedReranker,
    RERANK_BACKEND_FLASHRANK: FlashRankReranker,
}


def get_reranker(
    settings: Settings | None = None, *, enabled: bool | None = None, backend: str | None = None
):
    """Build the configured reranker, falling back to :class:`NullReranker`.

    Never raises: ``RERANK_ENABLED=false``, ``RERANK_BACKEND=none``, a missing library or a
    failed model download all produce pass-through reranking with a warning.
    """
    settings = settings or get_settings()

    wanted = settings.rerank_enabled if enabled is None else bool(enabled)
    if not wanted:
        logger.info("rerank.disabled", extra={"reason": "RERANK_ENABLED=false"})
        return NullReranker()

    choice = (backend or settings.rerank_backend or RERANK_BACKEND_FASTEMBED).strip().lower()
    if choice in {RERANK_BACKEND_NONE, "", "off", "null"}:
        logger.info("rerank.disabled", extra={"reason": "RERANK_BACKEND=none"})
        return NullReranker()

    factory = _BACKENDS.get(choice)
    if factory is None:
        logger.warning(
            "rerank.unknown_backend",
            extra={"backend": choice, "known": sorted(_BACKENDS)},
        )
        return NullReranker()

    try:
        return factory(settings.rerank_model)
    except ImportError:
        logger.warning("rerank.unavailable", extra={"backend": choice, "reason": "not installed"})
        return NullReranker()
    except Exception as exc:  # pragma: no cover - network / model download failure
        logger.warning("rerank.failed", extra={"backend": choice, "error": str(exc)})
        return NullReranker()


__all__ = [
    "DEFAULT_FASTEMBED_MODEL",
    "DEFAULT_MAX_LENGTH",
    "FASTEMBED_MODEL_ALIASES",
    "FLASHRANK_DEFAULT_MODEL",
    "MAX_PASSAGE_CHARS",
    "RERANK_BACKEND_FASTEMBED",
    "RERANK_BACKEND_FLASHRANK",
    "RERANK_BACKEND_NONE",
    "FastEmbedReranker",
    "FlashRankReranker",
    "NullReranker",
    "RerankedHit",
    "get_reranker",
    "resolve_fastembed_model",
]
