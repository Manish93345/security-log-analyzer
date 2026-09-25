"""Embedding-model factory.

Three backends, all selectable from ``.env``:

``fastembed`` (default)
    ONNX runtime, ~90 MB model, **no torch**, CPU-friendly. Installs in seconds —
    this is what makes "finish today" realistic. The same backend also runs on an NVIDIA
    GPU for free: install ``onnxruntime-gpu`` and set ``EMBEDDING_DEVICE=cuda``.
``sentence_transformers``
    Better quality on some benchmarks, but a ~2 GB torch download.
``openai``
    ``text-embedding-3-small`` — paid. Only used if you opt in.

All three are exposed to LangChain through the same ``Embeddings`` interface, so the
vector store and the retrieval code never know which one is active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..config import Settings, get_settings
from ..logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

logger = get_logger(__name__)

try:  # LangChain is optional at import time so unit tests run without it.
    from langchain_core.embeddings import Embeddings as _BaseEmbeddings
except ImportError:  # pragma: no cover

    class _BaseEmbeddings:  # type: ignore[no-redef]
        """Fallback base class so this module imports without LangChain installed."""

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            raise NotImplementedError("langchain-core is not installed")

        def embed_query(self, text: str) -> list[float]:
            raise NotImplementedError("langchain-core is not installed")


#: Known output dimensions, so we can validate a persisted index without loading a model.
_KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class FastEmbedEmbeddings(_BaseEmbeddings):
    """LangChain-compatible wrapper around ``fastembed.TextEmbedding`` (ONNX)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        device: str = "auto",
        threads: int | None = None,
    ):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = (device or "auto").strip().lower()
        self.threads = int(threads) if threads else None
        logger.info(
            "embeddings.load",
            extra={
                "backend": "fastembed",
                "model": model_name,
                "device": self.device,
                "threads": self.threads,
            },
        )
        kwargs: dict[str, Any] = {"model_name": model_name}
        if self.threads:
            kwargs["threads"] = self.threads
        # ``auto`` passes nothing: fastembed's own default, so an existing CPU-only install
        # is untouched. ``cuda`` fails loudly rather than silently falling back to the CPU,
        # because a silent fallback is how you end up timing the wrong thing for hours.
        if self.device in {"cuda", "cpu"}:
            kwargs["cuda"] = self.device == "cuda"
        self._model = TextEmbedding(**kwargs)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.embed(list(texts), batch_size=self.batch_size)
        return [list(map(float, vec)) for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = list(self._model.query_embed(text))
        return list(map(float, vectors[0]))


class SentenceTransformerEmbeddings(_BaseEmbeddings):
    """LangChain-compatible wrapper around ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        logger.info(
            "embeddings.load",
            extra={"backend": "sentence_transformers", "model": model_name},
        )
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True
        )
        return [list(map(float, vec)) for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return list(map(float, vec))


def _openai_embeddings(model_name: str):
    from langchain_openai import OpenAIEmbeddings

    settings = get_settings()
    return OpenAIEmbeddings(model=model_name, api_key=settings.openai_api_key or None)


_BACKENDS = {
    "fastembed": FastEmbedEmbeddings,
    "sentence_transformers": SentenceTransformerEmbeddings,
    "openai": _openai_embeddings,
}


class EmbeddingBackend:
    """Names of the supported backends (use instead of string literals)."""

    FASTEMBED = "fastembed"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"

    ALL = (FASTEMBED, SENTENCE_TRANSFORMERS, OPENAI)


def get_embedding_model(settings: Settings | None = None, backend: str | None = None):
    """Build the configured embedding model (lazy: downloads the model on first call)."""
    settings = settings or get_settings()
    backend = (backend or settings.embedding_backend).lower()
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND '{backend}'. Choose from: {list(_BACKENDS)}"
        )
    if backend == EmbeddingBackend.OPENAI:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai embedding backend.")
        return _openai_embeddings(settings.embedding_model)
    if backend == EmbeddingBackend.FASTEMBED:
        return FastEmbedEmbeddings(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
            threads=settings.embedding_threads,
        )
    return SentenceTransformerEmbeddings(
        model_name=settings.embedding_model, batch_size=settings.embedding_batch_size
    )


def available_providers() -> list[str]:
    """ONNX Runtime execution providers present here (``[]`` when ORT is missing)."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:  # pragma: no cover - ORT ships with fastembed; stay defensive anyway
        return []


def cuda_available() -> bool:
    """True only when ONNX Runtime can actually execute on a CUDA device here."""
    return "CUDAExecutionProvider" in available_providers()


def embedding_device_report(settings: Settings | None = None) -> dict[str, Any]:
    """Facts for ``slrag doctor``: configured device, live providers, effective device."""
    settings = settings or get_settings()
    providers = available_providers()
    has_cuda = "CUDAExecutionProvider" in providers
    configured = (settings.embedding_device or "auto").lower()
    if configured == "cuda":
        effective = "cuda" if has_cuda else "cuda (UNAVAILABLE — indexing will fail)"
    elif configured == "cpu":
        effective = "cpu"
    else:
        effective = "cuda" if has_cuda else "cpu"
    return {
        "configured": configured,
        "effective": effective,
        "onnxruntime_providers": providers,
        "cuda_available": has_cuda,
    }


def get_embedding_dimension(model_name: str) -> int:
    """Best-effort dimension lookup for a model name (used to validate a built index)."""
    if model_name in _KNOWN_DIMS:
        return _KNOWN_DIMS[model_name]
    raise ValueError(
        f"Unknown embedding dimension for '{model_name}'. Add it to _KNOWN_DIMS in "
        "src/slrag/providers/embeddings.py"
    )


__all__ = [
    "EmbeddingBackend",
    "FastEmbedEmbeddings",
    "SentenceTransformerEmbeddings",
    "available_providers",
    "cuda_available",
    "embedding_device_report",
    "get_embedding_dimension",
    "get_embedding_model",
]
