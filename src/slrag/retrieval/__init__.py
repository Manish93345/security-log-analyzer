"""Retrieval layer (Phase 3).

    question
        |
        +--> Chroma / dense  (BAAI/bge-small-en-v1.5)  --\\
        |                                                 >-- RRF (k=60) --> FlashRank --> top-k windows
        +--> BM25 / lexical  (search_text)             --/

Both retrievers are fed the *same* incident windows, so a window one retriever misses can
still be rescued by the other. That is the whole point of the hybrid: embeddings are good at
"what is this window about", BM25 is good at "does this window literally contain
``203.0.113.44``". Fusing them beats either alone, and the cross-encoder then re-reads only
the fused top-50 with full attention.

Everything heavy (chromadb, rank-bm25, flashrank, fastembed) is imported lazily inside the
classes, so importing this package — and running ``slrag version`` — never triggers a model
download or a slow import.
"""

from .bm25 import BM25Hit, BM25Index, tokenize
from .fusion import DEFAULT_RRF_K, FusionHit, reciprocal_rank_fusion
from .indexer import IndexResult, build_index, read_index_meta
from .pipeline import RetrievalEngine, RetrievalResult, RetrievedWindow, retrieve
from .rerank import FlashRankReranker, NullReranker, RerankedHit, get_reranker
from .vector_store import VectorHit, VectorStore

__all__ = [
    "DEFAULT_RRF_K",
    "BM25Hit",
    "BM25Index",
    "FlashRankReranker",
    "FusionHit",
    "IndexResult",
    "NullReranker",
    "RerankedHit",
    "RetrievalEngine",
    "RetrievalResult",
    "RetrievedWindow",
    "VectorHit",
    "VectorStore",
    "build_index",
    "get_reranker",
    "read_index_meta",
    "reciprocal_rank_fusion",
    "retrieve",
    "tokenize",
]
