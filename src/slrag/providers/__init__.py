"""Provider factories: chat models and embedding models, all free-tier friendly."""

from .embeddings import (
    EmbeddingBackend,
    get_embedding_model,
    get_embedding_dimension,
)
from .llm import build_chat_model, get_chat_model

__all__ = [
    "EmbeddingBackend",
    "build_chat_model",
    "get_chat_model",
    "get_embedding_dimension",
    "get_embedding_model",
]
