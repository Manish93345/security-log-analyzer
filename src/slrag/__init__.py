"""slrag — RAG-powered security log analyzer.

Package layout:
    slrag.config        settings loaded from .env
    slrag.providers     LLM + embedding factories (free providers)
    slrag.ingest        log parsing, normalization, incident-window chunking
    slrag.retrieval     Chroma + BM25 hybrid retrieval with reranking
    slrag.agent         LangChain tool-calling pipeline
    slrag.guardrails    prompt-injection screen, redaction, citation checks
    slrag.eval          retrieval eval harness
    slrag.api           FastAPI service
    slrag.ui            Streamlit app
"""

__version__ = "0.2.0"
__all__ = ["__version__"]
