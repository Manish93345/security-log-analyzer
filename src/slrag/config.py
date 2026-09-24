"""Central configuration.

Every tunable comes from environment variables (loaded from ``.env`` when present)
with safe defaults, so the package imports and the test suite runs with no .env at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # optional: tests run without python-dotenv installed
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    path = Path(_env(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Resolved application settings (immutable)."""

    # --- LLM ---------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "openai/gpt-oss-120b"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 4))
    llm_timeout_s: int = field(default_factory=lambda: _env_int("REQUEST_TIMEOUT_S", 60))
    llm_fallback_provider: str = field(
        default_factory=lambda: _env("LLM_FALLBACK_PROVIDER", "gemini")
    )
    llm_fallback_model: str = field(
        default_factory=lambda: _env("LLM_FALLBACK_MODEL", "gemini-3.5-flash")
    )

    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: _env("GOOGLE_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )

    # --- embeddings / rerank ----------------------------------------------
    embedding_backend: str = field(
        default_factory=lambda: _env("EMBEDDING_BACKEND", "fastembed")
    )
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_batch_size: int = field(
        default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 64)
    )
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("RERANK_ENABLED", True))
    rerank_model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", "ms-marco-MiniLM-L-6-v2")
    )
    rerank_top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 50))
    retrieve_top_k: int = field(default_factory=lambda: _env_int("RETRIEVE_TOP_K", 5))
    candidate_pool: int = field(default_factory=lambda: _env_int("CANDIDATE_POOL", 50))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))

    # --- paths -------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _env_path("DATA_DIR", "data"))
    raw_dir: Path = field(default_factory=lambda: _env_path("RAW_DIR", "data/raw"))
    processed_dir: Path = field(
        default_factory=lambda: _env_path("PROCESSED_DIR", "data/processed")
    )
    chroma_dir: Path = field(default_factory=lambda: _env_path("CHROMA_DIR", "data/chroma"))
    sqlite_path: Path = field(
        default_factory=lambda: _env_path("SQLITE_PATH", "data/processed/events.sqlite")
    )
    cache_dir: Path = field(default_factory=lambda: _env_path("CACHE_DIR", "data/cache"))
    reports_dir: Path = field(default_factory=lambda: _env_path("REPORTS_DIR", "reports"))

    # --- chunking ----------------------------------------------------------
    gap_minutes: int = field(default_factory=lambda: _env_int("CHUNK_GAP_MINUTES", 15))
    max_chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_MAX_CHARS", 4800))
    max_events_per_chunk: int = field(
        default_factory=lambda: _env_int("CHUNK_MAX_EVENTS", 120)
    )

    # --- guardrails --------------------------------------------------------
    guardrails_enabled: bool = field(
        default_factory=lambda: _env_bool("GUARDRAILS_ENABLED", True)
    )
    max_prompt_tokens: int = field(
        default_factory=lambda: _env_int("MAX_PROMPT_TOKENS", 12000)
    )

    # --- api ---------------------------------------------------------------
    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # ------------------------------------------------------------------ api
    def ensure_dirs(self) -> None:
        """Create the directories the pipeline writes into."""
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.chroma_dir,
            self.cache_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def llm_api_key(self) -> str:
        """The key that matches the selected provider ('' for local providers)."""
        return {
            "groq": self.groq_api_key,
            "gemini": self.google_api_key,
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
            "ollama": "ollama",
        }.get(self.llm_provider, "")

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — import this, do not instantiate Settings directly."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests that mutate the environment)."""
    get_settings.cache_clear()


__all__ = ["Settings", "get_settings", "reset_settings_cache", "PROJECT_ROOT"]
