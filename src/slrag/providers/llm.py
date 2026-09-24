"""Chat-model factory.

Free providers only by default:

============  ==========================================================
provider      how it is reached
============  ==========================================================
``groq``      ``ChatOpenAI`` with ``base_url=https://api.groq.com/openai/v1``
``openrouter````ChatOpenAI`` with ``base_url=https://openrouter.ai/api/v1``
``ollama``    ``ChatOpenAI`` with ``base_url=http://localhost:11434/v1``
``openai``    ``ChatOpenAI`` (paid — opt in via .env only)
``gemini``    ``ChatGoogleGenerativeAI`` (free tier)
============  ==========================================================

Every OpenAI-compatible provider goes through the same ``ChatOpenAI`` class, which is why
switching providers is a one-line ``.env`` change.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings, get_settings
from ..logging_setup import get_logger

logger = get_logger(__name__)

_OPENAI_COMPATIBLE_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
}

_KEY_ATTR = {
    "groq": "groq_api_key",
    "openrouter": "openrouter_api_key",
    "openai": "openai_api_key",
    "gemini": "google_api_key",
}


class ProviderConfigError(RuntimeError):
    """Raised when the selected provider has no usable credentials."""


def build_chat_model(
    provider: str | None = None,
    model: str | None = None,
    *,
    settings: Settings | None = None,
    temperature: float | None = None,
    **kwargs: Any,
):
    """Return a LangChain chat model for ``provider``/``model``.

    Imported lazily so that the package (and the fast unit tests) work without
    LangChain installed.
    """
    settings = settings or get_settings()
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model
    temperature = settings.llm_temperature if temperature is None else temperature

    if provider == "gemini":
        if not settings.google_api_key:
            raise ProviderConfigError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/app/apikey"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("llm.provider", extra={"provider": provider, "model": model})
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_s,
            **kwargs,
        )

    from langchain_openai import ChatOpenAI

    if provider == "ollama":
        base_url = settings.ollama_base_url
        api_key = "ollama"
    elif provider in _OPENAI_COMPATIBLE_BASE_URLS:
        base_url = _OPENAI_COMPATIBLE_BASE_URLS[provider]
        api_key = getattr(settings, _KEY_ATTR[provider], "")
        if not api_key:
            raise ProviderConfigError(
                f"{_KEY_ATTR[provider].upper()} is not set for provider '{provider}'. "
                "Groq free keys: https://console.groq.com/keys"
            )
    elif provider == "openai":
        base_url = None
        api_key = settings.openai_api_key
        if not api_key:
            raise ProviderConfigError("OPENAI_API_KEY is not set (this provider is paid).")
    else:
        raise ProviderConfigError(
            f"Unknown provider '{provider}'. "
            "Use one of: groq, gemini, openrouter, ollama, openai"
        )

    logger.info(
        "llm.provider", extra={"provider": provider, "model": model, "base_url": base_url}
    )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_s,
        **kwargs,
    )


def get_chat_model(settings: Settings | None = None, **kwargs: Any):
    """Chat model for the configured primary provider."""
    return build_chat_model(settings=settings, **kwargs)


def get_fallback_chat_model(settings: Settings | None = None, **kwargs: Any):
    """Chat model used when the primary provider returns 429 / 5xx.

    Returns ``None`` when the fallback is not configured or has no credentials, so
    callers can simply skip fallback handling.
    """
    settings = settings or get_settings()
    provider = settings.llm_fallback_provider.lower()
    if provider in {"", "none"}:
        return None
    key_attr = _KEY_ATTR.get(provider)
    if key_attr and not getattr(settings, key_attr, ""):
        logger.info("llm.fallback.unavailable", extra={"provider": provider})
        return None
    try:
        return build_chat_model(
            provider=provider, model=settings.llm_fallback_model, settings=settings, **kwargs
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("llm.fallback.failed", extra={"error": str(exc)})
        return None


__all__ = [
    "ProviderConfigError",
    "build_chat_model",
    "get_chat_model",
    "get_fallback_chat_model",
]
