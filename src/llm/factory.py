"""Construct the configured LLM provider.

Selecting a backend happens here and nowhere else, so the rest of the codebase
depends only on the :class:`~src.llm.base.LLMProvider` interface.
"""

from __future__ import annotations

from src.config import Provider, Settings
from src.llm.base import LLMProvider
from src.observability.trace import Tracer


def build_provider(settings: Settings, tracer: Tracer | None = None) -> LLMProvider:
    """Return the provider named by ``settings.provider``."""
    if settings.provider is Provider.GEMINI:
        from src.llm.gemini import GeminiProvider

        return GeminiProvider(settings, tracer)

    from src.llm.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(settings, tracer)
