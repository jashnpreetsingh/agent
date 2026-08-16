"""Exception hierarchy for the LLM layer.

The distinction that matters operationally is *retryable* vs *terminal*:
a 503 deserves another attempt, a 400 never will. The graph nodes catch
``LLMError`` and degrade gracefully rather than crashing the run.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all provider failures."""


class LLMTransportError(LLMError):
    """Network/HTTP failure talking to the provider."""

    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class LLMResponseError(LLMError):
    """The provider replied, but the payload was unusable."""


class LLMRefusalError(LLMError):
    """The provider blocked the request (safety filters, recitation, etc.).

    Worth its own type: a refusal is a *content* outcome, not an outage, so
    retrying the identical request is pointless.
    """

    def __init__(self, message: str, finish_reason: str | None = None):
        super().__init__(message)
        self.finish_reason = finish_reason


class SchemaValidationError(LLMError):
    """Structured output did not validate against the requested model."""

    def __init__(self, message: str, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


class CassetteMissError(LLMError):
    """Replay mode was asked for an interaction that was never recorded."""
