"""Provider-agnostic LLM interface.

The graph nodes depend only on this abstraction, so swapping Gemini for
another provider (or for the replay cassette) is a configuration change rather
than a rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)

Role = Literal["user", "model", "tool"]


@dataclass(slots=True)
class ToolCall:
    """A model-requested function invocation.

    ``thought_signature`` is an opaque token Gemini 3 attaches to function-call
    parts. It must be echoed back with the call when the conversation
    continues, or the next request fails with a 400 - which silently caps the
    tool loop at a single round.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    thought_signature: str | None = None


@dataclass(slots=True)
class ChatMessage:
    """One turn of conversation.

    A model turn that requested tools carries them in ``tool_calls``; the
    matching results come back as ``role="tool"`` turns. Both must be replayed
    to the provider on the next iteration or the model loses the thread of what
    it already asked for.
    """

    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_result: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(role="user", content=content)

    @classmethod
    def model(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> "ChatMessage":
        return cls(role="model", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, name: str, result: dict[str, Any], call_id: str | None = None) -> "ChatMessage":
        return cls(role="tool", name=name, tool_result=result, tool_call_id=call_id)


@dataclass(slots=True)
class Usage:
    """Token accounting for one call.

    ``thoughts_tokens`` captures Gemini's internal reasoning spend, which is
    billed but invisible in the response text - worth surfacing in traces so
    reasoning cost is not a black box.
    """

    prompt_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            thoughts_tokens=self.thoughts_tokens + other.thoughts_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(slots=True)
class LLMResponse:
    """Result of a generation call."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    model: str = ""
    latency_ms: float = 0.0
    from_cassette: bool = False

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """Capabilities the agent requires of any language-model backend."""

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        force_tool_use: bool = False,
        temperature: float | None = None,
        purpose: str = "generate",
    ) -> LLMResponse:
        """Free-form generation, optionally with tool calling enabled."""

    @abstractmethod
    def generate_structured(
        self,
        messages: list[ChatMessage],
        response_model: type[ModelT],
        *,
        system: str | None = None,
        temperature: float | None = None,
        purpose: str = "structured",
    ) -> ModelT:
        """Generation constrained to a Pydantic model's schema."""

    @abstractmethod
    def embed(self, texts: list[str], *, task_type: str = "query") -> list[list[float]]:
        """Embed texts for similarity ranking.

        Args:
            texts: Strings to embed.
            task_type: ``"query"`` for the question, ``"document"`` for the
                passages being searched. Retrieval models are asymmetric, and
                each adapter maps these onto its provider's own vocabulary.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
