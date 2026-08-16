"""Provider for OpenAI-compatible chat APIs (NVIDIA NIM, OpenAI, vLLM, …).

Implements the same :class:`~src.llm.base.LLMProvider` contract as the Gemini
client, so swapping backends is a configuration change. The differences that
matter are all in the wire format:

* the system prompt is a message, not a separate field;
* tools are ``{"type": "function", "function": {...}}`` and arguments arrive as
  a JSON *string* that has to be parsed (and can be malformed);
* structured output uses ``response_format={"type": "json_schema", ...}`` with
  strict-mode schema rules;
* tool results are ``role="tool"`` turns correlated by ``tool_call_id``.

Reasoning models served here may also return ``reasoning_content``; it is
traced rather than treated as answer text.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.cassette import CassetteStore, canonical_key
from src.config import Settings, TransportMode
from src.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolCall, Usage
from src.llm.errors import (
    CassetteMissError,
    LLMRefusalError,
    LLMResponseError,
    LLMTransportError,
    SchemaValidationError,
)
from src.llm.schema_utils import to_openai_schema
from src.observability.trace import Tracer

ModelT = TypeVar("ModelT", bound=BaseModel)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

#: Asymmetric retrieval models encode questions and documents differently;
#: passing the wrong side silently degrades ranking quality.
QUERY_INPUT_TYPE = "query"
PASSAGE_INPUT_TYPE = "passage"


class OpenAICompatProvider(LLMProvider):
    """Chat, tool calling, structured output, and embeddings over one API."""

    def __init__(
        self,
        settings: Settings,
        tracer: Tracer | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer or Tracer.null()
        self.mode = settings.llm_mode
        self._cassette = CassetteStore(settings.fixtures_dir, "llm")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=settings.request_timeout)
        self.total_usage = Usage()

        if self.mode in (TransportMode.LIVE, TransportMode.RECORD) and not settings.api_key:
            raise ValueError(
                f"An API key is required for llm_mode={self.mode.value}. Set "
                f"{settings.provider.upper()}_API_KEY in .env, or use LLM_MODE=replay."
            )

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
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
        body = self._build_body(messages, system=system, temperature=temperature)
        if tools:
            body["tools"] = [{"type": "function", "function": tool} for tool in tools]
            body["tool_choice"] = "required" if force_tool_use else "auto"
        raw = self._call("/chat/completions", body, purpose)
        return self._parse_response(raw, purpose=purpose)

    def generate_structured(
        self,
        messages: list[ChatMessage],
        response_model: type[ModelT],
        *,
        system: str | None = None,
        temperature: float | None = None,
        purpose: str = "structured",
    ) -> ModelT:
        body = self._build_body(messages, system=system, temperature=temperature)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": _schema_name(response_model),
                "schema": to_openai_schema(response_model.model_json_schema()),
                "strict": True,
            },
        }

        raw = self._call("/chat/completions", body, purpose)
        response = self._parse_response(raw, purpose=purpose)

        try:
            return self._validate(response.text, response_model)
        except SchemaValidationError as first_error:
            self.tracer.event(
                "llm.schema_retry",
                purpose=purpose,
                model=response_model.__name__,
                error=str(first_error)[:400],
            )
            repair = list(messages) + [
                ChatMessage.model(response.text[:2000]),
                ChatMessage.user(
                    f"That response failed schema validation with: {str(first_error)[:600]}\n"
                    "Return corrected JSON that satisfies the schema exactly. Output JSON only."
                ),
            ]
            retry_body = self._build_body(repair, system=system, temperature=0.0)
            retry_body["response_format"] = body["response_format"]
            raw2 = self._call("/chat/completions", retry_body, f"{purpose}.repair")
            return self._validate(
                self._parse_response(raw2, purpose=f"{purpose}.repair").text, response_model
            )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed(self, texts: list[str], *, task_type: str = "query") -> list[list[float]]:
        """Embed texts as either queries or passages.

        NVIDIA's retriever models are asymmetric and *require* ``input_type``;
        OpenAI has no such parameter and rejects the request outright. The
        difference is declared in settings rather than guessed from the model
        name, so a third endpoint can be configured without code changes.
        """
        if not texts:
            return []
        input_type = (
            PASSAGE_INPUT_TYPE
            if task_type.lower().startswith(("passage", "document"))
            else QUERY_INPUT_TYPE
        )
        body: dict[str, Any] = {
            "model": self.settings.embedding_model,
            "input": [text[:6000] for text in texts],
            "encoding_format": "float",
        }
        if self.settings.embedding_input_type:
            body["input_type"] = input_type
            body["truncate"] = "END"
        raw = self._call("/embeddings", body, f"embed.{input_type}")
        data = raw.get("data", [])
        if len(data) != len(texts):
            raise LLMResponseError(
                f"embedding count mismatch: asked {len(texts)}, got {len(data)}"
            )
        # The API does not guarantee ordering; sort by the echoed index.
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item.get("embedding", []) for item in ordered]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------
    def _build_body(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})
        wire.extend(self._to_messages(messages))
        body: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": wire,
            # Newer OpenAI models reject `max_tokens` outright and require
            # `max_completion_tokens`; NVIDIA takes the former.
            self.settings.max_tokens_field: self.settings.max_output_tokens,
        }
        if self.settings.send_temperature:
            # OpenAI reasoning models accept only the default temperature, so
            # this is omitted rather than risking a 400 on every call.
            body["temperature"] = (
                self.settings.temperature if temperature is None else temperature
            )
        if self.settings.enable_thinking:
            # Vendor extension: the OpenAI SDK sends these via `extra_body`,
            # which merges them into the top level of the request exactly as
            # done here. Off by default - OpenAI rejects unknown parameters.
            body["chat_template_kwargs"] = {"enable_thinking": True}
            if self.settings.reasoning_budget:
                body["reasoning_budget"] = self.settings.reasoning_budget
        return body

    @staticmethod
    def _to_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or msg.name or "call",
                        "name": msg.name or "tool",
                        "content": json.dumps(msg.tool_result or {}, default=str),
                    }
                )
                continue

            if msg.role == "model":
                entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.call_id or f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, default=str),
                            },
                        }
                        for index, call in enumerate(msg.tool_calls)
                    ]
                wire.append(entry)
                continue

            wire.append({"role": "user", "content": msg.content})
        return wire

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _call(self, path: str, body: dict[str, Any], purpose: str) -> dict[str, Any]:
        key = canonical_key({"path": path, "body": body, "model": self.settings.model_name})

        if self.mode is TransportMode.REPLAY:
            cached = self._cassette.get(key, purpose=purpose)
            if cached is None:
                raise CassetteMissError(
                    f"No recorded LLM response for {purpose} (key {key}). "
                    "Re-record with LLM_MODE=record, or run in live mode."
                )
            self.tracer.event("llm.replay", purpose=purpose, key=key)
            return cached

        raw = self._request_with_retries(path, body, purpose)

        if self.mode is TransportMode.RECORD:
            self._cassette.put(key, {"path": path, "body": body}, raw, meta={"purpose": purpose})
        return raw

    def _request_with_retries(self, path: str, body: dict[str, Any], purpose: str) -> dict[str, Any]:
        url = f"{self.settings.base_url.rstrip('/')}{path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key.get_secret_value()}",  # type: ignore[union-attr]
        }
        last_error: Exception | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            started = time.perf_counter()
            try:
                resp = self._client.post(url, json=body, headers=headers)
                elapsed_ms = (time.perf_counter() - started) * 1000

                if resp.status_code == 200:
                    self.tracer.event(
                        "llm.request",
                        purpose=purpose,
                        attempt=attempt,
                        status=200,
                        latency_ms=round(elapsed_ms, 1),
                    )
                    return resp.json()

                detail = self._error_detail(resp)
                retryable = resp.status_code in RETRYABLE_STATUS
                retry_after = _retry_after(resp)
                self.tracer.event(
                    "llm.http_error",
                    purpose=purpose,
                    attempt=attempt,
                    status=resp.status_code,
                    retryable=retryable,
                    server_retry_delay=retry_after,
                    detail=detail[:300],
                )
                if not retryable:
                    raise LLMTransportError(
                        f"provider returned {resp.status_code}: {detail}",
                        status_code=resp.status_code,
                        retryable=False,
                    )
                last_error = LLMTransportError(
                    f"provider returned {resp.status_code}: {detail}",
                    status_code=resp.status_code,
                    retryable=True,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self.tracer.event(
                    "llm.transport_error", purpose=purpose, attempt=attempt, detail=str(exc)[:300]
                )
                last_error = LLMTransportError(f"transport failure: {exc}", retryable=True)
                retry_after = None

            if attempt < self.settings.max_retries:
                time.sleep(self._backoff_seconds(attempt, retry_after))

        raise last_error or LLMTransportError("request failed with no recorded cause")

    def _backoff_seconds(self, attempt: int, server_delay: float | None) -> float:
        if server_delay is not None:
            return min(server_delay + 1.0, self.settings.max_retry_delay)
        delay = self.settings.retry_base_delay * (2 ** (attempt - 1))
        return min(delay * (0.5 + random.random() * 0.5), self.settings.max_retry_delay)

    @staticmethod
    def _error_detail(resp: httpx.Response) -> str:
        try:
            payload = resp.json()
        except ValueError:
            return resp.text[:300]
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message", error))
        return str(error or payload.get("detail") or payload.get("title") or resp.text[:300])

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, raw: dict[str, Any], *, purpose: str) -> LLMResponse:
        choices = raw.get("choices") or []
        if not choices:
            raise LLMResponseError("provider returned no choices")

        choice = choices[0]
        finish = choice.get("finish_reason")
        message = choice.get("message", {}) or {}

        if finish == "content_filter":
            raise LLMRefusalError("generation blocked by content filter", finish_reason=finish)

        if reasoning := message.get("reasoning_content"):
            # Visible only in the trace: it is the model's scratchpad, not the answer.
            self.tracer.event("llm.reasoning", purpose=purpose, chars=len(str(reasoning)))

        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) or {}
            tool_calls.append(
                ToolCall(
                    name=function.get("name", ""),
                    arguments=_parse_arguments(function.get("arguments")),
                    call_id=call.get("id"),
                )
            )

        usage_meta = raw.get("usage", {}) or {}
        completion_details = usage_meta.get("completion_tokens_details") or {}
        usage = Usage(
            prompt_tokens=usage_meta.get("prompt_tokens", 0),
            output_tokens=usage_meta.get("completion_tokens", 0),
            thoughts_tokens=completion_details.get("reasoning_tokens", 0),
            total_tokens=usage_meta.get("total_tokens", 0),
        )
        self.total_usage = self.total_usage + usage

        if finish == "length" and not tool_calls:
            self.tracer.event("llm.truncated", purpose=purpose)

        return LLMResponse(
            text=(message.get("content") or "").strip(),
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish,
            model=raw.get("model", self.settings.model_name),
        )

    @staticmethod
    def _validate(text: str, response_model: type[ModelT]) -> ModelT:
        cleaned = _strip_code_fence(text)
        if not cleaned:
            raise SchemaValidationError("model returned empty output", raw_text=text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"output was not valid JSON: {exc}", raw_text=text) from exc
        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            raise SchemaValidationError(
                f"output did not match {response_model.__name__}: {exc}", raw_text=text
            ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Parse tool arguments, which arrive as a JSON string.

    A malformed string is returned as an error payload rather than raising:
    the registry will reject it and hand the model a validation message it can
    act on, which is more useful than aborting the run.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"__malformed_arguments": str(raw)[:500]}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _retry_after(resp: httpx.Response) -> float | None:
    """Read a ``Retry-After`` header, if the provider sent one."""
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _schema_name(model: type[BaseModel]) -> str:
    """Schema names must be identifier-safe for strict mode."""
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in model.__name__)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()
