"""Gemini implementation of :class:`~src.llm.base.LLMProvider`.

Talks to the Generative Language REST API directly (no vendor SDK), which
keeps the dependency surface small and makes every request/response shape
visible in the trace - useful when the point of the exercise is to show how
the agent works.

Reliability behaviour:
  * exponential backoff with jitter on 429/500/503/504 and transport errors;
  * terminal failure on 400/401/403/404 - retrying a bad request wastes quota;
  * ``LLMRefusalError`` for safety/recitation finishes, which callers treat as
    a content outcome rather than an outage;
  * one automatic re-ask when structured output fails schema validation.
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
from src.llm.schema_utils import to_gemini_schema
from src.observability.trace import Tracer

ModelT = TypeVar("ModelT", bound=BaseModel)

#: HTTP statuses worth another attempt.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
#: Finish reasons that mean "the model declined", not "the call failed".
REFUSAL_FINISH_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}


class GeminiProvider(LLMProvider):
    """Structured generation, tool calling, and embeddings via Gemini."""

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

        if self.mode in (TransportMode.LIVE, TransportMode.RECORD) and not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for llm_mode=live/record. "
                "Set it in .env, or run with LLM_MODE=replay to use recorded fixtures."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self.settings.model_name

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
            body["tools"] = [{"functionDeclarations": tools}]
            body["toolConfig"] = {
                "functionCallingConfig": {"mode": "ANY" if force_tool_use else "AUTO"}
            }
        raw = self._call(f"models/{self.settings.model_name}:generateContent", body, purpose)
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
        """Generate and validate against ``response_model``.

        On a validation failure the model is re-asked once with the validation
        error appended - in practice this recovers the majority of the rare
        schema misses without a full node retry.
        """
        schema = to_gemini_schema(response_model.model_json_schema())
        body = self._build_body(messages, system=system, temperature=temperature)
        body["generationConfig"].update(
            {"responseMimeType": "application/json", "responseSchema": schema}
        )

        raw = self._call(f"models/{self.settings.model_name}:generateContent", body, purpose)
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
                    "That response failed schema validation with: "
                    f"{str(first_error)[:600]}\n"
                    "Return corrected JSON that satisfies the schema exactly. "
                    "Output JSON only."
                ),
            ]
            retry_body = self._build_body(repair, system=system, temperature=0.0)
            retry_body["generationConfig"].update(
                {"responseMimeType": "application/json", "responseSchema": schema}
            )
            raw2 = self._call(
                f"models/{self.settings.model_name}:generateContent", retry_body, f"{purpose}.repair"
            )
            response2 = self._parse_response(raw2, purpose=f"{purpose}.repair")
            return self._validate(response2.text, response_model)

    def embed(self, texts: list[str], *, task_type: str = "query") -> list[list[float]]:
        """Embed a batch of texts.

        Sent as individual requests inside one batch envelope so a single
        oversized document cannot fail the whole ranking step. The caller's
        task type is mapped onto Gemini's own retrieval vocabulary so that
        questions and passages are encoded for asymmetric retrieval.
        """
        if not texts:
            return []
        gemini_task = (
            "RETRIEVAL_DOCUMENT"
            if task_type.lower().startswith(("passage", "document"))
            else "RETRIEVAL_QUERY"
        )
        model = f"models/{self.settings.embedding_model}"
        body = {
            "requests": [
                {
                    "model": model,
                    "content": {"parts": [{"text": text[:8000]}]},
                    "taskType": gemini_task,
                }
                for text in texts
            ]
        }
        raw = self._call(
            f"models/{self.settings.embedding_model}:batchEmbedContents", body, "embed"
        )
        embeddings = raw.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise LLMResponseError(
                f"embedding count mismatch: asked {len(texts)}, got {len(embeddings)}"
            )
        return [item.get("values", []) for item in embeddings]

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
        body: dict[str, Any] = {
            "contents": self._to_contents(messages),
            "generationConfig": {
                "temperature": self.settings.temperature if temperature is None else temperature,
                "maxOutputTokens": self.settings.max_output_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    @staticmethod
    def _to_contents(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Map internal messages onto Gemini's ``contents`` array."""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                # Function results are delivered on a user turn in this API.
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.name or "tool",
                                    "response": msg.tool_result or {},
                                }
                            }
                        ],
                    }
                )
                continue

            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"text": msg.content})
            for call in msg.tool_calls:
                part: dict[str, Any] = {
                    "functionCall": {"name": call.name, "args": call.arguments}
                }
                # Gemini 3 rejects replayed function calls that arrive without
                # the signature it issued with them.
                if call.thought_signature:
                    part["thoughtSignature"] = call.thought_signature
                parts.append(part)
            if not parts:
                parts.append({"text": ""})
            contents.append({"role": "model" if msg.role == "model" else "user", "parts": parts})
        return contents

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _call(self, path: str, body: dict[str, Any], purpose: str) -> dict[str, Any]:
        """Execute a request honouring the configured transport mode."""
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
        url = f"{self.settings.gemini_base_url}/{path}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.settings.gemini_api_key.get_secret_value(),  # type: ignore[union-attr]
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
                server_delay = self._retry_delay(resp)
                self.tracer.event(
                    "llm.http_error",
                    purpose=purpose,
                    attempt=attempt,
                    status=resp.status_code,
                    retryable=retryable,
                    server_retry_delay=server_delay,
                    detail=detail[:300],
                )
                if not retryable:
                    raise LLMTransportError(
                        f"Gemini returned {resp.status_code}: {detail}",
                        status_code=resp.status_code,
                        retryable=False,
                    )
                last_error = LLMTransportError(
                    f"Gemini returned {resp.status_code}: {detail}",
                    status_code=resp.status_code,
                    retryable=True,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self.tracer.event(
                    "llm.transport_error", purpose=purpose, attempt=attempt, detail=str(exc)[:300]
                )
                last_error = LLMTransportError(f"transport failure: {exc}", retryable=True)
                server_delay = None

            if attempt < self.settings.max_retries:
                time.sleep(self._backoff_seconds(attempt, server_delay))

        raise last_error or LLMTransportError("request failed with no recorded cause")

    @staticmethod
    def _error_detail(resp: httpx.Response) -> str:
        try:
            return str(resp.json().get("error", {}).get("message", resp.text[:300]))
        except (ValueError, AttributeError):
            return resp.text[:300]

    @staticmethod
    def _retry_delay(resp: httpx.Response) -> float | None:
        """Read the server's own ``RetryInfo`` hint from a 429 body.

        Gemini states exactly how long to wait ("retryDelay": "7s"). Honouring
        it beats guessing: on the free tier's 5 requests/minute limit, blind
        exponential backoff burns all attempts inside the window and fails a
        request that would have succeeded after one accurate wait.
        """
        try:
            details = resp.json().get("error", {}).get("details", [])
        except (ValueError, AttributeError):
            return None
        for detail in details:
            if not str(detail.get("@type", "")).endswith("RetryInfo"):
                continue
            raw = str(detail.get("retryDelay", "")).strip()
            if raw.endswith("s"):
                try:
                    return float(raw[:-1])
                except ValueError:
                    return None
        return None

    def _backoff_seconds(self, attempt: int, server_delay: float | None) -> float:
        """Wait time before the next attempt.

        Uses the server's hint when present (plus a small buffer, since the
        quota window is measured server-side), otherwise exponential backoff
        with full jitter to avoid synchronised retries.
        """
        if server_delay is not None:
            return min(server_delay + 1.0, self.settings.max_retry_delay)
        delay = self.settings.retry_base_delay * (2 ** (attempt - 1))
        return min(delay * (0.5 + random.random() * 0.5), self.settings.max_retry_delay)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, raw: dict[str, Any], *, purpose: str) -> LLMResponse:
        if prompt_feedback := raw.get("promptFeedback", {}):
            if reason := prompt_feedback.get("blockReason"):
                raise LLMRefusalError(f"prompt blocked by provider: {reason}", finish_reason=reason)

        candidates = raw.get("candidates") or []
        if not candidates:
            raise LLMResponseError("provider returned no candidates")

        candidate = candidates[0]
        finish = candidate.get("finishReason")
        if finish in REFUSAL_FINISH_REASONS:
            raise LLMRefusalError(f"generation stopped: {finish}", finish_reason=finish)

        parts = candidate.get("content", {}).get("parts", []) or []
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        for part in parts:
            # Thought summaries are metadata, not answer text.
            if part.get("thought"):
                continue
            if "text" in part:
                text_chunks.append(part["text"])
            elif "functionCall" in part:
                call = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        name=call.get("name", ""),
                        arguments=call.get("args", {}) or {},
                        call_id=call.get("id"),
                        # Captured here so the next turn can replay it.
                        thought_signature=part.get("thoughtSignature"),
                    )
                )

        usage_meta = raw.get("usageMetadata", {}) or {}
        usage = Usage(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            output_tokens=usage_meta.get("candidatesTokenCount", 0),
            thoughts_tokens=usage_meta.get("thoughtsTokenCount", 0),
            total_tokens=usage_meta.get("totalTokenCount", 0),
        )
        self.total_usage = self.total_usage + usage

        text = "".join(text_chunks).strip()
        if finish == "MAX_TOKENS" and not tool_calls:
            # Surfaced rather than silently returning a truncated answer.
            self.tracer.event("llm.truncated", purpose=purpose, chars=len(text))

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish,
            model=self.settings.model_name,
            from_cassette=self.mode is TransportMode.REPLAY,
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


def _strip_code_fence(text: str) -> str:
    """Remove markdown fences some models add even in JSON mode."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()
