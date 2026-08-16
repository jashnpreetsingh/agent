"""Wire-format tests for the OpenAI-compatible adapter.

"OpenAI-compatible" is a family resemblance, not a contract. These endpoints
agree on the message and tool-call shape but disagree on parameter names and on
which vendor extensions they tolerate — and every one of those disagreements
surfaces as a 400 at runtime, against a real key, usually not the one you
developed on.

So the request body is asserted directly here, with no network involved.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.config import Provider, Settings, TransportMode
from src.llm.base import ChatMessage, ToolCall
from src.llm.openai_compat import OpenAICompatProvider
from src.schemas import ResearchPlan


def make_provider(tmp_path, provider: Provider, **overrides):
    """Build a provider whose transport captures the outgoing request."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        if request.url.path.endswith("/embeddings"):
            payload = {
                "data": [
                    {"index": i, "embedding": [0.1, 0.2, 0.3]}
                    for i in range(len(captured["body"]["input"]))
                ]
            }
        else:
            payload = {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"ok": true}'},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            }
        return httpx.Response(200, json=payload)

    settings = Settings(
        provider=provider,
        nvidia_api_key="nv-key",
        openai_api_key="oa-key",
        gemini_api_key="gm-key",
        llm_mode=TransportMode.LIVE,
        fixtures_dir=tmp_path / "fixtures",
        traces_dir=tmp_path / "traces",
        **overrides,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider(settings, client=client), captured


# ---------------------------------------------------------------------------
# Token-limit parameter
# ---------------------------------------------------------------------------
def test_nvidia_uses_max_tokens(tmp_path):
    provider, captured = make_provider(tmp_path, Provider.NVIDIA)
    provider.generate([ChatMessage.user("hi")])
    assert "max_tokens" in captured["body"]
    assert "max_completion_tokens" not in captured["body"]


def test_openai_uses_max_completion_tokens(tmp_path):
    """Newer OpenAI models reject `max_tokens` with a 400."""
    provider, captured = make_provider(tmp_path, Provider.OPENAI)
    provider.generate([ChatMessage.user("hi")])
    assert "max_completion_tokens" in captured["body"]
    assert "max_tokens" not in captured["body"]


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------
def test_openai_omits_temperature_by_default(tmp_path):
    """OpenAI reasoning models accept only the default temperature."""
    provider, captured = make_provider(tmp_path, Provider.OPENAI)
    provider.generate([ChatMessage.user("hi")])
    assert "temperature" not in captured["body"]


def test_temperature_can_be_re_enabled(tmp_path):
    provider, captured = make_provider(tmp_path, Provider.OPENAI, send_temperature=True)
    provider.generate([ChatMessage.user("hi")])
    assert captured["body"]["temperature"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Vendor extensions
# ---------------------------------------------------------------------------
def test_thinking_params_absent_by_default(tmp_path):
    """OpenAI 400s on unknown parameters, so these stay off unless asked for."""
    provider, captured = make_provider(tmp_path, Provider.OPENAI)
    provider.generate([ChatMessage.user("hi")])
    assert "chat_template_kwargs" not in captured["body"]
    assert "reasoning_budget" not in captured["body"]


def test_thinking_params_sent_when_enabled(tmp_path):
    provider, captured = make_provider(
        tmp_path, Provider.NVIDIA, enable_thinking=True, reasoning_budget=4096
    )
    provider.generate([ChatMessage.user("hi")])
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["body"]["reasoning_budget"] == 4096


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def test_nvidia_embeddings_send_input_type(tmp_path):
    """NVIDIA retriever models 400 without it: 'required for asymmetric models'."""
    provider, captured = make_provider(tmp_path, Provider.NVIDIA)
    provider.embed(["abstract text"], task_type="document")
    assert captured["body"]["input_type"] == "passage"
    assert captured["body"]["truncate"] == "END"


def test_openai_embeddings_omit_input_type(tmp_path):
    """`input_type` is not an OpenAI parameter and would be rejected."""
    provider, captured = make_provider(tmp_path, Provider.OPENAI)
    provider.embed(["abstract text"], task_type="document")
    assert "input_type" not in captured["body"]
    assert "truncate" not in captured["body"]


def test_query_and_passage_map_to_distinct_input_types(tmp_path):
    provider, captured = make_provider(tmp_path, Provider.NVIDIA)
    provider.embed(["q"], task_type="query")
    assert captured["body"]["input_type"] == "query"
    provider.embed(["d"], task_type="document")
    assert captured["body"]["input_type"] == "passage"


def test_embeddings_are_ordered_by_index(tmp_path):
    """The API does not promise ordering; results must be re-sorted."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [9.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            },
        )

    settings = Settings(
        provider=Provider.NVIDIA,
        nvidia_api_key="k",
        llm_mode=TransportMode.LIVE,
        fixtures_dir=tmp_path / "f",
        traces_dir=tmp_path / "t",
    )
    provider = OpenAICompatProvider(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert provider.embed(["a", "b"]) == [[1.0], [9.0]]


# ---------------------------------------------------------------------------
# Routing and auth
# ---------------------------------------------------------------------------
def test_each_provider_uses_its_own_key_and_host(tmp_path):
    nvidia, nv_captured = make_provider(tmp_path, Provider.NVIDIA)
    nvidia.generate([ChatMessage.user("hi")])
    assert nv_captured["auth"] == "Bearer nv-key"
    assert "integrate.api.nvidia.com" in nv_captured["url"]

    openai, oa_captured = make_provider(tmp_path, Provider.OPENAI)
    openai.generate([ChatMessage.user("hi")])
    assert oa_captured["auth"] == "Bearer oa-key"
    assert "api.openai.com" in oa_captured["url"]


# ---------------------------------------------------------------------------
# Message and tool shapes (identical across OpenAI-compatible providers)
# ---------------------------------------------------------------------------
def test_tool_results_are_correlated_by_call_id(tmp_path):
    provider, captured = make_provider(tmp_path, Provider.OPENAI)
    provider.generate(
        [
            ChatMessage.user("find trials"),
            ChatMessage.model("", [ToolCall("pubmed_search", {"query": "x"}, call_id="call_7")]),
            ChatMessage.tool("pubmed_search", {"results": []}, call_id="call_7"),
        ]
    )
    messages = captured["body"]["messages"]
    assistant = next(m for m in messages if m["role"] == "assistant")
    tool = next(m for m in messages if m["role"] == "tool")

    assert assistant["tool_calls"][0]["id"] == "call_7"
    # Arguments must be a JSON *string*, not an object.
    assert isinstance(assistant["tool_calls"][0]["function"]["arguments"], str)
    assert tool["tool_call_id"] == "call_7"


def test_system_prompt_is_a_message(tmp_path):
    provider, captured = make_provider(tmp_path, Provider.OPENAI)
    provider.generate([ChatMessage.user("hi")], system="You are a researcher.")
    assert captured["body"]["messages"][0] == {
        "role": "system",
        "content": "You are a researcher.",
    }


def test_structured_output_uses_strict_json_schema(tmp_path):
    provider, captured = make_provider(tmp_path, Provider.OPENAI)
    try:
        provider.generate_structured([ChatMessage.user("plan")], ResearchPlan)
    except Exception:
        # The mock returns `{"ok": true}`, which fails validation; the request
        # shape is what this test is about.
        pass

    schema_block = captured["body"]["response_format"]["json_schema"]
    assert schema_block["strict"] is True
    assert schema_block["name"] == "ResearchPlan"

    schema = schema_block["schema"]
    assert schema["additionalProperties"] is False
    # Strict mode requires every property to be listed as required.
    assert set(schema["required"]) == set(schema["properties"])
    assert "$ref" not in json.dumps(schema)


def test_malformed_tool_arguments_do_not_raise(tmp_path):
    """A truncated arguments string must reach the registry as data, not a crash."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "pubmed_search",
                                        "arguments": '{"query": "unterminated',
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        )

    settings = Settings(
        provider=Provider.NVIDIA,
        nvidia_api_key="k",
        llm_mode=TransportMode.LIVE,
        fixtures_dir=tmp_path / "f",
        traces_dir=tmp_path / "t",
    )
    provider = OpenAICompatProvider(
        settings, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    response = provider.generate([ChatMessage.user("hi")])

    assert response.tool_calls[0].name == "pubmed_search"
    assert "__malformed_arguments" in response.tool_calls[0].arguments
