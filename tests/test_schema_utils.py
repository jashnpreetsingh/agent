"""Tests for the Pydantic -> Gemini schema conversion.

Gemini rejects `$ref`, `$defs`, `anyOf`, and stray annotations with a 400, and
that failure only shows up at runtime against the real API - so it is worth
pinning here.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from src.llm.schema_utils import strip_unsupported, to_gemini_schema
from src.schemas import Critique, DraftAnswer, ResearchPlan


class Inner(BaseModel):
    label: str
    weight: float | None = None


class Outer(BaseModel):
    name: str = Field(description="the name")
    inner: Inner
    items: list[Inner] = Field(default_factory=list)
    optional_note: str | None = None


def test_refs_are_inlined():
    schema = to_gemini_schema(Outer.model_json_schema())
    blob = json.dumps(schema)
    assert "$ref" not in blob
    assert "$defs" not in blob
    assert schema["properties"]["inner"]["type"] == "object"
    assert schema["properties"]["inner"]["properties"]["label"]["type"] == "string"


def test_optional_becomes_nullable():
    schema = to_gemini_schema(Outer.model_json_schema())
    note = schema["properties"]["optional_note"]
    assert note["nullable"] is True
    assert note["type"] == "string"
    assert "anyOf" not in note


def test_nested_list_items_are_converted():
    schema = to_gemini_schema(Outer.model_json_schema())
    items = schema["properties"]["items"]["items"]
    assert items["type"] == "object"
    assert "label" in items["properties"]


def test_descriptions_survive():
    """Field descriptions are the model's only instruction about semantics."""
    schema = to_gemini_schema(Outer.model_json_schema())
    assert schema["properties"]["name"]["description"] == "the name"


def test_property_ordering_present():
    schema = to_gemini_schema(Outer.model_json_schema())
    assert schema["propertyOrdering"] == list(schema["properties"].keys())


def test_enum_is_preserved():
    schema = to_gemini_schema(ResearchPlan.model_json_schema())
    pub_types = schema["properties"]["searches"]["items"]["properties"]["pub_types"]
    assert "Meta-Analysis" in pub_types["items"]["enum"]


def test_production_models_emit_only_supported_keys():
    for model in (ResearchPlan, DraftAnswer, Critique):
        schema = to_gemini_schema(model.model_json_schema())
        assert schema == strip_unsupported(schema), f"{model.__name__} has unsupported keys"


def test_self_referential_model_terminates():
    class Node(BaseModel):
        value: str
        child: "Node | None" = None

    Node.model_rebuild()
    schema = to_gemini_schema(Node.model_json_schema())
    assert "$ref" not in json.dumps(schema)
