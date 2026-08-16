"""Translate Pydantic JSON Schema into Gemini's ``responseSchema`` dialect.

Gemini accepts a restricted subset of OpenAPI 3.0 schema objects. Pydantic,
by contrast, emits full JSON Schema: ``$defs`` with ``$ref`` pointers for
nested models and enums, ``anyOf`` for ``Optional[...]``, plus annotations
(``title``, ``default``, ``additionalProperties``) that Gemini rejects.

Sending ``model_json_schema()`` straight to the API fails with a 400, so this
module does three jobs:

1. inline every ``$ref`` (recursion-guarded),
2. collapse ``Optional[T]`` unions into ``nullable: true``,
3. drop unsupported keywords and emit ``propertyOrdering``, which measurably
   stabilises field order in Gemini's output.
"""

from __future__ import annotations

from typing import Any

#: Keywords Gemini understands on a schema node.
_SUPPORTED_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
    "propertyOrdering",
    "minItems",
    "maxItems",
}

#: JSON Schema string formats Gemini supports; others are dropped rather than
#: risking a 400 on an exotic one (e.g. "uuid", "email").
_SUPPORTED_FORMATS = {"date-time", "enum", "int32", "int64", "float", "double"}

_MAX_DEPTH = 24


def to_gemini_schema(schema: dict[str, Any], defs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a Pydantic JSON schema into a Gemini-compatible schema.

    Args:
        schema: Output of ``Model.model_json_schema()`` (or a sub-schema).
        defs: The ``$defs`` table; taken from ``schema`` on the first call.

    Returns:
        A schema dict safe to pass as ``generationConfig.responseSchema``.
    """
    if defs is None:
        defs = schema.get("$defs", {})
    return _convert(schema, defs, depth=0)


def _convert(node: dict[str, Any], defs: dict[str, Any], depth: int) -> dict[str, Any]:
    if depth > _MAX_DEPTH:
        # Self-referential model; degrade to a permissive string rather than
        # recursing forever.
        return {"type": "string"}

    node = _resolve_ref(node, defs)
    node = _flatten_nullable_union(node, defs)

    out: dict[str, Any] = {}
    node_type = node.get("type")

    # An enum arrives as {"enum": [...], "type": "string"} once the $ref to the
    # enum definition has been inlined.
    if "enum" in node:
        out["type"] = node_type or "string"
        out["enum"] = list(node["enum"])
        _copy_scalars(node, out)
        return out

    if node_type == "object" or "properties" in node:
        out["type"] = "object"
        props = node.get("properties", {})
        converted = {name: _convert(sub, defs, depth + 1) for name, sub in props.items()}
        if converted:
            out["properties"] = converted
            # Field order is not guaranteed by the API unless stated.
            out["propertyOrdering"] = list(converted.keys())
        required = [r for r in node.get("required", []) if r in converted]
        if required:
            out["required"] = required
        _copy_scalars(node, out)
        return out

    if node_type == "array" or "items" in node:
        out["type"] = "array"
        items = node.get("items")
        out["items"] = _convert(items, defs, depth + 1) if isinstance(items, dict) else {"type": "string"}
        for key in ("minItems", "maxItems"):
            if key in node:
                out[key] = node[key]
        _copy_scalars(node, out)
        return out

    # Scalars. Pydantic may omit "type" for permissive fields; default to string
    # because Gemini requires the key to be present.
    out["type"] = node_type if isinstance(node_type, str) else "string"
    if node.get("format") in _SUPPORTED_FORMATS:
        out["format"] = node["format"]
    _copy_scalars(node, out)
    return out


def _copy_scalars(node: dict[str, Any], out: dict[str, Any]) -> None:
    """Carry across the annotations Gemini accepts.

    ``description`` is not decoration: it is the only in-band instruction the
    model gets about a field's semantics, so it is preserved deliberately.
    """
    if desc := node.get("description"):
        out["description"] = desc
    if node.get("nullable"):
        out["nullable"] = True


def _resolve_ref(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Inline a ``$ref`` pointer, merging any sibling keywords."""
    ref = node.get("$ref")
    if not ref:
        return node
    name = ref.rsplit("/", 1)[-1]
    target = defs.get(name)
    if target is None:
        return {k: v for k, v in node.items() if k != "$ref"}
    merged = dict(target)
    # Sibling keys (commonly "description") override the definition's own.
    for key, value in node.items():
        if key != "$ref":
            merged[key] = value
    return merged


def _flatten_nullable_union(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Turn ``Optional[T]`` (``anyOf: [T, null]``) into ``T`` + ``nullable``.

    Non-null unions are not expressible in Gemini's dialect; the first concrete
    branch is used, which is the pragmatic choice for our schemas where unions
    only ever arise from ``Optional``.
    """
    union = node.get("anyOf") or node.get("oneOf")
    if not union:
        return node

    non_null = [b for b in union if b.get("type") != "null"]
    nullable = len(non_null) < len(union)
    if not non_null:
        return {"type": "string", "nullable": True}

    chosen = _resolve_ref(non_null[0], defs)
    merged = dict(chosen)
    for key, value in node.items():
        if key not in ("anyOf", "oneOf"):
            merged.setdefault(key, value)
    if nullable:
        merged["nullable"] = True
    return merged


def strip_unsupported(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove keys outside Gemini's supported set.

    A defensive final pass, used by tests to assert the converter's output is
    clean regardless of future Pydantic changes.
    """
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SUPPORTED_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: strip_unsupported(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = strip_unsupported(value)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# OpenAI-compatible dialect (NVIDIA NIM, OpenAI, and friends)
# ---------------------------------------------------------------------------
#: Keys permitted on a node in OpenAI strict structured-output schemas.
_OPENAI_KEYS = {
    "type",
    "description",
    "enum",
    "items",
    "properties",
    "required",
    "additionalProperties",
}


def to_openai_schema(schema: dict[str, Any], defs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a Pydantic JSON schema for OpenAI-style ``json_schema`` output.

    Strict mode has its own rules, different from Gemini's:

    * every object must set ``additionalProperties: false``;
    * every property must appear in ``required`` - optional fields are expressed
      as a nullable *type union* (``["string", "null"]``) instead of being left
      out;
    * ``$ref``/``$defs`` are inlined here for the same reason as the Gemini
      converter - one schema shape to reason about across providers.
    """
    if defs is None:
        defs = schema.get("$defs", {})
    return _convert_openai(schema, defs, depth=0)


def _convert_openai(node: dict[str, Any], defs: dict[str, Any], depth: int) -> dict[str, Any]:
    if depth > _MAX_DEPTH:
        return {"type": "string"}

    node = _resolve_ref(node, defs)
    node, nullable = _split_nullable(node, defs)

    out: dict[str, Any] = {}
    node_type = node.get("type")

    if "enum" in node:
        out["type"] = _with_null(node_type or "string", nullable)
        out["enum"] = list(node["enum"])
        if desc := node.get("description"):
            out["description"] = desc
        return out

    if node_type == "object" or "properties" in node:
        props = node.get("properties", {})
        converted = {name: _convert_openai(sub, defs, depth + 1) for name, sub in props.items()}
        out["type"] = _with_null("object", nullable)
        out["properties"] = converted
        # Strict mode requires *every* property to be listed as required.
        out["required"] = list(converted.keys())
        out["additionalProperties"] = False
        if desc := node.get("description"):
            out["description"] = desc
        return out

    if node_type == "array" or "items" in node:
        items = node.get("items")
        out["type"] = _with_null("array", nullable)
        out["items"] = (
            _convert_openai(items, defs, depth + 1) if isinstance(items, dict) else {"type": "string"}
        )
        if desc := node.get("description"):
            out["description"] = desc
        return out

    out["type"] = _with_null(node_type if isinstance(node_type, str) else "string", nullable)
    if desc := node.get("description"):
        out["description"] = desc
    return out


def _with_null(base: str, nullable: bool) -> Any:
    """Express nullability as a type union, which is what strict mode wants."""
    return [base, "null"] if nullable else base


def _split_nullable(node: dict[str, Any], defs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Separate ``Optional[T]`` into its concrete branch plus a nullable flag."""
    union = node.get("anyOf") or node.get("oneOf")
    if not union:
        return node, False

    non_null = [b for b in union if b.get("type") != "null"]
    nullable = len(non_null) < len(union)
    if not non_null:
        return {"type": "string"}, True

    chosen = _resolve_ref(non_null[0], defs)
    merged = dict(chosen)
    for key, value in node.items():
        if key not in ("anyOf", "oneOf"):
            merged.setdefault(key, value)
    return merged, nullable
