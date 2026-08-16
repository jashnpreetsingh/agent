"""Record/replay store for external calls.

Both the LLM provider and the PubMed client route their traffic through a
cassette. Recorded once, an entire agent run replays offline and
deterministically, which is what lets the evaluation suite run in CI with no
API key and no dependence on NCBI being up.

Cassettes are plain JSON keyed by a hash of the canonicalised request, so a
reviewer can open one and read exactly what was sent and returned.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

#: Request fields that must never be persisted to a committed fixture.
_SECRET_KEYS = {"key", "api_key", "apikey", "x-goog-api-key", "authorization", "token"}

_API_KEY_PATTERN = re.compile(r"(?i)(api[_-]?key|key)=([^&\s\"']+)")


def canonical_key(payload: Any) -> str:
    """Stable hash of a request payload, independent of dict ordering."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def scrub(payload: Any) -> Any:
    """Strip credentials from anything about to be written to disk."""
    if isinstance(payload, dict):
        return {
            k: ("***REDACTED***" if k.lower() in _SECRET_KEYS else scrub(v))
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return [scrub(item) for item in payload]
    if isinstance(payload, str):
        return _API_KEY_PATTERN.sub(r"\1=***REDACTED***", payload)
    return payload


#: Name of the ordered manifest written alongside the cassettes.
MANIFEST = "_manifest.json"


class CassetteStore:
    """A namespaced directory of recorded request/response pairs.

    Lookup is two-stage, because exact-match replay alone is too brittle for
    an LLM agent. The request body for turn N contains the model's own output
    from turn N-1, so a single token of drift changes the hash of every
    subsequent request and the whole replay collapses.

    So: try the exact request hash first (faithful), and on a miss fall back to
    the next unconsumed recording with the same *purpose* in the order it was
    recorded (robust). The fallback is what lets a recorded run replay
    end-to-end on a machine with no API key.
    """

    def __init__(self, root: Path, namespace: str) -> None:
        self.dir = Path(root) / namespace
        self.namespace = namespace
        self._memo: dict[str, dict[str, Any]] = {}
        self._manifest: list[dict[str, str]] | None = None
        self._consumed: set[int] = set()
        self.exact_hits = 0
        self.sequence_hits = 0

    # -- lookup ------------------------------------------------------------
    def path_for(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str, purpose: str | None = None) -> dict[str, Any] | None:
        """Return a recorded response, or ``None`` on a miss.

        Args:
            key: Hash of the canonicalised request.
            purpose: Logical step name (e.g. ``"planner"``). When supplied,
                enables the ordered fallback described in the class docstring.
        """
        exact = self._load(key)
        if exact is not None:
            self.exact_hits += 1
            return exact
        if purpose is None:
            return None
        return self._next_by_purpose(purpose)

    def _load(self, key: str) -> dict[str, Any] | None:
        if key in self._memo:
            return self._memo[key]
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt fixture is a miss, not a crash: live/record modes can
            # simply overwrite it.
            return None
        response = record.get("response")
        if response is not None:
            self._memo[key] = response
        return response

    def _next_by_purpose(self, purpose: str) -> dict[str, Any] | None:
        """Consume the next recording made for the same purpose."""
        for index, entry in enumerate(self._load_manifest()):
            if index in self._consumed or entry.get("purpose") != purpose:
                continue
            response = self._load(entry["key"])
            if response is None:
                continue
            self._consumed.add(index)
            self.sequence_hits += 1
            return response
        return None

    def _load_manifest(self) -> list[dict[str, str]]:
        if self._manifest is not None:
            return self._manifest
        path = self.dir / MANIFEST
        if not path.exists():
            self._manifest = []
            return self._manifest
        try:
            self._manifest = json.loads(path.read_text(encoding="utf-8")).get("entries", [])
        except (OSError, json.JSONDecodeError):
            self._manifest = []
        return self._manifest

    # -- write -------------------------------------------------------------
    def put(self, key: str, request: Any, response: Any, meta: dict[str, Any] | None = None) -> None:
        """Persist a request/response pair, credentials removed."""
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = meta or {}
        record = {
            "namespace": self.namespace,
            "key": key,
            "meta": meta,
            "request": scrub(request),
            "response": response,
        }
        self.path_for(key).write_text(
            json.dumps(record, indent=2, sort_keys=False, default=str),
            encoding="utf-8",
        )
        self._memo[key] = response
        self._append_manifest(key, str(meta.get("purpose") or meta.get("endpoint") or "unknown"))

    def _append_manifest(self, key: str, purpose: str) -> None:
        """Record call order so replay can fall back to it."""
        entries = list(self._load_manifest())
        entries.append({"key": key, "purpose": purpose})
        self._manifest = entries
        (self.dir / MANIFEST).write_text(
            json.dumps({"entries": entries}, indent=2), encoding="utf-8"
        )

    # -- introspection -----------------------------------------------------
    def count(self) -> int:
        return len(list(self.dir.glob("*.json"))) if self.dir.exists() else 0

    def keys(self) -> list[str]:
        if not self.dir.exists():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json"))
