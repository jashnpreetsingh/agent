"""Structured tracing for the agent loop.

Every node, tool call, retry, and guardrail decision emits an event to a
per-run JSONL file. The requirement is that a reader can reconstruct *why*
the agent answered as it did without re-running it, so events carry inputs
and outputs, not just labels.

The tracer is deliberately dependency-free and synchronous: it is a debugging
instrument, and one that can itself fail or block would be worse than useless.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("pubmed_agent.trace")

#: Values longer than this are truncated in the trace to keep files readable.
_MAX_VALUE_CHARS = 4000


@dataclass(slots=True)
class TraceEvent:
    """A single point-in-time record."""

    seq: int
    ts: str
    run_id: str
    name: str
    span: str | None = None
    duration_ms: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, ensure_ascii=False)


@dataclass(slots=True)
class Span:
    """An open timing scope; closed by the :meth:`Tracer.span` context manager."""

    name: str
    started: float
    data: dict[str, Any] = field(default_factory=dict)

    def annotate(self, **fields: Any) -> None:
        """Attach results to the span, recorded when it closes."""
        self.data.update(fields)


class Tracer:
    """Writes trace events to JSONL and keeps them in memory for the UI."""

    def __init__(
        self,
        run_id: str | None = None,
        traces_dir: Path | None = None,
        *,
        enabled: bool = True,
        echo: bool = False,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.enabled = enabled
        self.echo = echo
        self.events: list[TraceEvent] = []
        self._seq = 0
        self._stack: list[str] = []
        self._started = time.perf_counter()

        self.path: Path | None = None
        if enabled and traces_dir is not None:
            traces_dir.mkdir(parents=True, exist_ok=True)
            self.path = traces_dir / f"run-{self.run_id}.jsonl"

    @classmethod
    def null(cls) -> "Tracer":
        """A tracer that records nothing - used in unit tests and defaults."""
        return cls(enabled=False)

    # ------------------------------------------------------------------
    def event(self, name: str, **data: Any) -> TraceEvent:
        """Record a point event."""
        self._seq += 1
        evt = TraceEvent(
            seq=self._seq,
            ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            run_id=self.run_id,
            name=name,
            span=self._stack[-1] if self._stack else None,
            data=_truncate(data),
        )
        self._emit(evt)
        return evt

    @contextmanager
    def span(self, name: str, **data: Any) -> Iterator[Span]:
        """Time a scope, emitting ``<name>.start`` and ``<name>.end``.

        The end event is emitted even when the body raises, with the error
        attached - a failed node still has to leave a trail.
        """
        scope = Span(name=name, started=time.perf_counter(), data=dict(data))
        self._stack.append(name)
        self.event(f"{name}.start", **data)
        try:
            yield scope
        except Exception as exc:
            duration = (time.perf_counter() - scope.started) * 1000
            self._stack.pop()
            self._finish(name, duration, {**scope.data, "error": f"{type(exc).__name__}: {exc}"}, ok=False)
            raise
        else:
            duration = (time.perf_counter() - scope.started) * 1000
            self._stack.pop()
            self._finish(name, duration, scope.data, ok=True)

    def _finish(self, name: str, duration_ms: float, data: dict[str, Any], *, ok: bool) -> None:
        self._seq += 1
        evt = TraceEvent(
            seq=self._seq,
            ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            run_id=self.run_id,
            name=f"{name}.end",
            span=self._stack[-1] if self._stack else None,
            duration_ms=round(duration_ms, 1),
            data=_truncate({**data, "ok": ok}),
        )
        self._emit(evt)

    def _emit(self, evt: TraceEvent) -> None:
        self.events.append(evt)
        if not self.enabled:
            return
        line = evt.to_json()
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:  # pragma: no cover - disk issues
                logger.warning("could not write trace: %s", exc)
        if self.echo:
            logger.debug(line)

    # ------------------------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._started

    def summary(self) -> dict[str, Any]:
        """Aggregate counts used by the CLI footer and evaluation report."""
        tool_calls = [e for e in self.events if e.name == "tool.call.end"]
        llm_calls = [e for e in self.events if e.name == "llm.request"]
        return {
            "run_id": self.run_id,
            "events": len(self.events),
            "llm_requests": len(llm_calls),
            "tool_calls": len(tool_calls),
            "retries": len([e for e in self.events if e.name in ("llm.http_error", "llm.transport_error")]),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "trace_path": str(self.path) if self.path else None,
        }

    def to_markdown(self) -> str:
        """Render the run as a readable reasoning trail for the report."""
        lines = [f"# Agent trace `{self.run_id}`", ""]
        for evt in self.events:
            if evt.name.endswith(".start"):
                continue
            indent = "  " if evt.span else ""
            label = evt.name.removesuffix(".end")
            timing = f" _({evt.duration_ms:.0f} ms)_" if evt.duration_ms else ""
            lines.append(f"{indent}- **{label}**{timing}")
            for key, value in evt.data.items():
                if key == "ok" or value in (None, "", [], {}):
                    continue
                rendered = json.dumps(value, default=str, ensure_ascii=False)
                if len(rendered) > 400:
                    rendered = rendered[:400] + "…"
                lines.append(f"{indent}  - `{key}`: {rendered}")
        return "\n".join(lines)


def _truncate(data: dict[str, Any]) -> dict[str, Any]:
    """Bound the size of any single traced value."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > _MAX_VALUE_CHARS:
            out[key] = value[:_MAX_VALUE_CHARS] + f"…[+{len(value) - _MAX_VALUE_CHARS} chars]"
        else:
            out[key] = value
    return out


def load_trace(path: Path) -> list[TraceEvent]:
    """Read a JSONL trace back into events (used by the report tooling)."""
    events: list[TraceEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        events.append(TraceEvent(**raw))
    return events
