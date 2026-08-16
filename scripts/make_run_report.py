"""Render a recorded trace as a readable agent-run report.

    python scripts/make_run_report.py traces/run-<id>.jsonl -o docs/agent-run-report.md

Turns the raw JSONL trace into the reasoning chain / tool calls / final output
narrative that the brief asks for, so the report is generated from what
actually happened rather than transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability.trace import load_trace  # noqa: E402

#: Events that carry the narrative; everything else is supporting detail.
HEADLINE = {
    "run.start": "Run started",
    "guard.input.end": "Input guardrail",
    "node.planner.end": "Planner",
    "node.researcher.end": "Researcher",
    "tool.call.end": "Tool call",
    "node.rank.end": "Ranking (RAG)",
    "node.synthesizer.end": "Synthesizer",
    "node.critic.end": "Critic",
    "guard.output.end": "Output guardrail",
    "answer.final": "Final answer",
    "run.complete": "Run complete",
}


def fmt(value: Any, limit: int = 700) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " …"


def render(path: Path) -> str:
    events = load_trace(path)
    lines = [f"# Agent run trace — `{path.stem}`", ""]

    start = next((e for e in events if e.name == "run.start"), None)
    if start:
        lines += [
            f"**Question:** {start.data.get('question', '')}",
            "",
            f"- Model: `{start.data.get('model')}`",
            f"- LLM mode: `{start.data.get('llm_mode')}` · PubMed mode: `{start.data.get('pubmed_mode')}`",
            "",
            "---",
            "",
        ]

    step = 0
    for event in events:
        if event.name not in HEADLINE or event.name == "run.start":
            continue
        step += 1
        timing = f" _({event.duration_ms:.0f} ms)_" if event.duration_ms else ""
        lines.append(f"## {step}. {HEADLINE[event.name]}{timing}")
        lines.append("")
        for key, value in event.data.items():
            if key == "ok" or value in (None, "", [], {}):
                continue
            lines.append(f"- **{key}**: {fmt(value)}")
        lines.append("")

    retries = [e for e in events if e.name in ("llm.http_error", "llm.transport_error")]
    if retries:
        lines += ["---", "", "## Retries and recoveries", ""]
        for event in retries:
            data = event.data
            lines.append(
                f"- `{data.get('purpose')}` attempt {data.get('attempt')} → "
                f"HTTP {data.get('status', 'transport error')}"
                + (
                    f", server asked for {data['server_retry_delay']}s"
                    if data.get("server_retry_delay")
                    else ""
                )
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to a run-<id>.jsonl trace.")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.trace.exists():
        print(f"trace not found: {args.trace}", file=sys.stderr)
        return 2

    report = render(args.trace)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
