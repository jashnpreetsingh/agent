"""Command-line interface.

    python -m src.cli --query "What are the latest treatments for Type 2 diabetes?"
    python src/agent.py --domain healthcare --query "..."   # equivalent
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.agent import PubMedAgent, RunResult
from src.config import Settings, TransportMode, get_settings
from src.llm.errors import CassetteMissError, LLMError
from src.tools.pubmed import PubMedError

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pubmed-agent",
        description="Answer biomedical questions from PubMed literature, with citations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  python -m src.cli --query "Do SGLT2 inhibitors reduce heart failure hospitalisation?"\n'
            '  python -m src.cli --query "What about kidney outcomes?" --session-id demo\n'
            "  python -m src.cli --query \"...\" --llm-mode replay --pubmed-mode replay\n"
        ),
    )
    parser.add_argument("--query", "-q", required=True, help="The question to research.")
    parser.add_argument(
        "--domain",
        default="healthcare",
        choices=["healthcare"],
        help="Agent domain (only 'healthcare' is implemented).",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Persist this turn under a session id so follow-up questions have context.",
    )
    parser.add_argument("--model", default=None, help="Override the configured model.")
    parser.add_argument(
        "--llm-mode",
        choices=[m.value for m in TransportMode],
        default=None,
        help="live | record | replay (replay needs no API key).",
    )
    parser.add_argument(
        "--pubmed-mode",
        choices=[m.value for m in TransportMode],
        default=None,
        help="live | record | replay.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the answer as JSON.")
    parser.add_argument("--show-trace", action="store_true", help="Print the reasoning trace.")
    parser.add_argument(
        "--save-trace",
        type=Path,
        default=None,
        help="Write the trace to a markdown file.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def apply_overrides(args: argparse.Namespace) -> Settings:
    """Build settings, applying CLI overrides on top of the environment."""
    base = get_settings()
    overrides: dict[str, object] = {}
    if args.model:
        overrides["model_name"] = args.model
    if args.llm_mode:
        overrides["llm_mode"] = TransportMode(args.llm_mode)
    if args.pubmed_mode:
        overrides["pubmed_mode"] = TransportMode(args.pubmed_mode)
    if not overrides:
        return base
    return base.model_copy(update=overrides)


def render(result: RunResult, *, show_trace: bool) -> None:
    """Print the answer and a short run summary."""
    answer = result.answer

    if not answer.answered:
        console.print(
            Panel(
                Markdown(answer.summary),
                title=f"[bold yellow]Not answered — {answer.refusal_category.value if answer.refusal_category else 'blocked'}",
                border_style="yellow",
            )
        )
        return

    console.print(Panel(Markdown(answer.summary), title="[bold]Answer", border_style="cyan"))

    if answer.claims:
        table = Table(title="Findings", show_lines=False, expand=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Claim")
        table.add_column("Citations", style="green", width=24)
        table.add_column("Conf.", width=10)
        for index, claim in enumerate(answer.claims, start=1):
            table.add_row(
                str(index),
                claim.statement,
                ", ".join(claim.pmids),
                claim.confidence.value,
            )
        console.print(table)

    if answer.limitations:
        console.print(
            Panel(
                "\n".join(f"• {item}" for item in answer.limitations),
                title="[bold]Limitations",
                border_style="yellow",
            )
        )

    if answer.citations:
        sources = Table(title="Sources", expand=True)
        sources.add_column("PMID", style="green", width=12)
        sources.add_column("Citation")
        sources.add_column("Design", width=26)
        for article in answer.citations:
            sources.add_row(
                article.pmid,
                f"{article.title[:90]}\n[dim]{article.citation()}[/dim]",
                ", ".join(article.publication_types[:2]) or "—",
            )
        console.print(sources)

    audit = answer.citation_audit
    summary = result.trace_summary
    footer = (
        f"grade: [bold]{answer.overall_grade.value}[/bold]  |  "
        f"claims cited: {audit.cited_claims}/{audit.total_claims}  |  "
        f"fabricated PMIDs: {len(audit.hallucinated_pmids)}  |  "
        f"llm calls: {summary['llm_requests']}  |  tools: {summary['tool_calls']}  |  "
        f"{answer.elapsed_seconds}s"
    )
    console.print(Panel(footer, border_style="dim", title="Run"))

    if answer.notes:
        console.print("[dim]" + "\n".join(f"note: {n}" for n in answer.notes) + "[/dim]")
    if answer.degraded:
        console.print("[yellow]This answer used a fallback path; see the trace.[/yellow]")
    if summary["trace_path"]:
        console.print(f"[dim]trace: {summary['trace_path']}[/dim]")

    if show_trace:
        console.print(Panel(Markdown(result.tracer.to_markdown()), title="Reasoning trace"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = apply_overrides(args)
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2

    try:
        agent = PubMedAgent(settings)
        result = agent.run(args.query, session_id=args.session_id, echo_trace=args.verbose)
    except ValueError as exc:
        # Most commonly a missing API key.
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2
    except CassetteMissError as exc:
        console.print(f"[red]Replay miss:[/red] {exc}")
        return 3
    except (LLMError, PubMedError) as exc:
        console.print(f"[red]The run could not complete:[/red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        return 130

    if args.json:
        print(json.dumps(result.answer.model_dump(mode="json"), indent=2))
    else:
        render(result, show_trace=args.show_trace)

    if args.save_trace:
        args.save_trace.parent.mkdir(parents=True, exist_ok=True)
        args.save_trace.write_text(result.tracer.to_markdown(), encoding="utf-8")
        console.print(f"[dim]trace written to {args.save_trace}[/dim]")

    return 0 if result.answer.answered else 0


if __name__ == "__main__":
    sys.exit(main())
