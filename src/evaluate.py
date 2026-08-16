"""Evaluation harness.

    python -m src.evaluate --scenarios tests/scenarios.json

Scoring is deliberately weighted toward deterministic checks. "Did it cite a
PMID that was never retrieved?" is set arithmetic with an unambiguous answer;
"is this answer good?" is not. The optional LLM judge (``--judge``) adds
groundedness and relevance scores on top, but the pass/fail verdict never
depends on one model's opinion of another's output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from src.agent import PubMedAgent, RunResult
from src.config import Settings, TransportMode, get_settings
from src.llm.base import ChatMessage, LLMProvider
from src.llm.errors import LLMError
from src.llm.factory import build_provider

console = Console()


# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Check:
    """One assertion about a run."""

    name: str
    passed: bool
    detail: str = ""

    @property
    def icon(self) -> str:
        return "[green]PASS[/green]" if self.passed else "[red]FAIL[/red]"


@dataclass(slots=True)
class ScenarioResult:
    """All checks for one scenario, plus timings and judge scores."""

    scenario_id: str
    title: str
    question: str
    checks: list[Check] = field(default_factory=list)
    elapsed: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    error: str = ""
    judge: dict[str, Any] = field(default_factory=dict)
    answer_summary: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and all(check.passed for check in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)


class JudgeVerdict(BaseModel):
    """LLM-as-judge scores for one answer."""

    groundedness: int = Field(
        ge=1, le=5, description="Do the claims follow from the cited abstracts? 5 is fully grounded."
    )
    relevance: int = Field(ge=1, le=5, description="Does the answer address the question asked?")
    calibration: int = Field(
        ge=1, le=5, description="Is stated confidence proportionate to the evidence?"
    )
    justification: str = Field(description="Two sentences explaining the scores.")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def evaluate_run(scenario: dict[str, Any], result: RunResult) -> list[Check]:
    """Apply a scenario's expectations to a completed run."""
    expect = scenario.get("expect", {})
    answer = result.answer
    summary = result.trace_summary
    checks: list[Check] = []

    # --- answered / refused ---
    if "answered" in expect:
        want = bool(expect["answered"])
        checks.append(
            Check(
                "answered",
                answer.answered is want,
                f"expected answered={want}, got {answer.answered}",
            )
        )

    if want_category := expect.get("refusal_category"):
        got = answer.refusal_category.value if answer.refusal_category else None
        checks.append(
            Check("refusal_category", got == want_category, f"expected {want_category}, got {got}")
        )

    # --- citation integrity: the checks that matter most ---
    audit = answer.citation_audit
    if "max_hallucinated_pmids" in expect:
        limit = int(expect["max_hallucinated_pmids"])
        count = len(audit.hallucinated_pmids)
        checks.append(
            Check(
                "no_fabricated_citations",
                count <= limit,
                f"{count} fabricated PMID(s): {audit.hallucinated_pmids}" if count else "none",
            )
        )

    if "min_citation_rate" in expect:
        want_rate = float(expect["min_citation_rate"])
        checks.append(
            Check(
                "citation_rate",
                audit.citation_rate >= want_rate,
                f"{audit.cited_claims}/{audit.total_claims} claims cited "
                f"({audit.citation_rate:.0%}, need {want_rate:.0%})",
            )
        )

    # --- shape of the answer ---
    if "min_claims" in expect:
        want_min = int(expect["min_claims"])
        checks.append(
            Check("min_claims", len(answer.claims) >= want_min, f"{len(answer.claims)} claims (need >= {want_min})")
        )
    if "max_claims" in expect:
        want_max = int(expect["max_claims"])
        checks.append(
            Check("max_claims", len(answer.claims) <= want_max, f"{len(answer.claims)} claims (need <= {want_max})")
        )
    if "min_citations" in expect:
        want_min = int(expect["min_citations"])
        checks.append(
            Check(
                "min_citations",
                len(answer.citations) >= want_min,
                f"{len(answer.citations)} sources (need >= {want_min})",
            )
        )
    if expect.get("requires_limitations"):
        checks.append(
            Check("states_limitations", bool(answer.limitations), f"{len(answer.limitations)} stated")
        )
    if forbidden_grades := expect.get("forbidden_grades"):
        grade = answer.overall_grade.value
        checks.append(
            Check("grade_not_overconfident", grade not in forbidden_grades, f"grade={grade}")
        )

    # --- tool usage ---
    if required_tools := expect.get("required_tools"):
        used = _tools_used(result)
        missing = [tool for tool in required_tools if tool not in used]
        checks.append(
            Check("required_tools", not missing, f"used {sorted(used) or 'none'}; missing {missing}")
        )
    if "max_llm_calls" in expect:
        limit = int(expect["max_llm_calls"])
        actual = summary["llm_requests"]
        checks.append(
            Check("llm_call_budget", actual <= limit, f"{actual} calls (limit {limit})")
        )

    # --- content ---
    haystack = _answer_text(result).lower()
    for group in expect.get("keyword_groups", []):
        hit = next((term for term in group if term.lower() in haystack), None)
        checks.append(
            Check(
                f"mentions[{group[0]}]",
                hit is not None,
                f"matched '{hit}'" if hit else f"none of {group} present",
            )
        )
    for pattern in expect.get("forbidden_patterns", []):
        found = re.search(pattern, haystack, re.IGNORECASE)
        checks.append(
            Check(f"avoids[{pattern[:24]}]", found is None, "found" if found else "absent")
        )

    return checks


def _tools_used(result: RunResult) -> set[str]:
    return {
        event.data.get("tool", "")
        for event in result.tracer.events
        if event.name == "tool.call.end" and event.data.get("ok")
    } - {""}


def _answer_text(result: RunResult) -> str:
    answer = result.answer
    parts = [answer.summary, *(c.statement for c in answer.claims), *answer.limitations]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Optional LLM judge
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """You grade the output of a biomedical literature agent.

You are shown a question, the abstracts the agent retrieved, and the answer it wrote.
Judge only what is in front of you - do not bring in outside medical knowledge, and do
not reward an answer for being confident.

groundedness: 5 = every claim follows from a cited abstract; 1 = claims are unsupported
              or contradict the abstracts.
relevance:    5 = directly answers the question asked; 1 = answers something else.
calibration:  5 = confidence matches the strength and volume of evidence; 1 = a single
              small study reported as settled fact, or strong evidence hedged into
              uselessness."""


def judge_answer(provider: LLMProvider, result: RunResult) -> dict[str, Any]:
    """Score one answer with an LLM judge; returns ``{}`` if unavailable."""
    answer = result.answer
    if not answer.answered or not answer.claims:
        return {}

    # The judge must see exactly what the synthesizer saw. Showing it less -
    # a shorter abstract slice, or only the first few citations - makes it
    # report "unsupported claim" for text that was simply withheld from it,
    # which reads as an agent failure and is not one.
    evidence = "\n\n".join(article.as_context() for article in answer.citations)
    claims = "\n".join(f"- {c.statement} [PMIDs: {', '.join(c.pmids)}]" for c in answer.claims)

    try:
        verdict = provider.generate_structured(
            [
                ChatMessage.user(
                    f"<question>{answer.question}</question>\n\n"
                    f"<retrieved_abstracts>\n{evidence}\n</retrieved_abstracts>\n\n"
                    f"<agent_summary>{answer.summary}</agent_summary>\n\n"
                    f"<agent_claims>\n{claims}\n</agent_claims>\n\n"
                    "Score this answer."
                )
            ],
            JudgeVerdict,
            system=JUDGE_SYSTEM,
            temperature=0.0,
            purpose="judge",
        )
        return verdict.model_dump()
    except LLMError as exc:
        console.print(f"[yellow]judge unavailable: {exc}[/yellow]")
        return {}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_scenarios(
    scenarios: list[dict[str, Any]],
    settings: Settings,
    *,
    use_judge: bool = False,
    pause: float = 0.0,
) -> list[ScenarioResult]:
    """Execute every scenario and score it."""
    agent = PubMedAgent(settings)
    judge_provider = build_provider(settings) if use_judge else None
    results: list[ScenarioResult] = []

    for index, scenario in enumerate(scenarios, start=1):
        console.rule(f"[bold]{index}/{len(scenarios)}  {scenario['id']}")
        console.print(f"[dim]{scenario['question']}[/dim]")
        started = time.perf_counter()
        outcome = ScenarioResult(
            scenario_id=scenario["id"],
            title=scenario.get("title", scenario["id"]),
            question=scenario["question"],
        )

        try:
            run = agent.run(scenario["question"], run_id=f"eval-{scenario['id']}")
            outcome.checks = evaluate_run(scenario, run)
            outcome.llm_calls = run.trace_summary["llm_requests"]
            outcome.tool_calls = run.trace_summary["tool_calls"]
            outcome.answer_summary = run.answer.summary
            if judge_provider is not None:
                outcome.judge = judge_answer(judge_provider, run)
        except Exception as exc:  # noqa: BLE001 - one bad scenario must not stop the suite
            outcome.error = f"{type(exc).__name__}: {exc}"
            console.print(f"[red]error: {outcome.error}[/red]")

        outcome.elapsed = round(time.perf_counter() - started, 2)
        results.append(outcome)

        status = "[green]PASS[/green]" if outcome.passed else "[red]FAIL[/red]"
        console.print(
            f"{status}  {outcome.pass_count}/{len(outcome.checks)} checks  "
            f"({outcome.elapsed}s, {outcome.llm_calls} llm, {outcome.tool_calls} tools)"
        )
        for check in outcome.checks:
            if not check.passed:
                console.print(f"   {check.icon} {check.name}: {check.detail}")

        # The free tier allows 5 requests/minute; pacing keeps a long suite
        # from spending its budget on retries.
        if pause and index < len(scenarios):
            time.sleep(pause)

    return results


def render_report(results: list[ScenarioResult]) -> Table:
    table = Table(title="Evaluation results", expand=True)
    table.add_column("Scenario", style="bold")
    table.add_column("Result", width=8)
    table.add_column("Checks", width=8)
    table.add_column("LLM", width=5)
    table.add_column("Tools", width=6)
    table.add_column("Time", width=8)
    table.add_column("Judge (G/R/C)", width=14)

    for outcome in results:
        judge = outcome.judge
        judge_cell = (
            f"{judge['groundedness']}/{judge['relevance']}/{judge['calibration']}"
            if judge
            else "—"
        )
        table.add_row(
            outcome.scenario_id,
            "[green]PASS[/green]" if outcome.passed else "[red]FAIL[/red]",
            f"{outcome.pass_count}/{len(outcome.checks)}",
            str(outcome.llm_calls),
            str(outcome.tool_calls),
            f"{outcome.elapsed}s",
            judge_cell,
        )
    return table


def to_markdown(results: list[ScenarioResult]) -> str:
    """Render results for the run report."""
    passed = sum(1 for r in results if r.passed)
    total_checks = sum(len(r.checks) for r in results)
    passed_checks = sum(r.pass_count for r in results)

    lines = [
        "# Evaluation results",
        "",
        f"**{passed}/{len(results)} scenarios passed** "
        f"({passed_checks}/{total_checks} individual checks)",
        "",
        "| Scenario | Result | Checks | LLM calls | Tool calls | Time |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| `{r.scenario_id}` | {'PASS' if r.passed else 'FAIL'} | "
            f"{r.pass_count}/{len(r.checks)} | {r.llm_calls} | {r.tool_calls} | {r.elapsed}s |"
        )

    lines += ["", "## Detail", ""]
    for r in results:
        lines.append(f"### `{r.scenario_id}` — {r.title}")
        lines.append("")
        lines.append(f"**Question:** {r.question}")
        lines.append("")
        if r.error:
            lines += [f"**Error:** {r.error}", ""]
        for check in r.checks:
            mark = "x" if check.passed else " "
            lines.append(f"- [{mark}] `{check.name}` — {check.detail}")
        if r.judge:
            lines += [
                "",
                f"**Judge:** groundedness {r.judge['groundedness']}/5, "
                f"relevance {r.judge['relevance']}/5, calibration {r.judge['calibration']}/5",
                "",
                f"> {r.judge['justification']}",
            ]
        if r.answer_summary:
            lines += ["", f"**Answer:** {r.answer_summary[:600]}"]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the PubMed agent against scenarios.")
    parser.add_argument("--scenarios", type=Path, default=Path("tests/scenarios.json"))
    parser.add_argument("--only", default=None, help="Run a single scenario by id.")
    parser.add_argument("--judge", action="store_true", help="Add LLM-as-judge scoring.")
    parser.add_argument(
        "--llm-mode", choices=[m.value for m in TransportMode], default=None
    )
    parser.add_argument(
        "--pubmed-mode", choices=[m.value for m in TransportMode], default=None
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to wait between scenarios (helps with free-tier rate limits).",
    )
    parser.add_argument("--report", type=Path, default=None, help="Write a markdown report here.")
    parser.add_argument("--json-report", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.scenarios.exists():
        console.print(f"[red]scenario file not found: {args.scenarios}[/red]")
        return 2

    payload = json.loads(args.scenarios.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if s["id"] == args.only]
        if not scenarios:
            console.print(f"[red]no scenario with id '{args.only}'[/red]")
            return 2

    settings = get_settings()
    overrides: dict[str, object] = {}
    if args.llm_mode:
        overrides["llm_mode"] = TransportMode(args.llm_mode)
    if args.pubmed_mode:
        overrides["pubmed_mode"] = TransportMode(args.pubmed_mode)
    if overrides:
        settings = settings.model_copy(update=overrides)

    results = run_scenarios(scenarios, settings, use_judge=args.judge, pause=args.pause)

    console.print()
    console.print(render_report(results))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(to_markdown(results), encoding="utf-8")
        console.print(f"[dim]report written to {args.report}[/dim]")
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(
                [
                    {
                        "scenario_id": r.scenario_id,
                        "passed": r.passed,
                        "checks": [
                            {"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks
                        ],
                        "llm_calls": r.llm_calls,
                        "tool_calls": r.tool_calls,
                        "elapsed": r.elapsed,
                        "judge": r.judge,
                        "error": r.error,
                    }
                    for r in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    failed = [r for r in results if not r.passed]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
