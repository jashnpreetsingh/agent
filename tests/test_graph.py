"""End-to-end graph tests using fakes.

These prove the orchestration itself: that the tool loop cycles, that budgets
terminate it, that the critic can send work back, and that a failing node
degrades instead of crashing the run. No network, no API key.
"""

from __future__ import annotations

import pytest

from src.agent import PubMedAgent
from src.llm.errors import LLMTransportError
from src.memory.store import ConversationStore
from src.schemas import Claim, CritiqueVerdict, DraftAnswer, EvidenceGrade, RefusalCategory
from tests.conftest import FakeProvider, make_article


def build_agent(settings, registry, provider, tmp_path) -> PubMedAgent:
    return PubMedAgent(
        settings,
        provider=provider,
        registry=registry,
        store=ConversationStore(tmp_path / "mem.sqlite3"),
    )


def test_full_run_produces_cited_answer(settings, registry, tmp_path):
    provider = FakeProvider()
    agent = build_agent(settings, registry, provider, tmp_path)

    result = agent.run("What are the latest treatment options for Type 2 diabetes?")

    assert result.answer.answered
    assert result.answer.claims
    assert result.answer.citation_audit.is_clean
    assert [a.pmid for a in result.answer.citations] == ["11111111"]
    assert "planner" in provider.calls
    assert "synthesizer" in provider.calls


def test_tool_loop_executes_and_returns(settings, registry, tmp_path):
    provider = FakeProvider(tool_rounds=2)
    agent = build_agent(settings, registry, provider, tmp_path)

    result = agent.run("Treatments for type 2 diabetes?")

    tool_events = [e for e in result.tracer.events if e.name == "tool.call.end"]
    assert len(tool_events) == 2, "researcher should cycle through two tool rounds"
    assert result.answer.answered


def test_tool_loop_stops_at_iteration_budget(settings, registry, tmp_path):
    """A model that never stops asking for tools must still terminate."""
    provider = FakeProvider(tool_rounds=999)
    agent = build_agent(settings, registry, provider, tmp_path)

    result = agent.run("Treatments for type 2 diabetes?")

    budget_events = [e for e in result.tracer.events if e.name == "budget.exhausted"]
    assert budget_events, "expected a budget stop"
    assert result.answer.answered


def test_critic_revision_loop_is_bounded(settings, registry, tmp_path):
    provider = FakeProvider(verdict=CritiqueVerdict.REVISE)
    agent = build_agent(settings, registry, provider, tmp_path)

    result = agent.run("Treatments for type 2 diabetes?")

    assert result.answer.answered
    assert result.state["revisions"] <= settings.max_revisions + 1


def test_emergency_refusal_skips_the_pipeline(settings, registry, tmp_path):
    provider = FakeProvider()
    agent = build_agent(settings, registry, provider, tmp_path)

    result = agent.run("I am having crushing chest pain right now and can't breathe")

    assert not result.answer.answered
    assert result.answer.refusal_category is RefusalCategory.MEDICAL_EMERGENCY
    assert provider.calls == [], "a refusal must not spend LLM calls"
    assert result.trace_summary["tool_calls"] == 0


def test_planner_failure_degrades_to_direct_search(settings, registry, tmp_path):
    class NoPlanner(FakeProvider):
        def generate_structured(self, messages, response_model, **kwargs):
            from src.schemas import ResearchPlan

            if response_model is ResearchPlan:
                raise LLMTransportError("planner down", status_code=503, retryable=True)
            return super().generate_structured(messages, response_model, **kwargs)

    agent = build_agent(settings, registry, NoPlanner(), tmp_path)
    result = agent.run("Treatments for type 2 diabetes?")

    assert result.answer.answered, "a planner outage should not fail the run"
    assert result.answer.degraded
    assert any("planner" in err for err in result.state["errors"])


def test_synthesis_failure_returns_retrieved_records(settings, registry, tmp_path):
    class NoSynthesis(FakeProvider):
        def generate_structured(self, messages, response_model, **kwargs):
            if response_model is DraftAnswer:
                raise LLMTransportError("synthesis down", status_code=500, retryable=True)
            return super().generate_structured(messages, response_model, **kwargs)

    agent = build_agent(settings, registry, NoSynthesis(), tmp_path)
    result = agent.run("Treatments for type 2 diabetes?")

    assert result.answer.answered
    assert result.answer.degraded
    assert result.answer.claims, "fallback should still list retrieved records"
    assert result.answer.citation_audit.is_clean


def test_fabricated_citations_are_stripped_end_to_end(settings, registry, tmp_path):
    bad_draft = DraftAnswer(
        summary="Answer with a fabricated source.",
        claims=[
            Claim(statement="Real finding", pmids=["11111111"]),
            Claim(statement="Invented finding", pmids=["9999999999"]),
        ],
        overall_grade=EvidenceGrade.STRONG,
    )
    agent = build_agent(settings, registry, FakeProvider(draft=bad_draft), tmp_path)

    result = agent.run("Treatments for type 2 diabetes?")

    statements = [c.statement for c in result.answer.claims]
    assert "Invented finding" not in statements
    assert result.answer.citation_audit.hallucinated_pmids == ["9999999999"]


def test_no_evidence_yields_honest_answer(settings, tmp_path):
    from src.tools.registry import ToolRegistry
    from tests.conftest import FakePubMed

    empty_registry = ToolRegistry(FakePubMed(articles=[]), settings)  # type: ignore[arg-type]
    agent = build_agent(settings, empty_registry, FakeProvider(), tmp_path)

    result = agent.run("Treatments for an extremely obscure condition?")

    assert result.answer.overall_grade is EvidenceGrade.INSUFFICIENT
    assert result.answer.claims == []
    assert result.answer.citation_audit.is_clean


def test_trace_records_every_phase(settings, registry, tmp_path):
    agent = build_agent(settings, registry, FakeProvider(), tmp_path)
    result = agent.run("Treatments for type 2 diabetes?")

    names = {e.name for e in result.tracer.events}
    for expected in {
        "run.start",
        "guard.input.end",
        "node.planner.end",
        "node.researcher.end",
        "tool.call.end",
        "node.rank.end",
        "node.synthesizer.end",
        "guard.output.end",
        "answer.final",
    }:
        assert expected in names, f"missing trace event {expected}"

    assert result.tracer.path is not None and result.tracer.path.exists()


def test_session_memory_carries_context(settings, registry, tmp_path):
    store = ConversationStore(tmp_path / "mem.sqlite3")
    agent = PubMedAgent(settings, provider=FakeProvider(), registry=registry, store=store)

    agent.run("What are the treatment options for type 2 diabetes?", session_id="s1")
    context = store.context_for("s1")

    assert "type 2 diabetes" in context.lower()
    assert store.history("s1")[0].pmids == ["11111111"]


@pytest.mark.parametrize("question", ["", "   ", "ok"])
def test_malformed_questions_are_rejected(settings, registry, tmp_path, question):
    agent = build_agent(settings, registry, FakeProvider(), tmp_path)
    result = agent.run(question)
    assert not result.answer.answered
    assert result.answer.refusal_category is RefusalCategory.MALFORMED_INPUT


def test_revision_skips_researcher_when_tool_budget_is_spent(settings, registry, tmp_path):
    """A revision with no research budget must rewrite, not re-ask the researcher.

    Routing back to the researcher there costs one LLM call that cannot invoke
    a tool, plus a re-rank of an unchanged corpus, and changes nothing.
    """
    # Two iterations: one that calls a tool, one that stops - after which the
    # research budget is spent and a revision cannot gather anything new.
    tight = settings.model_copy(update={"max_research_iterations": 2, "max_tool_calls": 5})
    provider = FakeProvider(verdict=CritiqueVerdict.REVISE, tool_rounds=1)
    agent = PubMedAgent(
        tight,
        provider=provider,
        registry=registry,
        store=ConversationStore(tmp_path / "mem.sqlite3"),
    )

    result = agent.run("Treatments for type 2 diabetes?")

    names = [e.name for e in result.tracer.events]
    assert "critic.rewrite_only" in names
    # Two syntheses (original + rewrite), and no researcher turn between them.
    assert names.count("node.synthesizer.end") == 2
    synth_idx = [i for i, n in enumerate(names) if n == "node.synthesizer.end"]
    between = names[synth_idx[0] : synth_idx[1]]
    assert "node.researcher.end" not in between
    assert result.answer.answered
