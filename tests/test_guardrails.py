"""Tests for input screening and citation verification.

The citation tests are the most important in the suite: they cover the one
failure mode - a confident answer citing a PMID that does not exist - that
would make the whole system untrustworthy.
"""

from __future__ import annotations

import pytest

from src.guardrails.input_guard import InputGuard
from src.guardrails.output_guard import OutputGuard, verify_citations
from src.schemas import (
    AgentAnswer,
    Claim,
    DraftAnswer,
    EvidenceGrade,
    RefusalCategory,
)
from tests.conftest import make_article


@pytest.fixture
def guard() -> InputGuard:
    # No provider: exercises the deterministic rules, which must stand alone.
    return InputGuard(provider=None)


# ---------------------------------------------------------------------------
# Emergencies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "I am having crushing chest pain right now, what do I do?",
        "I can't breathe and my face is drooping",
        "I think I overdosed on my pills",
        "I want to kill myself",
        "my son is unconscious and won't wake up",
    ],
)
def test_emergencies_are_refused(guard: InputGuard, question: str):
    decision = guard.check(question)
    assert not decision.allowed
    assert decision.category is RefusalCategory.MEDICAL_EMERGENCY
    assert "911" in decision.user_message


def test_emergency_refusal_names_crisis_line(guard: InputGuard):
    decision = guard.check("I am suicidal and want to die")
    assert "988" in decision.user_message


# ---------------------------------------------------------------------------
# Personal advice
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "Should I stop taking my metformin?",
        "I was diagnosed with diabetes, what should I take?",
        "How much vitamin D should I take for my condition?",
        "Do I have diabetes?",
        "Is it safe for me to combine ibuprofen with my blood pressure medication?",
    ],
)
def test_personal_advice_is_refused(guard: InputGuard, question: str):
    decision = guard.check(question)
    assert not decision.allowed
    assert decision.category is RefusalCategory.PERSONAL_MEDICAL_ADVICE
    assert "clinician" in decision.user_message or "doctor" in decision.user_message


def test_advice_refusal_offers_a_usable_alternative(guard: InputGuard):
    """A refusal that leaves the user stuck is a bad refusal."""
    decision = guard.check("Should I switch from metformin to tirzepatide?")
    assert "What does the evidence say" in decision.user_message


# ---------------------------------------------------------------------------
# Legitimate questions must pass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "What are the latest treatment options for Type 2 diabetes?",
        "Do SGLT2 inhibitors reduce heart failure hospitalisation?",
        "What is the mechanism of action of metformin?",
        "How effective is vitamin D for preventing respiratory infections?",
        "What are the reported adverse effects of tirzepatide in clinical trials?",
    ],
)
def test_research_questions_are_allowed(guard: InputGuard, question: str):
    decision = guard.check(question)
    assert decision.allowed, f"false positive on: {question}"


def test_short_input_rejected(guard: InputGuard):
    decision = guard.check("hi")
    assert not decision.allowed
    assert decision.category is RefusalCategory.MALFORMED_INPUT


def test_overlong_input_rejected(guard: InputGuard):
    decision = guard.check("diabetes " * 500)
    assert not decision.allowed
    assert decision.category is RefusalCategory.MALFORMED_INPUT


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------
def test_verify_citations_flags_fabricated_pmid():
    draft = DraftAnswer(
        summary="s",
        claims=[
            Claim(statement="Real claim", pmids=["11111111"]),
            Claim(statement="Invented claim", pmids=["99999999"]),
        ],
    )
    audit = verify_citations(draft, {"11111111"})
    assert audit.hallucinated_pmids == ["99999999"]
    assert audit.cited_claims == 1
    assert audit.total_claims == 2
    assert not audit.is_clean


def test_verify_citations_clean_when_all_present():
    draft = DraftAnswer(summary="s", claims=[Claim(statement="c", pmids=["11111111"])])
    audit = verify_citations(draft, {"11111111", "22222222"})
    assert audit.is_clean
    assert audit.citation_rate == 1.0


def test_pmid_normalisation_strips_prefixes():
    """Models emit 'PMID: 123' and '123.' - both must compare equal to '123'."""
    claim = Claim(statement="c", pmids=["PMID: 11111111", "22222222."])
    assert claim.pmids == ["11111111", "22222222"]


def test_output_guard_drops_unverifiable_claims():
    articles = {"11111111": make_article("11111111")}
    draft = DraftAnswer(
        summary="Summary of findings.",
        claims=[
            Claim(statement="Supported", pmids=["11111111"]),
            Claim(statement="Fabricated", pmids=["99999999"]),
        ],
        overall_grade=EvidenceGrade.MODERATE,
    )
    answer = AgentAnswer(run_id="t", question="q", answered=True, summary=draft.summary)

    result = OutputGuard().apply(answer, draft, articles)

    assert [c.statement for c in result.claims] == ["Supported"]
    assert result.citation_audit.hallucinated_pmids == ["99999999"]
    assert [a.pmid for a in result.citations] == ["11111111"]
    assert any("dropped" in note for note in result.notes)


def test_output_guard_keeps_partially_valid_citations():
    """A claim citing one real and one invented PMID keeps its real support."""
    articles = {"11111111": make_article("11111111")}
    draft = DraftAnswer(
        summary="s",
        claims=[Claim(statement="Mixed", pmids=["11111111", "99999999"])],
    )
    answer = AgentAnswer(run_id="t", question="q", answered=True, summary="s")

    result = OutputGuard().apply(answer, draft, articles)

    assert len(result.claims) == 1
    assert result.claims[0].pmids == ["11111111"]


def test_output_guard_grades_empty_answer_insufficient():
    draft = DraftAnswer(summary="s", claims=[Claim(statement="Bogus", pmids=["99999999"])])
    answer = AgentAnswer(run_id="t", question="q", answered=True, summary="s")

    result = OutputGuard().apply(answer, draft, {})

    assert result.claims == []
    assert result.overall_grade is EvidenceGrade.INSUFFICIENT


def test_output_guard_neutralises_directive_advice():
    articles = {"11111111": make_article("11111111")}
    draft = DraftAnswer(
        summary="You should take metformin daily.",
        claims=[Claim(statement="c", pmids=["11111111"])],
    )
    answer = AgentAnswer(
        run_id="t", question="q", answered=True, summary="You should take metformin daily."
    )

    result = OutputGuard().apply(answer, draft, articles)

    assert "You should take" not in result.summary
