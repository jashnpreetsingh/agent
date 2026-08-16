"""The state object that flows through the graph.

Every node reads and writes this single typed structure. Nodes return partial
updates rather than mutating it, so each transition is an explicit, inspectable
diff - which is also what makes the trace faithful to what actually happened.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from src.llm.base import ChatMessage
from src.schemas import (
    AgentAnswer,
    Article,
    Critique,
    DraftAnswer,
    Evidence,
    GuardDecision,
    ResearchPlan,
)


def _replace(_current: Any, incoming: Any) -> Any:
    """Last write wins - the default for single-owner fields."""
    return incoming


def _append(current: list[Any] | None, incoming: list[Any] | None) -> list[Any]:
    """Accumulate across nodes (used for notes and errors)."""
    return (current or []) + (incoming or [])


class AgentState(TypedDict, total=False):
    """Shared state for one question.

    Grouped by the node that owns each field; ``total=False`` because the graph
    fills it in progressively.
    """

    # --- input ---
    question: str
    run_id: str
    conversation_context: str

    # --- guard_input ---
    guard_decision: GuardDecision

    # --- planner ---
    plan: ResearchPlan

    # --- researcher / tool loop ---
    messages: Annotated[list[ChatMessage], _replace]
    research_iterations: int
    tool_calls_made: int
    research_summary: str
    articles: dict[str, Article]

    # --- ranking ---
    evidence: list[Evidence]
    ranking_method: str

    # --- synthesis / critique ---
    draft: DraftAnswer
    critique: Critique
    revisions: int
    critic_feedback: str

    # --- output ---
    answer: AgentAnswer
    degraded: bool
    notes: Annotated[list[str], _append]
    errors: Annotated[list[str], _append]


def initial_state(question: str, run_id: str, conversation_context: str = "") -> AgentState:
    """Build the starting state for a run."""
    return AgentState(
        question=question,
        run_id=run_id,
        conversation_context=conversation_context,
        messages=[],
        research_iterations=0,
        tool_calls_made=0,
        research_summary="",
        articles={},
        evidence=[],
        ranking_method="",
        revisions=0,
        critic_feedback="",
        degraded=False,
        notes=[],
        errors=[],
    )
