"""Graph construction: nodes, edges, and the budget checks on every cycle.

The topology has two loops, and both are bounded:

* researcher <-> tool_executor, capped by ``max_research_iterations`` and
  ``max_tool_calls``;
* critic -> researcher, capped by ``max_revisions``.

The caps are global to a run rather than per-loop. A revision round therefore
draws from the same tool budget as the first pass, which is the conservative
choice: total spend per question stays bounded no matter which path the graph
takes.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.graph.nodes import AgentNodes
from src.graph.state import AgentState
from src.observability.trace import Tracer
from src.schemas import CritiqueVerdict


def build_graph(nodes: AgentNodes, settings: Settings, tracer: Tracer | None = None):
    """Wire the agent graph and compile it."""
    tracer = tracer or Tracer.null()
    graph = StateGraph(AgentState)

    graph.add_node("guard_input", nodes.guard_input)
    graph.add_node("refuse", nodes.refuse)
    graph.add_node("planner", nodes.plan)
    graph.add_node("researcher", nodes.research)
    graph.add_node("tool_executor", nodes.execute_tools)
    graph.add_node("rank", nodes.rank_evidence)
    graph.add_node("synthesizer", nodes.synthesize)
    graph.add_node("critic", nodes.critique)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "guard_input")

    # Blocked questions never reach the planner, so a refusal costs one LLM
    # call at most.
    graph.add_conditional_edges(
        "guard_input",
        _route_after_guard,
        {"refuse": "refuse", "plan": "planner"},
    )
    graph.add_edge("refuse", END)
    graph.add_edge("planner", "researcher")

    graph.add_conditional_edges(
        "researcher",
        _make_research_router(settings, tracer),
        {"tools": "tool_executor", "done": "rank"},
    )
    graph.add_edge("tool_executor", "researcher")
    graph.add_edge("rank", "synthesizer")
    graph.add_edge("synthesizer", "critic")

    graph.add_conditional_edges(
        "critic",
        _make_critic_router(settings, tracer),
        # Two revision paths: gather more evidence, or rewrite from what we
        # already have when the research budget is spent.
        {"revise": "researcher", "rewrite": "synthesizer", "accept": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
def _route_after_guard(state: AgentState) -> Literal["refuse", "plan"]:
    decision = state.get("guard_decision")
    return "plan" if decision is None or decision.allowed else "refuse"


def _make_research_router(settings: Settings, tracer: Tracer):
    """Continue the tool loop only while the model wants tools and budget remains."""

    def route(state: AgentState) -> Literal["tools", "done"]:
        messages = state.get("messages") or []
        wants_tool = bool(messages and messages[-1].tool_calls)

        if not wants_tool:
            return "done"

        iterations = state.get("research_iterations", 0)
        calls = state.get("tool_calls_made", 0)
        if iterations >= settings.max_research_iterations:
            tracer.event(
                "budget.exhausted", limit="max_research_iterations",
                value=iterations, action="proceeding to synthesis",
            )
            return "done"
        if calls >= settings.max_tool_calls:
            tracer.event(
                "budget.exhausted", limit="max_tool_calls",
                value=calls, action="proceeding to synthesis",
            )
            return "done"
        return "tools"

    return route


def _make_critic_router(settings: Settings, tracer: Tracer):
    """Allow a bounded number of revision rounds."""

    def route(state: AgentState) -> Literal["revise", "rewrite", "accept"]:
        critique = state.get("critique")
        if critique is None or critique.verdict is CritiqueVerdict.ACCEPT:
            return "accept"

        revisions = state.get("revisions", 0)
        if revisions > settings.max_revisions:
            tracer.event(
                "budget.exhausted", limit="max_revisions",
                value=revisions, action="accepting current draft",
            )
            return "accept"

        # A revision with no actionable direction would just repeat itself.
        if not critique.followup_queries and not critique.missing_aspects:
            tracer.event("critic.revise_without_direction", action="accepting current draft")
            return "accept"

        # Sending the researcher back with no tool budget left buys nothing:
        # it burns a call that cannot invoke a tool, then re-ranks an
        # unchanged corpus. Rewrite from existing evidence instead.
        research_exhausted = (
            state.get("research_iterations", 0) >= settings.max_research_iterations
            or state.get("tool_calls_made", 0) >= settings.max_tool_calls
        )
        if research_exhausted:
            tracer.event(
                "critic.rewrite_only",
                round=revisions,
                reason="research budget spent; revising with existing evidence",
            )
            return "rewrite"

        tracer.event("critic.revision_requested", round=revisions, gaps=critique.missing_aspects)
        return "revise"

    return route
