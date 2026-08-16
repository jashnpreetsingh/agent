"""Node implementations: one function per agent role.

Each node is a pure-ish function of ``AgentState`` returning a partial update.
They share a common failure philosophy: a node that cannot do its job degrades
to something usable and records why, rather than raising into the graph. A
research assistant that answers "the planner API timed out" with a stack trace
is less useful than one that runs the user's question as a literal search and
says so.
"""

from __future__ import annotations

from typing import Any

from src.config import Settings
from src.graph.state import AgentState
from src.llm.base import ChatMessage, LLMProvider
from src.llm.errors import LLMError, LLMRefusalError
from src.observability.trace import Tracer
from src.prompts.critic import CRITIC_SYSTEM, build_critic_prompt
from src.prompts.planner import PLANNER_SYSTEM, build_planner_prompt
from src.prompts.researcher import (
    RESEARCHER_SYSTEM,
    build_followup_prompt,
    build_researcher_prompt,
)
from src.prompts.synthesizer import SYNTHESIZER_SYSTEM, build_synthesis_prompt
from src.prompts import SHARED_PREAMBLE
from src.schemas import (
    AgentAnswer,
    Claim,
    Critique,
    CritiqueVerdict,
    DraftAnswer,
    EvidenceGrade,
    ResearchPlan,
    SearchSpec,
)
from src.guardrails.input_guard import InputGuard
from src.guardrails.output_guard import OutputGuard
from src.tools.rank import EvidenceRanker
from src.tools.registry import ToolRegistry


def _system(role_prompt: str) -> str:
    return f"{SHARED_PREAMBLE}\n\n{role_prompt}"


class AgentNodes:
    """Bundles the dependencies every node needs.

    Holding them on one object keeps node signatures graph-compatible
    (``state -> update``) while still allowing tests to inject fakes.
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        settings: Settings,
        tracer: Tracer | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.settings = settings
        self.tracer = tracer or Tracer.null()
        self.ranker = EvidenceRanker(provider, self.tracer)
        self.input_guard = InputGuard(
            provider, self.tracer, use_llm_scope_check=settings.use_llm_scope_check
        )
        self.output_guard = OutputGuard(self.tracer)

    # ==================================================================
    # 1. Input guardrail
    # ==================================================================
    def guard_input(self, state: AgentState) -> dict[str, Any]:
        decision = self.input_guard.check(state["question"])
        return {"guard_decision": decision}

    def refuse(self, state: AgentState) -> dict[str, Any]:
        """Terminal node for blocked questions."""
        decision = state["guard_decision"]
        answer = AgentAnswer(
            run_id=state["run_id"],
            question=state["question"],
            answered=False,
            summary=decision.user_message,
            refusal_category=decision.category,
            overall_grade=EvidenceGrade.INSUFFICIENT,
        )
        self.tracer.event(
            "answer.refused",
            category=decision.category.value if decision.category else None,
            reason=decision.reason,
        )
        return {"answer": answer}

    # ==================================================================
    # 2. Planner
    # ==================================================================
    def plan(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        with self.tracer.span("node.planner") as span:
            try:
                plan = self.provider.generate_structured(
                    [ChatMessage.user(build_planner_prompt(question, state.get("conversation_context", "")))],
                    ResearchPlan,
                    system=_system(PLANNER_SYSTEM),
                    purpose="planner",
                )
                span.annotate(
                    interpretation=plan.interpretation,
                    sub_questions=plan.sub_questions,
                    mesh_terms=plan.mesh_terms,
                    searches=[s.query for s in plan.searches],
                )
                return {"plan": plan}
            except LLMError as exc:
                # Degrade to searching the question verbatim. PubMed's own term
                # mapping is decent, so this still returns usable records.
                self.tracer.event("planner.fallback", detail=str(exc)[:250])
                span.annotate(fallback=True)
                fallback = ResearchPlan(
                    interpretation=f"Literal search for: {question}",
                    sub_questions=[question],
                    mesh_terms=[],
                    searches=[
                        SearchSpec(query=question, rationale="Planner unavailable; direct search.")
                    ],
                )
                return {
                    "plan": fallback,
                    "degraded": True,
                    "errors": [f"planner failed: {exc}"],
                    "notes": ["Research plan degraded to a direct search."],
                }

    # ==================================================================
    # 3. Researcher (tool loop)
    # ==================================================================
    def research(self, state: AgentState) -> dict[str, Any]:
        """Ask the model what to do next; it either calls a tool or stops."""
        messages = list(state.get("messages") or [])
        iteration = state.get("research_iterations", 0) + 1

        if not messages:
            messages.append(ChatMessage.user(build_researcher_prompt(state["question"], state["plan"])))
        elif state.get("critic_feedback") and state.get("revisions", 0) > 0 and messages[-1].role == "model":
            critique = state.get("critique")
            messages.append(
                ChatMessage.user(
                    build_followup_prompt(
                        gaps=(critique.missing_aspects if critique else []),
                        queries=(critique.followup_queries if critique else []),
                    )
                )
            )

        with self.tracer.span("node.researcher", iteration=iteration) as span:
            try:
                response = self.provider.generate(
                    messages,
                    system=_system(RESEARCHER_SYSTEM),
                    tools=self.registry.declarations(),
                    purpose=f"researcher.{iteration}",
                )
            except LLMError as exc:
                # Whatever was already retrieved still flows to synthesis.
                self.tracer.event("researcher.failed", detail=str(exc)[:250])
                span.annotate(error=str(exc)[:200])
                return {
                    "research_iterations": iteration,
                    "messages": messages,
                    "research_summary": "Research loop ended early due to a provider error.",
                    "degraded": True,
                    "errors": [f"researcher failed: {exc}"],
                }

            messages.append(ChatMessage.model(response.text, response.tool_calls))
            span.annotate(
                wants_tool=response.wants_tool,
                tools=[c.name for c in response.tool_calls],
                text=response.text[:300],
            )
            return {
                "messages": messages,
                "research_iterations": iteration,
                "research_summary": response.text or state.get("research_summary", ""),
            }

    def execute_tools(self, state: AgentState) -> dict[str, Any]:
        """Run every tool the model requested and feed results back."""
        messages = list(state["messages"])
        requested = messages[-1].tool_calls if messages else []
        calls_made = state.get("tool_calls_made", 0)

        for call in requested:
            result = self.registry.call(call.name, call.arguments)
            calls_made += 1
            messages.append(ChatMessage.tool(call.name, result.for_model(), call.call_id))

        return {
            "messages": messages,
            "tool_calls_made": calls_made,
            "articles": dict(self.registry.article_store),
        }

    # ==================================================================
    # 4. Ranking (RAG)
    # ==================================================================
    def rank_evidence(self, state: AgentState) -> dict[str, Any]:
        articles = list(self.registry.article_store.values())
        with self.tracer.span("node.rank", candidates=len(articles)) as span:
            outcome = self.ranker.rank(
                state["question"],
                articles,
                top_k=self.settings.max_evidence_items,
                char_budget=self.settings.evidence_char_budget,
            )
            span.annotate(
                method=outcome.method,
                selected=[e.pmid for e in outcome.evidence],
            )
            update: dict[str, Any] = {
                "evidence": outcome.evidence,
                "ranking_method": outcome.method,
                "articles": dict(self.registry.article_store),
            }
            if outcome.method == "lexical_fallback":
                update["notes"] = ["Semantic ranking unavailable; used lexical ranking."]
            return update

    # ==================================================================
    # 5. Synthesizer
    # ==================================================================
    def synthesize(self, state: AgentState) -> dict[str, Any]:
        evidence = state.get("evidence") or []
        with self.tracer.span("node.synthesizer", evidence_count=len(evidence)) as span:
            if not evidence:
                span.annotate(empty=True)
                return {
                    "draft": DraftAnswer(
                        summary=(
                            "I could not find published records addressing this question in "
                            "PubMed. This may mean the topic is not indexed under the terms "
                            "searched, or that little research has been published on it."
                        ),
                        claims=[],
                        limitations=["No relevant PubMed records were retrieved."],
                        overall_grade=EvidenceGrade.INSUFFICIENT,
                    )
                }
            try:
                draft = self.provider.generate_structured(
                    [
                        ChatMessage.user(
                            build_synthesis_prompt(
                                state["question"],
                                evidence,
                                conversation_context=state.get("conversation_context", ""),
                                critic_feedback=state.get("critic_feedback", ""),
                            )
                        )
                    ],
                    DraftAnswer,
                    system=_system(SYNTHESIZER_SYSTEM),
                    purpose="synthesizer",
                )
                span.annotate(
                    claims=len(draft.claims),
                    grade=draft.overall_grade.value,
                    summary=draft.summary[:300],
                )
                return {"draft": draft}
            except LLMRefusalError as exc:
                span.annotate(refused=True)
                return {
                    "draft": _evidence_only_draft(state, reason=f"provider refused: {exc}"),
                    "degraded": True,
                    "errors": [f"synthesis refused: {exc}"],
                    "notes": ["The model declined to synthesise; returning the retrieved records."],
                }
            except LLMError as exc:
                # The retrieval work is still valuable - hand back a listing
                # rather than nothing.
                span.annotate(error=str(exc)[:200])
                return {
                    "draft": _evidence_only_draft(state, reason=f"synthesis failed: {exc}"),
                    "degraded": True,
                    "errors": [f"synthesis failed: {exc}"],
                    "notes": ["Synthesis unavailable; returning ranked records without narrative."],
                }

    # ==================================================================
    # 6. Critic
    # ==================================================================
    def critique(self, state: AgentState) -> dict[str, Any]:
        draft = state["draft"]
        evidence = state.get("evidence") or []
        if not draft.claims:
            # Nothing to verify; revising cannot help.
            return {
                "critique": Critique(
                    verdict=CritiqueVerdict.ACCEPT,
                    reasoning="No claims to verify; accepting the low-evidence answer.",
                )
            }

        with self.tracer.span("node.critic", claims=len(draft.claims)) as span:
            try:
                critique = self.provider.generate_structured(
                    [ChatMessage.user(build_critic_prompt(state["question"], draft, evidence))],
                    Critique,
                    system=_system(CRITIC_SYSTEM),
                    purpose="critic",
                )
                span.annotate(
                    verdict=critique.verdict.value,
                    reasoning=critique.reasoning[:300],
                    unsupported=critique.unsupported_claims,
                    missing=critique.missing_aspects,
                )
                feedback = _format_feedback(critique)
                return {
                    "critique": critique,
                    "critic_feedback": feedback,
                    "revisions": state.get("revisions", 0) + (1 if critique.verdict is CritiqueVerdict.REVISE else 0),
                }
            except LLMError as exc:
                # Without a critique, accept: the deterministic citation audit
                # in the output guard still runs.
                span.annotate(error=str(exc)[:200])
                return {
                    "critique": Critique(
                        verdict=CritiqueVerdict.ACCEPT,
                        reasoning=f"Critic unavailable ({exc}); accepted without review.",
                    ),
                    "errors": [f"critic failed: {exc}"],
                    "notes": ["Self-review was skipped; citations were still verified."],
                }

    # ==================================================================
    # 7. Finalise
    # ==================================================================
    def finalize(self, state: AgentState) -> dict[str, Any]:
        draft = state["draft"]
        articles = dict(self.registry.article_store)

        answer = AgentAnswer(
            run_id=state["run_id"],
            question=state["question"],
            answered=True,
            summary=draft.summary,
            claims=list(draft.claims),
            limitations=list(draft.limitations),
            overall_grade=draft.overall_grade,
            degraded=state.get("degraded", False),
            notes=list(state.get("notes") or []),
        )
        answer = self.output_guard.apply(answer, draft, articles)

        self.tracer.event(
            "answer.final",
            claims=len(answer.claims),
            citations=len(answer.citations),
            grade=answer.overall_grade.value,
            citation_rate=round(answer.citation_audit.citation_rate, 3),
            degraded=answer.degraded,
        )
        return {"answer": answer}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _evidence_only_draft(state: AgentState, *, reason: str) -> DraftAnswer:
    """Fallback draft: report the retrieved records without narrative synthesis.

    Each record becomes a claim citing itself, so the answer stays verifiable
    and passes the same citation audit as a synthesised one.
    """
    evidence = state.get("evidence") or []
    claims = [
        Claim(
            statement=f"{item.article.title} ({item.article.journal or 'unknown journal'}, "
            f"{item.article.year or 'n.d.'}).",
            pmids=[item.pmid],
            confidence=EvidenceGrade.LIMITED,
        )
        for item in evidence[:6]
    ]
    return DraftAnswer(
        summary=(
            "I retrieved relevant published records but could not generate a narrative "
            "summary of them. The most relevant studies are listed below for direct review."
        ),
        claims=claims,
        limitations=[
            f"Automatic synthesis was unavailable ({reason}).",
            "Records are listed by ranking score and have not been interpreted.",
        ],
        overall_grade=EvidenceGrade.LIMITED,
    )


def _format_feedback(critique: Critique) -> str:
    """Render a critique as instructions for the next synthesis attempt."""
    parts = [critique.reasoning]
    if critique.unsupported_claims:
        parts.append("Unsupported claims: " + "; ".join(critique.unsupported_claims))
    if critique.missing_aspects:
        parts.append("Unanswered aspects: " + "; ".join(critique.missing_aspects))
    return "\n".join(parts)
