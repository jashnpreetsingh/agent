"""The agent façade: assembles dependencies and runs one question end to end.

Everything above this line is composable parts; this is where they become an
agent. Construction is explicit rather than global so tests and the evaluation
harness can substitute fakes for the provider or the tool registry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings
from src.graph.build import build_graph
from src.graph.nodes import AgentNodes
from src.graph.state import initial_state
from src.llm.base import LLMProvider
from src.llm.factory import build_provider
from src.memory.store import ConversationStore
from src.observability.trace import Tracer
from src.schemas import AgentAnswer, EvidenceGrade
from src.tools.pubmed import PubMedClient
from src.tools.registry import ToolRegistry


@dataclass(slots=True)
class RunResult:
    """An answer plus everything needed to inspect how it was produced."""

    answer: AgentAnswer
    tracer: Tracer
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def trace_summary(self) -> dict[str, Any]:
        return self.tracer.summary()


class PubMedAgent:
    """Answers biomedical questions from PubMed literature."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: LLMProvider | None = None,
        registry: ToolRegistry | None = None,
        store: ConversationStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._provider_override = provider
        self._registry_override = registry
        self.store = store if store is not None else ConversationStore(self.settings.memory_db)

    # ------------------------------------------------------------------
    def run(
        self,
        question: str,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        echo_trace: bool = False,
    ) -> RunResult:
        """Answer one question.

        Args:
            question: The user's question.
            session_id: When given, prior turns are loaded as context and this
                turn is persisted for the next one.
            run_id: Override the generated run identifier.
            echo_trace: Mirror trace events to the logger.

        Returns:
            The answer, its trace, and the final graph state.
        """
        run_id = run_id or uuid.uuid4().hex[:12]
        tracer = Tracer(run_id=run_id, traces_dir=self.settings.traces_dir, echo=echo_trace)

        # A fresh registry per run keeps each run's article store isolated,
        # which the citation audit depends on.
        provider = self._provider_override or build_provider(self.settings, tracer)
        registry = self._registry_override or ToolRegistry(
            PubMedClient(self.settings, tracer), self.settings, tracer
        )
        if self._registry_override is not None:
            self._registry_override.tracer = tracer
        if hasattr(provider, "tracer"):
            provider.tracer = tracer  # type: ignore[attr-defined]

        nodes = AgentNodes(provider, registry, self.settings, tracer)
        graph = build_graph(nodes, self.settings, tracer)

        context = self.store.context_for(session_id) if session_id else ""

        tracer.event(
            "run.start",
            question=question,
            model=provider.model_name,
            llm_mode=self.settings.llm_mode.value,
            pubmed_mode=self.settings.pubmed_mode.value,
            session_id=session_id,
            has_context=bool(context),
        )

        with tracer.span("run") as span:
            final_state = graph.invoke(initial_state(question, run_id, context))
            answer: AgentAnswer = final_state["answer"]
            answer.elapsed_seconds = round(tracer.elapsed_seconds, 2)
            span.annotate(
                answered=answer.answered,
                claims=len(answer.claims),
                citations=len(answer.citations),
            )

        usage = getattr(provider, "total_usage", None)
        tracer.event(
            "run.complete",
            **tracer.summary(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            thoughts_tokens=getattr(usage, "thoughts_tokens", 0),
        )

        if session_id:
            self.store.add_turn(
                session_id=session_id,
                run_id=run_id,
                question=question,
                summary=answer.summary,
                pmids=[c.pmid for c in answer.citations],
                grade=answer.overall_grade.value,
                answered=answer.answered,
            )

        return RunResult(answer=answer, tracer=tracer, state=dict(final_state))

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release provider/tool connections owned by this agent."""
        for candidate in (self._provider_override, self._registry_override):
            closer = getattr(candidate, "close", None)
            if callable(closer):
                closer()


def answer_question(question: str, settings: Settings | None = None) -> AgentAnswer:
    """One-shot convenience wrapper used by the tests and the README example."""
    return PubMedAgent(settings).run(question).answer


__all__ = ["PubMedAgent", "RunResult", "answer_question", "EvidenceGrade"]


if __name__ == "__main__":  # pragma: no cover
    # Supports the `python src/agent.py --domain healthcare --query "..."` form
    # shown in the assignment brief, alongside `python -m src.cli`.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.cli import main

    sys.exit(main())
