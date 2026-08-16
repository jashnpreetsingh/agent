"""Output guardrails: verify citations before the answer leaves the system.

The important check here is deterministic. Every PMID in the draft is compared
against the corpus actually retrieved this run; anything else is a fabrication,
and no amount of prompting removes that risk entirely. Because the check is set
arithmetic rather than a judgement call, it cannot be talked out of a finding.

Claims citing only fabricated PMIDs are dropped rather than shown with a
warning: an unsupported claim is worse than a shorter answer.
"""

from __future__ import annotations

import re

from src.observability.trace import Tracer
from src.schemas import (
    AgentAnswer,
    Article,
    CitationAudit,
    Claim,
    DraftAnswer,
    EvidenceGrade,
)

#: Phrasing that turns a literature summary into personal medical direction.
_ADVICE_PATTERNS = (
    re.compile(r"\byou should (take|start|stop|switch|discontinue|increase|decrease)\b", re.I),
    re.compile(r"\b(i recommend|my advice|you must take|you need to take)\b", re.I),
    re.compile(r"\byour (dose|dosage|treatment|prescription) should\b", re.I),
)

DISCLAIMER = (
    "This is a summary of published research, not medical advice. "
    "Discuss any treatment decision with a qualified clinician."
)


def verify_citations(draft: DraftAnswer, available_pmids: set[str]) -> CitationAudit:
    """Check every cited PMID against the retrieved corpus.

    Args:
        draft: The synthesized answer.
        available_pmids: PMIDs actually retrieved during this run.

    Returns:
        An audit naming uncited claims and fabricated PMIDs.
    """
    audit = CitationAudit(total_claims=len(draft.claims))
    hallucinated: set[str] = set()

    for claim in draft.claims:
        valid = [pmid for pmid in claim.pmids if pmid in available_pmids]
        invalid = [pmid for pmid in claim.pmids if pmid not in available_pmids]
        hallucinated.update(invalid)
        if valid:
            audit.cited_claims += 1
        else:
            audit.uncited_claims.append(claim.statement)

    audit.hallucinated_pmids = sorted(hallucinated)
    return audit


class OutputGuard:
    """Final pass over a draft answer before it is returned."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer.null()

    def apply(
        self,
        answer: AgentAnswer,
        draft: DraftAnswer,
        articles_by_pmid: dict[str, Article],
    ) -> AgentAnswer:
        """Verify, clean, and finalise ``answer`` in place."""
        with self.tracer.span("guard.output") as span:
            available = set(articles_by_pmid)
            audit = verify_citations(draft, available)
            answer.citation_audit = audit

            kept: list[Claim] = []
            dropped: list[str] = []
            for claim in draft.claims:
                valid = [pmid for pmid in claim.pmids if pmid in available]
                if not valid:
                    dropped.append(claim.statement)
                    continue
                # Keep only the verified subset, so a claim citing one real and
                # one invented PMID survives with its real support.
                kept.append(claim.model_copy(update={"pmids": valid}))

            answer.claims = kept
            answer.limitations = list(draft.limitations)

            if audit.hallucinated_pmids:
                self.tracer.event(
                    "guard.hallucinated_citations",
                    pmids=audit.hallucinated_pmids,
                    dropped_claims=len(dropped),
                )
                answer.notes.append(
                    f"{len(audit.hallucinated_pmids)} cited PMID(s) were not in the "
                    "retrieved set and were removed."
                )
            if dropped:
                answer.notes.append(
                    f"{len(dropped)} claim(s) were dropped for lacking verifiable citations."
                )
                answer.limitations.append(
                    "Some generated claims could not be verified against the retrieved "
                    "records and were removed from this answer."
                )

            answer.summary = _neutralise_advice(answer.summary)

            # Cite only what survived verification.
            cited_pmids = {pmid for claim in answer.claims for pmid in claim.pmids}
            answer.citations = [
                articles_by_pmid[pmid] for pmid in sorted(cited_pmids) if pmid in articles_by_pmid
            ]

            if not answer.claims:
                answer.overall_grade = EvidenceGrade.INSUFFICIENT
                if not answer.summary.strip():
                    answer.summary = (
                        "I could not produce a citation-backed answer from the records "
                        "retrieved for this question."
                    )

            if DISCLAIMER not in answer.summary:
                answer.notes.append(DISCLAIMER)

            span.annotate(
                claims_kept=len(answer.claims),
                claims_dropped=len(dropped),
                hallucinated_pmids=audit.hallucinated_pmids,
                citation_rate=round(audit.citation_rate, 3),
            )
            return answer


def _neutralise_advice(text: str) -> str:
    """Soften second-person clinical direction into descriptive phrasing."""
    cleaned = text
    for pattern in _ADVICE_PATTERNS:
        if pattern.search(cleaned):
            cleaned = pattern.sub("the literature suggests that patients may", cleaned)
    return cleaned
