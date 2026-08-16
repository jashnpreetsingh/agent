"""Critic prompt: the self-reflection pass over a draft answer."""

from __future__ import annotations

from src.schemas import DraftAnswer, Evidence

CRITIC_SYSTEM = """You are the Critic in a biomedical evidence-retrieval system.

You review a draft answer against the evidence it was built from. You are the last check
before a user sees it, and your bias is toward catching problems, not toward approving.

Check, in this order:

1. Grounding. Does each claim actually follow from the cited record's abstract? A claim
   that overstates, generalises beyond the studied population, or asserts something the
   abstract never says is unsupported - even when the PMID is real and relevant.
2. Citation validity. Does every claim cite at least one PMID from the evidence set?
3. Completeness. Does the answer address the question that was asked, or only part of it?
4. Calibration. Does stated confidence match study design and volume? A single small
   trial does not support "strong".
5. Honesty about gaps. If evidence is thin or conflicting, does the answer say so?

Verdict:
- "accept" - grounded and responsive. Minor wording issues are not grounds to revise.
- "revise" - a claim is unsupported, a material part of the question is unanswered, or
  confidence is badly miscalibrated.

Choose "revise" only when more searching or rewriting would plausibly fix the problem. If
the literature simply does not contain the answer, that is a legitimate outcome: accept
the draft and let its limitations section carry the message. Requesting a revision that
cannot succeed just burns budget.

When you request a revision, always supply followup_queries that would close the gap."""


def build_critic_prompt(question: str, draft: DraftAnswer, evidence: list[Evidence]) -> str:
    """Assemble the critique turn.

    The critic receives the **full abstracts**, not a title index. Its first
    job is deciding whether each claim follows from the text it cites, and a
    reviewer shown only titles cannot do that: it correctly observes that a
    title does not report an effect size, marks every quantitative claim
    unsupported, and demands revisions no rewrite can satisfy. Grounding
    review costs context by nature - the alternative is a critic that always
    says no.
    """
    available = "\n\n".join(
        f"[Record {i}]\n{item.article.as_context()}"
        for i, item in enumerate(evidence, start=1)
    ) or "  (none)"

    claims = "\n".join(
        f"  {i}. {claim.statement}\n     cites: {', '.join(claim.pmids) or 'NOTHING'} "
        f"| confidence: {claim.confidence.value}"
        for i, claim in enumerate(draft.claims, start=1)
    ) or "  (no claims)"

    limitations = "\n".join(f"  - {lim}" for lim in draft.limitations) or "  (none stated)"

    return (
        f"<question>{question}</question>\n\n"
        f"<evidence>\nThese are the full records the draft was written from. Judge each "
        f"claim against the abstract text below.\n\n{available}\n</evidence>\n\n"
        f"<draft_summary>{draft.summary}</draft_summary>\n\n"
        f"<draft_claims>\n{claims}\n</draft_claims>\n\n"
        f"<draft_limitations>\n{limitations}\n</draft_limitations>\n\n"
        f"<draft_grade>{draft.overall_grade.value}</draft_grade>\n\n"
        "Review this draft and return your critique as JSON matching the schema."
    )
