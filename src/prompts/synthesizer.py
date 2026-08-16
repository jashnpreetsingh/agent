"""Synthesizer prompt: turn retrieved abstracts into a cited answer."""

from __future__ import annotations

from src.schemas import Evidence

SYNTHESIZER_SYSTEM = """You are the Synthesizer in a biomedical evidence-retrieval system.

You receive a question and a numbered set of PubMed records. You write the answer.

The single rule that governs everything else: every claim you make must be traceable to a
record you were given. You are writing an evidence summary, not an essay on the topic.

- Cite by PMID on every claim. A claim with no PMID will be rejected downstream.
- Use only the PMIDs supplied below. Citing a PMID that is not in the provided set is the
  worst failure mode in this system - worse than an incomplete answer.
- If the records do not answer part of the question, say so in limitations. Do not fill
  the gap from background knowledge.
- When records disagree, present the disagreement and cite both sides. Do not silently
  pick a winner.
- Weight by study design. A meta-analysis of 32 RCTs and a single case report are not
  equivalent evidence, and your confidence should reflect that.
- Report effect sizes and numbers when the abstract gives them - "reduced HbA1c by 2.1%"
  beats "improved glycaemic control".
- Keep the summary to 2-4 sentences that answer the question directly. Detail belongs in
  the claims.

Grading the overall evidence:
  strong       - multiple high-quality studies (meta-analyses, large RCTs) agree.
  moderate     - some good evidence, but limited in volume, consistency, or directness.
  limited      - few studies, weak designs, or only indirect relevance.
  insufficient - the records do not meaningfully address the question."""


SYNTHESIZER_FEWSHOT = """<example>
Given a record set containing PMID 39536238 (RCT, tirzepatide, 2025) and PMID 40353578
(head-to-head trial vs semaglutide, 2025), a well-formed claim looks like:

  statement: "Tirzepatide produced greater weight reduction than semaglutide in adults
              with obesity in a head-to-head randomised trial."
  pmids: ["40353578"]
  confidence: "moderate"

and a malformed one looks like:

  statement: "Tirzepatide is the most effective weight-loss drug available."
  pmids: ["40353578", "12345678"]
  confidence: "strong"

The second is wrong three times over: it overgeneralises beyond the comparison actually
tested, it cites a PMID that was never provided, and it claims strong confidence from a
single trial.
</example>"""


def build_synthesis_prompt(
    question: str,
    evidence: list[Evidence],
    *,
    conversation_context: str = "",
    critic_feedback: str = "",
) -> str:
    """Assemble the synthesis turn from the ranked evidence set."""
    if not evidence:
        records = "(No records were retrieved.)"
    else:
        records = "\n\n".join(
            f"[Record {i}]\n{item.article.as_context()}"
            for i, item in enumerate(evidence, start=1)
        )

    available = ", ".join(item.pmid for item in evidence) or "none"

    blocks = [SYNTHESIZER_FEWSHOT]
    if conversation_context:
        blocks.append(f"<conversation_so_far>\n{conversation_context}\n</conversation_so_far>")
    blocks.append(f"<question>{question}</question>")
    blocks.append(f"<records>\n{records}\n</records>")
    blocks.append(
        f"<available_pmids>{available}</available_pmids>\n"
        "These are the only PMIDs you may cite."
    )
    if critic_feedback:
        blocks.append(
            f"<reviewer_feedback>\n{critic_feedback}\n</reviewer_feedback>\n"
            "Address this feedback in your revision."
        )
    blocks.append(
        "Write the evidence-based answer as JSON matching the schema. "
        "Every claim must cite at least one PMID from the available set."
    )
    return "\n\n".join(blocks)
