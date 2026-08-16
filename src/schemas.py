"""Structured contracts between agent roles.

Each node in the graph consumes and emits one of these models. Because every
hand-off is a validated Pydantic object rather than free text, a malformed LLM
response fails loudly at the boundary instead of quietly corrupting the answer
three steps later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
class PublicationType(str, Enum):
    """PubMed publication types, ordered by evidence-hierarchy weight."""

    META_ANALYSIS = "Meta-Analysis"
    SYSTEMATIC_REVIEW = "Systematic Review"
    RANDOMIZED_CONTROLLED_TRIAL = "Randomized Controlled Trial"
    PRACTICE_GUIDELINE = "Practice Guideline"
    CLINICAL_TRIAL = "Clinical Trial"
    OBSERVATIONAL_STUDY = "Observational Study"
    REVIEW = "Review"
    CASE_REPORTS = "Case Reports"


#: Weight used when ranking evidence. Higher = stronger study design.
EVIDENCE_WEIGHT: dict[str, float] = {
    PublicationType.META_ANALYSIS.value: 1.00,
    PublicationType.SYSTEMATIC_REVIEW.value: 0.95,
    PublicationType.PRACTICE_GUIDELINE.value: 0.90,
    PublicationType.RANDOMIZED_CONTROLLED_TRIAL.value: 0.85,
    PublicationType.CLINICAL_TRIAL.value: 0.70,
    PublicationType.OBSERVATIONAL_STUDY.value: 0.55,
    PublicationType.REVIEW.value: 0.50,
    PublicationType.CASE_REPORTS.value: 0.25,
}
DEFAULT_EVIDENCE_WEIGHT = 0.40


class EvidenceGrade(str, Enum):
    """How much confidence the retrieved corpus supports overall."""

    STRONG = "strong"
    MODERATE = "moderate"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


class RefusalCategory(str, Enum):
    """Why a question was blocked by the input guardrail."""

    MEDICAL_EMERGENCY = "medical_emergency"
    PERSONAL_MEDICAL_ADVICE = "personal_medical_advice"
    OUT_OF_SCOPE = "out_of_scope"
    UNSAFE_CONTENT = "unsafe_content"
    MALFORMED_INPUT = "malformed_input"


# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------
class SearchSpec(BaseModel):
    """One concrete PubMed query the researcher may execute."""

    query: str = Field(description="PubMed query using standard field syntax.")
    rationale: str = Field(description="Why this query targets the question.")
    pub_types: list[PublicationType] = Field(
        default_factory=list,
        description="Publication types to prefer; empty means no filter.",
    )
    date_from: int | None = Field(
        default=None, description="Earliest publication year to include."
    )


class ResearchPlan(BaseModel):
    """The planner's decomposition of a user question."""

    interpretation: str = Field(
        description="Restatement of what the user is actually asking."
    )
    sub_questions: list[str] = Field(
        description="2-4 focused sub-questions that together answer the query."
    )
    mesh_terms: list[str] = Field(
        default_factory=list,
        description="Controlled-vocabulary MeSH headings relevant to the topic.",
    )
    searches: list[SearchSpec] = Field(
        description="Initial PubMed searches to run, most important first."
    )

    @field_validator("sub_questions", "searches")
    @classmethod
    def _non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("planner must produce at least one entry")
        return v


# ---------------------------------------------------------------------------
# Retrieval artefacts
# ---------------------------------------------------------------------------
class Article(BaseModel):
    """A PubMed record, as parsed from an efetch response."""

    pmid: str
    title: str
    abstract: str = ""
    journal: str = ""
    year: int | None = None
    publication_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def design_weight(self) -> float:
        """Best evidence-hierarchy weight among this record's pub types."""
        if not self.publication_types:
            return DEFAULT_EVIDENCE_WEIGHT
        return max(
            (EVIDENCE_WEIGHT.get(pt, DEFAULT_EVIDENCE_WEIGHT) for pt in self.publication_types),
            default=DEFAULT_EVIDENCE_WEIGHT,
        )

    def citation(self) -> str:
        # Authors are stored as "Surname Initials", so the surname is the
        # leading token - taking the last one yields "LJ et al.".
        first = self.authors[0].split()[0] if self.authors else "Anon"
        suffix = " et al." if len(self.authors) > 1 else ""
        return f"{first}{suffix} ({self.year or 'n.d.'}), {self.journal or 'Unknown journal'}. PMID {self.pmid}"

    def as_context(self, max_abstract_chars: int = 2200) -> str:
        """Render the record for inclusion in a synthesis prompt."""
        abstract = self.abstract or "[No abstract available]"
        if len(abstract) > max_abstract_chars:
            abstract = abstract[:max_abstract_chars].rstrip() + " […truncated]"
        types = ", ".join(self.publication_types) or "Unclassified"
        return (
            f"PMID: {self.pmid}\n"
            f"Title: {self.title}\n"
            f"Journal/Year: {self.journal or 'Unknown'} ({self.year or 'n.d.'})\n"
            f"Publication types: {types}\n"
            f"Abstract: {abstract}"
        )


class Evidence(BaseModel):
    """An article selected for synthesis, with its ranking provenance."""

    article: Article
    relevance: float = Field(description="Cosine similarity to the question.")
    score: float = Field(description="Combined relevance + design + recency score.")
    matched_sub_question: str | None = None

    @property
    def pmid(self) -> str:
        return self.article.pmid


# ---------------------------------------------------------------------------
# Synthesis + critique
# ---------------------------------------------------------------------------
class Claim(BaseModel):
    """A single assertion, bound to the PMIDs that support it."""

    statement: str = Field(description="One factual claim, self-contained.")
    pmids: list[str] = Field(
        description="PMIDs from the provided evidence that support this claim."
    )
    confidence: EvidenceGrade = Field(
        default=EvidenceGrade.MODERATE,
        description="Confidence given the supporting study designs.",
    )

    @field_validator("pmids")
    @classmethod
    def _clean_pmids(cls, v: list[str]) -> list[str]:
        # Models occasionally emit "PMID: 12345" or "12345." - normalise early
        # so the citation verifier compares like with like.
        cleaned = []
        for raw in v:
            digits = "".join(ch for ch in str(raw) if ch.isdigit())
            if digits:
                cleaned.append(digits)
        return cleaned


class DraftAnswer(BaseModel):
    """The synthesizer's structured answer, before verification."""

    summary: str = Field(description="Direct 2-4 sentence answer to the question.")
    claims: list[Claim] = Field(description="Supporting claims, each cited.")
    limitations: list[str] = Field(
        default_factory=list,
        description="Gaps, conflicts, or caveats in the retrieved evidence.",
    )
    overall_grade: EvidenceGrade = EvidenceGrade.MODERATE


class CritiqueVerdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"


class Critique(BaseModel):
    """The critic's assessment of a draft answer."""

    verdict: CritiqueVerdict
    reasoning: str = Field(description="Why the draft passes or fails.")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claim statements not backed by the supplied evidence.",
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="Parts of the question the draft leaves unanswered.",
    )
    followup_queries: list[str] = Field(
        default_factory=list,
        description="PubMed queries that would close the gaps, if revising.",
    )


# ---------------------------------------------------------------------------
# Guardrails + final output
# ---------------------------------------------------------------------------
class GuardDecision(BaseModel):
    """Outcome of an input guardrail check."""

    allowed: bool
    category: RefusalCategory | None = None
    reason: str = ""
    user_message: str = ""


class CitationAudit(BaseModel):
    """Deterministic verification that citations refer to retrieved records."""

    total_claims: int = 0
    cited_claims: int = 0
    uncited_claims: list[str] = Field(default_factory=list)
    hallucinated_pmids: list[str] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.hallucinated_pmids and not self.uncited_claims

    @property
    def citation_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.cited_claims / self.total_claims


class AgentAnswer(BaseModel):
    """The final object returned to the caller."""

    run_id: str
    question: str
    answered: bool
    summary: str
    claims: list[Claim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    citations: list[Article] = Field(default_factory=list)
    overall_grade: EvidenceGrade = EvidenceGrade.INSUFFICIENT
    refusal_category: RefusalCategory | None = None
    citation_audit: CitationAudit = Field(default_factory=CitationAudit)
    degraded: bool = Field(
        default=False,
        description="True when a fallback path produced this answer.",
    )
    notes: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_markdown(self) -> str:
        """Human-readable rendering used by the CLI and the run report."""
        lines: list[str] = [f"# {self.question}", ""]
        if not self.answered:
            lines += [f"**Not answered** ({self.refusal_category.value if self.refusal_category else 'unknown'})", "", self.summary]
            return "\n".join(lines)

        lines += [self.summary, "", f"**Evidence grade:** {self.overall_grade.value}", ""]
        if self.claims:
            lines.append("## Findings")
            for claim in self.claims:
                refs = ", ".join(f"PMID {p}" for p in claim.pmids) or "uncited"
                lines.append(f"- {claim.statement} _({refs})_")
            lines.append("")
        if self.limitations:
            lines.append("## Limitations")
            lines += [f"- {item}" for item in self.limitations]
            lines.append("")
        if self.citations:
            lines.append("## Sources")
            for art in self.citations:
                lines.append(f"- {art.citation()} — {art.url}")
            lines.append("")
        lines.append(
            "---\n_Research summary of published literature. Not medical advice; "
            "consult a qualified clinician for individual care._"
        )
        return "\n".join(lines)
