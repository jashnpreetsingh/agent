"""Input guardrails.

Three checks run before any budget is spent:

1. **Validation** - length and content sanity.
2. **Safety triage** - medical emergencies and requests for personal medical
   advice.
3. **Scope** - is this a biomedical literature question at all?

Checks 1 and 2 are deliberately deterministic. A regex cannot be prompt-injected,
costs nothing, and still works when the LLM API is down - and "the safety check
was skipped because the provider returned 503" is not an acceptable failure mode.
Check 3 is fuzzier and genuinely benefits from a model, so it uses one, and
fails *open* into the normal pipeline (where the answer stays evidence-grounded
anyway) rather than blocking a legitimate question on an API error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.llm.base import ChatMessage, LLMProvider
from src.llm.errors import LLMError
from src.observability.trace import Tracer
from src.schemas import GuardDecision, RefusalCategory

MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 2000


@dataclass(frozen=True, slots=True)
class Rule:
    """A named detection pattern."""

    name: str
    pattern: re.Pattern[str]


def _compile(name: str, *alternatives: str) -> Rule:
    return Rule(name=name, pattern=re.compile("|".join(alternatives), re.IGNORECASE))


#: Symptoms that need emergency care now, not a literature review.
EMERGENCY_RULES: tuple[Rule, ...] = (
    _compile(
        "self_harm",
        r"\b(kill (myself|him|her|them)|suicidal|suicide|end my life|want to die)\b",
        r"\b(harm myself|hurt myself|self[- ]harm)\b",
    ),
    _compile(
        "cardiac",
        r"\b(chest pain|crushing chest|heart attack)\b.{0,40}\b(now|having|right now|currently|i am|i'm)\b",
        r"\b(i am|i'm|having)\b.{0,30}\b(chest pain|heart attack)\b",
    ),
    _compile(
        "stroke",
        r"\b(face (is )?drooping|slurred speech|sudden numbness|can'?t move (my|his|her) (arm|leg|side))\b",
    ),
    _compile(
        "respiratory",
        r"\b(can'?t breathe|cannot breathe|struggling to breathe|stopped breathing)\b",
    ),
    _compile(
        "overdose",
        r"\b(overdosed?|took too many pills|poisoned|swallowed bleach)\b",
    ),
    _compile(
        "obstetric_bleeding",
        r"\b(bleeding heavily|hemorrhaging|haemorrhaging|won'?t stop bleeding)\b",
    ),
    _compile("unresponsive", r"\b(unconscious|unresponsive|passed out and|not waking up)\b"),
)

#: Requests for individualised clinical decisions.
PERSONAL_ADVICE_RULES: tuple[Rule, ...] = (
    _compile(
        "should_i",
        r"\b(should|shall|can|must|ought) (i|my (husband|wife|son|daughter|mother|father|child|partner))\b",
        r"\bwhat should i (take|do|use|try)\b",
    ),
    _compile(
        "my_case",
        r"\b(i (have|was|am) (been )?(diagnosed|suffering|experiencing)|my (doctor|diagnosis|condition|symptoms|results?|scan|biopsy))\b",
        r"\bi'?m (currently )?(taking|on|prescribed)\b",
    ),
    _compile(
        "dosing",
        r"\b(how (much|many) (should|do) i|what dose (should|do) i|is it safe for me to)\b",
        r"\b(can i (take|combine|stop|mix)|should i stop taking)\b",
    ),
    _compile(
        "diagnose_me",
        r"\b(do i have|what'?s wrong with me|am i (having|going to)|diagnose me)\b",
    ),
)


class ScopeTriage(BaseModel):
    """LLM judgement on whether a question suits a PubMed literature search."""

    in_scope: bool = Field(
        description="True if answerable from published biomedical or clinical literature."
    )
    reason: str = Field(description="One sentence explaining the judgement.")


class InputGuard:
    """Screens a question before the agent spends any budget on it."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        tracer: Tracer | None = None,
        *,
        use_llm_scope_check: bool = True,
    ) -> None:
        self.provider = provider
        self.tracer = tracer or Tracer.null()
        self.use_llm_scope_check = use_llm_scope_check and provider is not None

    def check(self, question: str) -> GuardDecision:
        """Return the decision for ``question``."""
        with self.tracer.span("guard.input", chars=len(question)) as span:
            decision = self._check(question)
            span.annotate(
                allowed=decision.allowed,
                category=decision.category.value if decision.category else None,
                reason=decision.reason,
            )
            return decision

    # ------------------------------------------------------------------
    def _check(self, question: str) -> GuardDecision:
        text = (question or "").strip()

        # 1. Structural validation.
        if len(text) < MIN_QUESTION_CHARS:
            return GuardDecision(
                allowed=False,
                category=RefusalCategory.MALFORMED_INPUT,
                reason=f"question is under {MIN_QUESTION_CHARS} characters",
                user_message=(
                    "That question is too short for me to research. Please describe what "
                    "you would like to know in a full sentence."
                ),
            )
        if len(text) > MAX_QUESTION_CHARS:
            return GuardDecision(
                allowed=False,
                category=RefusalCategory.MALFORMED_INPUT,
                reason=f"question exceeds {MAX_QUESTION_CHARS} characters",
                user_message=(
                    f"That question is {len(text)} characters, above the "
                    f"{MAX_QUESTION_CHARS}-character limit. Please shorten it to the "
                    "specific clinical question you want evidence on."
                ),
            )

        # 2. Emergencies first: this outranks every other consideration.
        if rule := _first_match(text, EMERGENCY_RULES):
            return GuardDecision(
                allowed=False,
                category=RefusalCategory.MEDICAL_EMERGENCY,
                reason=f"emergency pattern matched: {rule}",
                user_message=(
                    "This sounds like it may be a medical emergency, and I am a literature "
                    "search tool - I cannot help with urgent care.\n\n"
                    "**Please contact emergency services now** (911 in the US, 999 in the UK, "
                    "112 in the EU) or go to your nearest emergency department.\n\n"
                    "If you are having thoughts of harming yourself, you can reach the 988 "
                    "Suicide & Crisis Lifeline (US) by calling or texting 988, or the "
                    "Samaritans (UK) on 116 123. Both are free and available 24/7."
                ),
            )

        # 3. Personal clinical decisions.
        if rule := _first_match(text, PERSONAL_ADVICE_RULES):
            return GuardDecision(
                allowed=False,
                category=RefusalCategory.PERSONAL_MEDICAL_ADVICE,
                reason=f"personal-advice pattern matched: {rule}",
                user_message=(
                    "I can summarise what the published literature says, but I cannot advise "
                    "on your individual care - that depends on your history, other "
                    "medications, and test results, which only your clinician can weigh.\n\n"
                    "Please raise this with your doctor or pharmacist. If it would help, I "
                    "can research the general evidence instead: ask something like *\"What "
                    "does the evidence say about the efficacy and risks of <treatment> for "
                    "<condition>?\"* and I will summarise the studies with citations."
                ),
            )

        # 4. Scope, where a model genuinely helps.
        if self.use_llm_scope_check:
            triage = self._llm_scope_check(text)
            if triage is not None and not triage.in_scope:
                return GuardDecision(
                    allowed=False,
                    category=RefusalCategory.OUT_OF_SCOPE,
                    reason=triage.reason,
                    user_message=(
                        "That question falls outside what I can research. I answer "
                        "biomedical and clinical questions using published literature from "
                        "PubMed - for example treatment efficacy, drug safety, diagnostic "
                        "accuracy, or disease mechanisms.\n\n"
                        f"({triage.reason})"
                    ),
                )

        return GuardDecision(allowed=True, reason="passed all input checks")

    def _llm_scope_check(self, question: str) -> ScopeTriage | None:
        """Classify scope, returning ``None`` when the check cannot run."""
        assert self.provider is not None
        try:
            return self.provider.generate_structured(
                [
                    ChatMessage.user(
                        "Decide whether this question can be answered from published "
                        "biomedical or clinical literature indexed in PubMed.\n\n"
                        f"Question: {question}\n\n"
                        "In scope: diseases, treatments, drugs, diagnostics, physiology, "
                        "epidemiology, public health, biology, and clinical research "
                        "methods - including general questions about how a treatment works "
                        "or what the evidence shows.\n"
                        "Out of scope: programming, finance, travel, general trivia, and "
                        "anything with no biomedical content."
                    )
                ],
                ScopeTriage,
                system=(
                    "You classify questions for a PubMed research agent. Be permissive: "
                    "only mark out_of_scope when the question has no biomedical content at "
                    "all. Wrongly rejecting a real clinical question is worse than "
                    "researching a borderline one."
                ),
                temperature=0.0,
                purpose="guard.scope",
            )
        except LLMError as exc:
            # Fail open: the downstream pipeline is still citation-grounded, so
            # an unclassifiable question degrades to a normal (possibly empty)
            # literature search rather than a false rejection.
            self.tracer.event("guard.scope_check_failed", detail=str(exc)[:200], action="fail_open")
            return None


def _first_match(text: str, rules: tuple[Rule, ...]) -> str | None:
    for rule in rules:
        if rule.pattern.search(text):
            return rule.name
    return None
