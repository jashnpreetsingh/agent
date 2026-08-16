"""Evidence ranking: the retrieval-augmented step.

PubMed's own relevance ordering is lexical and knows nothing about the user's
actual question, so raw ``esearch`` output is a poor context window. This
module re-ranks candidate articles semantically and selects a budgeted subset
for synthesis.

The score blends three signals rather than similarity alone:

``relevance`` (0.65)  cosine similarity between question and abstract;
``design``    (0.20)  evidence-hierarchy weight - a meta-analysis outranks a
                      case report at equal similarity;
``recency``   (0.15)  linear decay over a 12-year window, since "latest
                      treatment options" should not surface 1998 trials.

If the embedding API fails the ranker degrades to a lexical overlap score
instead of raising: a slightly worse ordering beats no answer at all.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from src.llm.base import LLMProvider
from src.llm.errors import LLMError
from src.observability.trace import Tracer
from src.schemas import Article, Evidence

#: Score component weights; must sum to 1.0.
W_RELEVANCE = 0.65
W_DESIGN = 0.20
W_RECENCY = 0.15

#: Publications older than this contribute no recency credit.
RECENCY_WINDOW_YEARS = 12

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the to was were
    what which with how does do can when who whom this these those there their been being""".split()
)


@dataclass(slots=True)
class RankingOutcome:
    """Ranked evidence plus a note on how it was produced."""

    evidence: list[Evidence]
    method: str
    considered: int


class EvidenceRanker:
    """Re-ranks and selects articles for the synthesis context window."""

    def __init__(self, provider: LLMProvider, tracer: Tracer | None = None) -> None:
        self.provider = provider
        self.tracer = tracer or Tracer.null()

    def rank(
        self,
        question: str,
        articles: list[Article],
        *,
        top_k: int = 8,
        char_budget: int = 24_000,
        current_year: int | None = None,
    ) -> RankingOutcome:
        """Return the best ``top_k`` articles that fit within ``char_budget``."""
        if not articles:
            return RankingOutcome(evidence=[], method="empty", considered=0)

        year_now = current_year or datetime.now(timezone.utc).year
        deduped = _dedupe(articles)
        relevances, method = self._relevance_scores(question, deduped)

        scored: list[Evidence] = []
        for article, relevance in zip(deduped, relevances):
            score = (
                W_RELEVANCE * relevance
                + W_DESIGN * article.design_weight
                + W_RECENCY * _recency(article.year, year_now)
            )
            scored.append(
                Evidence(article=article, relevance=round(relevance, 4), score=round(score, 4))
            )

        scored.sort(key=lambda e: e.score, reverse=True)
        selected = _apply_budget(scored, top_k=top_k, char_budget=char_budget)

        self.tracer.event(
            "rank.complete",
            method=method,
            considered=len(deduped),
            selected=len(selected),
            top_scores=[{"pmid": e.pmid, "score": e.score} for e in selected[:5]],
        )
        return RankingOutcome(evidence=selected, method=method, considered=len(deduped))

    # ------------------------------------------------------------------
    def _relevance_scores(self, question: str, articles: list[Article]) -> tuple[list[float], str]:
        """Cosine similarity via embeddings, falling back to lexical overlap.

        The question and the abstracts are embedded in *separate* calls with
        different task types. Retrieval embedding models are asymmetric: they
        encode a short question and a long passage differently on purpose, and
        embedding both sides identically quietly costs ranking quality.
        """
        documents = [f"{a.title}\n{a.abstract[:4000]}" for a in articles]
        try:
            question_vec = self.provider.embed([question], task_type="query")[0]
            doc_vecs = self.provider.embed(documents, task_type="document")
            # Cosine lives in [-1, 1]; rescale so the blend stays in [0, 1].
            scores = [(_cosine(question_vec, dv) + 1.0) / 2.0 for dv in doc_vecs]
            return scores, "embedding_cosine"
        except (LLMError, ValueError, IndexError) as exc:
            self.tracer.event("rank.embedding_failed", detail=str(exc)[:250], fallback="lexical")
            return [_lexical_overlap(question, doc) for doc in documents], "lexical_fallback"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_overlap(question: str, document: str) -> float:
    """Jaccard-style overlap on content words; the no-embeddings fallback."""
    q_tokens = {t for t in _TOKEN_RE.findall(question.lower()) if t not in _STOPWORDS and len(t) > 2}
    d_tokens = {t for t in _TOKEN_RE.findall(document.lower()) if t not in _STOPWORDS and len(t) > 2}
    if not q_tokens or not d_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)


def _recency(year: int | None, current_year: int) -> float:
    if not year:
        return 0.3  # unknown date: neither rewarded nor heavily punished
    age = max(0, current_year - year)
    if age >= RECENCY_WINDOW_YEARS:
        return 0.0
    return 1.0 - (age / RECENCY_WINDOW_YEARS)


def _dedupe(articles: list[Article]) -> list[Article]:
    """Drop repeat PMIDs, which overlapping searches produce routinely."""
    seen: set[str] = set()
    unique: list[Article] = []
    for article in articles:
        if article.pmid not in seen:
            seen.add(article.pmid)
            unique.append(article)
    return unique


def _apply_budget(scored: list[Evidence], *, top_k: int, char_budget: int) -> list[Evidence]:
    """Take the highest-scoring evidence that fits the context budget.

    Enforcing a character budget here is what keeps the synthesis prompt from
    growing without bound as retrieval widens.
    """
    selected: list[Evidence] = []
    used = 0
    for item in scored:
        if len(selected) >= top_k:
            break
        cost = len(item.article.as_context())
        if used + cost > char_budget and selected:
            continue
        selected.append(item)
        used += cost
    return selected
