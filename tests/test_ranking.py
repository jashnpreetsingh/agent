"""Tests for the evidence ranker (the RAG step)."""

from __future__ import annotations

from src.llm.errors import LLMTransportError
from src.tools.rank import EvidenceRanker, _cosine, _lexical_overlap, _recency
from tests.conftest import FakeProvider, make_article


def test_cosine_bounds():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_handles_degenerate_vectors():
    assert _cosine([], [1.0]) == 0.0
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert _cosine([1.0, 2.0], [1.0]) == 0.0


def test_recency_decays_and_floors():
    assert _recency(2024, 2024) == 1.0
    assert _recency(2012, 2024) == 0.0
    assert 0.0 < _recency(2018, 2024) < 1.0
    assert _recency(None, 2024) == 0.3


def test_lexical_overlap_ignores_stopwords():
    score = _lexical_overlap(
        "What are the treatments for diabetes?", "diabetes treatments include metformin"
    )
    assert score > 0.5


def test_ranking_prefers_stronger_design_at_equal_relevance():
    provider = FakeProvider()
    ranker = EvidenceRanker(provider)
    # Identical text so the embedding component is equal for both.
    weak = make_article("1", publication_types=["Case Reports"], abstract="same text")
    strong = make_article("2", publication_types=["Meta-Analysis"], abstract="same text")

    outcome = ranker.rank("question", [weak, strong], top_k=2)

    assert outcome.evidence[0].pmid == "2"


def test_ranking_deduplicates_pmids():
    ranker = EvidenceRanker(FakeProvider())
    article = make_article("42")
    outcome = ranker.rank("q", [article, article, article], top_k=5)
    assert len(outcome.evidence) == 1
    assert outcome.considered == 1


def test_char_budget_limits_selection():
    ranker = EvidenceRanker(FakeProvider())
    articles = [make_article(str(i), abstract="x" * 2000) for i in range(10)]
    outcome = ranker.rank("q", articles, top_k=10, char_budget=5000)
    assert 0 < len(outcome.evidence) < 10


def test_top_k_is_respected():
    ranker = EvidenceRanker(FakeProvider())
    articles = [make_article(str(i)) for i in range(10)]
    outcome = ranker.rank("q", articles, top_k=3)
    assert len(outcome.evidence) == 3


def test_falls_back_to_lexical_when_embeddings_fail():
    class BrokenEmbeddings(FakeProvider):
        def embed(self, texts, *, task_type="query"):
            raise LLMTransportError("embeddings unavailable", retryable=True)

    ranker = EvidenceRanker(BrokenEmbeddings())
    outcome = ranker.rank("diabetes treatment", [make_article("1"), make_article("2")], top_k=2)

    assert outcome.method == "lexical_fallback"
    assert len(outcome.evidence) == 2  # degraded, not empty


def test_empty_input_returns_empty():
    outcome = EvidenceRanker(FakeProvider()).rank("q", [], top_k=5)
    assert outcome.evidence == []
    assert outcome.method == "empty"
