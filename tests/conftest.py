"""Shared test fixtures.

The fakes here let the entire graph run in milliseconds with no API key and no
network, which is what makes the loop itself testable rather than only its
parts.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from src.config import Settings, TransportMode
from src.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolCall, Usage
from src.schemas import (
    Article,
    Claim,
    Critique,
    CritiqueVerdict,
    DraftAnswer,
    EvidenceGrade,
    ResearchPlan,
    SearchSpec,
)
from src.tools.pubmed import SearchResult
from src.tools.registry import ToolRegistry

SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>N Engl J Med</ISOAbbreviation>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Tirzepatide and <i>glycaemic</i> control in type 2 diabetes</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Tirzepatide is a dual agonist.</AbstractText>
          <AbstractText Label="RESULTS">HbA1c fell by 2.1% versus placebo.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Aronne</LastName><Initials>LJ</Initials></Author>
          <Author><LastName>Smith</LastName><Initials>AB</Initials></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Randomized Controlled Trial</PublicationType>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Diabetes Mellitus, Type 2</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
        <ArticleId IdType="doi">10.1056/REAL-DOI</ArticleId>
      </ArticleIdList>
      <ReferenceList>
        <Reference>
          <ArticleIdList>
            <ArticleId IdType="doi">10.9999/WRONG-REFERENCE-DOI</ArticleId>
          </ArticleIdList>
        </Reference>
      </ReferenceList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>87654321</PMID>
      <Article>
        <Journal>
          <ISOAbbreviation>Lancet</ISOAbbreviation>
          <JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>SGLT2 inhibitors and cardiovascular outcomes</ArticleTitle>
        <Abstract><AbstractText>Reduced heart failure hospitalisation.</AbstractText></Abstract>
        <PublicationTypeList>
          <PublicationType>Meta-Analysis</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def make_article(pmid: str, **overrides: Any) -> Article:
    data: dict[str, Any] = {
        "pmid": pmid,
        "title": f"Study {pmid}",
        "abstract": f"Abstract text for {pmid} about diabetes treatment outcomes.",
        "journal": "J Test",
        "year": 2024,
        "publication_types": ["Randomized Controlled Trial"],
        "authors": ["Doe J"],
    }
    data.update(overrides)
    return Article(**data)


class FakePubMed:
    """Stands in for :class:`PubMedClient` with deterministic results."""

    def __init__(self, articles: list[Article] | None = None) -> None:
        # `articles or [...]` would treat an explicit empty list as "not
        # supplied" and substitute the defaults, silently breaking the
        # no-results test.
        if articles is None:
            articles = [make_article("11111111"), make_article("22222222")]
        self.articles = articles
        self.searches: list[str] = []
        self.fail_next = False

    def search(self, query: str, **kwargs: Any) -> SearchResult:
        self.searches.append(query)
        return SearchResult(
            query=query,
            total_count=len(self.articles),
            pmids=[a.pmid for a in self.articles],
            translated_query=query,
        )

    def fetch(self, pmids: list[str]) -> list[Article]:
        return [a for a in self.articles if a.pmid in pmids]

    def lookup_mesh(self, term: str, **kwargs: Any) -> list[str]:
        return ["Diabetes Mellitus, Type 2"]


class FakeProvider(LLMProvider):
    """Scripted LLM: canned structured outputs and one tool-calling round."""

    def __init__(
        self,
        *,
        draft: DraftAnswer | None = None,
        verdict: CritiqueVerdict = CritiqueVerdict.ACCEPT,
        tool_rounds: int = 1,
    ) -> None:
        self.draft = draft
        self.verdict = verdict
        self.tool_rounds = tool_rounds
        self.calls: list[str] = []
        self.embed_calls = 0

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        force_tool_use: bool = False,
        temperature: float | None = None,
        purpose: str = "generate",
    ) -> LLMResponse:
        self.calls.append(purpose)
        rounds_done = sum(1 for m in messages if m.role == "tool")
        if rounds_done < self.tool_rounds:
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        name="pubmed_search",
                        arguments={"query": "type 2 diabetes treatment", "max_results": 5},
                        thought_signature="sig-abc",
                    )
                ],
                usage=Usage(prompt_tokens=10, output_tokens=5, total_tokens=15),
            )
        return LLMResponse(text="Found relevant trials covering all sub-questions.")

    def generate_structured(
        self,
        messages: list[ChatMessage],
        response_model: type[BaseModel],
        *,
        system: str | None = None,
        temperature: float | None = None,
        purpose: str = "structured",
    ) -> Any:
        self.calls.append(purpose)
        if response_model is ResearchPlan:
            return ResearchPlan(
                interpretation="Treatment options for T2D.",
                sub_questions=["Which drug classes are first-line?"],
                mesh_terms=["Diabetes Mellitus, Type 2"],
                searches=[SearchSpec(query="diabetes treatment", rationale="core query")],
            )
        if response_model is DraftAnswer:
            return self.draft or DraftAnswer(
                summary="Metformin remains first-line therapy.",
                claims=[
                    Claim(
                        statement="Metformin is first-line therapy.",
                        pmids=["11111111"],
                        confidence=EvidenceGrade.MODERATE,
                    )
                ],
                limitations=["Limited to two records."],
                overall_grade=EvidenceGrade.MODERATE,
            )
        if response_model is Critique:
            return Critique(
                verdict=self.verdict,
                reasoning="Checked against the evidence.",
                missing_aspects=["cardiovascular outcomes"] if self.verdict is CritiqueVerdict.REVISE else [],
                followup_queries=["SGLT2 cardiovascular"] if self.verdict is CritiqueVerdict.REVISE else [],
            )
        raise AssertionError(f"unexpected model {response_model}")

    def embed(self, texts: list[str], *, task_type: str = "query") -> list[list[float]]:
        self.embed_calls += 1
        # Deterministic pseudo-embeddings: length-derived, stable across runs.
        return [[float(len(t) % 7), float(len(t) % 5), 1.0] for t in texts]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        gemini_api_key="test-key",
        llm_mode=TransportMode.REPLAY,
        pubmed_mode=TransportMode.REPLAY,
        fixtures_dir=tmp_path / "fixtures",
        traces_dir=tmp_path / "traces",
        memory_db=tmp_path / "memory.sqlite3",
        max_research_iterations=3,
        max_tool_calls=5,
    )


@pytest.fixture
def fake_pubmed() -> FakePubMed:
    return FakePubMed()


@pytest.fixture
def registry(fake_pubmed: FakePubMed, settings: Settings) -> ToolRegistry:
    return ToolRegistry(fake_pubmed, settings)  # type: ignore[arg-type]
