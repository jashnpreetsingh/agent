"""Tool declarations and dispatch.

One Pydantic model per tool defines both the validation rules *and* the
function schema advertised to the model, so the two can never drift apart.

Dispatch never raises into the agent loop. A failed tool returns a
``ToolResult`` with ``ok=False`` and a message written for the model to read,
because "that query was malformed, try narrower terms" is something the
researcher node can actually recover from - whereas an exception ends the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from src.config import Settings
from src.llm.schema_utils import to_gemini_schema
from src.observability.trace import Tracer
from src.schemas import Article, PublicationType
from src.tools.pubmed import PubMedClient, PubMedError


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------
class PubMedSearchArgs(BaseModel):
    """Arguments for ``pubmed_search``."""

    query: str = Field(
        description=(
            "PubMed search expression. Field tags are supported, e.g. "
            '\'"Diabetes Mellitus, Type 2"[MeSH Terms] AND tirzepatide\'.'
        )
    )
    max_results: int = Field(
        default=10, ge=1, le=25, description="How many records to retrieve (1-25)."
    )
    date_from: int | None = Field(
        default=None, description="Restrict to publications from this year onward."
    )
    pub_types: list[PublicationType] = Field(
        default_factory=list,
        description="Restrict to these study designs. Omit to search all designs.",
    )


class MeshLookupArgs(BaseModel):
    """Arguments for ``mesh_lookup``."""

    term: str = Field(
        description="A word or phrase to map onto controlled MeSH vocabulary, "
        "e.g. 'sugar disease' or 'heart attack'."
    )


class FetchAbstractsArgs(BaseModel):
    """Arguments for ``fetch_abstracts``."""

    pmids: list[str] = Field(
        description="PMIDs to retrieve full abstracts for; from a prior search.",
        min_length=1,
        max_length=25,
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ToolResult:
    """Outcome of a single tool invocation."""

    name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0

    def for_model(self) -> dict[str, Any]:
        """The payload handed back to the LLM as a function response."""
        if self.ok:
            return self.data
        return {"error": self.error, "hint": "Adjust the arguments and try a different query."}


@dataclass(slots=True)
class ToolSpec:
    """A callable tool plus the schema advertised to the model."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], dict[str, Any]]

    def declaration(self) -> dict[str, Any]:
        """Gemini ``functionDeclaration`` for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": to_gemini_schema(self.args_model.model_json_schema()),
        }


class ToolRegistry:
    """Holds the tool set and the articles retrieved during a run."""

    def __init__(self, pubmed: PubMedClient, settings: Settings, tracer: Tracer | None = None) -> None:
        self.pubmed = pubmed
        self.settings = settings
        self.tracer = tracer or Tracer.null()
        #: Every article seen this run, keyed by PMID. The ranker and the
        #: citation verifier both read from here, which guarantees they judge
        #: the same corpus the model was shown.
        self.article_store: dict[str, Article] = {}
        self.call_count = 0
        self._specs: dict[str, ToolSpec] = {}
        self._register_defaults()

    # ------------------------------------------------------------------
    def _register_defaults(self) -> None:
        self.register(
            ToolSpec(
                name="pubmed_search",
                description=(
                    "Search PubMed for biomedical literature. Returns matching records "
                    "with PMID, title, year, and study design. Use MeSH terms and "
                    "publication-type filters to target high-quality evidence."
                ),
                args_model=PubMedSearchArgs,
                handler=self._handle_search,
            )
        )
        self.register(
            ToolSpec(
                name="mesh_lookup",
                description=(
                    "Map everyday or ambiguous wording onto official MeSH headings. "
                    "Call this before searching when the question uses lay language."
                ),
                args_model=MeshLookupArgs,
                handler=self._handle_mesh,
            )
        )
        self.register(
            ToolSpec(
                name="fetch_abstracts",
                description=(
                    "Retrieve the full abstracts of specific PMIDs returned by an "
                    "earlier search, when titles alone are not enough to judge relevance."
                ),
                args_model=FetchAbstractsArgs,
                handler=self._handle_fetch,
            )
        )

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    # ------------------------------------------------------------------
    def declarations(self) -> list[dict[str, Any]]:
        """Function declarations for every registered tool."""
        return [spec.declaration() for spec in self._specs.values()]

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Validate arguments and dispatch, converting failures into results."""
        started = time.perf_counter()
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(
                name=name,
                ok=False,
                error=f"Unknown tool '{name}'. Available tools: {', '.join(self._specs)}.",
            )

        with self.tracer.span("tool.call", tool=name, arguments=arguments) as span:
            try:
                args = spec.args_model.model_validate(arguments)
            except ValidationError as exc:
                # Hand the model the actual validation message: it usually
                # repairs its own arguments on the next turn.
                result = ToolResult(
                    name=name,
                    ok=False,
                    error=f"Invalid arguments for {name}: {exc.errors(include_url=False)}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                span.annotate(ok=False, error=result.error[:300])
                return result

            self.call_count += 1
            try:
                data = spec.handler(args)
                result = ToolResult(
                    name=name, ok=True, data=data,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                span.annotate(ok=True, result_summary=_summarise(data))
            except PubMedError as exc:
                result = ToolResult(
                    name=name, ok=False, error=str(exc),
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                span.annotate(ok=False, error=str(exc)[:300])
            return result

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _handle_search(self, args: PubMedSearchArgs) -> dict[str, Any]:
        """Search, then fetch the matching records in one round trip.

        The model gets a compact index (title/year/design) rather than full
        abstracts: enough to steer the next step, cheap in tokens. Full text is
        already cached in ``article_store`` for the ranker.
        """
        search = self.pubmed.search(
            args.query,
            max_results=min(args.max_results, self.settings.max_articles_per_search),
            date_from=args.date_from,
            pub_types=[pt.value for pt in args.pub_types],
        )
        articles = self.pubmed.fetch(search.pmids) if search.pmids else []
        for article in articles:
            self.article_store[article.pmid] = article

        return {
            "executed_query": search.query,
            "total_matches": search.total_count,
            "returned": len(articles),
            "results": [
                {
                    "pmid": a.pmid,
                    "title": a.title,
                    "year": a.year,
                    "journal": a.journal,
                    "publication_types": a.publication_types,
                    "has_abstract": bool(a.abstract),
                }
                for a in articles
            ],
            "note": (
                "No records matched. Try broader terms, remove filters, or widen the date range."
                if not articles
                else "Abstracts are retained for synthesis; no further fetch is required."
            ),
        }

    def _handle_mesh(self, args: MeshLookupArgs) -> dict[str, Any]:
        descriptors = self.pubmed.lookup_mesh(args.term)
        return {
            "term": args.term,
            "mesh_descriptors": descriptors,
            "usage_hint": (
                'Use as \'"<descriptor>"[MeSH Terms]\' in a search.'
                if descriptors
                else "No MeSH heading matched; search with free text instead."
            ),
        }

    def _handle_fetch(self, args: FetchAbstractsArgs) -> dict[str, Any]:
        wanted = [p for p in args.pmids if p not in self.article_store]
        if wanted:
            for article in self.pubmed.fetch(wanted):
                self.article_store[article.pmid] = article

        found = [self.article_store[p] for p in args.pmids if p in self.article_store]
        missing = [p for p in args.pmids if p not in self.article_store]
        return {
            "articles": [
                {
                    "pmid": a.pmid,
                    "title": a.title,
                    "year": a.year,
                    "publication_types": a.publication_types,
                    "abstract": a.abstract[:3000] or "[No abstract available]",
                }
                for a in found
            ],
            "missing_pmids": missing,
        }


def _summarise(data: dict[str, Any]) -> dict[str, Any]:
    """Compact tool output for the trace, dropping bulky text."""
    summary: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, list):
            summary[key] = f"[{len(value)} items]"
        elif isinstance(value, str) and len(value) > 200:
            summary[key] = value[:200] + "…"
        else:
            summary[key] = value
    return summary


def build_registry(settings: Settings, tracer: Tracer | None = None) -> ToolRegistry:
    """Construct a registry with a live PubMed client."""
    return ToolRegistry(PubMedClient(settings, tracer), settings, tracer)
