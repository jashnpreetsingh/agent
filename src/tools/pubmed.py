"""PubMed access via NCBI E-utilities.

Covers the three calls the agent needs:

``esearch``   - resolve a query to PMIDs;
``efetch``    - pull full records (abstract, MeSH, publication types) as XML;
``mesh``      - map lay phrasing onto controlled MeSH vocabulary.

Two details in here are easy to get wrong and worth flagging:

* DOIs must be read from ``PubmedData/ArticleIdList`` specifically. A blanket
  ``.//ArticleIdList/ArticleId`` also matches every entry in ``ReferenceList``,
  so a record with 60 references yields 60 wrong DOIs.
* ``PubDate`` carries either ``Year`` or a free-text ``MedlineDate``
  ("2023 Jan-Feb"), so the year parser has to handle both.

NCBI's usage policy caps unauthenticated clients at 3 requests/second; the
rate limiter here enforces that rather than relying on politeness.
"""

from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.cassette import CassetteStore, canonical_key
from src.config import Settings, TransportMode
from src.llm.errors import CassetteMissError
from src.observability.trace import Tracer
from src.schemas import Article

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_YEAR_RE = re.compile(r"(19|20)\d{2}")


class PubMedError(Exception):
    """Raised when PubMed cannot be reached or returns an unusable payload."""


@dataclass(slots=True)
class SearchResult:
    """Outcome of one ``esearch`` call."""

    query: str
    total_count: int
    pmids: list[str] = field(default_factory=list)
    translated_query: str = ""


class RateLimiter:
    """Minimal thread-safe request spacer."""

    def __init__(self, per_second: float) -> None:
        self._min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


class PubMedClient:
    """E-utilities client with retries, rate limiting, and record/replay."""

    def __init__(
        self,
        settings: Settings,
        tracer: Tracer | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer or Tracer.null()
        self.mode = settings.pubmed_mode
        self._cassette = CassetteStore(settings.fixtures_dir, "pubmed")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=settings.request_timeout)
        # NCBI allows 10 req/s with a key, 3 without.
        self._limiter = RateLimiter(10.0 if settings.ncbi_api_key else 3.0)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        date_from: int | None = None,
        pub_types: list[str] | None = None,
        sort: str = "relevance",
    ) -> SearchResult:
        """Resolve a query to PMIDs, applying date and study-design filters."""
        full_query = self.build_query(query, date_from=date_from, pub_types=pub_types)
        params = {
            "db": "pubmed",
            "term": full_query,
            "retmax": str(max(1, min(max_results, 50))),
            "retmode": "json",
            "sort": sort,
        }
        payload = self._get("esearch.fcgi", params)
        result = payload.get("esearchresult", {})
        if "ERROR" in result:
            raise PubMedError(f"PubMed rejected the query: {result['ERROR']}")

        return SearchResult(
            query=full_query,
            total_count=int(result.get("count", 0) or 0),
            pmids=list(result.get("idlist", [])),
            translated_query=result.get("querytranslation", ""),
        )

    def fetch(self, pmids: list[str]) -> list[Article]:
        """Fetch full records for up to a few dozen PMIDs."""
        if not pmids:
            return []
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        xml_text = self._get("efetch.fcgi", params, expect_json=False)
        return self.parse_articles(xml_text)

    def lookup_mesh(self, term: str, *, max_results: int = 5) -> list[str]:
        """Map a phrase onto MeSH descriptors.

        Uses two complementary signals: PubMed's own automatic term mapping
        (authoritative for lay phrasing) and a direct MeSH database search.
        Term mapping is what resolves "heart attack" to "myocardial infarction"
        and "water on the brain" to "hydrocephalus". Phrases absent from
        PubMed's synonym table degrade to their component words rather than
        failing, so callers should treat the result as a hint, not a guarantee.
        """
        descriptors: list[str] = []

        # 1. PubMed's automatic term mapping, read off the query translation.
        try:
            probe = self._get(
                "esearch.fcgi",
                {"db": "pubmed", "term": term, "retmax": "1", "retmode": "json"},
            )
            translation = probe.get("esearchresult", {}).get("querytranslation", "")
            descriptors.extend(re.findall(r'"([^"]+)"\[MeSH Terms\]', translation))
        except PubMedError:
            # Term mapping is a nicety; a failure here should not sink the tool.
            self.tracer.event("tool.mesh.mapping_unavailable", term=term)

        # 2. Direct MeSH descriptor search.
        try:
            found = self._get(
                "esearch.fcgi",
                {"db": "mesh", "term": term, "retmax": str(max_results), "retmode": "json"},
            )
            uids = found.get("esearchresult", {}).get("idlist", [])
            if uids:
                summary = self._get(
                    "esummary.fcgi",
                    {"db": "mesh", "id": ",".join(uids), "retmode": "json"},
                )
                records = summary.get("result", {})
                for uid in records.get("uids", []):
                    # ds_meshterms lists the preferred descriptor first,
                    # followed by entry-term synonyms we do not want.
                    terms = records.get(uid, {}).get("ds_meshterms") or []
                    if terms:
                        descriptors.append(terms[0])
        except PubMedError:
            self.tracer.event("tool.mesh.lookup_failed", term=term)

        seen: set[str] = set()
        unique: list[str] = []
        for descriptor in descriptors:
            key = descriptor.lower()
            if key not in seen:
                seen.add(key)
                unique.append(descriptor)
        return unique[:max_results]

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------
    @staticmethod
    def build_query(
        query: str,
        *,
        date_from: int | None = None,
        pub_types: list[str] | None = None,
    ) -> str:
        """Compose a PubMed query string with optional filters."""
        clauses = [f"({query})" if query.strip() else ""]
        if pub_types:
            joined = " OR ".join(f'"{pt}"[Publication Type]' for pt in pub_types)
            clauses.append(f"({joined})")
        if date_from:
            clauses.append(f'("{date_from}"[Date - Publication] : "3000"[Date - Publication])')
        return " AND ".join(c for c in clauses if c)

    # ------------------------------------------------------------------
    # XML parsing
    # ------------------------------------------------------------------
    @classmethod
    def parse_articles(cls, xml_text: str) -> list[Article]:
        """Parse an ``efetch`` XML payload into :class:`Article` records."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise PubMedError(f"could not parse PubMed XML: {exc}") from exc

        articles: list[Article] = []
        for node in root.findall(".//PubmedArticle"):
            parsed = cls._parse_article(node)
            if parsed is not None:
                articles.append(parsed)
        return articles

    @classmethod
    def _parse_article(cls, node: ET.Element) -> Article | None:
        pmid = node.findtext(".//MedlineCitation/PMID")
        if not pmid:
            return None
        article_el = node.find(".//MedlineCitation/Article")
        if article_el is None:
            return None

        title_el = article_el.find("ArticleTitle")
        title = _all_text(title_el) if title_el is not None else "[No title]"

        # DOI lives in PubmedData/ArticleIdList. Scoping matters: a generic
        # .//ArticleIdList search also matches every cited reference.
        doi = None
        for id_el in node.findall("./PubmedData/ArticleIdList/ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = (id_el.text or "").strip() or None
                break

        return Article(
            pmid=pmid.strip(),
            title=title,
            abstract=cls._parse_abstract(article_el),
            journal=(
                article_el.findtext("Journal/ISOAbbreviation")
                or article_el.findtext("Journal/Title")
                or ""
            ).strip(),
            year=cls._parse_year(article_el),
            publication_types=[
                (pt.text or "").strip()
                for pt in article_el.findall("PublicationTypeList/PublicationType")
                if (pt.text or "").strip()
            ],
            mesh_terms=[
                (m.text or "").strip()
                for m in node.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
                if (m.text or "").strip()
            ],
            authors=cls._parse_authors(article_el),
            doi=doi,
        )

    @staticmethod
    def _parse_abstract(article_el: ET.Element) -> str:
        """Flatten a structured abstract, preserving section labels."""
        chunks: list[str] = []
        for section in article_el.findall("Abstract/AbstractText"):
            text = _all_text(section)
            if not text:
                continue
            label = section.get("Label")
            chunks.append(f"{label}: {text}" if label else text)
        return "\n".join(chunks)

    @staticmethod
    def _parse_year(article_el: ET.Element) -> int | None:
        year = article_el.findtext("Journal/JournalIssue/PubDate/Year")
        if year and year.strip().isdigit():
            return int(year.strip())
        # MedlineDate holds free text such as "2023 Jan-Feb" or "2019-2020".
        medline = article_el.findtext("Journal/JournalIssue/PubDate/MedlineDate") or ""
        if match := _YEAR_RE.search(medline):
            return int(match.group(0))
        if article_date := article_el.findtext("ArticleDate/Year"):
            if article_date.strip().isdigit():
                return int(article_date.strip())
        return None

    @staticmethod
    def _parse_authors(article_el: ET.Element, limit: int = 12) -> list[str]:
        authors: list[str] = []
        for author in article_el.findall("AuthorList/Author")[:limit]:
            last = (author.findtext("LastName") or "").strip()
            initials = (author.findtext("Initials") or "").strip()
            if last:
                authors.append(f"{last} {initials}".strip())
            elif collective := (author.findtext("CollectiveName") or "").strip():
                authors.append(collective)
        return authors

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _get(self, endpoint: str, params: dict[str, str], expect_json: bool = True) -> Any:
        """Perform an E-utilities GET under the configured transport mode."""
        params = dict(params)
        params.setdefault("tool", self.settings.ncbi_tool)
        params.setdefault("email", self.settings.ncbi_email)

        key = canonical_key({"endpoint": endpoint, "params": params})

        if self.mode is TransportMode.REPLAY:
            cached = self._cassette.get(key, purpose=endpoint)
            if cached is None:
                raise CassetteMissError(
                    f"No recorded PubMed response for {endpoint} (key {key}). "
                    "Re-record with PUBMED_MODE=record."
                )
            self.tracer.event("pubmed.replay", endpoint=endpoint, key=key)
            return cached["body"] if isinstance(cached, dict) and "body" in cached else cached

        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key.get_secret_value()

        payload = self._request_with_retries(endpoint, params, expect_json)

        if self.mode is TransportMode.RECORD:
            self._cassette.put(
                key,
                {"endpoint": endpoint, "params": params},
                payload if expect_json else {"body": payload},
                meta={"endpoint": endpoint},
            )
        return payload

    def _request_with_retries(self, endpoint: str, params: dict[str, str], expect_json: bool) -> Any:
        url = f"{self.settings.ncbi_base_url}/{endpoint}"
        last_error: Exception | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            self._limiter.acquire()
            started = time.perf_counter()
            try:
                resp = self._client.get(url, params=params)
                elapsed_ms = (time.perf_counter() - started) * 1000

                if resp.status_code == 200:
                    self.tracer.event(
                        "pubmed.request",
                        endpoint=endpoint,
                        attempt=attempt,
                        latency_ms=round(elapsed_ms, 1),
                        bytes=len(resp.content),
                    )
                    return resp.json() if expect_json else resp.text

                retryable = resp.status_code in RETRYABLE_STATUS
                self.tracer.event(
                    "pubmed.http_error",
                    endpoint=endpoint,
                    attempt=attempt,
                    status=resp.status_code,
                    retryable=retryable,
                )
                if not retryable:
                    raise PubMedError(f"PubMed returned {resp.status_code} for {endpoint}")
                last_error = PubMedError(f"PubMed returned {resp.status_code} for {endpoint}")
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self.tracer.event(
                    "pubmed.transport_error", endpoint=endpoint, attempt=attempt, detail=str(exc)[:200]
                )
                last_error = PubMedError(f"could not reach PubMed: {exc}")
            except ValueError as exc:  # malformed JSON
                last_error = PubMedError(f"malformed PubMed response: {exc}")

            if attempt < self.settings.max_retries:
                time.sleep(self.settings.retry_base_delay * (2 ** (attempt - 1)))

        raise last_error or PubMedError("PubMed request failed")


def _all_text(element: ET.Element) -> str:
    """Flatten an element's text, including inline markup such as <i>/<sup>.

    PubMed titles and abstracts contain formatting tags; ``element.text``
    alone silently truncates at the first one.
    """
    return " ".join("".join(element.itertext()).split()).strip()
