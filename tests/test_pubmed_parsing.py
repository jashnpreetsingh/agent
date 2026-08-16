"""Tests for PubMed XML parsing and query construction."""

from __future__ import annotations

from src.tools.pubmed import PubMedClient
from tests.conftest import SAMPLE_XML


def test_parses_core_fields():
    articles = PubMedClient.parse_articles(SAMPLE_XML)
    assert len(articles) == 2

    first = articles[0]
    assert first.pmid == "12345678"
    assert first.journal == "N Engl J Med"
    assert first.year == 2024
    assert "Randomized Controlled Trial" in first.publication_types
    assert first.mesh_terms == ["Diabetes Mellitus, Type 2"]


def test_title_includes_inline_markup_text():
    """<i>glycaemic</i> must not truncate the title at the first tag."""
    first = PubMedClient.parse_articles(SAMPLE_XML)[0]
    assert first.title == "Tirzepatide and glycaemic control in type 2 diabetes"


def test_structured_abstract_keeps_section_labels():
    first = PubMedClient.parse_articles(SAMPLE_XML)[0]
    assert "BACKGROUND: Tirzepatide is a dual agonist." in first.abstract
    assert "RESULTS: HbA1c fell by 2.1% versus placebo." in first.abstract


def test_doi_is_not_taken_from_reference_list():
    """A generic .//ArticleIdList search would return the reference's DOI."""
    first = PubMedClient.parse_articles(SAMPLE_XML)[0]
    assert first.doi == "10.1056/REAL-DOI"
    assert first.doi != "10.9999/WRONG-REFERENCE-DOI"


def test_medline_date_year_is_extracted():
    """PubDate sometimes carries free text instead of a Year element."""
    second = PubMedClient.parse_articles(SAMPLE_XML)[1]
    assert second.year == 2019


def test_missing_optional_fields_do_not_crash():
    second = PubMedClient.parse_articles(SAMPLE_XML)[1]
    assert second.authors == []
    assert second.doi is None
    assert second.citation().startswith("Anon")


def test_citation_uses_surname_not_initials():
    first = PubMedClient.parse_articles(SAMPLE_XML)[0]
    assert first.citation().startswith("Aronne et al. (2024)")


def test_design_weight_prefers_strongest_type():
    articles = PubMedClient.parse_articles(SAMPLE_XML)
    assert articles[1].design_weight > articles[0].design_weight  # meta-analysis > RCT


def test_build_query_applies_filters():
    query = PubMedClient.build_query(
        "metformin", date_from=2020, pub_types=["Randomized Controlled Trial"]
    )
    assert "(metformin)" in query
    assert '"Randomized Controlled Trial"[Publication Type]' in query
    assert '"2020"[Date - Publication]' in query


def test_build_query_without_filters_is_bare():
    assert PubMedClient.build_query("metformin") == "(metformin)"


def test_malformed_xml_raises_pubmed_error():
    import pytest

    from src.tools.pubmed import PubMedError

    with pytest.raises(PubMedError):
        PubMedClient.parse_articles("<not-xml")
