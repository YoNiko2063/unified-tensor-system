"""
Tests for tensor/financial_ingestion.py

All HTTP calls are mocked — no real network requests.
"""

import sys
import json
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, '/home/nyoo/projects/unified-tensor-system')

import numpy as np
import pytest

from tensor.financial_ingestion import (
    AdaptiveElementStore,
    DEFAULT_SOURCE_PROFILES,
    EarningsTranscriptParser,
    FilingParser,
    FinancialArticle,
    FinancialIngestionRouter,
    SourceProfile,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_store(tmp_path):
    """AdaptiveElementStore backed by a temp directory."""
    return AdaptiveElementStore(base_dir=str(tmp_path / "profiles"))


@pytest.fixture
def router(tmp_store):
    """FinancialIngestionRouter with default profiles and temp store."""
    return FinancialIngestionRouter(adaptive_store=tmp_store)


@pytest.fixture
def transcript_parser():
    return EarningsTranscriptParser()


@pytest.fixture
def filing_parser():
    return FilingParser()


# =============================================================================
# 1. FinancialIngestionRouter — instantiation
# =============================================================================

class TestRouterInit:
    def test_default_profiles_loaded(self, router):
        """Router initialises with all DEFAULT_SOURCE_PROFILES."""
        assert "reuters.com" in router._profiles
        assert "sec.gov" in router._profiles
        assert "arxiv.org" in router._profiles

    def test_profile_count(self, router):
        """All 7 default profiles are registered."""
        assert len(router._profiles) == len(DEFAULT_SOURCE_PROFILES)

    def test_custom_profiles_override_defaults(self, tmp_store):
        """Passing custom profiles list replaces defaults."""
        custom = [SourceProfile("example.com", "Fetcher", 1.0, {"headline": "h1"})]
        r = FinancialIngestionRouter(profiles=custom, adaptive_store=tmp_store)
        assert "example.com" in r._profiles
        assert "reuters.com" not in r._profiles


# =============================================================================
# 2. _detect_domain
# =============================================================================

class TestDetectDomain:
    def test_reuters_detected(self, router):
        profile = router._detect_domain("https://reuters.com/article/abc-123")
        assert profile is not None
        assert profile.domain == "reuters.com"

    def test_sec_detected(self, router):
        profile = router._detect_domain("https://www.sec.gov/cgi-bin/browse-edgar")
        assert profile is not None
        assert profile.domain == "sec.gov"

    def test_unknown_returns_none(self, router):
        profile = router._detect_domain("https://unknown-site.com/news")
        assert profile is None

    def test_subdomain_matched(self, router):
        """Subdomain URLs still match the registered domain."""
        profile = router._detect_domain("https://markets.ft.com/data/equities")
        assert profile is not None
        assert profile.domain == "ft.com"

    def test_arxiv_detected(self, router):
        profile = router._detect_domain("https://arxiv.org/abs/2301.00001")
        assert profile is not None
        assert profile.domain == "arxiv.org"


# =============================================================================
# 3. register_source
# =============================================================================

class TestRegisterSource:
    def test_register_new_domain(self, router):
        new_profile = SourceProfile("bloomberg.com", "StealthyFetcher", 2.5,
                                    {"headline": "h1", "body": ".body-copy"})
        router.register_source(new_profile)
        assert "bloomberg.com" in router._profiles

    def test_override_existing_domain(self, router):
        """Registering a profile for an existing domain replaces it."""
        override = SourceProfile("reuters.com", "Fetcher", 0.5, {"headline": "h2"})
        router.register_source(override)
        p = router._profiles["reuters.com"]
        assert p.fetcher_class == "Fetcher"
        assert p.rate_limit_seconds == 0.5

    def test_override_preserves_other_profiles(self, router):
        override = SourceProfile("reuters.com", "Fetcher", 0.5, {})
        router.register_source(override)
        # Other profiles untouched
        assert "sec.gov" in router._profiles
        assert "ft.com" in router._profiles


# =============================================================================
# 4. _select_fetcher
# =============================================================================

class TestSelectFetcher:
    def test_returns_requests_fallback_when_scrapling_absent(self, router):
        """When no Scrapling class is importable, falls back to requests."""
        import tensor.financial_ingestion as fi
        # Temporarily patch all Scrapling classes to None
        orig = (fi._FETCHER_CLS, fi._STEALTHY_FETCHER_CLS,
                fi._DYNAMIC_FETCHER_CLS, fi._ASYNC_FETCHER_CLS)
        fi._FETCHER_CLS = None
        fi._STEALTHY_FETCHER_CLS = None
        fi._DYNAMIC_FETCHER_CLS = None
        fi._ASYNC_FETCHER_CLS = None
        # Rebuild the chain
        fi._FETCHER_CHAIN = [
            ("DynamicFetcher",  None),
            ("StealthyFetcher", None),
            ("Fetcher",         None),
        ]
        try:
            _, name = router._select_fetcher(router._profiles["reuters.com"])
            assert name == "requests"
        finally:
            fi._FETCHER_CLS, fi._STEALTHY_FETCHER_CLS, \
                fi._DYNAMIC_FETCHER_CLS, fi._ASYNC_FETCHER_CLS = orig
            fi._FETCHER_CHAIN = [
                ("DynamicFetcher",  fi._DYNAMIC_FETCHER_CLS),
                ("StealthyFetcher", fi._STEALTHY_FETCHER_CLS),
                ("Fetcher",         fi._FETCHER_CLS),
            ]

    def test_select_fetcher_with_none_profile(self, router):
        """select_fetcher handles None profile (unknown domain)."""
        _, name = router._select_fetcher(None)
        assert name in ("Fetcher", "StealthyFetcher", "DynamicFetcher", "requests")


# =============================================================================
# 5. fetch() — mocked HTTP
# =============================================================================

SIMPLE_HTML = """
<html><head><title>Test Page</title></head>
<body>
<h1>Apple $AAPL beats earnings estimates</h1>
<div class="article-body">Revenue grew 12% year-over-year.</div>
</body></html>
"""

REUTERS_HTML = """
<html><body>
<div data-testid="Heading">$MSFT Microsoft quarterly earnings</div>
<p data-testid="paragraph-0">Microsoft reported strong results.</p>
<time>2024-01-25</time>
</body></html>
"""


class TestFetch:
    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value=SIMPLE_HTML)
    def test_fetch_returns_financial_article(self, mock_req, router):
        article = router.fetch("https://unknown-site.com/news/apple")
        assert isinstance(article, FinancialArticle)

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value=SIMPLE_HTML)
    def test_fetch_sets_url(self, mock_req, router):
        url = "https://unknown-site.com/news/apple"
        article = router.fetch(url)
        assert article.url == url

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value=REUTERS_HTML)
    def test_fetch_detects_reuters_domain(self, mock_req, router):
        article = router.fetch("https://reuters.com/article/msft-earnings")
        assert article.domain == "reuters.com"

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value=REUTERS_HTML)
    def test_fetch_unknown_domain_is_unknown(self, mock_req, router):
        article = router.fetch("https://some-random-blog.com/post")
        assert article.domain == "unknown"

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value=SIMPLE_HTML)
    def test_fetch_word_count_positive(self, mock_req, router):
        article = router.fetch("https://unknown-site.com/news")
        assert article.word_count > 0

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value=SIMPLE_HTML)
    def test_fetch_fetcher_used_set(self, mock_req, router):
        article = router.fetch("https://unknown-site.com/news")
        assert article.fetcher_used in (
            "requests", "Fetcher", "StealthyFetcher", "DynamicFetcher", "AsyncFetcher"
        )


# =============================================================================
# 6. Ticker extraction
# =============================================================================

class TestTickerExtraction:
    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value='<html><body><h1>Apple $AAPL earnings beat</h1>'
                        '<div>Nothing else</div></body></html>')
    def test_tickers_aapl_extracted(self, mock_req, router):
        article = router.fetch("https://unknown-site.com/news")
        assert "AAPL" in article.tickers

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value='<html><body><h1>$MSFT and $GOOGL report results</h1>'
                        '<div>Both beat estimates</div></body></html>')
    def test_multiple_tickers_extracted(self, mock_req, router):
        article = router.fetch("https://unknown-site.com/news")
        assert "MSFT" in article.tickers
        assert "GOOGL" in article.tickers

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value='<html><body><h1>No tickers here</h1></body></html>')
    def test_no_tickers_returns_empty_list(self, mock_req, router):
        article = router.fetch("https://unknown-site.com/news")
        assert article.tickers == []

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value='<html><body><h1>$AAPL $AAPL double mention</h1></body></html>')
    def test_tickers_deduplicated(self, mock_req, router):
        article = router.fetch("https://unknown-site.com/news")
        assert article.tickers.count("AAPL") == 1


# =============================================================================
# 7. AdaptiveElementStore
# =============================================================================

class TestAdaptiveElementStore:
    def test_load_returns_empty_dict_for_new_domain(self, tmp_store):
        result = tmp_store.load("new-domain.com")
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_store):
        patterns = {"headline": ".art-head", "body": ".art-body"}
        tmp_store.save("reuters.com", patterns)
        loaded = tmp_store.load("reuters.com")
        assert loaded == patterns

    def test_get_selector_returns_none_for_missing(self, tmp_store):
        sel = tmp_store.get_selector("unknown.com", "headline")
        assert sel is None

    def test_get_selector_returns_stored_value(self, tmp_store):
        tmp_store.save("ft.com", {"headline": ".ft-head"})
        sel = tmp_store.get_selector("ft.com", "headline")
        assert sel == ".ft-head"

    def test_save_creates_file(self, tmp_store):
        tmp_store.save("example.com", {"body": "p"})
        # Path: base_dir/example_com.json
        p = Path(tmp_store._base) / "example_com.json"
        assert p.exists()

    def test_domain_dots_normalised_in_filename(self, tmp_store):
        tmp_store.save("a.b.com", {"k": "v"})
        p = Path(tmp_store._base) / "a_b_com.json"
        assert p.exists()

    def test_overwrite_updates_stored_value(self, tmp_store):
        tmp_store.save("site.com", {"headline": ".old"})
        tmp_store.save("site.com", {"headline": ".new"})
        assert tmp_store.get_selector("site.com", "headline") == ".new"


# =============================================================================
# 8. EarningsTranscriptParser
# =============================================================================

TRANSCRIPT_BODY = """
Welcome to our Q3 earnings call.

We expect revenue to reach $2.5 billion next quarter.
Management targets a 15% margin improvement.
We guide for EPS of $1.20 for full year.

Question-and-Answer Session

Q: Can you elaborate on the guidance?
A: We anticipate strong demand continuing into Q4.

EPS of $1.10 was reported this quarter.
Revenue came in at $500m.
"""


class TestEarningsTranscriptParser:
    def test_parse_returns_dict_with_required_keys(self, transcript_parser):
        result = transcript_parser.parse(TRANSCRIPT_BODY)
        assert "prepared_remarks" in result
        assert "qa_section" in result
        assert "management_guidance" in result
        assert "revenue_mentions" in result
        assert "eps_mentions" in result

    def test_management_guidance_contains_expect(self, transcript_parser):
        result = transcript_parser.parse(TRANSCRIPT_BODY)
        guidance = result["management_guidance"]
        assert any("expect" in line.lower() for line in guidance)

    def test_management_guidance_contains_target(self, transcript_parser):
        result = transcript_parser.parse(TRANSCRIPT_BODY)
        guidance = result["management_guidance"]
        assert any("target" in line.lower() for line in guidance)

    def test_qa_section_split_at_qa_boundary(self, transcript_parser):
        result = transcript_parser.parse(TRANSCRIPT_BODY)
        assert "Question" in result["qa_section"] or "question" in result["qa_section"].lower()

    def test_prepared_remarks_before_qa(self, transcript_parser):
        result = transcript_parser.parse(TRANSCRIPT_BODY)
        assert "Q3 earnings" in result["prepared_remarks"]

    def test_revenue_mentions_found(self, transcript_parser):
        result = transcript_parser.parse(TRANSCRIPT_BODY)
        assert len(result["revenue_mentions"]) > 0

    def test_eps_mentions_found(self, transcript_parser):
        result = transcript_parser.parse(TRANSCRIPT_BODY)
        assert len(result["eps_mentions"]) > 0

    def test_body_without_qa_boundary(self, transcript_parser):
        """No Q&A section — full body goes to prepared_remarks."""
        simple = "We expect to grow 10% next year."
        result = transcript_parser.parse(simple)
        assert result["qa_section"] == ""
        assert "We expect to grow" in result["prepared_remarks"]


# =============================================================================
# 9. FilingParser
# =============================================================================

class TestFilingParser:
    def test_detect_8k_from_url(self, filing_parser):
        url = "https://www.sec.gov/Archives/edgar/data/123456/000123/8-K.htm"
        result = filing_parser.parse("Some filing content.", url)
        assert result["filing_type"] == "8-K"

    def test_detect_10q_from_url(self, filing_parser):
        url = "https://www.sec.gov/Archives/10-Q/filing.htm"
        result = filing_parser.parse("Quarterly report.", url)
        assert result["filing_type"] == "10-Q"

    def test_detect_13f_from_url(self, filing_parser):
        url = "https://www.sec.gov/Archives/edgar/13F/filing.htm"
        result = filing_parser.parse("Holdings report.", url)
        assert result["filing_type"] == "13F"

    def test_event_type_item_for_8k(self, filing_parser):
        url = "https://sec.gov/8-K"
        body = "Item 2.02 Results of Operations and Financial Condition."
        result = filing_parser.parse(body, url)
        assert result["event_type"] == "Item 2.02"

    def test_event_type_none_for_non_8k(self, filing_parser):
        url = "https://sec.gov/10-K"
        result = filing_parser.parse("Annual report content.", url)
        assert result["event_type"] is None

    def test_key_figures_extracts_dollars(self, filing_parser):
        url = "https://sec.gov/8-K"
        body = "Revenue was $500 million. Profit was $50 million."
        result = filing_parser.parse(body, url)
        assert len(result["key_figures"]) > 0

    def test_unknown_filing_type_when_no_match(self, filing_parser):
        result = filing_parser.parse("Some random content.", "https://example.com/doc")
        assert result["filing_type"] == "unknown"


# =============================================================================
# 10. fetch_batch
# =============================================================================

class TestFetchBatch:
    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value=SIMPLE_HTML)
    def test_fetch_batch_returns_list(self, mock_req, router):
        urls = [
            "https://unknown-site.com/news/1",
            "https://unknown-site.com/news/2",
        ]
        results = asyncio.get_event_loop().run_until_complete(
            router.fetch_batch(urls)
        )
        assert isinstance(results, list)
        assert len(results) == 2

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value=SIMPLE_HTML)
    def test_fetch_batch_all_financial_articles(self, mock_req, router):
        urls = ["https://unknown-site.com/a", "https://unknown-site.com/b"]
        results = asyncio.get_event_loop().run_until_complete(
            router.fetch_batch(urls)
        )
        for article in results:
            assert isinstance(article, FinancialArticle)

    @patch("tensor.financial_ingestion.FinancialIngestionRouter._fetch_requests",
           return_value=SIMPLE_HTML)
    def test_fetch_batch_urls_preserved(self, mock_req, router):
        urls = [
            "https://unknown-site.com/x",
            "https://unknown-site.com/y",
        ]
        results = asyncio.get_event_loop().run_until_complete(
            router.fetch_batch(urls)
        )
        result_urls = {a.url for a in results}
        assert result_urls == set(urls)


# =============================================================================
# 11. DEFAULT_SOURCE_PROFILES sanity
# =============================================================================

class TestDefaultSourceProfiles:
    def test_all_profiles_have_domain(self):
        for p in DEFAULT_SOURCE_PROFILES:
            assert p.domain

    def test_all_profiles_have_selectors(self):
        for p in DEFAULT_SOURCE_PROFILES:
            assert isinstance(p.selectors, dict)

    def test_js_requiring_sites_flagged(self):
        for p in DEFAULT_SOURCE_PROFILES:
            if p.fetcher_class == "DynamicFetcher":
                assert p.requires_js is True

    def test_seeking_alpha_is_transcript_parser(self):
        profile = next(p for p in DEFAULT_SOURCE_PROFILES if p.domain == "seekingalpha.com")
        assert profile.parser_type == "transcript"

    def test_sec_is_filing_parser(self):
        profile = next(p for p in DEFAULT_SOURCE_PROFILES if p.domain == "sec.gov")
        assert profile.parser_type == "filing"
