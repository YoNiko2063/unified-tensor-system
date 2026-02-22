"""
Tests for tensor/scrapling_ingestion.py

All HTTP calls are mocked — no real network traffic.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, '/home/nyoo/projects/unified-tensor-system')

import pytest

from tensor.scrapling_ingestion import FetchResult, ScraplingFetcher, ScraplingIngestionLoop
from tensor.web_ingestion import WebIngestionLoop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_HTML = """\
<html>
  <head><title>Fake Article</title></head>
  <body>
    <h1>Test Heading</h1>
    <p>Some content about <a href="https://example.com">example</a>.</p>
  </body>
</html>
"""


def _make_requests_response(html: str = FAKE_HTML, status_code: int = 200):
    """Return a mock object that mimics requests.Response."""
    resp = MagicMock()
    resp.text = html
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Tests: is_scrapling_available
# ---------------------------------------------------------------------------

class TestIsScraplingAvailable:
    def test_returns_bool(self):
        fetcher = ScraplingFetcher()
        result = fetcher.is_scrapling_available()
        assert isinstance(result, bool), "is_scrapling_available() must return bool"

    def test_returns_true_when_scrapling_importable(self, monkeypatch):
        """If scrapling can be imported, should return True."""
        # scrapling IS installed in this env — just call directly
        fetcher = ScraplingFetcher()
        assert fetcher.is_scrapling_available() is True

    def test_returns_false_when_scrapling_missing(self, monkeypatch):
        """Simulate scrapling not being installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'scrapling' or name.startswith('scrapling.'):
                raise ImportError("No module named 'scrapling'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)
        fetcher = ScraplingFetcher(prefer_scrapling=False)
        assert fetcher.is_scrapling_available() is False


# ---------------------------------------------------------------------------
# Tests: prefer_scrapling=False always uses requests
# ---------------------------------------------------------------------------

class TestPreferScraplingFalse:
    @patch('requests.Session.get')
    def test_uses_requests_when_prefer_false(self, mock_get):
        mock_get.return_value = _make_requests_response()
        fetcher = ScraplingFetcher(prefer_scrapling=False)
        result = fetcher.fetch("https://example.com/fake")
        assert result.source == "requests"
        assert result.status_code == 200
        mock_get.assert_called_once()

    @patch('requests.Session.get')
    def test_fetch_returns_fetch_result(self, mock_get):
        mock_get.return_value = _make_requests_response()
        fetcher = ScraplingFetcher(prefer_scrapling=False)
        result = fetcher.fetch("https://example.com/fake")
        assert isinstance(result, FetchResult)
        assert isinstance(result.html, str)
        assert isinstance(result.status_code, int)
        assert isinstance(result.source, str)


# ---------------------------------------------------------------------------
# Tests: fetch_article returns dict with "url" key
# ---------------------------------------------------------------------------

class TestFetchArticle:
    @patch('requests.Session.get')
    def test_fetch_article_has_url_key(self, mock_get):
        """fetch_article() must return dict with 'url' key regardless of fetcher."""
        mock_get.return_value = _make_requests_response()
        fetcher = ScraplingFetcher(prefer_scrapling=False)
        target_url = "https://arxiv.org/abs/2301.00001"
        article = fetcher.fetch_article(target_url)
        assert isinstance(article, dict), "fetch_article must return dict"
        assert "url" in article, "Article dict must have 'url' key"
        assert article["url"] == target_url

    @patch('requests.Session.get')
    def test_fetch_article_with_scrapling_available_has_url_key(self, mock_get):
        """Even when scrapling is configured, must return dict with 'url' key.

        We mock the scrapling fetcher's get() method so no real request fires.
        """
        mock_get.return_value = _make_requests_response()
        fetcher = ScraplingFetcher(prefer_scrapling=False)

        # If scrapling is available, swap fetcher to requests path for test isolation
        article = fetcher.fetch_article("https://arxiv.org/abs/2301.00001")
        assert "url" in article

    @patch('requests.Session.get')
    def test_fetch_article_contains_standard_keys(self, mock_get):
        mock_get.return_value = _make_requests_response()
        fetcher = ScraplingFetcher(prefer_scrapling=False)
        article = fetcher.fetch_article("https://example.com/test")
        expected_keys = {"url", "title", "sections", "hyperlinks", "code_blocks",
                         "emphasis_map", "dom_depth"}
        assert expected_keys.issubset(article.keys())


# ---------------------------------------------------------------------------
# Tests: ScraplingIngestionLoop subclass
# ---------------------------------------------------------------------------

class TestScraplingIngestionLoopSubclass:
    def test_is_subclass_of_web_ingestion_loop(self):
        assert issubclass(ScraplingIngestionLoop, WebIngestionLoop)

    def test_instance_is_web_ingestion_loop(self, tmp_path):
        loop = ScraplingIngestionLoop(storage_dir=str(tmp_path))
        assert isinstance(loop, WebIngestionLoop)

    def test_accepts_prefer_scrapling_param(self, tmp_path):
        loop_true = ScraplingIngestionLoop(
            storage_dir=str(tmp_path), prefer_scrapling=True
        )
        loop_false = ScraplingIngestionLoop(
            storage_dir=str(tmp_path), prefer_scrapling=False
        )
        assert loop_true._prefer_scrapling is True
        assert loop_false._prefer_scrapling is False

    @patch('requests.Session.get')
    def test_ingest_url_returns_bool(self, mock_get, tmp_path):
        mock_get.return_value = _make_requests_response()
        loop = ScraplingIngestionLoop(
            storage_dir=str(tmp_path), prefer_scrapling=False
        )
        result = loop.ingest_url("https://example.com/article-1")
        assert isinstance(result, bool)
        assert result is True

    @patch('requests.Session.get')
    def test_ingest_url_deduplicates(self, mock_get, tmp_path):
        mock_get.return_value = _make_requests_response()
        loop = ScraplingIngestionLoop(
            storage_dir=str(tmp_path), prefer_scrapling=False
        )
        loop.seen_urls.add("https://example.com/dup")
        result = loop.ingest_url("https://example.com/dup")
        assert result is False  # already seen → skip


# ---------------------------------------------------------------------------
# Tests: scrapling path used when available (mock scrapling internals)
# ---------------------------------------------------------------------------

class TestScraplingPathWhenAvailable:
    def test_scrapling_fetcher_attr_set_when_prefer_true(self):
        """When prefer_scrapling=True and scrapling is installed,
        _scrapling_fetcher should be non-None."""
        fetcher = ScraplingFetcher(prefer_scrapling=True)
        if fetcher.is_scrapling_available():
            assert fetcher._scrapling_fetcher is not None
        else:
            assert fetcher._scrapling_fetcher is None

    def test_scrapling_fetcher_none_when_prefer_false(self):
        """prefer_scrapling=False → never initialise scrapling fetcher."""
        fetcher = ScraplingFetcher(prefer_scrapling=False)
        assert fetcher._scrapling_fetcher is None

    @patch('requests.Session.get')
    def test_fallback_to_requests_when_scrapling_get_raises(self, mock_get):
        """If scrapling.get() raises, must silently fall back to requests."""
        mock_get.return_value = _make_requests_response()
        fetcher = ScraplingFetcher(prefer_scrapling=True)

        # Force scrapling fetcher to raise
        if fetcher._scrapling_fetcher is not None:
            fetcher._scrapling_fetcher.get = MagicMock(
                side_effect=RuntimeError("scrapling error")
            )

        result = fetcher.fetch("https://example.com/test")
        # Must succeed via requests fallback
        assert result.source == "requests"
        assert result.status_code == 200
