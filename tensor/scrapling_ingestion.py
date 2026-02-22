"""
Scrapling-first web ingestion with requests fallback.

Classes:
  - FetchResult:           Named result tuple from fetch()
  - ScraplingFetcher:      Scrapling-first HTTP fetcher with requests fallback
  - ScraplingIngestionLoop: WebIngestionLoop subclass using ScraplingFetcher
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional

from tensor.web_ingestion import ArticleParser, ResearchConceptExtractor, WebIngestionLoop


@dataclass
class FetchResult:
    """Result of a single HTTP fetch operation.

    Attributes:
        html:        Raw HTML string of the response body.
        status_code: HTTP status code (e.g. 200, 404).
        source:      Which fetcher was used: "scrapling" or "requests".
    """
    html: str
    status_code: int
    source: str  # "scrapling" or "requests"


class ScraplingFetcher:
    """Scrapling-first HTTP fetcher with requests fallback.

    Priority:
      1. Try Scrapling (fetchers.Fetcher) if installed and prefer_scrapling=True
      2. Fall back to requests.Session

    Methods:
        fetch(url)          -> FetchResult
        fetch_article(url)  -> Dict  (full article parse via ArticleParser)
        is_scrapling_available() -> bool
    """

    def __init__(self, prefer_scrapling: bool = True) -> None:
        self._prefer_scrapling = prefer_scrapling
        self._parser = ArticleParser()
        self._scrapling_fetcher = None  # lazy init

        # Attempt to import and configure scrapling once
        if prefer_scrapling:
            self._scrapling_fetcher = self._try_init_scrapling()

    # ── Public API ─────────────────────────────────────────────────────────

    def is_scrapling_available(self) -> bool:
        """Return True if scrapling package is importable."""
        try:
            from scrapling import fetchers as _  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch(self, url: str) -> FetchResult:
        """Fetch URL, returning (html, status_code, source).

        Uses Scrapling when available and preferred, otherwise requests.
        """
        if self._prefer_scrapling and self._scrapling_fetcher is not None:
            result = self._fetch_scrapling(url)
            if result is not None:
                return result
            # Scrapling failed — fall through to requests

        return self._fetch_requests(url)

    def fetch_article(self, url: str) -> Dict:
        """Fetch URL and parse the HTML into a structured article dict.

        Returns dict with keys: url, title, sections, hyperlinks,
        code_blocks, emphasis_map, dom_depth.
        """
        result = self.fetch(url)
        return self._parser.parse(result.html, url)

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _try_init_scrapling():
        """Attempt to create a scrapling Fetcher instance. Returns None on failure."""
        try:
            from scrapling.fetchers import Fetcher
            return Fetcher()
        except Exception:
            return None

    def _fetch_scrapling(self, url: str) -> Optional[FetchResult]:
        """Attempt fetch via scrapling. Returns None on any error."""
        try:
            response = self._scrapling_fetcher.get(url, timeout=15)
            html = str(response.text) if response.text else ""
            status = int(response.status)
            return FetchResult(html=html, status_code=status, source="scrapling")
        except Exception:
            return None

    @staticmethod
    def _fetch_requests(url: str) -> FetchResult:
        """Fetch via requests.Session (always available fallback)."""
        import requests
        session = requests.Session()
        try:
            response = session.get(url, timeout=10)
            return FetchResult(
                html=response.text,
                status_code=response.status_code,
                source="requests",
            )
        except Exception as exc:
            return FetchResult(html="", status_code=0, source="requests")


class ScraplingIngestionLoop(WebIngestionLoop):
    """WebIngestionLoop that uses ScraplingFetcher instead of raw requests.

    Overrides ingest_url() to:
      - Use ScraplingFetcher (scrapling-first with requests fallback)
      - Log which fetcher was used per article
      - Apply a 2-second rate limit between requests (conservative for Scrapling)

    Constructor params:
        storage_dir (str):         Where to write ingested JSON files.
        prefer_scrapling (bool):   If False, always use the requests fallback.
    """

    def __init__(
        self,
        storage_dir: str = "tensor/data/ingested",
        prefer_scrapling: bool = True,
    ) -> None:
        super().__init__(storage_dir=storage_dir)
        self._fetcher = ScraplingFetcher(prefer_scrapling=prefer_scrapling)
        self._prefer_scrapling = prefer_scrapling

    # ── Override ───────────────────────────────────────────────────────────

    def ingest_url(self, url: str) -> bool:
        """Fetch single URL with ScraplingFetcher, parse, extract, store.

        Returns True if successfully ingested, False if duplicate or error.
        Rate-limits to 2 s between requests.
        """
        if url in self.seen_urls:
            return False  # duplicate

        try:
            fetch_result = self._fetcher.fetch(url)
            print(
                f"[ScraplingIngestion] Fetched via {fetch_result.source}: "
                f"{url} (HTTP {fetch_result.status_code})"
            )

            if fetch_result.status_code == 0 or not fetch_result.html:
                print(f"[ScraplingIngestion] Empty/failed response for {url}")
                return False

            article = self.parser.parse(fetch_result.html, url)
            concepts = self.extractor.extract(article)

            self._store_ingested(url, article, concepts)

            self.seen_urls.add(url)
            self._save_seen_urls()

            # Conservative rate limit for Scrapling
            time.sleep(2)
            return True

        except Exception as exc:
            print(f"[ScraplingIngestion] Failed {url}: {exc}")
            return False
