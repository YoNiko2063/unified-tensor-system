"""
Financial-domain web ingestion with adaptive Scrapling element relocation.

Classes:
  - SourceProfile:              Per-domain fetcher + selector configuration
  - AdaptiveElementStore:       Persist learned CSS selector patterns per domain
  - FinancialArticle:           Structured output of a fetched financial page
  - EarningsTranscriptParser:   Specialized parser for earnings call transcripts
  - FilingParser:               Specialized parser for SEC filings
  - FinancialIngestionRouter:   Routes URLs to correct fetcher tier + parser
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import numpy as np

# BeautifulSoup is expected to be available (same dep as web_ingestion.py)
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# ── Scrapling tier imports (all guarded) ─────────────────────────────────────

_FETCHER_CLS = None
_STEALTHY_FETCHER_CLS = None
_ASYNC_FETCHER_CLS = None
_DYNAMIC_FETCHER_CLS = None
_STEALTHY_SESSION_CLS = None

try:
    from scrapling.fetchers import Fetcher as _F
    _FETCHER_CLS = _F
except ImportError:
    pass

try:
    from scrapling.fetchers import StealthyFetcher as _SF
    _STEALTHY_FETCHER_CLS = _SF
except ImportError:
    pass

try:
    from scrapling.fetchers import AsyncFetcher as _AF
    _ASYNC_FETCHER_CLS = _AF
except ImportError:
    pass

try:
    from scrapling.fetchers import DynamicFetcher as _DF
    _DYNAMIC_FETCHER_CLS = _DF
except ImportError:
    pass

try:
    from scrapling.fetchers import StealthySession as _SS
    _STEALTHY_SESSION_CLS = _SS
except ImportError:
    pass

# ── Ticker extraction regex ───────────────────────────────────────────────────

_TICKER_RE = re.compile(r'\$([A-Z]{1,5})\b')


# =============================================================================
# SourceProfile
# =============================================================================

@dataclass
class SourceProfile:
    """Per-domain fetcher and element selector configuration.

    Attributes:
        domain:             Hostname substring, e.g. "reuters.com".
        fetcher_class:      One of "Fetcher"|"StealthyFetcher"|"DynamicFetcher"|
                            "AsyncFetcher".
        rate_limit_seconds: Minimum seconds between requests to this domain.
        selectors:          CSS selectors keyed by element name
                            (e.g. {"headline": "h1", "body": ".article-body"}).
        adaptive:           If True, allow Scrapling adaptive element relocation.
        parser_type:        One of "news"|"transcript"|"filing"|"social"|"academic".
        requires_js:        True when the page requires JavaScript rendering.
    """

    domain: str
    fetcher_class: str
    rate_limit_seconds: float
    selectors: Dict[str, str]
    adaptive: bool = True
    parser_type: str = "news"
    requires_js: bool = False


# =============================================================================
# DEFAULT_SOURCE_PROFILES
# =============================================================================

DEFAULT_SOURCE_PROFILES: List[SourceProfile] = [
    SourceProfile(
        "arxiv.org", "Fetcher", 1.5,
        {"headline": ".title", "body": ".abstract"},
        parser_type="academic",
    ),
    SourceProfile(
        "sec.gov", "Fetcher", 1.0,
        {"body": ".formContent", "date": ".filing-date"},
        parser_type="filing",
    ),
    SourceProfile(
        "reuters.com", "StealthyFetcher", 2.0,
        {
            "headline": "[data-testid='Heading']",
            "body": "[data-testid='paragraph-0']",
            "date": "time",
        },
        adaptive=True, parser_type="news",
    ),
    SourceProfile(
        "ft.com", "StealthyFetcher", 2.0,
        {
            "headline": ".article-headline",
            "body": ".article-body__content",
            "date": "time",
        },
        adaptive=True, parser_type="news",
    ),
    SourceProfile(
        "seekingalpha.com", "DynamicFetcher", 3.0,
        {"headline": "h1", "body": ".sa-art", "date": "time"},
        adaptive=True, parser_type="transcript", requires_js=True,
    ),
    SourceProfile(
        "motleyfool.com", "DynamicFetcher", 3.0,
        {"headline": "h1", "body": ".article-body", "date": "time"},
        adaptive=True, parser_type="transcript", requires_js=True,
    ),
    SourceProfile(
        "finviz.com", "StealthyFetcher", 1.5,
        {"headline": ".news-link-container", "body": ".news-link"},
        adaptive=False, parser_type="social",
    ),
]


# =============================================================================
# AdaptiveElementStore
# =============================================================================

class AdaptiveElementStore:
    """Persist learned Scrapling element patterns per domain.

    Storage layout::

        base_dir/
            reuters_com.json    # {"headline": ".art-head", "body": ".art-body"}
            sec_gov.json
            ...

    Domain names are normalised (dots → underscores) to form filenames.
    """

    def __init__(self, base_dir: str = "tensor/data/scrapling_profiles") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _path(self, domain: str) -> Path:
        safe = domain.replace(".", "_").replace("/", "_")
        return self._base / f"{safe}.json"

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, domain: str) -> Dict:
        """Load saved element patterns for *domain*. Returns ``{}`` if not found."""
        p = self._path(domain)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, domain: str, patterns: Dict) -> None:
        """Persist *patterns* for *domain* to disk."""
        p = self._path(domain)
        p.write_text(json.dumps(patterns, indent=2))

    def get_selector(self, domain: str, element_name: str) -> Optional[str]:
        """Return best known CSS selector for *element_name* on *domain*, or None."""
        patterns = self.load(domain)
        return patterns.get(element_name)


# =============================================================================
# FinancialArticle
# =============================================================================

@dataclass
class FinancialArticle:
    """Structured result of fetching and parsing a financial page.

    Attributes:
        url:          Source URL.
        domain:       Matched domain from SourceProfile (or "unknown").
        headline:     Extracted article headline.
        body:         Extracted body text.
        publish_date: ISO-format date string, or None.
        parser_type:  One of "news"|"transcript"|"filing"|"social"|"academic".
        tickers:      Ticker symbols found via ``$AAPL`` pattern.
        fetcher_used: Which fetcher tier produced the response.
        word_count:   Number of words in body.
        hdv_encoding: Optional HDV vector (set when hdv_system provided).
    """

    url: str
    domain: str
    headline: str
    body: str
    publish_date: Optional[str]
    parser_type: str
    tickers: List[str]
    fetcher_used: str
    word_count: int
    hdv_encoding: Optional[np.ndarray] = None


# =============================================================================
# EarningsTranscriptParser
# =============================================================================

class EarningsTranscriptParser:
    """Extracts structured sections from earnings call transcripts.

    Splits the body at the first occurrence of "Question" (Q&A boundary),
    then scans for forward-looking guidance lines and dollar / EPS figures.
    """

    # Matches $123m, $1.2B, $4.5 billion, $500 million
    _REVENUE_RE = re.compile(
        r'\$\s*[\d,]+(?:\.\d+)?\s*(?:m(?:illion)?|b(?:illion)?|k)?',
        re.IGNORECASE,
    )

    # Matches "EPS of $X.XX" or "earnings per share X.XX"
    _EPS_RE = re.compile(
        r'(?:eps|earnings\s+per\s+share)[^\d$]*[\$]?[\d]+(?:\.\d+)?',
        re.IGNORECASE,
    )

    # Forward-looking signal words
    _GUIDANCE_WORDS = ("expect", "guide", "target", "anticipate", "forecast",
                       "outlook", "project", "estimate")

    def parse(self, body: str) -> Dict:
        """Return structured transcript data.

        Returns dict with keys:
          - prepared_remarks  (str)
          - qa_section        (str)
          - management_guidance (List[str])
          - revenue_mentions  (List[str])
          - eps_mentions      (List[str])
        """
        # Split at Q&A boundary (case-insensitive)
        split_pattern = re.compile(
            r'question[\s\-]*and[\s\-]*answer|q\s*&\s*a\s+session|q\s*&\s*a:',
            re.IGNORECASE,
        )
        match = split_pattern.search(body)
        if match:
            prepared = body[:match.start()].strip()
            qa = body[match.start():].strip()
        else:
            prepared = body.strip()
            qa = ""

        # Guidance lines: sentences containing guidance words
        sentences = re.split(r'(?<=[.!?])\s+', body)
        guidance = [
            s.strip() for s in sentences
            if any(w in s.lower() for w in self._GUIDANCE_WORDS)
        ]

        revenue_mentions = self._REVENUE_RE.findall(body)
        eps_mentions = self._EPS_RE.findall(body)

        return {
            "prepared_remarks": prepared,
            "qa_section": qa,
            "management_guidance": guidance,
            "revenue_mentions": revenue_mentions,
            "eps_mentions": eps_mentions,
        }


# =============================================================================
# FilingParser
# =============================================================================

class FilingParser:
    """Parses SEC filings for key financial signals.

    Detects filing type from URL path or body content, identifies the 8-K
    Item that was triggered, and extracts dollar amounts / percentages from
    the first 500 words.
    """

    _FILING_TYPES = ("10-K", "10-Q", "8-K", "13F", "S-1", "DEF 14A", "424B")
    _ITEM_RE = re.compile(r'item\s+(\d+\.\d+)', re.IGNORECASE)
    _DOLLAR_RE = re.compile(r'\$\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?',
                            re.IGNORECASE)
    _PCT_RE = re.compile(r'[\d]+(?:\.\d+)?\s*%')

    def parse(self, body: str, url: str) -> Dict:
        """Return structured filing data.

        Returns dict with keys:
          - filing_type  (str or "unknown")
          - event_type   (str or None)  — e.g. "Item 2.02" for 8-K earnings
          - key_figures  (List[str])    — dollar amounts + percentages
        """
        filing_type = self._detect_filing_type(body, url)
        event_type = self._detect_event_type(body, filing_type)

        # Key figures from first 500 words
        first_500 = " ".join(body.split()[:500])
        dollars = self._DOLLAR_RE.findall(first_500)
        pcts = self._PCT_RE.findall(first_500)
        key_figures = dollars + pcts

        return {
            "filing_type": filing_type,
            "event_type": event_type,
            "key_figures": key_figures,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_filing_type(self, body: str, url: str) -> str:
        """Infer filing type from URL then body content."""
        url_upper = url.upper()
        for ft in self._FILING_TYPES:
            if ft.replace("-", "") in url_upper.replace("-", ""):
                return ft
        # Fall back to body scan (first 200 words)
        head = " ".join(body.split()[:200]).upper()
        for ft in self._FILING_TYPES:
            if ft.replace("-", "") in head.replace("-", ""):
                return ft
        return "unknown"

    def _detect_event_type(self, body: str, filing_type: str) -> Optional[str]:
        """For 8-K filings, identify which Item was triggered."""
        if filing_type != "8-K":
            return None
        m = self._ITEM_RE.search(body)
        if m:
            return f"Item {m.group(1)}"
        return None


# =============================================================================
# FinancialIngestionRouter
# =============================================================================

# Fetcher fallback chain (most capable → least capable)
_FETCHER_CHAIN = [
    ("DynamicFetcher",  _DYNAMIC_FETCHER_CLS),
    ("StealthyFetcher", _STEALTHY_FETCHER_CLS),
    ("Fetcher",         _FETCHER_CLS),
]


class FinancialIngestionRouter:
    """Routes financial URLs to the correct Scrapling fetcher tier + parser.

    Priority:
      1. Match URL domain to SourceProfile in registry.
      2. Select fetcher class from profile (with fallback chain).
      3. Extract elements using profile selectors + adaptive store.
      4. Parse extracted content using parser_type-specific parser.
      5. Return FinancialArticle with structured fields.
    """

    def __init__(
        self,
        profiles: Optional[List[SourceProfile]] = None,
        adaptive_store: Optional[AdaptiveElementStore] = None,
        hdv_system=None,
    ) -> None:
        self._profiles: Dict[str, SourceProfile] = {}
        for p in (profiles if profiles is not None else DEFAULT_SOURCE_PROFILES):
            self._profiles[p.domain] = p

        self._store = adaptive_store if adaptive_store is not None else AdaptiveElementStore()
        self._hdv = hdv_system

        # Parser singletons
        self._transcript_parser = EarningsTranscriptParser()
        self._filing_parser = FilingParser()

        # Rate-limit tracking: domain → last request time
        self._last_request: Dict[str, float] = {}

        # Cached fetcher instances keyed by class name
        self._fetcher_instances: Dict[str, object] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def register_source(self, profile: SourceProfile) -> None:
        """Add or override a source profile."""
        self._profiles[profile.domain] = profile

    def fetch(self, url: str) -> FinancialArticle:
        """Fetch and parse a financial URL using the appropriate fetcher tier.

        Fetcher fallback chain: DynamicFetcher → StealthyFetcher → Fetcher → requests
        """
        profile = self._detect_domain(url)
        fetcher, fetcher_name = self._select_fetcher(profile)

        # Apply rate limiting
        if profile is not None:
            self._rate_limit(profile)

        html = self._do_fetch(fetcher, fetcher_name, url)
        elements = self._extract_elements(html, profile)
        article = self._parse_article(elements, profile, url)
        article.fetcher_used = fetcher_name

        # Optional HDV encoding
        if self._hdv is not None:
            combined_text = f"{article.headline} {article.body}"
            domain_label = profile.domain if profile else "financial"
            try:
                article.hdv_encoding = self._hdv.structural_encode(
                    combined_text, domain_label
                )
            except Exception:
                pass

        return article

    async def fetch_batch(self, urls: List[str]) -> List[FinancialArticle]:
        """Concurrent batch fetch using AsyncFetcher where available.

        Falls back to sequential fetch when AsyncFetcher is unavailable.
        """
        if _ASYNC_FETCHER_CLS is not None:
            return await self._fetch_batch_async(urls)
        # Sequential fallback
        loop = asyncio.get_event_loop()
        results = []
        for url in urls:
            article = await loop.run_in_executor(None, self.fetch, url)
            results.append(article)
        return results

    def _select_fetcher(self, profile: Optional[SourceProfile]):
        """Return (fetcher_instance, class_name_str) for the best available tier.

        Walks the fallback chain starting from the profile's requested class.
        Always ends with requests as the final fallback.
        """
        desired = profile.fetcher_class if profile is not None else "Fetcher"

        # Build ordered chain starting from desired class
        chain = []
        desired_found = False
        for name, cls in _FETCHER_CHAIN:
            if name == desired:
                desired_found = True
            if desired_found:
                chain.append((name, cls))

        # If desired class not in chain (e.g. "AsyncFetcher"), use full chain
        if not chain:
            chain = list(_FETCHER_CHAIN)

        for name, cls in chain:
            if cls is not None:
                inst = self._get_fetcher_instance(name, cls)
                if inst is not None:
                    return inst, name

        # Ultimate fallback: raw requests
        return None, "requests"

    def _detect_domain(self, url: str) -> Optional[SourceProfile]:
        """Match URL to registered SourceProfile by domain substring."""
        try:
            hostname = urlparse(url).hostname or ""
        except Exception:
            hostname = ""

        for domain, profile in self._profiles.items():
            if domain in hostname:
                return profile
        return None

    # ── Extraction + parsing ──────────────────────────────────────────────────

    def _extract_elements(self, html: str, profile: Optional[SourceProfile]) -> Dict[str, str]:
        """Extract elements from HTML using profile selectors.

        Uses BeautifulSoup with CSS selectors. Falls back to adaptive store
        when the primary selector yields nothing.
        """
        result: Dict[str, str] = {}

        if not _BS4_AVAILABLE or not html:
            return result

        soup = BeautifulSoup(html, "html.parser")

        selectors = profile.selectors if profile is not None else {}
        domain = profile.domain if profile is not None else ""

        for element_name, selector in selectors.items():
            text = self._try_selector(soup, selector)

            # Adaptive store fallback
            if not text and domain and profile is not None and profile.adaptive:
                stored_selector = self._store.get_selector(domain, element_name)
                if stored_selector and stored_selector != selector:
                    text = self._try_selector(soup, stored_selector)

            result[element_name] = text

        # Generic title fallback for headline
        if not result.get("headline"):
            title_tag = soup.find("title")
            if title_tag:
                result["headline"] = title_tag.get_text(strip=True)
            else:
                h1 = soup.find("h1")
                result["headline"] = h1.get_text(strip=True) if h1 else ""

        # Generic body fallback
        if not result.get("body"):
            body_tag = soup.find("body")
            result["body"] = body_tag.get_text(separator=" ", strip=True) if body_tag else ""

        return result

    def _parse_article(
        self, elements: Dict[str, str], profile: Optional[SourceProfile], url: str
    ) -> FinancialArticle:
        """Convert extracted elements into a FinancialArticle."""
        headline = elements.get("headline", "")
        body = elements.get("body", "")
        publish_date = elements.get("date") or None
        parser_type = profile.parser_type if profile is not None else "news"
        domain = profile.domain if profile is not None else "unknown"

        # Ticker extraction
        combined = f"{headline} {body}"
        tickers = list(dict.fromkeys(_TICKER_RE.findall(combined)))  # deduplicated

        word_count = len(body.split()) if body else 0

        article = FinancialArticle(
            url=url,
            domain=domain,
            headline=headline,
            body=body,
            publish_date=publish_date,
            parser_type=parser_type,
            tickers=tickers,
            fetcher_used="requests",  # overwritten by fetch()
            word_count=word_count,
        )
        return article

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _try_selector(soup, selector: str) -> str:
        """Apply a CSS selector and return stripped text, or '' on failure."""
        try:
            tag = soup.select_one(selector)
            if tag:
                return tag.get_text(separator=" ", strip=True)
        except Exception:
            pass
        return ""

    def _get_fetcher_instance(self, name: str, cls) -> Optional[object]:
        """Return a cached fetcher instance, creating one if needed."""
        if name not in self._fetcher_instances:
            try:
                self._fetcher_instances[name] = cls()
            except Exception:
                self._fetcher_instances[name] = None
        return self._fetcher_instances[name]

    def _do_fetch(self, fetcher, fetcher_name: str, url: str) -> str:
        """Dispatch to the correct fetcher and return raw HTML string."""
        if fetcher_name == "requests" or fetcher is None:
            return self._fetch_requests(url)

        try:
            response = fetcher.get(url, timeout=15)
            html = str(response.text) if response.text else ""
            return html
        except Exception:
            # Fallback to requests on any scrapling error
            return self._fetch_requests(url)

    @staticmethod
    def _fetch_requests(url: str) -> str:
        """Fallback HTTP fetch via requests."""
        import requests as _requests
        try:
            resp = _requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (compatible; FinancialIngestion/1.0)"
            })
            return resp.text
        except Exception:
            return ""

    def _rate_limit(self, profile: SourceProfile) -> None:
        """Sleep if needed to honour profile.rate_limit_seconds."""
        domain = profile.domain
        last = self._last_request.get(domain, 0.0)
        elapsed = time.time() - last
        if elapsed < profile.rate_limit_seconds:
            time.sleep(profile.rate_limit_seconds - elapsed)
        self._last_request[domain] = time.time()

    async def _fetch_batch_async(self, urls: List[str]) -> List[FinancialArticle]:
        """Use AsyncFetcher for concurrent batch fetching."""
        async_fetcher = _ASYNC_FETCHER_CLS()  # type: ignore[misc]
        results: List[FinancialArticle] = []

        async def fetch_one(url: str) -> FinancialArticle:
            profile = self._detect_domain(url)
            try:
                response = await async_fetcher.get(url, timeout=15)
                html = str(response.text) if response.text else ""
            except Exception:
                html = ""
            elements = self._extract_elements(html, profile)
            article = self._parse_article(elements, profile, url)
            article.fetcher_used = "AsyncFetcher"
            return article

        tasks = [fetch_one(u) for u in urls]
        results = list(await asyncio.gather(*tasks, return_exceptions=False))
        return results
