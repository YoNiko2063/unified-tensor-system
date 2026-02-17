"""
FICUTS Layer 6: Web Ingestion + Knowledge Extraction

Classes:
  - ArticleParser:           HTML → structured dict (Task 6.1)
  - ResearchConceptExtractor: Extract equations/params/terms (Task 6.2)
  - WebIngestionLoop:        Continuous RSS ingestion + dedup (Task 6.3)
"""

import hashlib
import json
import re
import time
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional


class ArticleParser:
    """
    Extract structural information from HTML articles/papers.

    Captures:
    - DOM hierarchy (h1/h2/h3 structure)
    - Hyperlink graph (which concepts link where)
    - Code blocks (equations, algorithms)
    - Emphasis patterns (bold, italic, quotes)
    """

    def parse(self, html: str, url: str) -> Dict:
        soup = BeautifulSoup(html, 'html.parser')

        return {
            'url': url,
            'title': self._extract_title(soup),
            'sections': self._extract_sections(soup),
            'hyperlinks': self._extract_hyperlinks(soup),
            'code_blocks': self._extract_code(soup),
            'emphasis_map': self._extract_emphasis(soup),
            'dom_depth': self._compute_dom_depth(soup),
        }

    def _extract_title(self, soup) -> str:
        """Get title from <title> or <h1>"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.text.strip()
        h1 = soup.find('h1')
        return h1.text.strip() if h1 else 'Untitled'

    def _extract_sections(self, soup) -> List[Dict]:
        """Extract h1/h2/h3 hierarchy"""
        sections = []
        for tag in soup.find_all(['h1', 'h2', 'h3']):
            sections.append({
                'level': int(tag.name[1]),
                'text': tag.text.strip(),
                'position': len(sections),
            })
        return sections

    def _extract_hyperlinks(self, soup) -> List[Dict]:
        """Extract all <a> links with anchor text"""
        links = []
        for a in soup.find_all('a', href=True):
            links.append({
                'href': a['href'],
                'text': a.text.strip(),
                'outbound': not a['href'].startswith('#'),
            })
        return links

    def _extract_code(self, soup) -> List[str]:
        """Extract <pre> code blocks and inline <code>"""
        code_blocks = [block.text.strip() for block in soup.find_all('pre')]
        inline_code = [code.text.strip() for code in soup.find_all('code')]
        return code_blocks + inline_code

    def _extract_emphasis(self, soup) -> Dict:
        """Count bold, italic, blockquote usage"""
        return {
            'bold_count': len(soup.find_all(['b', 'strong'])),
            'italic_count': len(soup.find_all(['i', 'em'])),
            'quote_count': len(soup.find_all('blockquote')),
        }

    def _compute_dom_depth(self, soup) -> int:
        """Maximum nesting depth of DOM tree"""
        def depth(elem):
            if not hasattr(elem, 'children'):
                return 0
            children_depths = [depth(c) for c in elem.children if hasattr(c, 'name')]
            return 1 + max(children_depths) if children_depths else 0
        return depth(soup)


class ResearchConceptExtractor:
    """
    Extract mathematical/scientific concepts from parsed articles.

    Extracts:
    - Equations (LaTeX patterns: \\frac, \\int, \\partial, etc.)
    - Parameters (τ = 5ms, α = 0.01, etc.)
    - Technical terms (Title Case phrases)
    - Experimental indicators (procedure, measured, etc.)
    """

    # Greek letter names to match in English
    _GREEK = (
        r'[α-ωΑ-Ω]'
        r'|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa'
        r'|lambda|mu|nu|xi|rho|sigma|tau|upsilon|phi|chi|psi|omega'
    )

    # LaTeX markers that identify an equation
    _LATEX_INDICATORS = [
        r'\frac', r'\int', r'\sum', r'\partial', r'\Delta',
        r'\nabla', r'\lambda', r'\infty', r'\cdot', r'\times',
    ]

    def extract(self, article: Dict) -> Dict:
        return {
            'equations': self._extract_equations(article),
            'parameters': self._extract_parameters(article),
            'technical_terms': self._extract_technical_terms(article),
            'has_experiment': self._detect_experiment(article),
        }

    def _extract_equations(self, article: Dict) -> List[str]:
        """Find LaTeX or math patterns in code blocks."""
        equations = []
        for code in article.get('code_blocks', []):
            if any(ind in code for ind in self._LATEX_INDICATORS):
                eq = ' '.join(code.split())
                equations.append(eq)
        return equations

    def _extract_parameters(self, article: Dict) -> List[Dict]:
        """
        Find parameter assignments like:
        - τ = 5ms
        - learning rate α = 0.01
        - sigma = 2.5
        """
        params = []
        text = ' '.join(s['text'] for s in article.get('sections', []))

        # pattern: greek_symbol = number [optional units]
        pattern = (
            rf'({self._GREEK})'          # symbol
            r'\s*=\s*'
            r'([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)'   # value
            r'\s*([a-zA-Z]*)'            # optional units
        )

        for match in re.finditer(pattern, text, re.IGNORECASE):
            symbol = match.group(1)
            try:
                value = float(match.group(2))
            except ValueError:
                continue
            units = match.group(3) or None
            context = text[max(0, match.start() - 50):match.end() + 50]
            params.append({
                'symbol': symbol,
                'value': value,
                'units': units,
                'context': context,
            })

        return params

    def _extract_technical_terms(self, article: Dict) -> List[str]:
        """Heuristic: Title Case multi-word phrases (2-4 words)."""
        text = ' '.join(s['text'] for s in article.get('sections', []))
        pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b'
        terms = re.findall(pattern, text)
        return list(set(terms))

    def _detect_experiment(self, article: Dict) -> bool:
        """Does article describe experimental procedure?"""
        indicators = [
            'we measured', 'experiment', 'experimental setup',
            'procedure', 'method', 'methodology',
            'data collection', 'results', 'we observed',
        ]
        text = ' '.join(s['text'] for s in article.get('sections', [])).lower()
        return any(ind in text for ind in indicators)


class WebIngestionLoop:
    """
    Continuously ingest from web sources:
    - RSS feeds (arXiv cs.AI, cs.LG, physics, etc.)
    - News aggregators
    - Research blogs

    Deduplicate by URL, parse, extract concepts, store.
    """

    def __init__(self, storage_dir: str = 'tensor/data/ingested'):
        self.parser = ArticleParser()
        self.extractor = ResearchConceptExtractor()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Load seen URLs
        self.seen_file = self.storage_dir / 'seen_urls.json'
        if self.seen_file.exists():
            self.seen_urls = set(json.loads(self.seen_file.read_text()))
        else:
            self.seen_urls = set()

    def ingest_url(self, url: str) -> bool:
        """
        Fetch single URL, parse, extract, store.

        Returns: True if successfully ingested, False if duplicate or error.
        """
        if url in self.seen_urls:
            return False  # duplicate

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            article = self.parser.parse(response.text, url)
            concepts = self.extractor.extract(article)

            self._store_ingested(url, article, concepts)

            self.seen_urls.add(url)
            self._save_seen_urls()

            print(f"[WebIngestion] Ingested: {url}")
            return True

        except Exception as e:
            print(f"[WebIngestion] Failed {url}: {e}")
            return False

    def run_continuous(self, feed_urls: List[str], interval_seconds: int = 3600):
        """
        Continuously ingest from RSS feeds.

        feed_urls: List of RSS feed URLs (e.g. arXiv RSS)
        interval_seconds: How often to check feeds (default 1 hour)
        """
        print(f"[WebIngestion] Starting continuous loop with {len(feed_urls)} feeds")

        while True:
            for feed_url in feed_urls:
                articles = self._fetch_feed(feed_url)
                print(f"[WebIngestion] Found {len(articles)} articles in {feed_url}")
                for article_url in articles:
                    self.ingest_url(article_url)

            print(f"[WebIngestion] Sleeping {interval_seconds}s until next check")
            time.sleep(interval_seconds)

    def _fetch_feed(self, feed_url: str) -> List[str]:
        """Parse RSS/Atom feed, return article URLs."""
        try:
            feed = feedparser.parse(feed_url)
            return [entry.link for entry in feed.entries if hasattr(entry, 'link')]
        except Exception as e:
            print(f"[WebIngestion] Feed parse error {feed_url}: {e}")
            return []

    def _store_ingested(self, url: str, article: Dict, concepts: Dict):
        """Store parsed article + extracted concepts as JSON."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        filename = self.storage_dir / f"{url_hash}.json"

        data = {
            'url': url,
            'article': article,
            'concepts': concepts,
            'ingested_at': time.time(),
        }
        filename.write_text(json.dumps(data, indent=2))

    def _save_seen_urls(self):
        """Persist seen URLs to disk."""
        self.seen_file.write_text(json.dumps(list(self.seen_urls)))

    def get_ingested_count(self) -> int:
        """Count how many articles are stored (excludes seen_urls.json)."""
        return len([f for f in self.storage_dir.glob('*.json')
                    if f.name != 'seen_urls.json'])
