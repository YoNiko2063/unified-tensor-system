"""
FICUTS Layer 6: Web Ingestion + Knowledge Extraction

Classes:
  - ArticleParser:           HTML → structured dict (Task 6.1)
  - ResearchConceptExtractor: Extract equations/params/terms (Task 6.2)
  - WebIngestionLoop:        Continuous RSS ingestion + dedup (Task 6.3)
"""

import re

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
