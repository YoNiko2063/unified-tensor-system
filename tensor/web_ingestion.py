"""
FICUTS Layer 6: Web Ingestion + Knowledge Extraction

Classes:
  - ArticleParser:           HTML → structured dict (Task 6.1)
  - ResearchConceptExtractor: Extract equations/params/terms (Task 6.2)
  - WebIngestionLoop:        Continuous RSS ingestion + dedup (Task 6.3)
"""

from bs4 import BeautifulSoup
from typing import Dict, List


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
