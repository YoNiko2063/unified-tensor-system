"""
FICUTS Task 6.4: ArXiv LaTeX Source Parser

Downloads LaTeX source (NOT rendered HTML) from arXiv's e-print endpoint.
Extracts equations from .tex files.

arXiv e-print endpoint: https://arxiv.org/e-print/{paper_id}
Returns: tar.gz archive containing .tex, .bib, .sty, image files.

Why this matters:
  - arXiv abstract pages (/abs/) render equations as MathJax → invisible
  - /e-print/ returns the original LaTeX → all equations accessible
  - This is the only reliable way to extract math from arXiv papers

URL types handled:
  - https://arxiv.org/abs/2602.13213
  - https://arxiv.org/pdf/2602.13213.pdf
  - 2602.13213  (bare paper ID)
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


class ArxivPDFSourceParser:
    """
    Download arXiv LaTeX source and extract equations.

    Handles tar.gz archives, gzipped single files, and raw .tex.
    Rate-limited to respect arXiv's server.
    """

    # Equation environments to extract
    _EQUATION_PATTERNS = [
        r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
        r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
        r"\\begin\{eqnarray\*?\}(.*?)\\end\{eqnarray\*?\}",
        r"\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}",
        r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}",
        r"\\\[(.*?)\\\]",
        r"\$\$(.*?)\$\$",
    ]

    def __init__(self, rate_limit_seconds: float = 1.5):
        """
        Args:
            rate_limit_seconds: minimum seconds between HTTP requests.
                                arXiv asks for >= 1s; 1.5s is conservative.
        """
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request: float = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "FICUTSResearchBot/1.0 (educational)"

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse_arxiv_paper(self, url: str) -> Optional[Dict]:
        """
        Download and parse equations from an arXiv paper.

        Args:
            url: arxiv.org/abs/... or arxiv.org/pdf/... URL, or bare paper ID.

        Returns:
            {'paper_id': str, 'equations': List[str], 'num_equations': int}
            or None if download/parse fails.
        """
        paper_id = self._extract_paper_id(url)
        if not paper_id:
            return None

        self._rate_limit()
        source_url = f"https://arxiv.org/e-print/{paper_id}"
        try:
            resp = self.session.get(source_url, timeout=30)
        except Exception:
            return None

        if resp.status_code != 200:
            return None

        equations = self._extract_from_content(resp.content)

        return {
            "paper_id": paper_id,
            "equations": equations,
            "num_equations": len(equations),
        }

    def extract_equations_from_latex(self, latex: str) -> List[str]:
        """
        Public helper: extract equations from a raw LaTeX string.
        Useful for testing and direct LaTeX processing.
        """
        return self._extract_equations(latex)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _extract_paper_id(self, url: str) -> Optional[str]:
        """
        Extract the arXiv paper ID from various URL formats.

        Strips version suffix (e.g. 'v1') so we always get latest.
        """
        url = url.strip()
        # Standard URL forms
        for marker in ("/abs/", "/pdf/", "/e-print/"):
            if marker in url:
                paper_id = url.split(marker)[-1]
                paper_id = paper_id.replace(".pdf", "").split("v")[0]
                return paper_id.strip("/")
        # Bare ID: 2602.13213 or old-style hep-ph/9901332
        if re.match(r"^\d{4}\.\d{4,5}$", url):
            return url
        if re.match(r"^[a-z-]+/\d+$", url):
            return url
        return None

    def _rate_limit(self):
        """Sleep to respect arXiv rate limits."""
        now = time.time()
        wait = self.rate_limit_seconds - (now - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()

    def _extract_from_content(self, content: bytes) -> List[str]:
        """
        Extract equations from raw response bytes.

        Tries formats in order: tar.gz → plain gzip → raw LaTeX.
        """
        # 1. Try tar.gz (most common)
        equations = self._try_tar_gz(content)
        if equations is not None:
            return equations

        # 2. Try plain gzip (single .tex file)
        equations = self._try_gzip(content)
        if equations is not None:
            return equations

        # 3. Try raw bytes as UTF-8 LaTeX
        equations = self._try_raw_latex(content)
        if equations is not None:
            return equations

        return []

    def _try_tar_gz(self, content: bytes) -> Optional[List[str]]:
        """Attempt to parse content as tar.gz archive."""
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                equations = []
                for member in tar.getmembers():
                    if not member.name.endswith(".tex"):
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    latex = f.read().decode("utf-8", errors="ignore")
                    equations.extend(self._extract_equations(latex))
                return equations
        except (tarfile.TarError, Exception):
            return None

    def _try_gzip(self, content: bytes) -> Optional[List[str]]:
        """Attempt to parse content as a single gzipped file."""
        try:
            latex = gzip.decompress(content).decode("utf-8", errors="ignore")
            if "\\begin{document}" in latex or "\\documentclass" in latex:
                return self._extract_equations(latex)
        except Exception:
            pass
        return None

    def _try_raw_latex(self, content: bytes) -> Optional[List[str]]:
        """Attempt to parse content as raw LaTeX text."""
        try:
            latex = content.decode("utf-8", errors="ignore")
            if "\\begin{document}" in latex or "\\documentclass" in latex:
                return self._extract_equations(latex)
        except Exception:
            pass
        return None

    def _extract_equations(self, latex: str) -> List[str]:
        """
        Extract equation environments from LaTeX source.

        Matches: equation, align, eqnarray, multline, gather, \[...\], $$...$$
        Filters: trivially short or empty matches.
        """
        equations = []
        for pattern in self._EQUATION_PATTERNS:
            matches = re.findall(pattern, latex, re.DOTALL)
            for m in matches:
                m = m.strip()
                # Skip trivially empty or noise
                if m and len(m) > 3 and not m.isspace():
                    equations.append(m)
        return equations
