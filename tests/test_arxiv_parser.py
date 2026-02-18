"""Tests for FICUTS Task 6.4: ArxivPDFSourceParser"""

import gzip
import io
import tarfile
import time
from unittest.mock import MagicMock, patch

import pytest

from tensor.arxiv_pdf_parser import ArxivPDFSourceParser


# ── Helpers ────────────────────────────────────────────────────────────────────

SAMPLE_LATEX = r"""
\documentclass{article}
\begin{document}

Lyapunov energy is defined as:
\begin{equation}
E(\theta) = \alpha \|\theta\|^2 + \beta \cdot \text{coupling}
\end{equation}

The isometric constraint:
\[
\|z_1 - z_2\| \approx \|f(z_1) - f(z_2)\|
\]

Golden ratio coupling:
\begin{align}
\tau / \gamma &= \phi = 1.618 \\
E_{\text{new}} &< E_{\text{old}}
\end{align}

Display math:
$$\frac{d}{dt}E = -\gamma E + u(t)$$

\end{document}
"""


def _make_tar_gz(latex_content: str) -> bytes:
    """Build an in-memory tar.gz containing one .tex file."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        content_bytes = latex_content.encode("utf-8")
        info = tarfile.TarInfo(name="main.tex")
        info.size = len(content_bytes)
        tar.addfile(info, io.BytesIO(content_bytes))
    return buf.getvalue()


def _make_gzip(latex_content: str) -> bytes:
    return gzip.compress(latex_content.encode("utf-8"))


# ── _extract_paper_id ──────────────────────────────────────────────────────────

def test_extract_paper_id_abs_url():
    p = ArxivPDFSourceParser()
    assert p._extract_paper_id("https://arxiv.org/abs/2602.13213") == "2602.13213"


def test_extract_paper_id_pdf_url():
    p = ArxivPDFSourceParser()
    assert p._extract_paper_id("https://arxiv.org/pdf/2602.13213.pdf") == "2602.13213"


def test_extract_paper_id_bare():
    p = ArxivPDFSourceParser()
    assert p._extract_paper_id("2602.13213") == "2602.13213"


def test_extract_paper_id_strips_version():
    p = ArxivPDFSourceParser()
    assert p._extract_paper_id("https://arxiv.org/abs/2602.13213v1") == "2602.13213"


def test_extract_paper_id_invalid():
    p = ArxivPDFSourceParser()
    assert p._extract_paper_id("https://google.com/search") is None


def test_extract_paper_id_eprint_url():
    p = ArxivPDFSourceParser()
    assert p._extract_paper_id("https://arxiv.org/e-print/2602.13213") == "2602.13213"


# ── _extract_equations ────────────────────────────────────────────────────────

def test_extract_equations_equation_env():
    p = ArxivPDFSourceParser()
    latex = r"\begin{equation}E = mc^2\end{equation}"
    eqs = p.extract_equations_from_latex(latex)
    assert len(eqs) >= 1
    assert "E = mc^2" in eqs[0] or "mc^2" in eqs[0]


def test_extract_equations_align_env():
    p = ArxivPDFSourceParser()
    latex = r"\begin{align}a &= b \\ c &= d\end{align}"
    eqs = p.extract_equations_from_latex(latex)
    assert len(eqs) >= 1


def test_extract_equations_display_math():
    p = ArxivPDFSourceParser()
    latex = r"\[E(\theta) = \alpha \|\theta\|^2\]"
    eqs = p.extract_equations_from_latex(latex)
    assert len(eqs) >= 1


def test_extract_equations_dollar_dollar():
    p = ArxivPDFSourceParser()
    latex = r"$$\phi = 1.618$$"
    eqs = p.extract_equations_from_latex(latex)
    assert len(eqs) >= 1


def test_extract_equations_sample_latex():
    p = ArxivPDFSourceParser()
    eqs = p.extract_equations_from_latex(SAMPLE_LATEX)
    assert len(eqs) >= 4  # equation, \[...\], align, $$...$$


def test_extract_equations_empty_latex():
    p = ArxivPDFSourceParser()
    eqs = p.extract_equations_from_latex("No math here at all.")
    assert eqs == []


def test_extract_equations_skips_trivially_short():
    p = ArxivPDFSourceParser()
    latex = r"\begin{equation}a\end{equation}"
    eqs = p.extract_equations_from_latex(latex)
    # "a" has len 1 → filtered out (< 3 chars)
    assert all(len(e) > 3 for e in eqs)


# ── _try_tar_gz ────────────────────────────────────────────────────────────────

def test_try_tar_gz_success():
    p = ArxivPDFSourceParser()
    tar_content = _make_tar_gz(SAMPLE_LATEX)
    eqs = p._try_tar_gz(tar_content)
    assert eqs is not None
    assert len(eqs) >= 4


def test_try_tar_gz_invalid_returns_none():
    p = ArxivPDFSourceParser()
    result = p._try_tar_gz(b"this is not a tar file at all")
    assert result is None


# ── _try_gzip ─────────────────────────────────────────────────────────────────

def test_try_gzip_success():
    p = ArxivPDFSourceParser()
    gz_content = _make_gzip(SAMPLE_LATEX)
    eqs = p._try_gzip(gz_content)
    assert eqs is not None
    assert len(eqs) >= 4


def test_try_gzip_invalid_returns_none():
    p = ArxivPDFSourceParser()
    result = p._try_gzip(b"not gzip data")
    assert result is None


# ── _try_raw_latex ─────────────────────────────────────────────────────────────

def test_try_raw_latex_success():
    p = ArxivPDFSourceParser()
    eqs = p._try_raw_latex(SAMPLE_LATEX.encode("utf-8"))
    assert eqs is not None
    assert len(eqs) >= 4


def test_try_raw_latex_non_latex_returns_none():
    p = ArxivPDFSourceParser()
    result = p._try_raw_latex(b"just some plain text without LaTeX markers")
    assert result is None


# ── parse_arxiv_paper (mocked HTTP) ───────────────────────────────────────────

def test_parse_arxiv_paper_tar_gz(tmp_path):
    p = ArxivPDFSourceParser(rate_limit_seconds=0)
    tar_content = _make_tar_gz(SAMPLE_LATEX)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = tar_content

    with patch.object(p.session, "get", return_value=mock_resp):
        result = p.parse_arxiv_paper("https://arxiv.org/abs/2602.13213")

    assert result is not None
    assert result["paper_id"] == "2602.13213"
    assert result["num_equations"] >= 4
    assert isinstance(result["equations"], list)


def test_parse_arxiv_paper_returns_none_on_404():
    p = ArxivPDFSourceParser(rate_limit_seconds=0)
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch.object(p.session, "get", return_value=mock_resp):
        result = p.parse_arxiv_paper("https://arxiv.org/abs/9999.99999")

    assert result is None


def test_parse_arxiv_paper_returns_none_on_network_error():
    p = ArxivPDFSourceParser(rate_limit_seconds=0)
    with patch.object(p.session, "get", side_effect=Exception("timeout")):
        result = p.parse_arxiv_paper("https://arxiv.org/abs/2602.13213")
    assert result is None


def test_parse_arxiv_paper_invalid_url():
    p = ArxivPDFSourceParser(rate_limit_seconds=0)
    result = p.parse_arxiv_paper("https://google.com/notarxiv")
    assert result is None


# ── Rate limiting ──────────────────────────────────────────────────────────────

def test_rate_limit_enforced():
    p = ArxivPDFSourceParser(rate_limit_seconds=0.1)
    p._last_request = time.time()

    start = time.time()
    p._rate_limit()
    elapsed = time.time() - start

    assert elapsed >= 0.05  # at least half the limit (allow timing slack)
