"""Tests for FICUTS Tasks 11.1 & 11.2: DeepWiki + GitHub API"""

import base64
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tensor.integrated_hdv import IntegratedHDVSystem
from tensor.deepwiki_integration import DeepWikiWorkflowParser
from tensor.github_api_fallback import GitHubAPICapabilityExtractor


HDV = 500
MODES = 5
EMBED = 32


@pytest.fixture
def hdv_sys(tmp_path):
    return IntegratedHDVSystem(
        hdv_dim=HDV, n_modes=MODES, embed_dim=EMBED,
        library_path=str(tmp_path / "lib.json"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DeepWikiWorkflowParser
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeepWikiWorkflowParser:

    @pytest.fixture
    def parser(self):
        return DeepWikiWorkflowParser(rate_limit_seconds=0)

    # ── _parse_owner_repo ────────────────────────────────────────────────────

    def test_parse_owner_repo_standard(self, parser):
        result = parser._parse_owner_repo("https://github.com/wmjordan/PDFPatcher")
        assert result == ("wmjordan", "PDFPatcher")

    def test_parse_owner_repo_trailing_slash(self, parser):
        result = parser._parse_owner_repo("https://github.com/numpy/numpy/")
        assert result == ("numpy", "numpy")

    def test_parse_owner_repo_invalid(self, parser):
        result = parser._parse_owner_repo("https://google.com")
        assert result is None

    # ── encode_workflow_to_hdv ────────────────────────────────────────────────

    def test_encode_workflow_to_hdv_shape(self, parser, hdv_sys):
        workflow_data = {
            "intent": "PDF manipulation tool",
            "workflow": ["load pdf", "extract pages", "merge", "save output"],
            "components": [],
            "dependencies": [],
        }
        vec = parser.encode_workflow_to_hdv(workflow_data, hdv_sys)
        assert vec.shape == (HDV,)
        assert vec.dtype == np.float32

    def test_encode_workflow_to_hdv_nonzero(self, parser, hdv_sys):
        workflow_data = {
            "intent": "3D printer slicer",
            "workflow": ["parse stl", "slice layers", "generate gcode"],
        }
        vec = parser.encode_workflow_to_hdv(workflow_data, hdv_sys)
        assert vec.sum() > 0

    def test_encode_workflow_different_workflows_differ(self, parser, hdv_sys):
        wf1 = {"intent": "PDF tool", "workflow": ["load", "merge", "save"]}
        wf2 = {"intent": "3D printer", "workflow": ["stl", "slice", "gcode"]}
        v1 = parser.encode_workflow_to_hdv(wf1, hdv_sys)
        v2 = parser.encode_workflow_to_hdv(wf2, hdv_sys)
        assert not np.array_equal(v1, v2)

    def test_encode_empty_workflow(self, parser, hdv_sys):
        wf = {"intent": "", "workflow": []}
        vec = parser.encode_workflow_to_hdv(wf, hdv_sys)
        assert vec.shape == (HDV,)

    # ── parse_deepwiki_summary (mocked) ───────────────────────────────────────

    SAMPLE_HTML = """
    <html><body>
    <p>PDFPatcher is a comprehensive PDF manipulation tool that allows
    users to merge, split, and extract pages from PDF documents with ease.</p>
    <ul>
      <li>Load PDF document</li>
      <li>Extract specific pages</li>
      <li>Merge multiple files</li>
      <li>Save merged output</li>
    </ul>
    <code>class PdfReader { ... }</code>
    <p>import iTextSharp for PDF processing</p>
    </body></html>
    """

    def test_parse_deepwiki_returns_structured_data(self, parser):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = self.SAMPLE_HTML

        with patch.object(parser.session, "get", return_value=mock_resp):
            result = parser.parse_deepwiki_summary(
                "https://github.com/wmjordan/PDFPatcher"
            )

        assert result is not None
        assert "intent" in result
        assert "workflow" in result
        assert isinstance(result["workflow"], list)
        assert result["source"] in ("deepwiki", "deepwiki_regex_fallback")

    def test_parse_deepwiki_returns_none_on_404(self, parser):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(parser.session, "get", return_value=mock_resp):
            result = parser.parse_deepwiki_summary(
                "https://github.com/nonexistent/repo"
            )

        assert result is None

    def test_parse_deepwiki_returns_none_on_network_error(self, parser):
        with patch.object(parser.session, "get", side_effect=Exception("timeout")):
            result = parser.parse_deepwiki_summary("https://github.com/x/y")
        assert result is None

    def test_parse_deepwiki_caches_result(self, parser):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = self.SAMPLE_HTML

        with patch.object(parser.session, "get", return_value=mock_resp) as mock_get:
            parser.parse_deepwiki_summary("https://github.com/wmjordan/PDFPatcher")
            parser.parse_deepwiki_summary("https://github.com/wmjordan/PDFPatcher")
            # Second call should use cache, not hit network
            assert mock_get.call_count == 1

    def test_parse_deepwiki_invalid_github_url(self, parser):
        result = parser.parse_deepwiki_summary("https://notgithub.com/x")
        assert result is None

    # ── batch_process_repos (mocked) ─────────────────────────────────────────

    def test_batch_process_saves_to_json(self, parser, hdv_sys, tmp_path):
        save_path = str(tmp_path / "workflows.json")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = self.SAMPLE_HTML

        with patch.object(parser.session, "get", return_value=mock_resp):
            results = parser.batch_process_repos(
                ["https://github.com/wmjordan/PDFPatcher"],
                hdv_sys,
                save_path=save_path,
            )

        assert Path(save_path).exists()
        saved = json.loads(Path(save_path).read_text())
        assert len(saved) >= 1

    def test_batch_process_skips_cached(self, parser, hdv_sys, tmp_path):
        save_path = str(tmp_path / "workflows.json")
        # Pre-populate cache file
        existing = {
            "https://github.com/wmjordan/PDFPatcher": {
                "capability": {"intent": "PDF tool", "workflow": ["load", "save"]},
                "hdv": [0.0] * HDV,
            }
        }
        Path(save_path).write_text(json.dumps(existing))

        with patch.object(parser.session, "get") as mock_get:
            results = parser.batch_process_repos(
                ["https://github.com/wmjordan/PDFPatcher"],
                hdv_sys,
                save_path=save_path,
            )
            # Should not make any HTTP requests (already cached)
            assert mock_get.call_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# GitHubAPICapabilityExtractor
# ═══════════════════════════════════════════════════════════════════════════════

class TestGitHubAPICapabilityExtractor:

    @pytest.fixture
    def extractor(self, tmp_path):
        return GitHubAPICapabilityExtractor(
            token=None,
            templates_path=str(tmp_path / "empty_templates.json"),
            rate_limit_seconds=0,
        )

    # ── _parse_owner_repo ────────────────────────────────────────────────────

    def test_parse_owner_repo(self, extractor):
        assert extractor._parse_owner_repo(
            "https://github.com/numpy/numpy"
        ) == ("numpy", "numpy")

    def test_parse_owner_repo_invalid(self, extractor):
        assert extractor._parse_owner_repo("not_a_url") is None

    # ── _infer_intent ────────────────────────────────────────────────────────

    def test_infer_intent_from_description(self, extractor):
        intent = extractor._infer_intent("A fast PDF merger library", "")
        assert "PDF" in intent

    def test_infer_intent_from_readme(self, extractor):
        readme = "# MyLib\n\nMyLib is a library that computes fast Fourier transforms."
        intent = extractor._infer_intent("", readme)
        assert len(intent) > 10

    # ── _infer_workflow ───────────────────────────────────────────────────────

    def test_infer_workflow_from_usage_section(self, extractor):
        readme = """# Repo
## Usage
1. Install the library
2. Import and initialize
3. Call process()
4. Get output
"""
        workflow = extractor._infer_workflow(readme, [])
        assert len(workflow) >= 1

    def test_infer_workflow_fallback(self, extractor):
        workflow = extractor._infer_workflow("No usage section", [])
        assert isinstance(workflow, list)
        assert len(workflow) > 0

    # ── _extract_file_patterns ────────────────────────────────────────────────

    def test_extract_file_patterns_python(self, extractor):
        files = ["src/main.py", "tests/test_main.py", "requirements.txt"]
        patterns = extractor._extract_file_patterns(files)
        assert "requirements.txt" in patterns

    def test_extract_file_patterns_deep_files_excluded(self, extractor):
        files = ["src/a/b/c/d/deep.py"]
        patterns = extractor._extract_file_patterns(files)
        # Too deep (>3 parts) → excluded
        assert "src/a/b/c/d/deep.py" not in patterns

    # ── _extract_dependencies ────────────────────────────────────────────────

    def test_extract_dependencies_pip(self, extractor):
        readme = "pip install numpy scipy pandas"
        deps = extractor._extract_dependencies(readme, "Python")
        assert "numpy" in deps
        assert "scipy" in deps

    # ── extract_capability_via_api (mocked) ──────────────────────────────────

    def _make_meta(self):
        return {
            "description": "Fast PDF manipulation library",
            "language": "Python",
            "stargazers_count": 1234,
        }

    def _make_readme(self, text: str) -> dict:
        return {"content": base64.b64encode(text.encode()).decode()}

    def _make_tree(self) -> dict:
        return {
            "tree": [
                {"path": "main.py", "type": "blob"},
                {"path": "requirements.txt", "type": "blob"},
            ]
        }

    def test_extract_capability_full(self, extractor):
        readme_text = (
            "# PDFTool\nPDFTool merges and splits PDF files.\n"
            "## Usage\n1. Load files\n2. Merge\n3. Save\n"
        )

        responses = [
            MagicMock(status_code=200, json=lambda: self._make_meta()),
            MagicMock(status_code=200, json=lambda: self._make_readme(readme_text)),
            MagicMock(status_code=200, json=lambda: self._make_tree()),
        ]
        responses[0].json = lambda: self._make_meta()
        responses[1].json = lambda: self._make_readme(readme_text)
        responses[2].json = lambda: self._make_tree()

        call_count = [0]
        def fake_get(url, **kwargs):
            r = responses[call_count[0]]
            call_count[0] += 1
            return r

        with patch.object(extractor.session, "get", side_effect=fake_get):
            result = extractor.extract_capability_via_api(
                "https://github.com/wmjordan/PDFPatcher"
            )

        assert result is not None
        assert result["language"] == "Python"
        assert result["stars"] == 1234
        assert "intent" in result
        assert isinstance(result["workflow"], list)
        assert result["source"] == "github_api"

    def test_extract_capability_returns_none_on_404(self, extractor):
        with patch.object(
            extractor.session, "get",
            return_value=MagicMock(status_code=404),
        ):
            result = extractor.extract_capability_via_api(
                "https://github.com/nonexistent/repo99999"
            )
        assert result is None

    def test_extract_capability_invalid_url(self, extractor):
        result = extractor.extract_capability_via_api("https://notgithub.com/x")
        assert result is None

    # ── match_to_deepwiki_template ────────────────────────────────────────────

    def test_match_to_template_no_templates(self, extractor, hdv_sys):
        cap = {"workflow": ["load", "process", "save"]}
        result = extractor.match_to_deepwiki_template(cap, hdv_sys)
        assert result is None  # no templates loaded

    def test_match_to_template_with_similar(self, hdv_sys, tmp_path):
        """Build a template, then match a very similar workflow."""
        # Create extractor with a pre-built template
        template_hdv = hdv_sys.encode_workflow(
            ["load pdf", "merge", "save output"], domain="behavioral"
        )
        templates_path = tmp_path / "templates.json"
        templates_path.write_text(json.dumps({
            "https://github.com/template/repo": {
                "capability": {"workflow": ["load pdf", "merge", "save output"]},
                "hdv": template_hdv.tolist(),
            }
        }))

        extractor = GitHubAPICapabilityExtractor(
            templates_path=str(templates_path),
            rate_limit_seconds=0,
        )
        # Very similar workflow
        cap = {"workflow": ["load pdf", "merge", "save output"]}
        result = extractor.match_to_deepwiki_template(cap, hdv_sys)
        assert result == "https://github.com/template/repo"
