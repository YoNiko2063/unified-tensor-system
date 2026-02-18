"""Tests for tensor/deepwiki_navigator.py (Phase 1 of CLAUDE_CODE_IMPLEMENTATION_PLAN)

All HTTP calls are mocked so tests are fast and offline.
"""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tensor.deepwiki_navigator import (
    CapabilityDiscovery,
    DeepWikiBookExtractor,
    DeepWikiChallengeExtractor,
    DeepWikiNavigator,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def b64(text: str) -> str:
    """Base-64 encode a string (mimics GitHub API content field)."""
    return base64.b64encode(text.encode()).decode()


def mock_resp(status_code: int = 200, json_data=None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text
    return resp


@pytest.fixture
def tmp_nav(tmp_path):
    return DeepWikiNavigator(
        cache_dir=str(tmp_path / "cache"),
        rate_limit_seconds=0,
    )


# ── DeepWikiNavigator ────────────────────────────────────────────────────────


class TestDeepWikiNavigator:
    def test_navigate_returns_dict_always(self, tmp_nav):
        """navigate_repo never raises; returns dict even on total HTTP failure."""
        with patch.object(tmp_nav.session, "get", return_value=mock_resp(404)):
            data = tmp_nav.navigate_repo("owner", "repo")
        assert isinstance(data, dict)
        for key in ("repo_url", "summary", "file_tree", "key_files",
                    "dependencies", "insights"):
            assert key in data

    def test_navigate_fallback_summary(self, tmp_nav):
        """When DeepWiki and GitHub both fail, summary defaults to 'repo repository'."""
        with patch.object(tmp_nav.session, "get", return_value=mock_resp(404)):
            data = tmp_nav.navigate_repo("owner", "myrepo")
        assert "myrepo" in data["summary"].lower()

    def test_navigate_uses_github_description(self, tmp_nav):
        """Falls back to GitHub repo description when DeepWiki returns nothing."""
        with patch.object(tmp_nav, "_try_deepwiki_summary", return_value=None), \
             patch.object(tmp_nav, "_get_github_meta",
                          return_value={"description": "My cool library",
                                        "language": "Python",
                                        "stargazers_count": 10}), \
             patch.object(tmp_nav, "_get_github_tree", return_value=[]), \
             patch.object(tmp_nav, "_infer_dependencies", return_value=[]):
            data = tmp_nav.navigate_repo("owner", "repo")
        assert "cool library" in data["summary"].lower()

    def test_navigate_uses_cache(self, tmp_nav):
        """Second call returns cached data without any HTTP requests."""
        cached = {
            "repo_url": "owner/repo",
            "summary": "cached!",
            "file_tree": [],
            "key_files": [],
            "dependencies": [],
            "insights": {},
        }
        cache_file = tmp_nav.cache_dir / "owner_repo.json"
        cache_file.write_text(json.dumps(cached))

        with patch.object(tmp_nav.session, "get") as mock_get:
            data = tmp_nav.navigate_repo("owner", "repo")
            mock_get.assert_not_called()

        assert data["summary"] == "cached!"

    def test_navigate_caps_file_tree(self, tmp_nav):
        """File tree is capped at 200 entries."""
        tree_items = [{"type": "blob", "path": f"f{i}.py"} for i in range(300)]
        with patch.object(tmp_nav, "_try_deepwiki_summary", return_value=None), \
             patch.object(tmp_nav, "_get_github_meta", return_value=None), \
             patch.object(tmp_nav, "_get_github_tree",
                          return_value=[{"path": f"f{i}.py", "type": "file"}
                                        for i in range(300)]), \
             patch.object(tmp_nav, "_infer_dependencies", return_value=[]):
            data = tmp_nav.navigate_repo("owner", "repo")
        assert len(data["file_tree"]) <= 200

    def test_navigate_saves_cache(self, tmp_nav):
        """navigate_repo writes a JSON cache file."""
        with patch.object(tmp_nav, "_try_deepwiki_summary", return_value="hi"), \
             patch.object(tmp_nav, "_get_github_meta", return_value=None), \
             patch.object(tmp_nav, "_get_github_tree", return_value=[]), \
             patch.object(tmp_nav, "_infer_dependencies", return_value=[]):
            tmp_nav.navigate_repo("owner", "newrepo")

        cache_file = tmp_nav.cache_dir / "owner_newrepo.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["summary"] == "hi"

    def test_identify_key_files_readme(self, tmp_nav):
        """README.md and pyproject.toml are detected as key files."""
        tree = [
            {"path": "README.md",       "type": "file"},
            {"path": "src/core.py",     "type": "file"},
            {"path": "pyproject.toml",  "type": "file"},
            {"path": "tests/test_a.py", "type": "file"},
        ]
        keys = tmp_nav._identify_key_files(tree)
        assert "README.md"      in keys
        assert "pyproject.toml" in keys
        assert "tests/test_a.py" not in keys

    def test_identify_key_files_package_json(self, tmp_nav):
        tree = [
            {"path": "package.json", "type": "file"},
            {"path": "index.js",     "type": "file"},
            {"path": "lib/util.js",  "type": "file"},
        ]
        keys = tmp_nav._identify_key_files(tree)
        assert "package.json" in keys
        assert "index.js"     in keys

    def test_parse_deps_requirements(self, tmp_nav):
        content = "numpy>=1.20\npandas==1.3\n# comment\nscipy\n"
        deps = tmp_nav._parse_deps(content, "requirements.txt")
        assert "numpy"  in deps
        assert "pandas" in deps
        assert "scipy"  in deps

    def test_parse_deps_package_json(self, tmp_nav):
        pkg = json.dumps({"dependencies": {"react": "^18.0", "axios": "1.0"}})
        deps = tmp_nav._parse_deps(pkg, "package.json")
        assert "react" in deps
        assert "axios" in deps

    def test_compute_insights_language_from_extensions(self, tmp_nav):
        tree = [{"path": f"src/f{i}.py", "type": "file"} for i in range(5)]
        tree.append({"path": "main.js", "type": "file"})
        ins = tmp_nav._compute_insights(tree, None)
        assert ins["language"] == "py"

    def test_compute_insights_github_meta_overrides_language(self, tmp_nav):
        tree = [{"path": "main.py", "type": "file"}]
        meta = {"language": "Python", "stargazers_count": 42}
        ins = tmp_nav._compute_insights(tree, meta)
        assert ins["language"] == "Python"
        assert ins["star_count"] == 42

    def test_compute_insights_empty_tree(self, tmp_nav):
        """No crash on empty file tree."""
        ins = tmp_nav._compute_insights([], None)
        assert ins["language"] == "unknown"
        assert ins["star_count"] == 0

    def test_compute_insights_patterns_from_top_dirs(self, tmp_nav):
        tree = [
            {"path": "src/core.py",   "type": "file"},
            {"path": "tests/test.py", "type": "file"},
            {"path": "docs/guide.md", "type": "file"},
        ]
        ins = tmp_nav._compute_insights(tree, None)
        assert "src"   in ins["patterns"]
        assert "tests" in ins["patterns"]
        assert "docs"  in ins["patterns"]

    def test_try_deepwiki_summary_404(self, tmp_nav):
        """Returns None when requests fallback returns 404 (Scrapling disabled)."""
        # Disable Scrapling so we test the requests path deterministically
        with patch("tensor.deepwiki_navigator._SCRAPLING_AVAILABLE", False), \
             patch.object(tmp_nav.session, "get", return_value=mock_resp(404)):
            result = tmp_nav._try_deepwiki_summary("owner", "repo")
        assert result is None

    def test_try_deepwiki_summary_empty_html(self, tmp_nav):
        """Returns None when page has no long sentences (requests fallback)."""
        with patch("tensor.deepwiki_navigator._SCRAPLING_AVAILABLE", False), \
             patch.object(tmp_nav.session, "get",
                          return_value=mock_resp(200, text="<html><body></body></html>")):
            result = tmp_nav._try_deepwiki_summary("owner", "repo")
        assert result is None

    def test_try_deepwiki_summary_extracts_text(self, tmp_nav):
        """Extracts a long sentence from HTML (requests fallback)."""
        html = "<html><body><p>This is a very long description about the project purpose and goals which has over 50 characters.</p></body></html>"
        with patch("tensor.deepwiki_navigator._SCRAPLING_AVAILABLE", False), \
             patch.object(tmp_nav.session, "get",
                          return_value=mock_resp(200, text=html)):
            result = tmp_nav._try_deepwiki_summary("owner", "repo")
        assert result is not None
        assert len(result) > 30

    def test_try_deepwiki_summary_scrapling_path(self, tmp_nav):
        """Uses Scrapling when available and returns paragraph text."""
        fake_para = MagicMock()
        fake_para.text = "This is a detailed description with more than fifty characters total here."
        fake_resp = MagicMock()
        fake_resp.status = 200
        fake_resp.find_all.return_value = [fake_para]

        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = fake_resp

        with patch("tensor.deepwiki_navigator._SCRAPLING_AVAILABLE", True), \
             patch("tensor.deepwiki_navigator._ScraplingFetcher", return_value=mock_fetcher):
            result = tmp_nav._try_deepwiki_summary("owner", "repo")

        assert result is not None
        assert len(result) > 30

    def test_try_deepwiki_summary_scrapling_falls_back_on_exception(self, tmp_nav):
        """Falls back to requests when Scrapling raises an exception."""
        html = "<html><body><p>Fallback path paragraph that is definitely longer than fifty chars.</p></body></html>"
        with patch("tensor.deepwiki_navigator._SCRAPLING_AVAILABLE", True), \
             patch("tensor.deepwiki_navigator._ScraplingFetcher", side_effect=RuntimeError("no browser")), \
             patch.object(tmp_nav.session, "get", return_value=mock_resp(200, text=html)):
            result = tmp_nav._try_deepwiki_summary("owner", "repo")
        assert result is not None

    def test_get_github_tree_404(self, tmp_nav):
        with patch.object(tmp_nav.session, "get", return_value=mock_resp(404)):
            tree = tmp_nav._get_github_tree("owner", "repo")
        assert tree == []

    def test_get_github_tree_parses_response(self, tmp_nav):
        tree_data = {
            "tree": [
                {"type": "blob", "path": "README.md"},
                {"type": "tree", "path": "src"},
                {"type": "blob", "path": "src/main.py"},
            ]
        }
        with patch.object(tmp_nav.session, "get",
                          return_value=mock_resp(200, json_data=tree_data)):
            tree = tmp_nav._get_github_tree("owner", "repo")
        assert len(tree) == 3
        types = {item["path"]: item["type"] for item in tree}
        assert types["src"]      == "dir"
        assert types["README.md"] == "file"


# ── DeepWikiChallengeExtractor ────────────────────────────────────────────────


@pytest.fixture
def challenge_ext(tmp_nav):
    return DeepWikiChallengeExtractor(tmp_nav, rate_limit_seconds=0)


class TestDeepWikiChallengeExtractor:
    def test_extract_challenges_empty_tree(self, challenge_ext, tmp_nav):
        with patch.object(tmp_nav, "navigate_repo",
                          return_value={"file_tree": [], "summary": "test"}):
            result = challenge_ext.extract_challenges()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_infer_difficulty_basic(self, challenge_ext):
        assert challenge_ext._infer_difficulty("src/basic/foo") == "basic"

    def test_infer_difficulty_intermediate(self, challenge_ext):
        assert challenge_ext._infer_difficulty("medium-challenges/bar") == "intermediate"

    def test_infer_difficulty_advanced(self, challenge_ext):
        assert challenge_ext._infer_difficulty("src/advanced/hard") == "advanced"

    def test_infer_difficulty_default(self, challenge_ext):
        assert challenge_ext._infer_difficulty("misc/unknown") == "basic"

    def test_infer_pattern_sorting(self, challenge_ext):
        assert challenge_ext._infer_pattern("Sort the array in ascending order") == "sorting"

    def test_infer_pattern_string(self, challenge_ext):
        assert challenge_ext._infer_pattern("Reverse the given string") == "string_manipulation"

    def test_infer_pattern_recursion(self, challenge_ext):
        assert challenge_ext._infer_pattern("Compute factorial recursively") == "recursion"

    def test_infer_pattern_graph(self, challenge_ext):
        assert challenge_ext._infer_pattern("Traverse the graph using BFS") == "graph_traversal"

    def test_infer_pattern_default(self, challenge_ext):
        assert challenge_ext._infer_pattern("Do something general") == "general"

    def test_parse_test_cases(self, challenge_ext):
        content = "assert foo(5) == 120\nassertEqual(bar(3), 9)\n# comment"
        cases = challenge_ext._parse_test_cases(content)
        assert len(cases) >= 1

    def test_challenges_sorted_by_difficulty(self, challenge_ext, tmp_nav):
        """Returned challenges are sorted basic → intermediate → advanced."""
        fake_tree = [
            {"path": "src/advanced/c",      "type": "dir"},
            {"path": "src/basic/a",         "type": "dir"},
            {"path": "src/intermediate/b",  "type": "dir"},
        ]
        fake_challenges = [
            {"id": "c", "title": "C", "difficulty": "advanced",
             "description": "", "test_cases": [], "solution_pattern": "general"},
            {"id": "a", "title": "A", "difficulty": "basic",
             "description": "", "test_cases": [], "solution_pattern": "general"},
            {"id": "b", "title": "B", "difficulty": "intermediate",
             "description": "", "test_cases": [], "solution_pattern": "general"},
        ]
        with patch.object(tmp_nav, "navigate_repo",
                          return_value={"file_tree": fake_tree, "summary": ""}):
            with patch.object(challenge_ext, "_extract_from_dir",
                              side_effect=fake_challenges):
                result = challenge_ext.extract_challenges()

        order = {"basic": 1, "intermediate": 2, "advanced": 3}
        diffs = [r["difficulty"] for r in result]
        for i in range(len(diffs) - 1):
            assert order[diffs[i]] <= order[diffs[i + 1]]

    def test_extract_from_dir_404(self, challenge_ext, tmp_nav):
        """Returns None when GitHub API returns 404."""
        with patch.object(tmp_nav.session, "get", return_value=mock_resp(404)):
            result = challenge_ext._extract_from_dir("owner", "repo", "src/challenge/a")
        assert result is None


# ── DeepWikiBookExtractor ────────────────────────────────────────────────────


@pytest.fixture
def book_ext(tmp_nav):
    return DeepWikiBookExtractor(tmp_nav, rate_limit_seconds=0)


class TestDeepWikiBookExtractor:
    def test_extract_empty_tree(self, book_ext, tmp_nav):
        with patch.object(tmp_nav, "navigate_repo",
                          return_value={"file_tree": [], "summary": ""}):
            result = book_ext.extract_book_curriculum()
        assert isinstance(result, list)

    def test_infer_format_pdf(self, book_ext):
        assert book_ext._infer_format("http://example.com/book.pdf") == "PDF"

    def test_infer_format_epub(self, book_ext):
        assert book_ext._infer_format("http://example.com/book.epub") == "EPUB"

    def test_infer_format_html(self, book_ext):
        assert book_ext._infer_format("https://example.com/guide") == "HTML"

    def test_extract_from_md_parses_links(self, book_ext, tmp_nav):
        md = (
            "# Books\n"
            "- [Learn Python](https://example.com/python.pdf)\n"
            "- [JavaScript Guide](https://example.com/js.html)\n"
            "- [Plain text](plain_no_url)\n"
        )
        resp_data = {"content": b64(md)}
        with patch.object(tmp_nav.session, "get",
                          return_value=mock_resp(200, json_data=resp_data)):
            books = book_ext._extract_from_md("owner", "repo", "books.md", "programming")

        titles = [b["title"] for b in books]
        assert "Learn Python"     in titles
        assert "JavaScript Guide" in titles
        assert all(b["topic"] == "programming" for b in books)

    def test_extract_from_md_404(self, book_ext, tmp_nav):
        with patch.object(tmp_nav.session, "get", return_value=mock_resp(404)):
            books = book_ext._extract_from_md("owner", "repo", "books.md", "prog")
        assert books == []

    def test_extract_only_top_level_md(self, book_ext, tmp_nav):
        """Only top-level .md files are processed."""
        tree = [
            {"path": "README.md",          "type": "file"},   # top-level, included
            {"path": "docs/guide.md",      "type": "file"},   # nested, excluded
            {"path": "free-books.md",      "type": "file"},   # top-level, included
        ]
        md_content = "[A Book](https://example.com/a.pdf)\n"
        resp_data = {"content": b64(md_content)}

        call_count = {"n": 0}
        def fake_get(url, **kwargs):
            call_count["n"] += 1
            return mock_resp(200, json_data=resp_data)

        with patch.object(tmp_nav, "navigate_repo",
                          return_value={"file_tree": tree, "summary": ""}):
            with patch.object(tmp_nav.session, "get", side_effect=fake_get):
                book_ext.extract_book_curriculum(max_files=5)

        # Only 2 top-level md files → at most 2 API calls
        assert call_count["n"] <= 2


# ── CapabilityDiscovery ──────────────────────────────────────────────────────


@pytest.fixture
def disc(tmp_nav):
    return CapabilityDiscovery(tmp_nav, hdv_system=None)


class TestCapabilityDiscovery:
    def test_identify_gaps_returns_all_types(self, disc):
        gaps = disc._identify_gaps()
        assert isinstance(gaps, list)
        expected = {"gcode_generation", "frontend_frameworks", "data_processing",
                    "computer_vision", "natural_language"}
        assert set(gaps) == expected

    def test_gap_to_dimension_physical(self, disc):
        assert disc._gap_to_dimension("gcode_generation") == "physical"

    def test_gap_to_dimension_behavioral(self, disc):
        assert disc._gap_to_dimension("natural_language") == "behavioral"
        assert disc._gap_to_dimension("computer_vision")  == "behavioral"

    def test_gap_to_dimension_execution(self, disc):
        assert disc._gap_to_dimension("data_processing") == "execution"

    def test_compute_relevance_high(self, disc):
        score = disc._compute_relevance(
            "gcode_generation",
            "3D printing slicer toolpath gcode generator cnc"
        )
        assert score > 0.5

    def test_compute_relevance_zero(self, disc):
        score = disc._compute_relevance("gcode_generation", "machine learning transformer")
        assert score == 0.0

    def test_compute_relevance_range(self, disc):
        score = disc._compute_relevance("natural_language", "NLP tokenize text corpus")
        assert 0.0 <= score <= 1.0

    def test_compute_relevance_unknown_gap(self, disc):
        score = disc._compute_relevance("nonexistent_gap", "anything")
        assert score == 0.0

    def test_discover_repos_structure(self, disc, tmp_nav):
        """discover_repos_for_gaps returns correct structure."""
        fake_data = {
            "repo_url": "Ultimaker/Cura",
            "summary":  "3D printing slicer gcode toolpath cnc",
            "file_tree": [], "key_files": [], "dependencies": [], "insights": {},
        }
        with patch.object(tmp_nav, "navigate_repo", return_value=fake_data):
            discoveries = disc.discover_repos_for_gaps(max_per_gap=1)

        assert isinstance(discoveries, list)
        if discoveries:
            assert "gap"   in discoveries[0]
            assert "repos" in discoveries[0]

    def test_discover_repos_sorted_by_relevance(self, disc, tmp_nav):
        """Repos within a gap are sorted descending by relevance."""
        summaries = [
            "nothing relevant here",
            "gcode slicer toolpath 3d print cnc",
        ]
        idx = {"i": 0}
        def fake_nav(owner, repo):
            s = summaries[idx["i"] % len(summaries)]
            idx["i"] += 1
            return {"repo_url": f"{owner}/{repo}", "summary": s,
                    "file_tree": [], "key_files": [], "dependencies": [], "insights": {}}

        with patch.object(tmp_nav, "navigate_repo", side_effect=fake_nav):
            results = disc._search_for_gap("gcode_generation", max_repos=2)

        if len(results) >= 2:
            assert results[0]["relevance"] >= results[1]["relevance"]

    def test_hdv_system_sparse_dim_prioritised(self, tmp_nav, tmp_path):
        """With hdv_system, the sparsest dimension's gap appears first."""
        from tensor.integrated_hdv import IntegratedHDVSystem
        hdv = IntegratedHDVSystem(
            hdv_dim=500, n_modes=5, embed_dim=32,
            library_path=str(tmp_path / "lib.json")
        )
        # Populate behavioral so it has higher density than physical (still 0)
        hdv.structural_encode("workflow test pattern", "behavioral")

        disc = CapabilityDiscovery(tmp_nav, hdv_system=hdv)
        gaps = disc._identify_gaps()

        # physical-mapped gaps should appear before behavioral-mapped gaps
        phys_indices = [i for i, g in enumerate(gaps)
                        if disc._gap_to_dimension(g) == "physical"]
        beh_indices  = [i for i, g in enumerate(gaps)
                        if disc._gap_to_dimension(g) == "behavioral"]

        if phys_indices and beh_indices:
            assert min(phys_indices) <= min(beh_indices)
