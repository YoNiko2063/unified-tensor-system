"""
FICUTS: DeepWiki Navigator

Phase 1 of CLAUDE_CODE_IMPLEMENTATION_PLAN.

Navigates DeepWiki + GitHub API to extract structured learning content
for curriculum training.

Note: DeepWiki pages are heavily JS-rendered. This navigator falls back to
GitHub API for file tree and structural data; DeepWiki is only attempted
for the repo summary/intent paragraph.

Classes:
    DeepWikiNavigator         — fetch + cache per-repo structured data
    DeepWikiChallengeExtractor — extract coding challenges (freeCodeCamp)
    DeepWikiBookExtractor     — extract book lists (free-programming-books)
    CapabilityDiscovery       — find repos that fill current capability gaps
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Scrapling: better HTML/JS scraping — optional, falls back to requests if absent
try:
    from scrapling.fetchers import Fetcher as _ScraplingFetcher
    _SCRAPLING_AVAILABLE = True
except Exception:
    _SCRAPLING_AVAILABLE = False


# ---------------------------------------------------------------------------
# DeepWikiNavigator
# ---------------------------------------------------------------------------

class DeepWikiNavigator:
    """
    Navigate DeepWiki + GitHub API to extract structured repo data.

    DeepWiki is JS-rendered so structural data comes from GitHub API.
    DeepWiki is tried first for a summary paragraph; GitHub description
    is used as fallback.

    Schema returned by navigate_repo():
        repo_url     : "owner/repo"
        summary      : str
        file_tree    : [{"path": str, "type": "file"|"dir"}]  (cap 200)
        key_files    : [str]           # README, setup.py, etc.
        dependencies : [str]           # from requirements.txt / package.json
        insights     : {"complexity": str, "patterns": [str], "language": str,
                         "star_count": int}
    """

    DEEPWIKI_BASE = "https://deepwiki.com"
    GITHUB_API    = "https://api.github.com"

    def __init__(
        self,
        cache_dir: str = "tensor/data/deepwiki_cache",
        rate_limit_seconds: float = 1.5,
        github_token: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request: float = 0.0

        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (compatible; FICUTSNavigator/1.0)"
        )
        if github_token:
            self.session.headers["Authorization"] = f"token {github_token}"

    # ── Public API ──────────────────────────────────────────────────────────

    def navigate_repo(self, owner: str, repo: str) -> Dict:
        """
        Fetch structured information about a GitHub repo.

        Returns a dict matching the schema described in the class docstring.
        Never raises — returns partial data on failure.
        """
        cache_file = self.cache_dir / f"{owner}_{repo}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

        summary = self._try_deepwiki_summary(owner, repo)
        meta    = self._get_github_meta(owner, repo)

        if not summary and meta:
            summary = meta.get("description") or f"{repo} repository"
        elif not summary:
            summary = f"{repo} repository"

        file_tree    = self._get_github_tree(owner, repo)
        key_files    = self._identify_key_files(file_tree)
        dependencies = self._infer_dependencies(owner, repo, file_tree)
        insights     = self._compute_insights(file_tree, meta)

        data: Dict = {
            "repo_url":     f"{owner}/{repo}",
            "summary":      summary,
            "file_tree":    file_tree[:200],
            "key_files":    key_files,
            "dependencies": dependencies,
            "insights":     insights,
        }

        try:
            cache_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

        return data

    # ── DeepWiki ────────────────────────────────────────────────────────────

    def _try_deepwiki_summary(self, owner: str, repo: str) -> Optional[str]:
        """
        Attempt to extract a summary paragraph from the DeepWiki page.

        Uses Scrapling (Fetcher) when available for better extraction from
        JS-rendered pages. Falls back to requests + BeautifulSoup/regex.
        """
        url = f"{self.DEEPWIKI_BASE}/{owner}/{repo}"
        self._rate_limit()

        # ── Path 1: Scrapling (handles JS-rendered content better) ─────────
        if _SCRAPLING_AVAILABLE:
            try:
                fetcher = _ScraplingFetcher()
                resp = fetcher.get(url, timeout=15)
                if resp and resp.status == 200:
                    # Use Scrapling's built-in element finder — returns real
                    # post-render DOM, not just raw HTML
                    paras = resp.find_all("p")
                    for p in paras:
                        text = p.text.strip() if hasattr(p, "text") else ""
                        if len(text) > 50:
                            return text[:400]
                    # If no paragraphs found via element API, try html_content
                    html = getattr(resp, "html_content", None) or ""
                    if html:
                        return self._extract_summary_from_html(html)
            except Exception:
                pass  # Fall through to requests

        # ── Path 2: requests + BeautifulSoup/regex ──────────────────────────
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return None
        except Exception:
            return None

        return self._extract_summary_from_html(resp.text)

    def _extract_summary_from_html(self, html: str) -> Optional[str]:
        """Extract the first substantive paragraph from raw HTML."""
        # Try BeautifulSoup
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 50:
                    return text[:400]
        except ImportError:
            pass

        # Regex fallback
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        for sentence in re.split(r"[.!?]", text):
            if len(sentence.strip()) > 50:
                return sentence.strip()[:400]

        return None

    # ── GitHub API ──────────────────────────────────────────────────────────

    def _get_github_meta(self, owner: str, repo: str) -> Optional[Dict]:
        """Fetch repo metadata (description, language, stars) from GitHub API."""
        self._rate_limit()
        try:
            resp = self.session.get(
                f"{self.GITHUB_API}/repos/{owner}/{repo}", timeout=15
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def _get_github_tree(self, owner: str, repo: str) -> List[Dict]:
        """Fetch full file tree via GitHub tree API."""
        self._rate_limit()
        try:
            resp = self.session.get(
                f"{self.GITHUB_API}/repos/{owner}/{repo}"
                "/git/trees/HEAD?recursive=1",
                timeout=20,
            )
            if resp.status_code == 200:
                tree = []
                for item in resp.json().get("tree", [])[:500]:
                    tree.append({
                        "path": item["path"],
                        "type": "dir" if item.get("type") == "tree" else "file",
                    })
                return tree
        except Exception:
            pass
        return []

    def _identify_key_files(self, file_tree: List[Dict]) -> List[str]:
        """Return paths that match README / entry-point / config patterns."""
        key_patterns = [
            r"^README",
            r"^setup\.py$",
            r"^pyproject\.toml$",
            r"^package\.json$",
            r"^main\.py$",
            r"^index\.js$",
            r"^app\.py$",
            r"^CMakeLists\.txt$",
            r"^Makefile$",
        ]
        key_files = []
        for item in file_tree:
            if item["type"] == "file":
                name = Path(item["path"]).name
                if any(re.match(p, name, re.IGNORECASE) for p in key_patterns):
                    key_files.append(item["path"])
        return key_files[:10]

    def _infer_dependencies(
        self, owner: str, repo: str, file_tree: List[Dict]
    ) -> List[str]:
        """Fetch a dependency file and parse package names."""
        dep_files = {"requirements.txt", "package.json", "pyproject.toml", "Pipfile"}
        for item in file_tree:
            if item["type"] == "file" and Path(item["path"]).name in dep_files:
                self._rate_limit()
                try:
                    resp = self.session.get(
                        f"{self.GITHUB_API}/repos/{owner}/{repo}"
                        f"/contents/{item['path']}",
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        content_b64 = resp.json().get("content", "")
                        content = base64.b64decode(content_b64).decode(
                            "utf-8", errors="ignore"
                        )
                        return self._parse_deps(content, item["path"])
                except Exception:
                    continue
        return []

    def _parse_deps(self, content: str, filename: str) -> List[str]:
        """Parse a dependency file into a list of package names."""
        if "package.json" in filename:
            try:
                pkg = json.loads(content)
                deps = list(pkg.get("dependencies", {}).keys())
                deps += list(pkg.get("devDependencies", {}).keys())
                return deps[:20]
            except Exception:
                return []

        # requirements.txt / Pipfile style
        deps = []
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pkg = re.split(r"[>=<!;\[]", line)[0].strip()
                if pkg:
                    deps.append(pkg)
        return deps[:20]

    def _compute_insights(
        self, file_tree: List[Dict], meta: Optional[Dict]
    ) -> Dict:
        """Infer language, top-level dirs (patterns), and star count."""
        extensions: Dict[str, int] = {}
        for item in file_tree:
            if item["type"] == "file":
                ext = Path(item["path"]).suffix.lstrip(".")
                if ext:
                    extensions[ext] = extensions.get(ext, 0) + 1

        language = max(extensions, key=extensions.get) if extensions else "unknown"
        if meta and meta.get("language"):
            language = meta["language"]

        top_dirs = {
            Path(item["path"]).parts[0]
            for item in file_tree
            if len(Path(item["path"]).parts) > 1
        }

        return {
            "complexity":  "medium",
            "patterns":    sorted(top_dirs)[:10],
            "language":    language,
            "star_count":  meta.get("stargazers_count", 0) if meta else 0,
        }

    # ── Rate limiting ───────────────────────────────────────────────────────

    def _rate_limit(self):
        now  = time.time()
        wait = self.rate_limit_seconds - (now - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()


# ---------------------------------------------------------------------------
# DeepWikiChallengeExtractor
# ---------------------------------------------------------------------------

class DeepWikiChallengeExtractor:
    """
    Extract coding challenges from freeCodeCamp (or similar repos).

    Uses DeepWikiNavigator for the file tree, then GitHub Contents API
    for README/test file content within each challenge directory.
    """

    def __init__(
        self,
        navigator: DeepWikiNavigator,
        rate_limit_seconds: float = 1.0,
    ):
        self.navigator = navigator
        self._session = navigator.session
        self._rate_limit_seconds = rate_limit_seconds
        self._last_request: float = 0.0

    def extract_challenges(
        self,
        owner: str = "freeCodeCamp",
        repo: str = "freeCodeCamp",
        max_challenges: int = 20,
    ) -> List[Dict]:
        """
        Return challenges sorted by difficulty (basic → advanced).

        Each entry: {id, title, difficulty, description, test_cases,
                      solution_pattern}
        """
        repo_data = self.navigator.navigate_repo(owner, repo)
        if not repo_data:
            return []

        challenge_dirs = [
            item for item in repo_data["file_tree"]
            if item["type"] == "dir"
            and any(
                kw in item["path"].lower()
                for kw in ["challenge", "exercise", "problem"]
            )
            and len(Path(item["path"]).parts) <= 3
        ]

        challenges = []
        for cd in challenge_dirs[:max_challenges]:
            ch = self._extract_from_dir(owner, repo, cd["path"])
            if ch:
                challenges.append(ch)

        difficulty_order = {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}
        challenges.sort(
            key=lambda c: difficulty_order.get(c.get("difficulty", "basic"), 0)
        )
        return challenges

    # ── Internals ───────────────────────────────────────────────────────────

    def _extract_from_dir(
        self, owner: str, repo: str, path: str
    ) -> Optional[Dict]:
        self._rate()
        try:
            resp = self._session.get(
                f"{DeepWikiNavigator.GITHUB_API}/repos/{owner}/{repo}"
                f"/contents/{path}",
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            files = resp.json()
            if not isinstance(files, list):
                return None
        except Exception:
            return None

        desc_content = ""
        test_cases: List[Dict] = []

        for f in files:
            fname = f.get("name", "").lower()
            if "readme" in fname or "description" in fname:
                desc_content = self._fetch_file_text(f.get("url", ""))
            elif "test" in fname or "spec" in fname:
                test_content = self._fetch_file_text(f.get("url", ""))
                test_cases = self._parse_test_cases(test_content)

        challenge_id = Path(path).name
        return {
            "id":               challenge_id,
            "title":            challenge_id.replace("-", " ").replace("_", " ").title(),
            "difficulty":       self._infer_difficulty(path),
            "description":      desc_content[:500],
            "test_cases":       test_cases[:5],
            "solution_pattern": self._infer_pattern(desc_content),
        }

    def _fetch_file_text(self, url: str) -> str:
        if not url:
            return ""
        self._rate()
        try:
            resp = self._session.get(url, timeout=10)
            if resp.status_code == 200:
                content_b64 = resp.json().get("content", "")
                return base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        except Exception:
            pass
        return ""

    def _parse_test_cases(self, content: str) -> List[Dict]:
        cases = []
        for m in re.finditer(r"assert\w*\(([^)]{1,200})\)", content):
            cases.append({"assertion": m.group(1)[:100]})
        return cases[:5]

    def _infer_difficulty(self, path: str) -> str:
        pl = path.lower()
        if any(k in pl for k in ["basic", "easy", "beginner", "intro"]):
            return "basic"
        if any(k in pl for k in ["intermediate", "medium"]):
            return "intermediate"
        if any(k in pl for k in ["advanced", "hard", "expert"]):
            return "advanced"
        return "basic"

    def _infer_pattern(self, description: str) -> str:
        dl = description.lower()
        if any(k in dl for k in ["sort", "order", "array"]):
            return "sorting"
        if any(k in dl for k in ["string", "text", "reverse"]):
            return "string_manipulation"
        if any(k in dl for k in ["tree", "graph", "node"]):
            return "graph_traversal"
        if any(k in dl for k in ["recursi", "factorial", "fibonacci"]):
            return "recursion"
        return "general"

    def _rate(self):
        now  = time.time()
        wait = self._rate_limit_seconds - (now - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()


# ---------------------------------------------------------------------------
# DeepWikiBookExtractor
# ---------------------------------------------------------------------------

class DeepWikiBookExtractor:
    """
    Extract programming book lists from EbookFoundation/free-programming-books.

    Uses GitHub Contents API to fetch markdown files and parse book links.
    """

    def __init__(
        self,
        navigator: DeepWikiNavigator,
        rate_limit_seconds: float = 1.0,
    ):
        self.navigator = navigator
        self._session = navigator.session
        self._rate_limit_seconds = rate_limit_seconds
        self._last_request: float = 0.0

    def extract_book_curriculum(
        self,
        owner: str = "EbookFoundation",
        repo: str = "free-programming-books",
        max_files: int = 5,
    ) -> List[Dict]:
        """
        Return books: [{title, topic, url, format}].

        Reads top-level .md files from the repo.
        """
        repo_data = self.navigator.navigate_repo(owner, repo)
        if not repo_data:
            return []

        md_files = [
            item for item in repo_data["file_tree"]
            if item["type"] == "file"
            and item["path"].endswith(".md")
            and "/" not in item["path"]       # top-level only
        ][:max_files]

        books: List[Dict] = []
        for md_file in md_files:
            topic = Path(md_file["path"]).stem.lower().replace("-", "_")
            books.extend(
                self._extract_from_md(owner, repo, md_file["path"], topic)
            )
        return books

    # ── Internals ───────────────────────────────────────────────────────────

    def _extract_from_md(
        self, owner: str, repo: str, filepath: str, topic: str
    ) -> List[Dict]:
        self._rate()
        try:
            resp = self._session.get(
                f"{DeepWikiNavigator.GITHUB_API}/repos/{owner}/{repo}"
                f"/contents/{filepath}",
                timeout=10,
            )
            if resp.status_code != 200:
                return []
            content_b64 = resp.json().get("content", "")
            content = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        except Exception:
            return []

        books = []
        for title, url in re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", content):
            if any(
                ext in url.lower()
                for ext in [".pdf", ".epub", ".html", "http"]
            ):
                books.append({
                    "title":  title[:100],
                    "topic":  topic,
                    "url":    url,
                    "format": self._infer_format(url),
                })
        return books[:50]

    def _infer_format(self, url: str) -> str:
        ul = url.lower()
        if ".pdf"  in ul: return "PDF"
        if ".epub" in ul: return "EPUB"
        return "HTML"

    def _rate(self):
        now  = time.time()
        wait = self._rate_limit_seconds - (now - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()


# ---------------------------------------------------------------------------
# CapabilityDiscovery
# ---------------------------------------------------------------------------

class CapabilityDiscovery:
    """
    Discover repos on DeepWiki/GitHub that fill current capability gaps.

    Strategy:
    1. Identify gaps from HDV system pattern density (sparse = gap)
    2. Navigate predefined high-quality repos per gap type
    3. Rank by keyword-based relevance
    4. Return ranked list for curriculum ingestion
    """

    GAP_REPOS: Dict[str, List[Tuple[str, str]]] = {
        "gcode_generation": [
            ("Ultimaker",      "Cura"),
            ("prusa3d",        "PrusaSlicer"),
            ("slic3r",         "Slic3r"),
        ],
        "frontend_frameworks": [
            ("facebook",  "react"),
            ("vuejs",     "vue"),
            ("sveltejs",  "svelte"),
        ],
        "data_processing": [
            ("pandas-dev", "pandas"),
            ("numpy",      "numpy"),
            ("apache",     "spark"),
        ],
        "computer_vision": [
            ("opencv",          "opencv"),
            ("isl-org",         "Open3D"),
            ("facebookresearch","detectron2"),
        ],
        "natural_language": [
            ("huggingface", "transformers"),
            ("explosion",   "spaCy"),
            ("nltk",        "nltk"),
        ],
    }

    GAP_KEYWORDS: Dict[str, List[str]] = {
        "gcode_generation":    ["gcode", "3d print", "slicer", "toolpath", "cnc"],
        "frontend_frameworks": ["react", "component", "ui", "frontend", "dom"],
        "data_processing":     ["dataframe", "array", "processing", "transform", "etl"],
        "computer_vision":     ["vision", "image", "detection", "recognition", "pixel"],
        "natural_language":    ["nlp", "language", "text", "tokenize", "corpus"],
    }

    _DIM_MAP: Dict[str, str] = {
        "gcode_generation":    "physical",
        "frontend_frameworks": "behavioral",
        "data_processing":     "execution",
        "computer_vision":     "behavioral",
        "natural_language":    "behavioral",
    }

    def __init__(
        self,
        navigator: DeepWikiNavigator,
        hdv_system=None,
    ):
        self.navigator  = navigator
        self.hdv_system = hdv_system
        self.capability_gaps = self._identify_gaps()

    def _identify_gaps(self) -> List[str]:
        """
        Return gap list, sorted by need (sparsest dimension first).

        If hdv_system is provided, checks domain_masks density to prioritise.
        """
        all_gaps = list(self.GAP_REPOS.keys())

        if self.hdv_system is None:
            return all_gaps

        def density(gap: str) -> float:
            dim  = self._gap_to_dimension(gap)
            mask = self.hdv_system.domain_masks.get(dim)
            if mask is None:
                return 0.0
            return float(mask.sum()) / len(mask)

        return sorted(all_gaps, key=density)

    def _gap_to_dimension(self, gap: str) -> str:
        return self._DIM_MAP.get(gap, "behavioral")

    def discover_repos_for_gaps(self, max_per_gap: int = 3) -> List[Dict]:
        """
        Return: [{"gap": str, "repos": [{"owner", "name", "relevance", "summary"}]}]
        """
        discoveries = []
        for gap in self.capability_gaps:
            repos = self._search_for_gap(gap, max_repos=max_per_gap)
            if repos:
                discoveries.append({"gap": gap, "repos": repos})
        return discoveries

    def _search_for_gap(self, gap: str, max_repos: int = 3) -> List[Dict]:
        """Navigate repos, compute relevance, return sorted list."""
        results = []
        for owner, repo in self.GAP_REPOS.get(gap, [])[:max_repos]:
            data = self.navigator.navigate_repo(owner, repo)
            if data:
                relevance = self._compute_relevance(gap, data.get("summary", ""))
                results.append({
                    "owner":     owner,
                    "name":      repo,
                    "relevance": relevance,
                    "summary":   data.get("summary", "")[:200],
                })
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results

    def _compute_relevance(self, gap: str, summary: str) -> float:
        """Keyword match → relevance in [0, 1]."""
        keywords = self.GAP_KEYWORDS.get(gap, [])
        if not keywords:
            return 0.0
        sl = summary.lower()
        return sum(1 for kw in keywords if kw in sl) / len(keywords)
