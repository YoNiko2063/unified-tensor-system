"""
FICUTS Task 11.1: DeepWiki Workflow Parser

DeepWiki (deepwiki.com) provides AI-generated analysis of GitHub repos.
URL format: https://deepwiki.com/{owner}/{repo}

Why DeepWiki first:
  - Pre-analyzed: DeepWiki has already extracted structure, intent, workflows
  - High signal: curated AI summaries vs. raw file listings
  - Fast: fetch one page instead of cloning a repo
  - Scale: covers 1M+ repos already analyzed

What we extract:
  - Intent   : what the repo does (1–2 sentences)
  - Workflow  : ordered sequence of operations (Load → Process → Output)
  - Components: key classes/functions
  - Dependencies: libraries used

How workflows become math:
  Workflow = directed sequence (mathematical object)
  Encode as position-weighted HDV → cosine similarity → pattern matching
  This makes "PDF load→merge→save" similar to "image read→transform→export"
  in HDV space, revealing the underlying universal pattern.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


class DeepWikiWorkflowParser:
    """
    Fetch and parse DeepWiki repo analysis pages.

    Encodes extracted workflows as HDV vectors via IntegratedHDVSystem
    for cross-dimensional pattern matching.
    """

    _DEEPWIKI_BASE = "https://deepwiki.com"

    def __init__(self, rate_limit_seconds: float = 2.0):
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request: float = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; FICUTSBot/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        self._cache: Dict[str, dict] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse_deepwiki_summary(self, repo_url: str) -> Optional[Dict]:
        """
        Fetch and parse DeepWiki summary for a GitHub repo.

        Args:
            repo_url: GitHub URL, e.g. "https://github.com/owner/repo"

        Returns:
            {
              'repo_url':     str,
              'intent':       str,        # what repo does
              'workflow':     List[str],  # ordered operation steps
              'components':   List[str],  # key classes/functions
              'dependencies': List[str],  # libraries
              'source':       str,        # 'deepwiki' or fallback variant
            }
            or None if unreachable / JS-only rendered.
        """
        if repo_url in self._cache:
            return self._cache[repo_url]

        owner_repo = self._parse_owner_repo(repo_url)
        if not owner_repo:
            return None

        owner, repo = owner_repo
        deepwiki_url = f"{self._DEEPWIKI_BASE}/{owner}/{repo}"

        self._rate_limit()
        try:
            resp = self.session.get(deepwiki_url, timeout=15)
        except Exception:
            return None

        if resp.status_code != 200:
            return None

        result = self._parse_html(resp.text, repo_url)
        if result:
            self._cache[repo_url] = result
        return result

    def encode_workflow_to_hdv(
        self, workflow_data: Dict, hdv_system
    ):
        """
        Encode a capability dict into HDV space.

        Combines intent words + workflow steps into a position-weighted
        sequence, then delegates to hdv_system.encode_workflow().

        Returns: np.ndarray [hdv_dim]
        """
        intent_words = workflow_data.get("intent", "").split()[:10]
        workflow_steps = workflow_data.get("workflow", [])
        full_sequence = intent_words + workflow_steps
        return hdv_system.encode_workflow(full_sequence, domain="behavioral")

    def batch_process_repos(
        self,
        repo_urls: List[str],
        hdv_system,
        save_path: str = "tensor/data/deepwiki_workflows.json",
    ) -> Dict[str, dict]:
        """
        Process multiple repos: fetch DeepWiki → encode → persist.

        Skips repos already in the save file.

        Returns: {repo_url: {'capability': dict, 'hdv': List[float]}}
        """
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        existing: Dict[str, dict] = {}
        if save_path_obj.exists():
            try:
                existing = json.loads(save_path_obj.read_text())
            except Exception:
                existing = {}

        results: Dict[str, dict] = {}
        for repo_url in repo_urls:
            if repo_url in existing:
                results[repo_url] = existing[repo_url]
                continue

            capability = self.parse_deepwiki_summary(repo_url)
            if capability:
                hdv_vec = self.encode_workflow_to_hdv(capability, hdv_system)
                results[repo_url] = {
                    "capability": capability,
                    "hdv": hdv_vec.tolist(),
                }
                name = repo_url.rstrip("/").split("/")[-1]
                print(f"[DeepWiki] {name}: "
                      f"{len(capability['workflow'])} steps, "
                      f"intent='{capability['intent'][:60]}'")
            else:
                print(f"[DeepWiki] Unavailable (JS-rendered?): {repo_url}")

        all_results = {**existing, **results}
        save_path_obj.write_text(json.dumps(all_results, indent=2))
        print(f"[DeepWiki] Saved {len(all_results)} capability maps → {save_path}")
        return results

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_html(self, html: str, repo_url: str) -> Optional[Dict]:
        """Parse DeepWiki HTML response."""
        if _BS4_AVAILABLE:
            return self._parse_with_bs4(html, repo_url)
        return self._parse_with_regex(html, repo_url)

    def _parse_with_bs4(self, html: str, repo_url: str) -> Optional[Dict]:
        """BeautifulSoup-based parser for DeepWiki pages."""
        soup = BeautifulSoup(html, "html.parser")

        intent = ""
        workflow: List[str] = []
        components: List[str] = []
        dependencies: List[str] = []

        # Intent: first substantial paragraph
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50 and not intent:
                intent = text[:300]
                break

        # Workflow: list items (short = steps, long = descriptions)
        for lst in soup.find_all(["ul", "ol"]):
            items = [li.get_text(strip=True) for li in lst.find_all("li")]
            items = [it for it in items if it and 5 < len(it) < 120]
            if items and not workflow:
                workflow = items[:10]

        # Components: code blocks with class/def/function signatures
        for cb in soup.find_all("code"):
            text = cb.get_text(strip=True)
            if re.match(r"^(class|def|function|public|private|static)\s+\w+", text):
                components.append(text[:100])

        # Dependencies: import patterns in full text
        full_text = soup.get_text()
        for pat in [r"import\s+([\w.]+)", r"from\s+([\w.]+)\s+import"]:
            for m in re.findall(pat, full_text):
                if len(m) > 2:
                    dependencies.append(m)

        # DeepWiki is heavily JS-rendered — if empty, return None
        if not intent and not workflow:
            return None

        return {
            "repo_url": repo_url,
            "intent": intent or f"Repository: {repo_url.rstrip('/').split('/')[-1]}",
            "workflow": workflow or ["initialize", "process", "output"],
            "components": list(dict.fromkeys(components))[:10],
            "dependencies": list(dict.fromkeys(dependencies))[:10],
            "source": "deepwiki",
        }

    def _parse_with_regex(self, html: str, repo_url: str) -> Optional[Dict]:
        """Regex-based fallback when BeautifulSoup is unavailable."""
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 100:
            return None

        # First substantial sentence as intent
        sentences = re.split(r"[.!?]", text)
        intent = next((s.strip() for s in sentences if len(s.strip()) > 40), "")

        return {
            "repo_url": repo_url,
            "intent": intent[:300],
            "workflow": ["load", "process", "output"],
            "components": [],
            "dependencies": [],
            "source": "deepwiki_regex_fallback",
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _parse_owner_repo(self, url: str) -> Optional[tuple]:
        """'https://github.com/owner/repo' → ('owner', 'repo')"""
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return None

    def _rate_limit(self):
        now = time.time()
        wait = self.rate_limit_seconds - (now - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()
