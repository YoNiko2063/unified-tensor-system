"""
FICUTS Task 11.2: GitHub API Capability Extractor

Fallback for repos not covered by DeepWiki.
Uses GitHub REST API only — no git clone, no local storage.

Strategy:
  1. GET /repos/{owner}/{repo}                          → metadata, language
  2. GET /repos/{owner}/{repo}/readme                   → README (intent/usage)
  3. GET /repos/{owner}/{repo}/git/trees/HEAD?recursive=1 → file structure
  4. Compare encoded structure to DeepWiki templates    → fast categorization
  5. No match → analyze README directly                 → new capability map

Rate limits:
  - Unauthenticated: 60 requests/hour
  - Authenticated (GITHUB_TOKEN): 5000 requests/hour

Set env var GITHUB_TOKEN for higher rate limits.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests


class GitHubAPICapabilityExtractor:
    """
    Extract capability maps from GitHub repos via REST API.

    Produces the same capability map format as DeepWikiWorkflowParser
    so both sources feed the same downstream pipeline.
    """

    _API_BASE = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        templates_path: str = "tensor/data/deepwiki_workflows.json",
        rate_limit_seconds: Optional[float] = None,
    ):
        """
        Args:
            token:          GitHub personal access token (optional).
                            If not provided, checks GITHUB_TOKEN env var.
            templates_path: path to saved DeepWiki workflow templates.
        """
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "FICUTSBot/1.0",
        })

        # Auth (optional)
        token = token or os.environ.get("GITHUB_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"token {token}"
            default_rl = 0.72  # ~5000/hr → 1 per 0.72s
        else:
            default_rl = 60.0  # 60/hr → 1 per 60s (conservative)

        self._rate_limit_seconds = rate_limit_seconds if rate_limit_seconds is not None else default_rl

        self._last_request: float = 0.0

        # Load DeepWiki templates for similarity matching
        self._templates: Dict[str, dict] = {}
        templates_path_obj = Path(templates_path)
        if templates_path_obj.exists():
            try:
                raw = json.loads(templates_path_obj.read_text())
                self._templates = {k: v for k, v in raw.items() if "hdv" in v}
            except Exception:
                self._templates = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_capability_via_api(self, repo_url: str) -> Optional[Dict]:
        """
        Extract capability map from a GitHub repo using only the REST API.

        Args:
            repo_url: e.g. "https://github.com/owner/repo"

        Returns:
            {
              'repo_url':      str,
              'intent':        str,
              'workflow':      List[str],
              'language':      str,
              'stars':         int,
              'dependencies':  List[str],
              'file_patterns': List[str],
              'source':        'github_api',
            }
            or None on failure.
        """
        owner_repo = self._parse_owner_repo(repo_url)
        if not owner_repo:
            return None
        owner, repo = owner_repo

        # 1. Repo metadata
        self._rate_limit()
        meta_resp = self.session.get(f"{self._API_BASE}/repos/{owner}/{repo}", timeout=10)
        if meta_resp.status_code != 200:
            return None
        meta = meta_resp.json()

        description = meta.get("description", "") or ""
        language = meta.get("language", "unknown") or "unknown"
        stars = meta.get("stargazers_count", 0)

        # 2. README for intent + usage patterns
        self._rate_limit()
        readme_resp = self.session.get(
            f"{self._API_BASE}/repos/{owner}/{repo}/readme", timeout=10
        )
        readme_text = ""
        if readme_resp.status_code == 200:
            try:
                encoded = readme_resp.json().get("content", "")
                readme_text = base64.b64decode(encoded).decode("utf-8", errors="ignore")
            except Exception:
                pass

        # 3. File tree for structure patterns
        self._rate_limit()
        tree_resp = self.session.get(
            f"{self._API_BASE}/repos/{owner}/{repo}/git/trees/HEAD?recursive=1",
            timeout=15,
        )
        file_patterns: List[str] = []
        if tree_resp.status_code == 200:
            try:
                tree = tree_resp.json()
                files = [
                    item["path"]
                    for item in tree.get("tree", [])
                    if item.get("type") == "blob"
                ]
                file_patterns = self._extract_file_patterns(files)
            except Exception:
                pass

        intent = self._infer_intent(description, readme_text)
        workflow = self._infer_workflow(readme_text, file_patterns)
        dependencies = self._extract_dependencies(readme_text, language)

        return {
            "repo_url": repo_url,
            "intent": intent,
            "workflow": workflow,
            "language": language,
            "stars": stars,
            "dependencies": dependencies,
            "file_patterns": file_patterns[:20],
            "source": "github_api",
        }

    def match_to_deepwiki_template(
        self, capability: Dict, hdv_system
    ) -> Optional[str]:
        """
        Compare a new repo's workflow to learned DeepWiki templates.

        If cosine similarity in HDV space > 0.75 → apply existing template.
        This avoids redundant analysis for repos with familiar patterns.

        Returns: matched template repo_url, or None if no good match.
        """
        if not self._templates or hdv_system is None:
            return None

        new_hdv = hdv_system.encode_workflow(
            capability.get("workflow", []), domain="behavioral"
        )

        best_sim, best_match = 0.0, None
        for template_url, template_data in self._templates.items():
            try:
                t_hdv = np.array(template_data["hdv"], dtype=np.float32)
            except Exception:
                continue
            if len(t_hdv) != len(new_hdv):
                continue
            n1, n2 = np.linalg.norm(new_hdv), np.linalg.norm(t_hdv)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            sim = float(np.dot(new_hdv, t_hdv) / (n1 * n2))
            if sim > best_sim:
                best_sim, best_match = sim, template_url

        return best_match if best_sim > 0.75 else None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _infer_intent(self, description: str, readme: str) -> str:
        """Infer repo's purpose from description + README intro."""
        if description and len(description) > 20:
            return description[:300]
        # First substantive README paragraph (skip headings + badges)
        for para in readme.split("\n\n")[:8]:
            para = para.strip()
            if para.startswith("#") or "![" in para or len(para) < 30:
                continue
            return para[:300]
        return f"GitHub repository ({description or 'no description'})"

    def _infer_workflow(self, readme: str, file_patterns: List[str]) -> List[str]:
        """Infer workflow steps from README usage/examples section."""
        workflow: List[str] = []

        # Extract Usage / Quick Start section
        usage_match = re.search(
            r"(?:##\s*(?:Usage|Quick[- ]?[Ss]tart|Getting [Ss]tarted|Example))"
            r"(.*?)(?=##|\Z)",
            readme, re.DOTALL | re.IGNORECASE,
        )
        if usage_match:
            usage_text = usage_match.group(1)
            for m in re.findall(r"(?:^\d+\.\s*(.+)|#\s*(.+))$", usage_text, re.MULTILINE):
                step = (m[0] or m[1]).strip()
                if step and len(step) > 5:
                    workflow.append(step[:80])
                    if len(workflow) >= 8:
                        break

        # Fallback: file-structure heuristic
        if not workflow:
            lf = [f.lower() for f in file_patterns]
            if any("load" in f or "read" in f or "input" in f for f in lf):
                workflow.append("load_input")
            if any("process" in f or "transform" in f or "train" in f for f in lf):
                workflow.append("process")
            if any("output" in f or "export" in f or "save" in f for f in lf):
                workflow.append("output")

        return workflow or ["initialize", "execute", "output"]

    def _extract_file_patterns(self, files: List[str]) -> List[str]:
        """Key file paths that reveal structure (small Python files, config)."""
        keep = set()
        for f in files:
            parts = f.split("/")
            if f.endswith(".py") and len(parts) <= 3:
                keep.add(f)
            elif f in {
                "requirements.txt", "setup.py", "pyproject.toml",
                "Cargo.toml", "package.json", "go.mod", "CMakeLists.txt",
            }:
                keep.add(f)
        return sorted(keep)[:30]

    def _extract_dependencies(self, readme: str, language: str) -> List[str]:
        """Extract dependency names from README install instructions."""
        deps: set = set()
        # pip install can list multiple packages on one line
        for line in re.findall(r"pip install ([\w\s@/-]+)", readme):
            for pkg in line.split():
                if re.match(r"^[\w-]{2,}$", pkg):
                    deps.add(pkg)
        for m in re.findall(r"npm install ([\w@/-]+)", readme):
            deps.add(m)
        for m in re.findall(r"`([\w-]{3,20})`", readme):
            deps.add(m)
        return list(deps)[:10]

    def _parse_owner_repo(self, url: str) -> Optional[tuple]:
        """'https://github.com/owner/repo' → ('owner', 'repo')"""
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return None

    def _rate_limit(self):
        now = time.time()
        wait = self._rate_limit_seconds - (now - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()
