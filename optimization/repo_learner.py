"""
RepoLearner — ingest a GitHub repository, extract Python functions,
classify their mathematical structure, and store experiences in
KoopmanExperienceMemory.

Pipeline:
    GitHub repo URL
        → GitHubAPICapabilityExtractor  (file tree)
        → extract .py files (via raw content API)
        → split into function definitions
        → ASTMathClassifier (static)
        → CodeHardwareProfiler.store()
        → KoopmanExperienceMemory  (domain="hardware_static")

After ingestion, the memory acts as a math→hardware retrieval index:
    new function source
        → analyse_static() → KoopmanInvariantDescriptor
        → retrieve_candidates() → nearest stored functions
        → best_params["fn_name"], ["complexity"], ["hints"]

No code is executed during ingestion (static only, safe for arbitrary repos).
Dynamic profiling is available separately via CodeHardwareProfiler.analyse_dynamic().
"""

from __future__ import annotations

import ast
import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from optimization.code_profiler import ASTMathClassifier, CodeHardwareProfiler, profile_and_store
from optimization.koopman_memory import KoopmanExperienceMemory


# ── GitHub raw content fetcher ────────────────────────────────────────────────


class GitHubFileFetcher:
    """
    Fetch Python file contents from a public GitHub repo via the REST API.

    No git clone required — uses:
      GET /repos/{owner}/{repo}/contents/{path}   (file tree)
      GET {raw_url}                                (file content)

    Rate limits (without token): 60 requests/hour.
    With GITHUB_TOKEN: 5000 requests/hour.
    """

    _API_BASE = "https://api.github.com"
    _MAX_FILES = 50     # cap per repo to avoid runaway fetching

    def __init__(
        self,
        token: Optional[str] = None,
        rate_limit_seconds: Optional[float] = None,
    ) -> None:
        if not _HAS_REQUESTS:
            raise ImportError("requests library required: pip install requests")
        import requests as _req
        self._session = _req.Session()
        self._session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "FICUTSRepoLearner/1.0",
        })
        tok = token or os.environ.get("GITHUB_TOKEN")
        if tok:
            self._session.headers["Authorization"] = f"token {tok}"
            default_rl = 0.72
        else:
            default_rl = 60.0
        self._rl = rate_limit_seconds if rate_limit_seconds is not None else default_rl
        self._last: float = 0.0

    def _get(self, url: str) -> Optional[dict]:
        elapsed = time.monotonic() - self._last
        if elapsed < self._rl:
            time.sleep(self._rl - elapsed)
        try:
            r = self._session.get(url, timeout=10)
            self._last = time.monotonic()
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def _get_raw(self, url: str) -> Optional[str]:
        elapsed = time.monotonic() - self._last
        if elapsed < self._rl:
            time.sleep(self._rl - elapsed)
        try:
            r = self._session.get(url, timeout=15)
            self._last = time.monotonic()
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
        return None

    def parse_owner_repo(self, url: str) -> Tuple[str, str]:
        """Extract (owner, repo) from a GitHub URL."""
        parts = url.rstrip("/").split("/")
        # github.com/owner/repo or owner/repo
        if "github.com" in url:
            idx = parts.index("github.com") if "github.com" in parts else -1
            if idx >= 0 and idx + 2 < len(parts):
                return parts[idx + 1], parts[idx + 2]
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        raise ValueError(f"Cannot parse GitHub URL: {url!r}")

    def list_python_files(self, owner: str, repo: str, path: str = "") -> List[dict]:
        """
        Recursively list all .py files in a repo (up to _MAX_FILES).

        Returns list of dicts: {name, path, download_url, size}
        """
        url = f"{self._API_BASE}/repos/{owner}/{repo}/contents/{path}"
        data = self._get(url)
        if not isinstance(data, list):
            return []

        results = []
        for item in data:
            if len(results) >= self._MAX_FILES:
                break
            if item.get("type") == "file" and item.get("name", "").endswith(".py"):
                results.append({
                    "name":         item["name"],
                    "path":         item["path"],
                    "download_url": item.get("download_url", ""),
                    "size":         item.get("size", 0),
                })
            elif item.get("type") == "dir":
                sub = self.list_python_files(owner, repo, item["path"])
                results.extend(sub[: self._MAX_FILES - len(results)])

        return results

    def fetch_file(self, download_url: str) -> Optional[str]:
        """Download raw file content."""
        return self._get_raw(download_url)


# ── Function extractor ────────────────────────────────────────────────────────


@dataclass
class FunctionRecord:
    fn_name: str
    source: str       # dedented function source
    file_path: str
    repo: str
    lineno: int


def extract_functions(source: str, file_path: str = "", repo: str = "") -> List[FunctionRecord]:
    """
    Parse Python source, return one FunctionRecord per top-level function.

    Skips:
      - Functions shorter than 3 lines (trivial wrappers / property stubs)
      - Functions with names starting with "_" (private internals)
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    records = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue   # skip private
        try:
            end = node.end_lineno
        except AttributeError:
            end = node.lineno + 10   # fallback for old Python
        fn_lines = source_lines[node.lineno - 1: end]
        if len(fn_lines) < 3:
            continue
        fn_source = textwrap.dedent("\n".join(fn_lines))
        records.append(FunctionRecord(
            fn_name=node.name,
            source=fn_source,
            file_path=file_path,
            repo=repo,
            lineno=node.lineno,
        ))

    return records


# ── Repo learner ──────────────────────────────────────────────────────────────


@dataclass
class LearningResult:
    repo: str
    files_fetched: int
    functions_found: int
    functions_stored: int
    errors: List[str] = field(default_factory=list)
    pattern_counts: Dict[str, int] = field(default_factory=dict)


class RepoLearner:
    """
    Learn from a GitHub repository: extract Python functions, classify
    their mathematical structure, and store in KoopmanExperienceMemory.

    After learning, the memory can answer:
      "What kind of computation is this function doing?"
      "What optimization approach worked for similar computations?"

    Usage:
        memory = KoopmanExperienceMemory()
        learner = RepoLearner(token=os.environ.get("GITHUB_TOKEN"),
                              rate_limit_seconds=0)  # 0 for local/test repos
        result = learner.learn("https://github.com/numpy/numpy", memory)
        print(result)
    """

    def __init__(
        self,
        token: Optional[str] = None,
        rate_limit_seconds: Optional[float] = None,
        max_functions: int = 100,
    ) -> None:
        self._fetcher = GitHubFileFetcher(token=token, rate_limit_seconds=rate_limit_seconds)
        self._profiler = CodeHardwareProfiler()
        self._classifier = ASTMathClassifier()
        self.max_functions = max_functions

    def learn(
        self,
        repo_url: str,
        memory: KoopmanExperienceMemory,
    ) -> LearningResult:
        """
        Ingest a GitHub repo into memory.

        Fetches .py files via GitHub API, extracts function definitions,
        classifies each statically, and stores in memory with domain="hardware_static".
        """
        try:
            owner, repo = self._fetcher.parse_owner_repo(repo_url)
        except ValueError as e:
            return LearningResult(repo=repo_url, files_fetched=0, functions_found=0,
                                  functions_stored=0, errors=[str(e)])

        py_files = self._fetcher.list_python_files(owner, repo)
        result = LearningResult(repo=f"{owner}/{repo}",
                                files_fetched=len(py_files),
                                functions_found=0,
                                functions_stored=0)

        n_stored = 0
        for file_info in py_files:
            if n_stored >= self.max_functions:
                break
            content = self._fetcher.fetch_file(file_info["download_url"])
            if not content:
                result.errors.append(f"fetch failed: {file_info['path']}")
                continue

            fns = extract_functions(content, file_path=file_info["path"],
                                    repo=f"{owner}/{repo}")
            result.functions_found += len(fns)

            for fn_rec in fns:
                if n_stored >= self.max_functions:
                    break
                try:
                    inv = profile_and_store(
                        source=fn_rec.source,
                        memory=memory,
                        fn_name=f"{owner}/{repo}:{fn_rec.file_path}:{fn_rec.fn_name}",
                        hints={
                            "repo":      f"{owner}/{repo}",
                            "file":      fn_rec.file_path,
                            "fn":        fn_rec.fn_name,
                            "lineno":    fn_rec.lineno,
                        },
                    )
                    pattern = self._classifier.classify(fn_rec.source)
                    cc = pattern.complexity_class
                    result.pattern_counts[cc] = result.pattern_counts.get(cc, 0) + 1
                    n_stored += 1
                except Exception as exc:
                    result.errors.append(f"{fn_rec.fn_name}: {exc}")

        result.functions_stored = n_stored
        return result

    def learn_from_source(
        self,
        source: str,
        memory: KoopmanExperienceMemory,
        repo_name: str = "local",
        file_name: str = "inline",
    ) -> LearningResult:
        """
        Learn directly from a Python source string (no GitHub required).
        Useful for testing and for ingesting local code.
        """
        fns = extract_functions(source, file_path=file_name, repo=repo_name)
        result = LearningResult(repo=repo_name, files_fetched=1,
                                functions_found=len(fns), functions_stored=0)
        for fn_rec in fns:
            try:
                profile_and_store(
                    source=fn_rec.source,
                    memory=memory,
                    fn_name=f"{repo_name}:{fn_rec.fn_name}",
                    hints={"repo": repo_name, "fn": fn_rec.fn_name},
                )
                pattern = self._classifier.classify(fn_rec.source)
                cc = pattern.complexity_class
                result.pattern_counts[cc] = result.pattern_counts.get(cc, 0) + 1
                result.functions_stored += 1
            except Exception as exc:
                result.errors.append(f"{fn_rec.fn_name}: {exc}")
        return result


# ── Retrieval helper ──────────────────────────────────────────────────────────


def find_similar_code(
    query_source: str,
    memory: KoopmanExperienceMemory,
    top_n: int = 5,
) -> List[dict]:
    """
    Find the most similar stored functions to a query function.

    Returns list of dicts: {fn_name, complexity, distance, hints}
    sorted by ascending distance (most similar first).
    """
    profiler = CodeHardwareProfiler()
    query_inv = profiler.analyse_static(query_source)
    candidates = memory.retrieve_candidates(query_inv, top_n=top_n)

    results = []
    q_vec = query_inv.to_query_vector()
    for entry in candidates:
        ev = entry.invariant.to_query_vector()
        dist = float(sum((a - b) ** 2 for a, b in zip(q_vec, ev)) ** 0.5)
        bp = entry.experience.best_params
        results.append({
            "fn_name":    bp.get("fn_name", "unknown"),
            "complexity": bp.get("complexity", "unknown"),
            "distance":   dist,
            "hints":      {k: v for k, v in bp.items()
                           if k not in ("fn_name", "complexity")},
        })

    return results
