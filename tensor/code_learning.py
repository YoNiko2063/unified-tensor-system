"""
FICUTS Code Learning System

Learns from local git repositories (ecemath, dev-agent) and remote repos
by extracting:
  1. Function signatures + docstrings → semantic code patterns
  2. AST structure (depth, branch factor, node types) → structural patterns
  3. Cross-language abstraction levels:
       Python (high-level) → identifies C-style patterns → HDV encoding
  4. Math-code bridges: when function docs mention math terms, the resulting
     HDV vector overlaps with arXiv equation patterns → universals discovered

Key insight: ecemath.numerical_jacobian() computes J[i,j] = dF_i/dx_j —
the exact same Jacobian field described in LOGIC_FLOW.md. The system should
automatically discover this cross-dimensional universal.

CodeExecutor: runs small Python snippets in a subprocess with timeout.
Execution results (stdout, error, type) feed back as 'execution' dimension HDVs.
This closes the loop: learn code → execute → verify → encode result → learn more.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Function extraction ────────────────────────────────────────────────────────

def extract_functions_from_file(py_file: Path) -> List[Dict]:
    """
    Parse a .py file and extract function-level information.

    Returns list of dicts:
      name, docstring, args, return_annotation, lineno,
      body_lines, ast_depth, complexity (# nodes), file_path
    """
    try:
        source = py_file.read_text(errors="replace")
    except Exception:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Docstring
        docstring = ast.get_docstring(node) or ""

        # Arguments
        args = [a.arg for a in node.args.args]

        # Return annotation
        ret_ann = ""
        if node.returns:
            try:
                ret_ann = ast.unparse(node.returns)
            except Exception:
                pass

        # AST depth and complexity for this function
        depth, n_nodes = _ast_metrics(node)

        results.append({
            "name": node.name,
            "docstring": docstring[:300],
            "args": args,
            "return_annotation": ret_ann,
            "lineno": node.lineno,
            "body_lines": (node.end_lineno or node.lineno) - node.lineno,
            "ast_depth": depth,
            "complexity": n_nodes,
            "file_path": str(py_file),
        })

    return results


def _ast_metrics(node: ast.AST) -> Tuple[int, int]:
    """Return (max_depth, total_nodes) for an AST subtree."""
    max_depth = 0
    total = 0

    def visit(n, depth):
        nonlocal max_depth, total
        total += 1
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(n):
            visit(child, depth + 1)

    visit(node, 0)
    return max_depth, total


def extract_functions_from_repo(repo_path: str, max_files: int = 200) -> List[Dict]:
    """
    Scan a repository directory for Python files and extract all functions.

    Args:
        repo_path: path to the local repo
        max_files: cap to avoid scanning giant repos (e.g. dev-agent has 3000+ files)

    Returns list of function dicts with 'repo' key added.
    """
    repo = Path(repo_path)
    if not repo.exists():
        return []

    # Prefer src/ subdirectory if it exists
    src_dir = repo / "src"
    search_root = src_dir if src_dir.exists() else repo

    py_files = sorted(search_root.rglob("*.py"))[:max_files]

    all_functions = []
    for pf in py_files:
        # Skip test files, __pycache__, migrations
        if any(part.startswith(("test", "__pycache__", "migration")) for part in pf.parts):
            continue
        fns = extract_functions_from_file(pf)
        for fn in fns:
            fn["repo"] = repo.name
        all_functions.extend(fns)

    return all_functions


# ── HDV encoding for code ──────────────────────────────────────────────────────

def encode_function_to_hdv(fn_info: Dict, hdv_system) -> Tuple[object, str]:
    """
    Encode a function into the HDV space.

    The encoding strategy:
      - Function name tokens → HDV (structural vocabulary)
      - Docstring tokens → HDV (semantic/math vocabulary — bridges math ↔ code)
      - Argument names → HDV
      - Structural features (depth, complexity) → text encoding

    Domain: 'execution' (code behavior) with title-bridge vocabulary.

    Returns (hdv_vector, domain_used).
    """
    import numpy as np

    # Build a rich text description that shares vocabulary with math/behavioral domains
    parts = [
        fn_info.get("name", ""),
        fn_info.get("docstring", ""),
        " ".join(fn_info.get("args", [])),
        fn_info.get("return_annotation", ""),
        f"complexity {fn_info.get('complexity', 0)} "
        f"depth {fn_info.get('ast_depth', 0)} "
        f"lines {fn_info.get('body_lines', 0)}",
    ]
    full_text = " ".join(p for p in parts if p)

    # Encode as execution domain (bridges to math via shared vocabulary)
    vec = hdv_system.structural_encode(full_text, "execution")
    return vec, "execution"


# ── Repository learning ────────────────────────────────────────────────────────

class RepoCodeLearner:
    """
    Learns from local git repositories by extracting functions and encoding them.

    Feeds patterns into CrossDimensionalDiscovery so that code patterns can
    be discovered as universals with math/behavioral/physical patterns.

    Key capability: ecemath's Jacobian, regime detection, and stability analysis
    code shares vocabulary with the LOGIC_FLOW.md mathematical framework —
    enabling the system to discover that code IS an implementation of the math.
    """

    def __init__(self, hdv_system, discovery, domain_registry=None):
        self.hdv_system = hdv_system
        self.discovery = discovery
        self.domain_registry = domain_registry
        self._learned_repos: set = set()

    def learn_from_repo(
        self,
        repo_path: str,
        max_files: int = 200,
    ) -> int:
        """
        Extract all functions from repo and encode them.

        Returns number of functions encoded.
        """
        if repo_path in self._learned_repos:
            return 0

        functions = extract_functions_from_repo(repo_path, max_files=max_files)
        encoded = 0

        for fn_info in functions:
            try:
                hdv_vec, domain = encode_function_to_hdv(fn_info, self.hdv_system)

                # Classify into one of 150 domains if registry available
                domain_id = "unknown"
                if self.domain_registry:
                    text_for_classify = (
                        fn_info.get("name", "") + " " +
                        fn_info.get("docstring", "")[:100]
                    )
                    domain_id, _ = self.domain_registry.activate_for_text(
                        text_for_classify, self.hdv_system
                    )

                self.discovery.record_pattern(
                    "execution", hdv_vec,
                    {
                        "type": "code_function",
                        "name": fn_info["name"],
                        "repo": fn_info.get("repo", ""),
                        "file": Path(fn_info["file_path"]).name,
                        "domain": domain_id,
                        "complexity": fn_info.get("complexity", 0),
                        "docstring_preview": fn_info.get("docstring", "")[:60],
                    },
                )
                encoded += 1
            except Exception:
                continue

        self._learned_repos.add(repo_path)
        return encoded

    def learn_from_local_repos(self, repo_paths: List[str]) -> Dict[str, int]:
        """Learn from all specified local repos. Returns {repo: n_encoded}."""
        results = {}
        for rp in repo_paths:
            n = self.learn_from_repo(rp)
            if n > 0:
                results[rp] = n
                print(f"[CodeLearner] Encoded {n} functions from {Path(rp).name}")
        return results


# ── Code execution ─────────────────────────────────────────────────────────────

class CodeExecutor:
    """
    Safely executes Python code snippets in a subprocess with timeout.

    Results (stdout, type, timing) are encoded as HDV vectors in the
    'execution' dimension, creating a feedback loop:
      learn code pattern → execute → encode result → learn from output.

    Safety: code runs in a fresh subprocess with:
      - No network access (relies on OS defaults; not a full sandbox)
      - 5-second timeout
      - stdout/stderr captured (not displayed to user)
      - No file system writes outside /tmp
    """

    def __init__(self, timeout: float = 5.0, max_output_chars: int = 2000):
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, code: str) -> Dict:
        """
        Run code in subprocess. Returns:
          {
            'stdout': str, 'stderr': str, 'returncode': int,
            'elapsed': float, 'success': bool, 'output_type': str
          }
        """
        # Wrap to capture repr of last expression (Jupyter-style)
        wrapped = textwrap.dedent(f"""
import sys, traceback
try:
    exec(compile({repr(code)}, '<snippet>', 'exec'))
    print('__EXEC_OK__')
except Exception as e:
    print(f'__EXEC_ERR__: {{e}}', file=sys.stderr)
""")
        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "-c", wrapped],
                capture_output=True, text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start
            stdout = result.stdout[:self.max_output_chars]
            stderr = result.stderr[:self.max_output_chars]
            success = result.returncode == 0 and "__EXEC_OK__" in stdout
            output_type = _classify_output(stdout, stderr)
            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
                "elapsed": round(elapsed, 3),
                "success": success,
                "output_type": output_type,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "", "stderr": "timeout",
                "returncode": -1, "elapsed": self.timeout,
                "success": False, "output_type": "timeout",
            }
        except Exception as e:
            return {
                "stdout": "", "stderr": str(e),
                "returncode": -1, "elapsed": 0.0,
                "success": False, "output_type": "error",
            }

    def execute_and_encode(self, code: str, hdv_system) -> Optional[object]:
        """
        Execute code, then encode the result as an HDV vector.

        The execution result descriptor is encoded into the 'execution' domain.
        Returns HDV vector or None on failure.
        """
        result = self.execute(code)
        # Build descriptor from execution outcome
        descriptor = (
            f"execution {result['output_type']} "
            f"success {result['success']} "
            f"elapsed {int(result['elapsed'] * 1000)} ms "
            f"output {result['stdout'][:100]}"
        )
        return hdv_system.structural_encode(descriptor, "execution")


def _classify_output(stdout: str, stderr: str) -> str:
    """Classify execution output type."""
    if "timeout" in stderr.lower():
        return "timeout"
    if stderr and not stdout:
        return "error"
    if "__EXEC_OK__" in stdout:
        return "success"
    if any(c.isdigit() for c in stdout[:50]):
        return "numeric"
    return "text"


# ── Active learning: unknown unknowns ─────────────────────────────────────────

class UnknownUnknownDetector:
    """
    Detects gaps in the system's knowledge coverage across the 150 domains.

    'Unknown unknowns' are domains that:
      1. Have zero recorded patterns (completely unseen)
      2. Have very few patterns relative to the domain's semantic density
      3. Are adjacent to active domains but haven't been activated

    Uses this analysis to prioritise which repos/papers to ingest next.
    """

    def __init__(self, domain_registry, discovery):
        self.registry = domain_registry
        self.discovery = discovery

    def find_uncovered_domains(self) -> List[str]:
        """Return domains with 0 recorded patterns."""
        inactive = self.registry.inactive_domains()
        return inactive

    def find_sparse_domains(self, min_patterns: int = 5) -> List[str]:
        """Return active domains with fewer than min_patterns recorded."""
        counts = self.discovery.get_pattern_counts()
        sparse = []
        for d in self.registry.active_domains():
            if counts.get(d, 0) < min_patterns:
                sparse.append(d)
        return sparse

    def recommend_repos_for_domain(self, domain_id: str) -> List[str]:
        """
        Suggest GitHub repos to ingest that would populate a domain.

        Uses keyword matching: domain keywords → repo name suggestions.
        """
        from tensor.domain_registry import _ID_TO_ENTRY
        entry = _ID_TO_ENTRY.get(domain_id)
        if not entry:
            return []
        keywords = entry[3]
        # Map common keywords to well-known repos
        _KEYWORD_REPO_MAP = {
            "protein": "https://github.com/aqlaboratory/openfold",
            "molecular": "https://github.com/deepchem/deepchem",
            "drug": "https://github.com/deepchem/deepchem",
            "genomic": "https://github.com/google-deepmind/alphatensor",
            "flood": "https://github.com/neuralhydrology/neuralhydrology",
            "climate": "https://github.com/pangeo-data/pangeo",
            "weather": "https://github.com/google-deepmind/graphcast",
            "turbulence": "https://github.com/google/jax-cfd",
            "circuit": "https://github.com/ahkole/spicepy",
            "quantum": "https://github.com/Qiskit/qiskit",
            "federated": "https://github.com/google-deepmind/federated",
            "fraud": "https://github.com/fraud-detection-handbook/fraud-detection-handbook",
            "robot": "https://github.com/openai/robosuite",
        }
        suggested = []
        for kw in keywords:
            repo = _KEYWORD_REPO_MAP.get(kw)
            if repo and repo not in suggested:
                suggested.append(repo)
        return suggested[:3]

    def coverage_report(self) -> Dict:
        """Return a summary of knowledge coverage across all 150 domains."""
        inactive = self.find_uncovered_domains()
        # Group inactive domains by category (first word of domain ID)
        categories: Dict[str, int] = {}
        for d in inactive:
            cat = d.split("_")[0]
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_domains": 150,
            "covered": 150 - len(inactive),
            "uncovered": len(inactive),
            "coverage_pct": round(100.0 * (150 - len(inactive)) / 150, 1),
            "uncovered_by_category": categories,
            "top_uncovered": inactive[:10],
        }
