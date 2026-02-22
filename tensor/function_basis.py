"""
FICUTS Layer 8: Universal Function Basis

Classes:
  - EquationParser:        LaTeX → SymPy, classify, extract params  (Task 8.1)
  - FunctionBasisLibrary:  Aggregate equations from papers           (Task 8.2)
  - FunctionBasisToHDV:    Map library → HDV dimension mask          (Task 8.3)
  - DEQCanonicalizer:      Filter noise, detect derivatives, tag DEQ structure
"""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sympy as sp

# Lazy-import parse_latex so tests can still run if antlr4 is missing
try:
    from sympy.parsing.latex import parse_latex as _parse_latex
    _LATEX_AVAILABLE = True
except Exception:
    _LATEX_AVAILABLE = False


# ── Task 8.1 ──────────────────────────────────────────────────────────────────

class EquationParser:
    """Parse LaTeX equations → SymPy, classify function type, extract params."""

    _INDEPENDENT = {'t', 'x', 'y', 'z', 'T', 'X', 'Y', 'Z'}

    def parse(self, latex_string: str) -> Optional[sp.Expr]:
        """Convert LaTeX → SymPy expression."""
        if _LATEX_AVAILABLE:
            try:
                cleaned = latex_string.replace(r'\,', ' ').replace(r'\;', ' ')
                return _parse_latex(cleaned)
            except Exception:
                pass
        return self._parse_simple(latex_string)

    def _parse_simple(self, latex_string: str) -> Optional[sp.Expr]:
        """Fallback: recognise exponential e^{...} patterns."""
        match = re.search(r'e\^\{([^}]+)\}', latex_string)
        if match:
            try:
                return sp.exp(sp.sympify(match.group(1)))
            except Exception:
                pass
        return None

    _E_SYM = sp.Symbol('e')

    def classify_function_type(self, expr: Optional[sp.Expr]) -> str:
        if expr is None:
            return 'unknown'
        # SymPy type hierarchy (most reliable)
        if expr.has(sp.exp):
            return 'exponential'
        # parse_latex("e^{...}") produces Pow(Symbol('e'), ...) not sp.exp
        if isinstance(expr, sp.Pow) and expr.base in (sp.E, self._E_SYM):
            return 'exponential'
        # Recurse into args for nested expressions
        if any(isinstance(a, sp.Pow) and a.base in (sp.E, self._E_SYM)
               for a in sp.preorder_traversal(expr)):
            return 'exponential'
        s = str(expr)
        if 'E**' in s:
            return 'exponential'
        if any(f in s for f in ('sin', 'cos', 'tan', 'Sin', 'Cos', 'Tan')):
            return 'trigonometric'
        if any(f in s for f in ('log', 'ln', 'Log')):
            return 'logarithmic'
        if 'Pow' in s or '**' in s:
            return 'power_law' if any(v in s for v in ('t', 'x')) else 'polynomial'
        return 'algebraic'

    def extract_parameters(self, expr: Optional[sp.Expr]) -> List[str]:
        if expr is None:
            return []
        return sorted(
            str(sym) for sym in expr.free_symbols
            if str(sym) not in self._INDEPENDENT
        )


# ── DEQ Canonicalization ──────────────────────────────────────────────────────

# LaTeX derivative signals: any of these in the raw latex → candidate DEQ entry
_DERIVATIVE_LATEX_SIGNALS = (
    r'\partial', r'\nabla', r'\Delta', r'\dot', r'\ddot',
    r'\frac{d', r'\frac{\partial', r'd/dt', r'\mathcal{L}',
    r'\square',  # d'Alembertian
)

# SymPy node types that indicate a derivative
_SYMPY_DERIVATIVE_TYPES = (sp.Derivative,)

# Noise patterns in symbolic_str that flag formatting debris, not math
_NOISE_PATTERNS = (
    'text*', 'mathbf', 'mathrm', 'mathcal*', 'widetilde*',
    'hat*', 'bar*', 'tilde*', 'overline',
    # Recursive ASCII-expansion artifacts from parse failures
    'r*(', 'e*(s*(s', 'o*(m*(p',
)


def _is_noise(symbolic_str: str) -> bool:
    """Return True if the symbolic string looks like a LaTeX formatting artifact."""
    return any(p in symbolic_str for p in _NOISE_PATTERNS)


def _has_derivative_latex(latex: str) -> bool:
    """Return True if the raw LaTeX string contains explicit derivative notation."""
    return any(sig in latex for sig in _DERIVATIVE_LATEX_SIGNALS)


def _has_derivative_sympy(expr: sp.Expr) -> bool:
    """Return True if the SymPy expression tree contains a Derivative node."""
    if expr is None:
        return False
    return any(isinstance(n, _SYMPY_DERIVATIVE_TYPES) for n in sp.preorder_traversal(expr))


class DEQCanonicalizer:
    """
    Stage-2 canonicalization layer for the function library.

    Responsibilities:
    1. Filter noise entries (LaTeX macro expansion artifacts).
    2. Detect derivative operators in both raw LaTeX and SymPy trees.
    3. Promote qualifying entries to structured DEQ form with metadata:
           equation_type:  'ODE' | 'PDE' | 'algebraic' | 'unknown'
           order_time:     int (0 if no time derivative)
           order_space:    int (0 if no spatial derivative)
           has_nonlinearity: bool
           operator_terms: list[str]  e.g. ['laplacian', 'diffusion', 'cubic']
    4. Tag operator_terms from recognisable canonical patterns.

    Usage:
        canon = DEQCanonicalizer()
        meta = canon.canonicalize(latex_str, sympy_expr)
        # meta is None → discard entry (noise)
        # meta is dict  → merge into library entry
    """

    # Operator-term detection: (latex_signal, operator_label)
    # NOTE: '^2' is intentionally omitted — it fires on '\nabla^2' (false positive).
    #       Quadratic/cubic nonlinearity is detected separately in _has_nonlinearity.
    _OPERATOR_SIGNALS: list = [
        (r'\nabla^2',      'laplacian'),
        (r'\nabla^{2}',    'laplacian'),
        (r'\Delta',        'laplacian'),
        (r'\nabla \cdot',  'divergence'),
        (r'\nabla \times', 'curl'),
        # Time derivatives (order matters: check 2nd before 1st)
        (r'\ddot{',        'second_order_time'),
        (r'\frac{\partial^2}{\partial t^2}', 'second_order_time'),
        (r'\partial^2}{\partial t', 'second_order_time'),
        (r'\dot{',         'time_derivative'),
        (r'\partial_t',    'time_derivative'),
        (r'\partial_{t}',  'time_derivative'),
        (r'\partial t',    'time_derivative'),   # \frac{\partial X}{\partial t}
        (r'\frac{\partial}{\partial t}', 'time_derivative'),
        (r'\frac{d}{dt}',  'time_derivative'),
        # Spatial derivatives
        (r'\frac{\partial}{\partial x}', 'spatial_derivative'),
        (r'\partial_x',    'spatial_derivative'),
        (r'\partial x',    'spatial_derivative'),
        # Nonlinear terms (only explicit cubic — quadratic via SymPy in _has_nonlinearity)
        (r'^3',            'cubic'),
        # Transcendental / other
        (r'\sin',          'trigonometric'),
        (r'\cos',          'trigonometric'),
        (r'e^{',           'exponential'),
        (r'\exp',          'exponential'),
        (r'\log',          'logarithmic'),
        (r'\ln',           'logarithmic'),
        (r'\sigma',        'sigmoid_like'),
        (r'\tanh',         'sigmoid_like'),
        (r'\mathcal{H}',   'hamiltonian'),
        (r'\mathcal{L}',   'lagrangian'),
    ]

    # Additional noise tokens not covered by _NOISE_PATTERNS
    _EXTRA_NOISE = (
        'columnwidth', 'resizebox', 'label*', 'begin*', 'emph*',
        'caption', 'textbf', 'textit', 'frac{', 'cdot', 'vspace',
        'hspace', 'phantom', 'rule{', 'includegraphics',
    )

    def canonicalize(
        self,
        latex: str,
        expr: Optional[sp.Expr],
    ) -> Optional[dict]:
        """
        Attempt to canonicalize one equation.

        Returns:
            None if the entry should be discarded (noise).
            dict with DEQ metadata fields otherwise.
        """
        # 1. Reject noise — check ORIGINAL latex BEFORE SymPy reorders symbols
        if _is_noise(latex) or any(t in latex for t in self._EXTRA_NOISE):
            return None
        symbolic_str = str(expr) if expr is not None else latex
        # Also check post-sympify representation (belt-and-suspenders)
        if _is_noise(symbolic_str):
            return None
        # Additional noise guard: suspiciously short symbolic strings
        if len(symbolic_str) < 4:
            return None

        # 2. Detect derivative presence
        has_deriv = _has_derivative_latex(latex) or _has_derivative_sympy(expr)

        # 3. Detect operator terms
        operator_terms = self._extract_operator_terms(latex)

        # 4. Infer time/space order
        order_time = self._infer_order(latex, time=True)
        order_space = self._infer_order(latex, time=False)

        # 5. Classify equation type
        if has_deriv:
            if order_space > 0:
                eq_type = 'PDE'
            else:
                eq_type = 'ODE'
        else:
            eq_type = 'algebraic'

        # 6. Detect nonlinearity (crude but effective)
        has_nonlinearity = self._has_nonlinearity(latex, expr)

        return {
            'equation_type': eq_type,
            'order_time': order_time,
            'order_space': order_space,
            'has_nonlinearity': has_nonlinearity,
            'operator_terms': operator_terms,
            'is_canonical_deq': has_deriv,
        }

    def _extract_operator_terms(self, latex: str) -> list:
        seen = set()
        terms = []
        for signal, label in self._OPERATOR_SIGNALS:
            if signal in latex and label not in seen:
                terms.append(label)
                seen.add(label)
        # Combined check: \partial^2 + \partial t → second-order time derivative
        # Catches \frac{\partial^2 u}{\partial t^2} where intermediate {} blocks simple search
        if ('second_order_time' not in seen
                and r'\partial^2' in latex
                and r'\partial t' in latex):
            terms.append('second_order_time')
            seen.add('second_order_time')
            # Upgrade time_derivative → second_order_time if already present
            if 'time_derivative' in seen:
                terms = [t for t in terms if t != 'time_derivative']
                seen.discard('time_derivative')
        return terms

    # Pattern for second-order time: \partial^2 appears AND \partial t also appears
    # (covers \frac{\partial^2 u}{\partial t^2} without being fooled by spatial partials)
    _RE_PARTIAL2_T = re.compile(r'\\partial\^2')

    def _infer_order(self, latex: str, time: bool) -> int:
        """Infer highest derivative order for time (time=True) or space (time=False)."""
        if time:
            # Second-order time: \ddot or \partial^2 + \partial t (wave equation form)
            if (r'\ddot' in latex
                    or r'\frac{\partial^2}{\partial t^2}' in latex
                    or (r'\partial^2' in latex and r'\partial t' in latex)):
                return 2
            # First-order time (\partial t catches \frac{\partial X}{\partial t})
            if (r'\dot' in latex
                    or r'\partial_t' in latex
                    or r'\partial_{t}' in latex
                    or r'\partial t' in latex
                    or r'\frac{\partial}{\partial t}' in latex
                    or r'\frac{d}{dt}' in latex
                    or r'\partial_{t' in latex):
                return 1
            return 0
        else:
            # Laplacian (∇²) → order 2; simple spatial partial → 1
            if r'\nabla^2' in latex or r'\Delta' in latex or r'\nabla^{2}' in latex:
                return 2
            if (r'\partial_x' in latex
                    or r'\partial_{x}' in latex
                    or r'\frac{\partial}{\partial x}' in latex
                    or r'\partial x' in latex
                    or r'\nabla' in latex):
                return 1
            return 0

    def _has_nonlinearity(self, latex: str, expr: Optional[sp.Expr]) -> bool:
        """Return True if the equation contains nonlinear terms."""
        # Cubic or higher powers
        if any(f in latex for f in ('^3', '^4', '^5', r'\sin', r'\cos', r'\tanh', r'\sigma')):
            return True
        if expr is not None:
            # Check for Pow with exponent > 1 and a non-constant base
            for node in sp.preorder_traversal(expr):
                if isinstance(node, sp.Pow):
                    base, exp = node.args
                    if exp.is_number and float(exp) > 1 and not base.is_number:
                        return True
        return False

    def filter_library(self, library: dict) -> dict:
        """
        Filter and annotate an existing library dict in-place.

        Returns the same dict with:
        - Noise entries removed (keys deleted)
        - DEQ metadata merged into surviving entries
        """
        parser = EquationParser()
        to_delete = []
        for key, entry in library.items():
            latex = entry.get('symbolic_str', '')
            try:
                expr = sp.sympify(latex)
            except Exception:
                expr = None

            meta = self.canonicalize(latex, expr)
            if meta is None:
                to_delete.append(key)
            else:
                entry.update(meta)

        for key in to_delete:
            del library[key]

        return library


# ── Task 8.2 ──────────────────────────────────────────────────────────────────

class FunctionBasisLibrary:
    """
    Aggregated function library discovered from ingested papers.

    Each entry:
      symbolic_str  : str(SymPy expr)
      type          : exponential | trigonometric | …
      parameters    : list of symbol names
      domains       : set of domain strings
      source_papers : list of paper_id strings
      classification: 'experimental' | 'foundational'
      discovered_at : float (unix timestamp)
    """

    def __init__(self, library_path: str = 'tensor/data/function_library.json'):
        self.library_path = Path(library_path)
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        self.parser = EquationParser()
        self.canonicalizer = DEQCanonicalizer()

        if self.library_path.exists():
            raw = json.loads(self.library_path.read_text())
            # Restore domains as sets
            self.library: Dict[str, dict] = {
                k: {**v, 'domains': set(v['domains'])} for k, v in raw.items()
            }
        else:
            self.library: Dict[str, dict] = {}

    def ingest_papers_from_storage(self, storage_dir: str = 'tensor/data/ingested'):
        storage = Path(storage_dir)
        paper_files = [f for f in storage.glob('*.json') if f.name != 'seen_urls.json']
        print(f"[FunctionLibrary] Processing {len(paper_files)} papers")

        for paper_file in paper_files:
            paper_data = json.loads(paper_file.read_text())
            paper_id = paper_file.stem
            url = paper_data.get('url', '')
            title = paper_data.get('article', {}).get('title', '')
            domain = self._infer_domain(url, title)

            for eq_latex in paper_data.get('concepts', {}).get('equations', []):
                self._add_equation(paper_id, eq_latex, domain)

        self._save_library()
        print(f"[FunctionLibrary] Library now has {len(self.library)} functions")

    def add_equation_direct(self, paper_id: str, latex: str, domain: str):
        """Add a single equation directly (used in tests)."""
        self._add_equation(paper_id, latex, domain)
        self._save_library()

    def _add_equation(self, paper_id: str, latex: str, domain: str):
        expr = self.parser.parse(latex)
        if expr is None:
            return
        func_type = self.parser.classify_function_type(expr)
        params = self.parser.extract_parameters(expr)

        # DEQ canonicalization: annotate entry + filter noise
        deq_meta = self.canonicalizer.canonicalize(latex, expr)
        if deq_meta is None:
            return  # noise entry — discard

        existing = self._find_matching_function(expr)

        if existing:
            self.library[existing]['domains'].add(domain)
            if paper_id not in self.library[existing]['source_papers']:
                self.library[existing]['source_papers'].append(paper_id)
        else:
            name = f"{func_type}_{len(self.library)}"
            entry = {
                'symbolic_str': str(expr),
                'type': func_type,
                'parameters': params,
                'domains': {domain},
                'source_papers': [paper_id],
                'classification': 'experimental',
                'discovered_at': time.time(),
            }
            entry.update(deq_meta)
            self.library[name] = entry

    def canonicalize_library(self) -> int:
        """
        Run DEQ canonicalization over the entire existing library.

        Removes noise entries and annotates survivors with DEQ metadata.
        Returns the number of entries removed.

        Use this once after bulk ingestion to upgrade legacy algebraic entries.
        """
        before = len(self.library)
        self.canonicalizer.filter_library(self.library)
        removed = before - len(self.library)
        self._save_library()
        print(f"[DEQ] Canonicalization complete: {removed} noise entries removed, "
              f"{len(self.library)} entries annotated")
        return removed

    def _find_matching_function(self, expr: sp.Expr) -> Optional[str]:
        for name, data in self.library.items():
            try:
                existing = sp.sympify(data['symbolic_str'])
                if sp.simplify(expr - existing) == 0:
                    return name
            except Exception:
                continue
        return None

    def _infer_domain(self, url: str, title: str) -> str:
        text = (url + ' ' + title).lower()
        if any(k in text for k in ('circuit', 'electronic', 'vlsi', 'semiconductor')):
            return 'ece'
        if any(k in text for k in ('biology', 'neuron', 'synapse', 'cell')):
            return 'biology'
        if any(k in text for k in ('finance', 'market', 'trading', 'economics')):
            return 'finance'
        if any(k in text for k in ('physics', 'mechanics', 'quantum')):
            return 'physics'
        return 'general'

    def get_universal_functions(self, min_domains: int = 3) -> List[str]:
        return [n for n, d in self.library.items() if len(d['domains']) >= min_domains]

    def promote_to_foundational(self, func_name: str):
        if func_name in self.library:
            self.library[func_name]['classification'] = 'foundational'
            self._save_library()

    def _save_library(self):
        serializable = {
            k: {**v, 'domains': list(v['domains'])}
            for k, v in self.library.items()
        }
        self.library_path.write_text(json.dumps(serializable, indent=2))


# ── Task 8.3 ──────────────────────────────────────────────────────────────────

class FunctionBasisToHDV:
    """Map function library → HDV dimension masks."""

    def __init__(self, function_library: FunctionBasisLibrary, hdv_dim: int = 10000):
        self.library = function_library
        self.hdv_dim = hdv_dim
        self.dim_assignments: Dict[str, object] = {}  # name → int or List[int]
        self.next_free_dim = 0

    def assign_dimensions(self):
        funcs = list(self.library.library.items())

        def priority(item):
            _, data = item
            return (
                0 if data['classification'] == 'foundational' else 1,
                -len(data['domains']),
            )

        funcs.sort(key=priority)

        for func_name, func_data in funcs:
            if func_data['classification'] == 'foundational':
                self.dim_assignments[func_name] = self.next_free_dim
                self.next_free_dim += 1
            elif len(func_data['domains']) >= 3:
                dims = list(range(self.next_free_dim, self.next_free_dim + 5))
                self.dim_assignments[func_name] = dims
                self.next_free_dim += 5
            else:
                self.dim_assignments[func_name] = self.next_free_dim
                self.next_free_dim += 1

        print(f"[HDVMapping] Assigned {self.next_free_dim} dims to {len(funcs)} functions")

    def get_domain_mask(self, domain: str) -> np.ndarray:
        mask = np.zeros(self.hdv_dim, dtype=bool)
        for func_name, func_data in self.library.library.items():
            if domain in func_data.get('domains', set()):
                dims = self.dim_assignments.get(func_name, [])
                if isinstance(dims, int):
                    dims = [dims]
                for d in dims:
                    if d < self.hdv_dim:
                        mask[d] = True
        return mask

    def get_overlap_dimensions(self) -> set:
        all_domains: set = set()
        for data in self.library.library.values():
            all_domains.update(data.get('domains', set()))

        usage = np.zeros(self.hdv_dim, dtype=int)
        for domain in all_domains:
            usage += self.get_domain_mask(domain).astype(int)

        return set(np.where(usage >= 2)[0].tolist())


# ── Task 8.4 ──────────────────────────────────────────────────────────────────

def populate_library_from_arxiv(
    ingested_dir: str = 'tensor/data/ingested',
    library_path: str = 'tensor/data/function_library.json',
    max_papers: Optional[int] = None,
) -> FunctionBasisLibrary:
    """
    Populate the function library from already-ingested arXiv papers.

    Reads paper URLs from tensor/data/ingested/*.json, downloads each paper's
    LaTeX source from arXiv's /e-print/ endpoint, extracts equations, and adds
    them to the FunctionBasisLibrary.

    This bridges the gap between:
      Layer 6 (web ingestion — stores paper metadata) and
      Layer 8 (function basis — needs real equations)

    Args:
        ingested_dir: directory containing *.json ingested paper files
        library_path: path to save the populated function library
        max_papers:   cap on papers processed (None = all)

    Returns:
        Populated FunctionBasisLibrary instance.
    """
    from tensor.arxiv_pdf_parser import ArxivPDFSourceParser

    parser = ArxivPDFSourceParser(rate_limit_seconds=1.5)
    library = FunctionBasisLibrary(library_path)
    storage = Path(ingested_dir)

    paper_files = [
        f for f in storage.glob('*.json')
        if f.name != 'seen_urls.json'
    ]
    if max_papers is not None:
        paper_files = paper_files[:max_papers]

    print(f"[Library] Processing {len(paper_files)} ingested papers")

    success, failed = 0, 0
    for paper_file in paper_files:
        try:
            data = json.loads(paper_file.read_text())
        except Exception:
            continue

        url = data.get('url', '')
        if 'arxiv.org' not in url:
            failed += 1
            continue

        title = data.get('article', {}).get('title', '')
        domain = library._infer_domain(url, title)

        result = parser.parse_arxiv_paper(url)
        if result and result['equations']:
            for eq_latex in result['equations']:
                library._add_equation(
                    paper_id=result['paper_id'],
                    latex=eq_latex,
                    domain=domain,
                )
            success += 1
            if success % 10 == 0:
                library._save_library()
                print(f"[Library] {success} papers parsed, "
                      f"{len(library.library)} functions")
        else:
            failed += 1

    library._save_library()
    print(f"[Library] Complete: {success} parsed, {failed} failed, "
          f"{len(library.library)} total functions")
    return library


# ── Task 8.5: Threaded Ingestion ─────────────────────────────────────────────


class ThreadedArxivIngester:
    """
    Multi-threaded arXiv paper ingestion pipeline.

    Architecture:
      N agent threads, each processing a batch of papers end-to-end.
      A shared rate limiter ensures <= 1 HTTP request per `rate_limit` seconds
      across all agents (respecting arXiv's rate policy).
      Library writes are synchronized via a separate lock.

    Each agent thread:
      1. Reads paper metadata from its batch
      2. Downloads LaTeX source (serialized by rate limiter)
      3. Extracts equations + runs SymPy classification (concurrent — overlaps
         with other agents' downloads during rate-limit waits)
      4. Batch-writes results to the shared library under lock

    This gives ~N× speedup on the CPU-bound SymPy parsing while keeping
    downloads correctly rate-limited.
    """

    def __init__(
        self,
        library: FunctionBasisLibrary,
        n_agents: int = 4,
        rate_limit: float = 1.5,
        journal_path: str = 'tensor/data/ingestion_journal.json',
    ):
        self.library = library
        self.n_agents = n_agents
        self.rate_limit = rate_limit
        self.journal_path = Path(journal_path)

        # Shared download session + rate limiter
        self._download_lock = threading.Lock()
        self._last_request = 0.0

        # Library write lock
        self._library_lock = threading.Lock()

        # Journal lock (for thread-safe checkpoint writes)
        self._journal_lock = threading.Lock()

        # Aggregated stats
        self._stats_lock = threading.Lock()
        self._stats = {'success': 0, 'failed': 0, 'new_eqs': 0}

        # Load journal (set of already-processed paper file stems)
        self._processed: set = set()
        if self.journal_path.exists():
            try:
                journal = json.loads(self.journal_path.read_text())
                self._processed = set(journal.get('processed', []))
            except Exception:
                pass

    def _save_journal(self):
        """Persist journal to disk (call under _journal_lock)."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path.write_text(json.dumps({
            'processed': sorted(self._processed),
            'stats': self._stats,
            'timestamp': time.time(),
        }, indent=2))

    def _mark_processed(self, paper_stem: str):
        """Thread-safe journal update."""
        with self._journal_lock:
            self._processed.add(paper_stem)

    def ingest(
        self,
        ingested_dir: str = 'tensor/data/ingested',
        max_papers: Optional[int] = None,
        resume: bool = True,
    ) -> dict:
        """
        Run threaded ingestion across N agent threads.

        Args:
            ingested_dir: directory containing *.json ingested paper files
            max_papers:   cap on papers processed (None = all)
            resume:       if True, skip papers already in the journal

        Returns:
            dict with keys: success, failed, new_eqs, total_library_size
        """
        storage = Path(ingested_dir)
        paper_files = [
            f for f in storage.glob('*.json')
            if f.name != 'seen_urls.json'
        ]
        if max_papers is not None:
            paper_files = paper_files[:max_papers]

        # Filter out already-processed papers when resuming
        if resume and self._processed:
            before = len(paper_files)
            paper_files = [f for f in paper_files if f.stem not in self._processed]
            skipped = before - len(paper_files)
            if skipped > 0:
                print(f"[ThreadedIngester] Resuming: skipping {skipped} "
                      f"already-processed papers")

        if not paper_files:
            print("[ThreadedIngester] No new papers to process")
            return {**self._stats, 'total_library_size': len(self.library.library)}

        # Distribute papers across agents (round-robin for balanced load)
        batches: List[List[Path]] = [[] for _ in range(self.n_agents)]
        for i, pf in enumerate(paper_files):
            batches[i % self.n_agents].append(pf)

        n_active = sum(1 for b in batches if b)
        print(f"[ThreadedIngester] {len(paper_files)} papers → "
              f"{n_active} agent threads")

        try:
            with ThreadPoolExecutor(
                max_workers=n_active,
                thread_name_prefix="ingest-agent",
            ) as pool:
                futures = {
                    pool.submit(self._agent_worker, agent_id, batch): agent_id
                    for agent_id, batch in enumerate(batches)
                    if batch
                }

                for future in as_completed(futures):
                    agent_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"[Agent-{agent_id}] ERROR: {exc}")
        except (KeyboardInterrupt, SystemExit):
            print("\n[ThreadedIngester] Interrupted — saving checkpoint...")
        finally:
            # Always save state on exit (graceful shutdown)
            self.library._save_library()
            with self._journal_lock:
                self._save_journal()

        result = {
            **self._stats,
            'total_library_size': len(self.library.library),
        }
        print(f"[ThreadedIngester] Done: {result['success']} parsed, "
              f"{result['failed']} failed, {result['new_eqs']} new equations, "
              f"{result['total_library_size']} total functions")
        return result

    def _rate_limited_download(self, url: str, session: 'requests.Session'):
        """Download with shared rate limiter across all agent threads."""
        with self._download_lock:
            now = time.time()
            wait = self.rate_limit - (now - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()
            try:
                resp = session.get(url, timeout=30)
                return resp if resp.status_code == 200 else None
            except Exception:
                return None

    def _agent_worker(self, agent_id: int, paper_files: List[Path]):
        """
        Single agent thread: downloads + parses its batch of papers.

        Each agent has its own ArxivPDFSourceParser (for extraction logic)
        and requests.Session, but downloads go through the shared rate limiter.
        """
        import requests as req

        session = req.Session()
        session.headers["User-Agent"] = (
            f"FICUTSResearchBot/1.0 agent-{agent_id} (educational)"
        )

        # Per-thread parsers (stateless, no sharing issues)
        arxiv_parser = _ArxivExtractorLocal()
        eq_parser = EquationParser()
        canonicalizer = DEQCanonicalizer()

        local_success, local_failed, local_new = 0, 0, 0

        for paper_file in paper_files:
            try:
                data = json.loads(paper_file.read_text())
            except Exception:
                continue

            url = data.get('url', '')
            if 'arxiv.org' not in url:
                local_failed += 1
                continue

            title = data.get('article', {}).get('title', '')
            domain = self.library._infer_domain(url, title)
            paper_id = paper_file.stem

            # Extract arXiv paper ID
            pid = arxiv_parser.extract_paper_id(url)
            if not pid:
                local_failed += 1
                continue

            # Rate-limited download (blocks until slot available)
            source_url = f"https://arxiv.org/e-print/{pid}"
            resp = self._rate_limited_download(source_url, session)
            if resp is None:
                local_failed += 1
                continue

            # CPU-bound: extract equations from LaTeX (runs concurrently
            # with other agents' downloads during rate-limit waits)
            equations = arxiv_parser.extract_from_content(resp.content)
            if not equations:
                local_failed += 1
                continue

            # Parse each equation with SymPy + DEQ canonicalization
            parsed_entries = []
            for eq_latex in equations:
                expr = eq_parser.parse(eq_latex)
                if expr is None:
                    continue

                func_type = eq_parser.classify_function_type(expr)
                params = eq_parser.extract_parameters(expr)

                deq_meta = canonicalizer.canonicalize(eq_latex, expr)
                if deq_meta is None:
                    continue

                parsed_entries.append({
                    'expr': expr,
                    'symbolic_str': str(expr),
                    'raw_latex': eq_latex,
                    'func_type': func_type,
                    'params': params,
                    'deq_meta': deq_meta,
                })

            # Batch write to library under lock
            with self._library_lock:
                for entry in parsed_entries:
                    existing = self.library._find_matching_function(entry['expr'])
                    if existing:
                        self.library.library[existing]['domains'].add(domain)
                        if paper_id not in self.library.library[existing]['source_papers']:
                            self.library.library[existing]['source_papers'].append(paper_id)
                        if (entry['deq_meta'].get('is_canonical_deq')
                                and not self.library.library[existing].get('is_canonical_deq')):
                            self.library.library[existing].update(entry['deq_meta'])
                            self.library.library[existing]['raw_latex'] = entry['raw_latex']
                    else:
                        name = f"{entry['func_type']}_{len(self.library.library)}"
                        lib_entry = {
                            'symbolic_str': entry['symbolic_str'],
                            'raw_latex': entry['raw_latex'],
                            'type': entry['func_type'],
                            'parameters': entry['params'],
                            'domains': {domain},
                            'source_papers': [paper_id],
                            'classification': 'experimental',
                            'discovered_at': time.time(),
                        }
                        lib_entry.update(entry['deq_meta'])
                        self.library.library[name] = lib_entry
                        local_new += 1

            local_success += 1

            # Journal this paper as processed (survives interrupts)
            self._mark_processed(paper_file.stem)

            # Periodic save + progress
            with self._stats_lock:
                self._stats['success'] += 1
                self._stats['new_eqs'] += local_new
                total_s = self._stats['success']
                if total_s % 10 == 0:
                    self.library._save_library()
                    with self._journal_lock:
                        self._save_journal()
                    print(f"  [Agent-{agent_id}] {total_s} papers done, "
                          f"{len(self.library.library)} functions")
                local_new = 0  # reset local counter after flushing

        # Flush remaining stats
        with self._stats_lock:
            self._stats['failed'] += local_failed
            self._stats['new_eqs'] += local_new

        print(f"  [Agent-{agent_id}] Finished batch: "
              f"{local_success} ok, {local_failed} failed")


class _ArxivExtractorLocal:
    """
    Lightweight equation extractor — no HTTP, no rate limiter.

    Replicates ArxivPDFSourceParser's extraction logic without the
    download machinery, so agent threads can use it without sharing state.
    """

    _EQUATION_PATTERNS = [
        r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
        r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
        r"\\begin\{eqnarray\*?\}(.*?)\\end\{eqnarray\*?\}",
        r"\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}",
        r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}",
        r"\\\[(.*?)\\\]",
        r"\$\$(.*?)\$\$",
    ]

    _ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
    _OLD_ID_RE = re.compile(r"^[a-z-]+/\d+$")

    def extract_paper_id(self, url: str) -> Optional[str]:
        url = url.strip()
        for marker in ("/abs/", "/pdf/", "/e-print/"):
            if marker in url:
                pid = url.split(marker)[-1]
                pid = pid.replace(".pdf", "").split("v")[0]
                return pid.strip("/")
        if self._ID_RE.match(url):
            return url
        if self._OLD_ID_RE.match(url):
            return url
        return None

    def extract_from_content(self, content: bytes) -> List[str]:
        import gzip
        import io
        import tarfile

        # Try tar.gz
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
        except Exception:
            pass

        # Try plain gzip
        try:
            latex = gzip.decompress(content).decode("utf-8", errors="ignore")
            if "\\begin{document}" in latex or "\\documentclass" in latex:
                return self._extract_equations(latex)
        except Exception:
            pass

        # Try raw LaTeX
        try:
            latex = content.decode("utf-8", errors="ignore")
            if "\\begin{document}" in latex or "\\documentclass" in latex:
                return self._extract_equations(latex)
        except Exception:
            pass

        return []

    def _extract_equations(self, latex: str) -> List[str]:
        equations = []
        for pattern in self._EQUATION_PATTERNS:
            for m in re.findall(pattern, latex, re.DOTALL):
                m = m.strip()
                if m and len(m) > 3 and not m.isspace():
                    equations.append(m)
        return equations
