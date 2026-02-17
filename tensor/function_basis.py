"""
FICUTS Layer 8: Universal Function Basis

Classes:
  - EquationParser:        LaTeX → SymPy, classify, extract params  (Task 8.1)
  - FunctionBasisLibrary:  Aggregate equations from papers           (Task 8.2)
  - FunctionBasisToHDV:    Map library → HDV dimension mask          (Task 8.3)
"""

import json
import re
import time
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
        existing = self._find_matching_function(expr)

        if existing:
            self.library[existing]['domains'].add(domain)
            if paper_id not in self.library[existing]['source_papers']:
                self.library[existing]['source_papers'].append(paper_id)
        else:
            name = f"{func_type}_{len(self.library)}"
            self.library[name] = {
                'symbolic_str': str(expr),
                'type': func_type,
                'parameters': params,
                'domains': {domain},
                'source_papers': [paper_id],
                'classification': 'experimental',
                'discovered_at': time.time(),
            }

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
