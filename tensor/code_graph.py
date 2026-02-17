"""Code graph: represents a codebase as an MNA-equivalent circuit.

Files/modules   = nodes
Import relations = capacitive edges (state storage)
Function calls   = resistive edges (energy flow)
Class inheritance = inductive edges (momentum/memory)
Complexity       = edge weight

Physical interpretation:
  High-complexity modules = high resistance = energy sink
  Many imports = high capacitance = slow to change
  Deep call chains = inductive loops = oscillatory risk
  Circular imports = short circuits = instability
"""
import ast
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

from core.matrix import MNASystem
from core.graph import CircuitGraph, Node, Edge
from core.components import Resistor, Capacitor, Inductor


@dataclass
class ModuleInfo:
    """Parsed information about a Python module."""
    path: str
    name: str
    imports: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    class_bases: List[str] = field(default_factory=list)
    complexity: int = 1
    n_functions: int = 0
    n_classes: int = 0
    lines: int = 0
    defined_functions: List[str] = field(default_factory=list)
    defined_classes: List[str] = field(default_factory=list)


def _cyclomatic_complexity(tree: ast.AST) -> int:
    """Estimate cyclomatic complexity from AST.

    Count: if, elif, for, while, except, with, and, or, assert,
    comprehensions, ternary expressions.
    """
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                             ast.With, ast.Assert)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                               ast.GeneratorExp)):
            complexity += 1
        elif isinstance(node, ast.IfExp):
            complexity += 1
    return complexity


def _extract_imports(tree: ast.AST) -> List[str]:
    """Extract imported module names."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split('.')[0])
    return imports


def _extract_calls(tree: ast.AST) -> List[str]:
    """Extract function/method call names."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return calls


def _extract_class_bases(tree: ast.AST) -> List[str]:
    """Extract class inheritance base names."""
    bases = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
    return bases


def _count_definitions(tree: ast.AST) -> Tuple[int, int]:
    """Count function and class definitions."""
    n_func = sum(1 for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    n_class = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    return n_func, n_class


def _extract_definitions(tree: ast.AST) -> Tuple[List[str], List[str]]:
    """Extract names of functions and classes defined in a module."""
    funcs = [n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    return funcs, classes


class CodeGraph:
    """Represents codebase as MNA-equivalent circuit graph."""

    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self._node_ids: Dict[str, int] = {}
        self._edges: List[Tuple[str, str, str, float]] = []  # (from, to, type, weight)

    @classmethod
    def from_directory(cls, path: str, max_files: int = 500) -> 'CodeGraph':
        """Parse all .py files via AST, build code graph.

        Args:
            path: Root directory to scan.
            max_files: Safety limit on number of files to parse.
        """
        graph = cls()
        py_files = []
        for root, dirs, files in os.walk(path):
            # Skip hidden dirs, __pycache__, .git, etc.
            dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
            for f in files:
                if f.endswith('.py') and not f.startswith('.'):
                    py_files.append(os.path.join(root, f))
                    if len(py_files) >= max_files:
                        break
            if len(py_files) >= max_files:
                break

        # Parse each file
        for fpath in py_files:
            rel = os.path.relpath(fpath, path)
            mod_name = rel.replace(os.sep, '.').removesuffix('.py')
            if mod_name.endswith('.__init__'):
                mod_name = mod_name.removesuffix('.__init__')

            try:
                with open(fpath, 'r', errors='replace') as fh:
                    source = fh.read()
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            lines = source.count('\n') + 1
            imports = _extract_imports(tree)
            calls = _extract_calls(tree)
            bases = _extract_class_bases(tree)
            complexity = _cyclomatic_complexity(tree)
            n_func, n_class = _count_definitions(tree)
            defined_funcs, defined_classes = _extract_definitions(tree)

            graph.modules[mod_name] = ModuleInfo(
                path=fpath, name=mod_name,
                imports=imports, function_calls=calls,
                class_bases=bases, complexity=complexity,
                n_functions=n_func, n_classes=n_class,
                lines=lines,
                defined_functions=defined_funcs,
                defined_classes=defined_classes,
            )

        # Assign node IDs
        for i, name in enumerate(sorted(graph.modules.keys())):
            graph._node_ids[name] = i

        # Build edges
        graph._build_edges()
        return graph

    def _build_edges(self):
        """Build typed edges from parsed module info."""
        self._edges = []
        mod_names = set(self.modules.keys())
        # Short module name map (for import resolution)
        short_to_full: Dict[str, List[str]] = {}
        for name in mod_names:
            short = name.split('.')[-1]
            short_to_full.setdefault(short, []).append(name)

        # Definition→module map: function/class name → modules that define it.
        # Used for call and inheritance resolution. Note: ambiguous names
        # (e.g. to_mna defined in 3 modules) create edges to ALL defining
        # modules — not precise dependency tracking, but produces a connected
        # graph needed for meaningful FIM/eigenvalue analysis.
        def_to_module: Dict[str, List[str]] = {}
        for name, info in self.modules.items():
            for func_name in info.defined_functions:
                def_to_module.setdefault(func_name, []).append(name)
            for class_name in info.defined_classes:
                def_to_module.setdefault(class_name, []).append(name)

        for mod_name, info in self.modules.items():
            # Import edges (capacitive) — deduplicated per module
            for imp in set(info.imports):
                targets = short_to_full.get(imp, [])
                if not targets and imp in mod_names:
                    targets = [imp]
                for target in targets:
                    if target != mod_name:
                        self._edges.append((mod_name, target, 'import', 1.0))

            # Call edges (resistive) — resolved through definition map
            call_targets: Set[str] = set()
            for call in info.function_calls:
                targets = def_to_module.get(call, [])
                for t in targets:
                    if t != mod_name and t not in call_targets:
                        call_targets.add(t)
                        weight = max(1.0, info.complexity / 10.0)
                        self._edges.append((mod_name, t, 'call', weight))

            # Inheritance edges (inductive) — resolved through definition map
            inherit_targets: Set[str] = set()
            for base in info.class_bases:
                targets = def_to_module.get(base, [])
                for t in targets:
                    if t != mod_name and t not in inherit_targets:
                        inherit_targets.add(t)
                        self._edges.append((mod_name, t, 'inheritance', 1.0))

    def to_mna(self) -> MNASystem:
        """Convert code graph to MNA-equivalent matrix.

        Import edges → capacitive (C matrix)
        Call edges → resistive (G matrix)
        Inheritance → inductive (branch variables via extended MNA)
        """
        n = len(self.modules)
        if n == 0:
            return MNASystem(
                C=np.zeros((1, 1)), G=np.zeros((1, 1)),
                n_nodes=1, n_branches=0, n_total=1,
                node_map={0: 0}, branch_map={}, branch_info=[],
            )

        G = np.zeros((n, n))
        C = np.zeros((n, n))

        for src, dst, etype, weight in self._edges:
            i = self._node_ids.get(src)
            j = self._node_ids.get(dst)
            if i is None or j is None or i == j:
                continue

            if etype == 'call':
                # Resistive: conductance = 1/weight
                g = 1.0 / max(weight, 0.01)
                G[i, i] += g
                G[j, j] += g
                G[i, j] -= g
                G[j, i] -= g
            elif etype == 'import':
                # Capacitive
                c = weight
                C[i, i] += c
                C[j, j] += c
                C[i, j] -= c
                C[j, i] -= c
            elif etype == 'inheritance':
                # Model as weak conductance (avoids extended MNA complexity)
                g = 0.1 * weight
                G[i, i] += g
                G[j, j] += g
                G[i, j] -= g
                G[j, i] -= g

        # Add small diagonal loading for numerical stability
        diag_load = 1e-6
        G += diag_load * np.eye(n)
        C += diag_load * np.eye(n)

        # Fix L2 eigenvalue gap collapse: if gap < 0.01, add φ-scaled perturbation
        PHI = 1.6180339887
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(G)))[::-1]
        if len(eigvals) >= 2 and abs(eigvals[0]) > 1e-30:
            gap = (eigvals[0] - eigvals[1]) / eigvals[0]
            if gap < 0.01:
                perturbation = PHI * 1e-4 * np.abs(G.diagonal()).mean()
                G += perturbation * np.eye(n)

        node_map = {i: i for i in range(n)}
        return MNASystem(
            C=C, G=G, n_nodes=n, n_branches=0, n_total=n,
            node_map=node_map, branch_map={}, branch_info=[],
        )

    def circular_imports(self) -> List[List[str]]:
        """Detect circular import cycles via DFS."""
        # Build adjacency from import edges only
        adj: Dict[str, Set[str]] = {m: set() for m in self.modules}
        for src, dst, etype, _ in self._edges:
            if etype == 'import' and src in adj and dst in adj:
                adj[src].add(dst)

        cycles = []
        visited: Set[str] = set()
        path_set: Set[str] = set()
        path: List[str] = []

        def dfs(node: str):
            if node in path_set:
                # Found cycle
                idx = path.index(node)
                cycle = path[idx:] + [node]
                cycles.append(cycle)
                return
            if node in visited:
                return
            visited.add(node)
            path_set.add(node)
            path.append(node)
            for nxt in adj.get(node, []):
                dfs(nxt)
            path.pop()
            path_set.discard(node)

        for mod in self.modules:
            if mod not in visited:
                dfs(mod)

        return cycles

    def complexity_hotspots(self, top_k: int = 5) -> List[str]:
        """Return top-k modules by cyclomatic complexity."""
        sorted_mods = sorted(self.modules.items(),
                             key=lambda x: x[1].complexity, reverse=True)
        return [name for name, _ in sorted_mods[:top_k]]

    @property
    def node_names(self) -> List[str]:
        """Module names in node-ID order."""
        inv = {v: k for k, v in self._node_ids.items()}
        return [inv[i] for i in range(len(inv))]

    @property
    def n_modules(self) -> int:
        return len(self.modules)

    @property
    def n_edges(self) -> int:
        return len(self._edges)
