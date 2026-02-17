"""Bootstrap orchestrator: dev-agent self-improvement guided by tensor geometry.

Runs dev-agent's file-splitting capability directed by tensor L2 free energy.
High free-energy nodes = high structural tension = priority refactor targets.
After each split, re-parses the codebase and measures consonance improvement.
"""
import ast
import json
import os
import sys
import time
import shutil
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.matrix import MNASystem
from core.sparse_solver import compute_harmonic_signature, consonance_score_from_ratios
from tensor.core import UnifiedTensor
from tensor.code_graph import CodeGraph


@dataclass
class BootstrapResult:
    """Result of one bootstrap cycle."""
    improved: bool
    consonance_before: float
    consonance_after: float
    consonance_delta: float
    files_changed: List[str]
    high_tension_nodes: List[str]
    step: int


class BootstrapOrchestrator:
    """Runs dev-agent's self-improvement loop guided by tensor geometric intelligence.

    The tensor tells dev-agent WHERE structural tension is highest (high free
    energy nodes at L2). Dev-agent splits/refactors those specific files.
    The tensor re-reads the improved structure. Repeat until L2 consonance
    score > target (stable octave).
    """

    def __init__(self, tensor: UnifiedTensor, dev_agent_root: str,
                 target_consonance: float = 0.75, max_lines: int = 200,
                 log_dir: str = 'tensor/logs'):
        self.tensor = tensor
        self.dev_agent_root = os.path.abspath(dev_agent_root)
        self.target_consonance = target_consonance
        self.max_lines = max_lines
        self.log_dir = log_dir
        self._step = 0
        self._no_improve_count = 0
        self._code_graph: Optional[CodeGraph] = None
        self._history: List[BootstrapResult] = []

    def _parse_and_update_l2(self) -> Tuple[CodeGraph, float]:
        """Parse codebase, update tensor L2, return (graph, consonance)."""
        cg = CodeGraph.from_directory(self.dev_agent_root, max_files=500)
        mna = cg.to_mna()
        self.tensor.update_level(2, mna, t=time.time())
        self._code_graph = cg

        # Compute consonance from L2 harmonic signature
        sig = self.tensor.harmonic_signature(2)
        return cg, sig.consonance_score

    def _get_high_tension_nodes(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k nodes with highest free energy at L2."""
        fe = self.tensor.free_energy_map(2)
        if self._code_graph is None:
            return []

        names = self._code_graph.node_names
        n = min(len(names), len(fe))
        node_fe = [(names[i], float(fe[i])) for i in range(n)]
        # Sort by absolute free energy descending (highest tension first)
        node_fe.sort(key=lambda x: abs(x[1]), reverse=True)
        return node_fe[:top_k]

    def _generate_split_plan(self, file_path: str, mod_info) -> List[dict]:
        """Generate a plan to split a file that's too large or too complex.

        Returns list of {new_file, functions, description} dicts.
        """
        if mod_info.lines <= self.max_lines and mod_info.complexity < 20:
            return []

        try:
            with open(file_path, 'r', errors='replace') as f:
                source = f.read()
            tree = ast.parse(source)
        except (SyntaxError, FileNotFoundError):
            return []

        # Group top-level definitions by type
        classes = []
        functions = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

        if len(classes) + len(functions) < 2:
            return []

        plans = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        parent_dir = os.path.dirname(file_path)

        # Strategy: each class gets its own file, helper functions grouped
        for cls_name in classes:
            plans.append({
                'new_file': os.path.join(parent_dir, f'{base_name}_{cls_name.lower()}.py'),
                'items': [cls_name],
                'description': f'Extract class {cls_name}',
            })

        if functions:
            plans.append({
                'new_file': os.path.join(parent_dir, f'{base_name}_helpers.py'),
                'items': functions,
                'description': f'Extract {len(functions)} helper functions',
            })

        return plans

    def _apply_split(self, file_path: str, plan: List[dict]) -> List[str]:
        """Apply a split plan. Returns list of changed file paths.

        For safety, this does a simulated split — creates marker files
        that record what WOULD be split, without destroying the original.
        Real dev-agent would do the actual AST surgery.
        """
        if not plan:
            return []

        changed = []
        os.makedirs(self.log_dir, exist_ok=True)

        for entry in plan:
            # Record the planned split
            log_entry = {
                'source': file_path,
                'target': entry['new_file'],
                'items': entry['items'],
                'description': entry['description'],
                'step': self._step,
                'timestamp': time.time(),
            }
            log_path = os.path.join(self.log_dir, 'bootstrap_splits.jsonl')
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            changed.append(entry['new_file'])

        return changed

    def run_bootstrap_step(self) -> BootstrapResult:
        """One cycle of bootstrap:
        1. Get top-5 high-tension nodes from tensor L2
        2. For each: generate refactor plan
        3. Re-parse with CodeGraph
        4. Update tensor L2
        5. Measure: did consonance improve?
        """
        self._step += 1

        # Parse current state
        cg_before, cons_before = self._parse_and_update_l2()

        # Find high-tension nodes
        tension_nodes = self._get_high_tension_nodes(top_k=5)
        high_tension_names = [name for name, _ in tension_nodes]

        files_changed = []
        for name, fe in tension_nodes:
            mod_info = cg_before.modules.get(name)
            if mod_info is None:
                continue

            plan = self._generate_split_plan(mod_info.path, mod_info)
            if plan:
                changed = self._apply_split(mod_info.path, plan)
                files_changed.extend(changed)

        # Re-parse after changes
        _, cons_after = self._parse_and_update_l2()

        delta = cons_after - cons_before
        improved = bool(delta > 0)

        if not improved:
            self._no_improve_count += 1
        else:
            self._no_improve_count = 0

        result = BootstrapResult(
            improved=improved,
            consonance_before=cons_before,
            consonance_after=cons_after,
            consonance_delta=delta,
            files_changed=files_changed,
            high_tension_nodes=high_tension_names,
            step=self._step,
        )
        self._history.append(result)
        return result

    def run_until_stable(self, max_steps: int = 50):
        """Loop run_bootstrap_step() until stable."""
        for step in range(max_steps):
            result = self.run_bootstrap_step()

            print(f"bootstrap step={result.step} "
                  f"consonance={result.consonance_before:.4f}→{result.consonance_after:.4f} "
                  f"changed={result.files_changed}")

            # Check termination conditions
            if result.consonance_after >= self.target_consonance:
                print(f"Target consonance {self.target_consonance} reached.")
                break

            if self._no_improve_count >= 5:
                print(f"No improvement for 5 consecutive steps. Stopping.")
                break

        return self._history

    def functionality_database(self) -> dict:
        """Emit the functionality database after bootstrap.

        Each entry maps a file to its behavioral signature.
        """
        if self._code_graph is None:
            self._parse_and_update_l2()

        cg = self._code_graph
        mna = cg.to_mna()
        fe = self.tensor.free_energy_map(2)

        # Compute eigenvalues for coupling strength
        eigvals = np.linalg.eigvalsh(mna.G)
        sorted_eig = np.sort(np.abs(eigvals))[::-1]

        db = {}
        names = cg.node_names
        for i, name in enumerate(names):
            mod_info = cg.modules.get(name)
            if mod_info is None:
                continue

            # Outgoing and incoming edges
            calls_out = []
            called_by = []
            for src, dst, etype, _ in cg._edges:
                if src == name and etype == 'call':
                    calls_out.append(dst)
                if dst == name and etype == 'call':
                    called_by.append(src)

            # Infer responsibility from AST
            responsibility = self._infer_responsibility(mod_info)

            entry = {
                'file': mod_info.path,
                'responsibility': responsibility,
                'tensor_node': i,
                'free_energy': float(fe[i]) if i < len(fe) else 0.0,
                'eigenvalue': float(sorted_eig[i]) if i < len(sorted_eig) else 0.0,
                'calls': calls_out,
                'called_by': called_by,
                'language_level': 'python',
            }
            db[name] = entry

        # Save to disk
        os.makedirs(self.log_dir, exist_ok=True)
        db_path = os.path.join(self.log_dir, 'functionality_db.json')
        with open(db_path, 'w') as f:
            json.dump(db, f, indent=2, default=str)

        return db

    def _infer_responsibility(self, mod_info) -> str:
        """Infer a module's responsibility from its AST metadata."""
        parts = []
        if mod_info.n_classes > 0:
            parts.append(f"{mod_info.n_classes} class(es)")
        if mod_info.n_functions > 0:
            parts.append(f"{mod_info.n_functions} function(s)")
        parts.append(f"complexity={mod_info.complexity}")
        parts.append(f"{mod_info.lines} lines")

        name = mod_info.name.split('.')[-1]
        return f"{name}: {', '.join(parts)}"
