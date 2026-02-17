"""Bridge between UnifiedTensor and dev-agent proposal ranking.

Translates tensor L2 (code structure) state into proposal weights,
closing the self-improvement loop. Proposals addressing high free-energy
nodes get higher weights — dev-agent prioritizes improvements where the
codebase has the most structural tension.
"""
import json
import os
import sys
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional

_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

from core.matrix import MNASystem
from core.coarsening import CoarseGrainingOperator
from core.sparse_solver import (
    compute_free_energy, compute_harmonic_signature, harmonic_distance,
    nearest_consonant, HarmonicSignature, node_entropy,
)

# Re-import from sibling modules
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tensor.core import UnifiedTensor
from tensor.code_graph import CodeGraph


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + np.exp(-x))


class DevAgentBridge:
    """Connects UnifiedTensor to dev-agent's proposal ranking."""

    def __init__(self, tensor: UnifiedTensor, dev_agent_root: str,
                 max_files: int = 200, log_dir: str = 'tensor/logs'):
        self.tensor = tensor
        self.dev_agent_root = dev_agent_root
        self.log_dir = log_dir
        self.max_files = max_files
        self._code_graph: Optional[CodeGraph] = None
        self._mna: Optional[MNASystem] = None
        self._coarsener: Optional[CoarseGrainingOperator] = None
        self._last_free_energies: Optional[np.ndarray] = None
        self._last_tensions: Optional[np.ndarray] = None
        self._hotspots: List[str] = []
        self._cycles: List[List[str]] = []
        self._cycle_modules: set = set()
        self.refresh()

    def refresh(self):
        """Reparse codebase, update CodeGraph and tensor L2."""
        self._code_graph = CodeGraph.from_directory(
            self.dev_agent_root, max_files=self.max_files)
        self._mna = self._code_graph.to_mna()
        self.tensor.update_level(2, self._mna, t=time.time())

        self._hotspots = self._code_graph.complexity_hotspots(top_k=10)
        self._cycles = self._code_graph.circular_imports()
        self._cycle_modules = set()
        for cyc in self._cycles:
            self._cycle_modules.update(cyc)

        # Compute free energies and tensions
        n = self._mna.n_total
        if n >= 3:
            k = max(1, min(n - 1, int(np.sqrt(n))))
            try:
                self._coarsener = CoarseGrainingOperator(self._mna, k=k, tolerance=1.0)
                self._coarsener.coarsen()
                x = np.zeros(n)
                firing = compute_free_energy(self._mna, x, self._coarsener)
                self._last_free_energies = firing.free_energies
                self._last_tensions = firing.harmonic_tensions
            except ValueError:
                self._last_free_energies = np.zeros(n)
                self._last_tensions = np.zeros(n)
        else:
            self._last_free_energies = np.zeros(max(n, 1))
            self._last_tensions = np.zeros(max(n, 1))

    def _module_to_idx(self, module: str) -> Optional[int]:
        """Find node index for a module name (exact or suffix match)."""
        if self._code_graph is None:
            return None
        ids = self._code_graph._node_ids
        if module in ids:
            return ids[module]
        for name, idx in ids.items():
            if name.endswith('.' + module) or name.split('.')[-1] == module:
                return idx
        return None

    def proposal_weights(self, proposals: List[str]) -> Dict[str, float]:
        """Compute tensor-derived weight for each proposal target module.

        weight = sigmoid(F + H) with boosts for hotspots and cycles.
        """
        weights: Dict[str, float] = {}
        cross_res = 0.5
        try:
            if self.tensor._mna.get(0) is not None:
                cross_res = self.tensor.cross_level_resonance(2, 0)
        except Exception:
            pass

        for proposal in proposals:
            idx = self._module_to_idx(proposal)
            if idx is None or self._last_free_energies is None:
                weights[proposal] = 0.5
                continue

            F = float(self._last_free_energies[idx]) if idx < len(self._last_free_energies) else 0.0
            H = float(self._last_tensions[idx]) if idx < len(self._last_tensions) else 0.0

            w = _sigmoid(F + H)

            # Boost for hotspots
            if proposal in self._hotspots or any(
                    proposal == h.split('.')[-1] for h in self._hotspots):
                w = min(1.0, w + 0.15)

            # Boost for circular import involvement
            if proposal in self._cycle_modules or any(
                    proposal == c.split('.')[-1] for c in self._cycle_modules):
                w = min(1.0, w + 0.1)

            # Boost if cross-level resonance is low (code-market misalignment)
            if cross_res < 0.5:
                w = min(1.0, w + 0.05)

            weights[proposal] = round(float(w), 4)

        return weights

    def improvement_priority_map(self) -> List[dict]:
        """Full priority map of all parsed modules, sorted by weight desc."""
        if self._code_graph is None:
            return []

        names = self._code_graph.node_names
        items = []

        cross_res = 0.5
        try:
            if self.tensor._mna.get(0) is not None:
                cross_res = self.tensor.cross_level_resonance(2, 0)
        except Exception:
            pass

        for name in names:
            idx = self._code_graph._node_ids.get(name)
            if idx is None:
                continue

            F = float(self._last_free_energies[idx]) if (
                self._last_free_energies is not None and idx < len(self._last_free_energies)
            ) else 0.0
            H = float(self._last_tensions[idx]) if (
                self._last_tensions is not None and idx < len(self._last_tensions)
            ) else 0.0

            mod_info = self._code_graph.modules.get(name)
            complexity = mod_info.complexity if mod_info else 0

            w = _sigmoid(F + H)
            reasons = []

            if name in self._hotspots:
                w = min(1.0, w + 0.15)
                reasons.append(f"High resistance node (complexity={complexity})")

            if name in self._cycle_modules:
                w = min(1.0, w + 0.1)
                reasons.append("Circular import involvement")

            if cross_res < 0.5:
                w = min(1.0, w + 0.05)
                reasons.append("Low cross-level resonance: market-code misalignment")

            if H > 0.5:
                sig = compute_harmonic_signature(self._mna.G, k=min(self._mna.n_total, 5))
                reasons.append(f"Dissonant eigenvalue neighborhood ({sig.dominant_interval})")
            elif not reasons:
                reasons.append("Stable node, low structural tension")

            items.append({
                'module': name,
                'free_energy': round(F, 4),
                'harmonic_tension': round(H, 4),
                'weight': round(float(w), 4),
                'reason': '; '.join(reasons),
            })

        items.sort(key=lambda x: x['weight'], reverse=True)
        return items

    def on_improvement_applied(self, module: str, outcome: str):
        """Called after dev-agent applies an improvement.

        Updates tensor L2 and logs the event.
        """
        # Capture before state
        idx = self._module_to_idx(module)
        fe_before = float(self._last_free_energies[idx]) if (
            idx is not None and self._last_free_energies is not None
            and idx < len(self._last_free_energies)) else 0.0

        # Refresh to pick up changes
        self.refresh()

        # Capture after state
        idx = self._module_to_idx(module)
        fe_after = float(self._last_free_energies[idx]) if (
            idx is not None and self._last_free_energies is not None
            and idx < len(self._last_free_energies)) else 0.0

        # Log
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, 'improvement_history.jsonl')
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'module': module,
            'free_energy_before': fe_before,
            'free_energy_after': fe_after,
            'outcome': outcome,
            'delta': round(fe_after - fe_before, 6),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        return entry
