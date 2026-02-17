"""Unified Tensor T in R^(L x N x N x t).

Stacks all system levels so they inform each other through a shared
mathematical structure. Each level's MNA matrix occupies T[l, :, :, t].

Levels:
  0: Market/signal   (trading bot)
  1: Neural/SNN      (ECEMath ML layer)
  2: Code structure   (dev-agent)
  3: Hardware topology (future)
"""
import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any

# Add ecemath to path for imports
_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

from core.matrix import MNASystem
from core.coarsening import CoarseGrainingOperator, CoarseGrainResult
from core.sparse_solver import (
    compute_free_energy, compute_harmonic_signature,
    HarmonicSignature, CorrectedLifter, HarmonicEnsemblePredictor,
    consonance_score_from_ratios,
)

_CONDITION_THRESHOLD = 1e12

LEVEL_NAMES = {
    0: 'market',
    1: 'neural',
    2: 'code',
    3: 'hardware',
}


class UnifiedTensor:
    """T in R^(L x N x N x t).

    Sparse: only stores non-zero slices. Each level can have a different
    actual node count (<= max_nodes), padded with zeros.
    """

    def __init__(self, max_nodes: int = 64, n_levels: int = 4,
                 history_len: int = 100):
        self.max_nodes = max_nodes
        self.n_levels = n_levels
        self.history_len = history_len

        # Tensor storage: T[level][t_idx] = (G, C) matrices
        # Only store occupied slots to save memory
        self._slices: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {
            l: {} for l in range(n_levels)
        }
        # MNA systems per level (latest)
        self._mna: Dict[int, Optional[MNASystem]] = {l: None for l in range(n_levels)}
        # Time index tracking
        self._t_idx: Dict[int, int] = {l: 0 for l in range(n_levels)}
        self._t_map: Dict[int, Dict[int, float]] = {l: {} for l in range(n_levels)}
        # Cached harmonic signatures
        self._signatures: Dict[int, Dict[int, HarmonicSignature]] = {
            l: {} for l in range(n_levels)
        }
        # Cached coarsening results: (level_fine, level_coarse) -> result
        self._coarsen_cache: Dict[Tuple[int, int], CoarseGrainResult] = {}
        self._coarsen_version: Dict[int, int] = {l: 0 for l in range(n_levels)}
        self._coarsen_source_version: Dict[Tuple[int, int], int] = {}
        # Corrected lifters
        self._lifters: Dict[Tuple[int, int], CorrectedLifter] = {}
        # State vectors per level (latest)
        self._state: Dict[int, Optional[np.ndarray]] = {l: None for l in range(n_levels)}

    def _validate_level(self, level: int):
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"Level {level} out of range [0, {self.n_levels})")

    def _pad_matrix(self, M: np.ndarray) -> np.ndarray:
        """Pad matrix to max_nodes x max_nodes with zeros."""
        n = M.shape[0]
        if n == self.max_nodes:
            return M.copy()
        padded = np.zeros((self.max_nodes, self.max_nodes))
        padded[:n, :n] = M
        return padded

    def _get_active_size(self, level: int) -> int:
        """Actual node count at level (before padding)."""
        mna = self._mna.get(level)
        return mna.n_total if mna is not None else 0

    def update_level(self, level: int, mna: MNASystem, t: float):
        """Insert MNA matrices into T[level, :n, :n, t_idx]."""
        self._validate_level(level)

        self._mna[level] = mna
        t_idx = self._t_idx[level]

        G_pad = self._pad_matrix(mna.G)
        C_pad = self._pad_matrix(mna.C)
        self._slices[level][t_idx] = (G_pad, C_pad)
        self._t_map[level][t_idx] = t

        # Compute and cache harmonic signature
        sig = compute_harmonic_signature(mna.G, k=min(mna.n_total, 10))
        self._signatures[level][t_idx] = sig

        # Advance time index (circular buffer)
        self._t_idx[level] = (t_idx + 1) % self.history_len

        # Bump version for cache invalidation
        self._coarsen_version[level] = self._coarsen_version.get(level, 0) + 1

        # Default state vector
        if self._state[level] is None or len(self._state[level]) != mna.n_total:
            self._state[level] = np.zeros(mna.n_total)

    def _resolve_t_idx(self, level: int, t: float) -> int:
        """Resolve t to nearest time index. t=-1 means latest."""
        if t < 0:
            # Latest: one before current write pointer
            idx = (self._t_idx[level] - 1) % self.history_len
            if idx in self._slices[level]:
                return idx
            # Find any populated index
            if self._slices[level]:
                return max(self._slices[level].keys())
            return 0
        # Find nearest timestamp
        best_idx = 0
        best_dist = float('inf')
        for idx, ts in self._t_map[level].items():
            d = abs(ts - t)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def _get_G_at(self, level: int, t: float = -1) -> Optional[np.ndarray]:
        """Get G matrix at level/time."""
        t_idx = self._resolve_t_idx(level, t)
        pair = self._slices[level].get(t_idx)
        if pair is None:
            return None
        return pair[0]

    def coarsen_to(self, level_fine: int, level_coarse: int,
                   k: Optional[int] = None) -> CoarseGrainResult:
        """Apply phi between two levels."""
        self._validate_level(level_fine)
        self._validate_level(level_coarse)

        mna_fine = self._mna.get(level_fine)
        if mna_fine is None:
            raise ValueError(f"Level {level_fine} has no MNA system")

        # Check cache
        key = (level_fine, level_coarse)
        src_ver = self._coarsen_version[level_fine]
        if key in self._coarsen_cache and self._coarsen_source_version.get(key) == src_ver:
            return self._coarsen_cache[key]

        if k is None:
            k = max(1, int(np.sqrt(mna_fine.n_total)))

        k = min(k, mna_fine.n_total - 1)
        phi = CoarseGrainingOperator(mna_fine, k=k, tolerance=0.3)
        result = phi.coarsen()

        # Update coarse level with the reduced system
        t_latest = self._t_map[level_fine].get(
            self._resolve_t_idx(level_fine, -1), 0.0)
        self.update_level(level_coarse, result.mna_coarse, t_latest)

        # Cache
        self._coarsen_cache[key] = result
        self._coarsen_source_version[key] = src_ver

        # Build corrected lifter
        self._lifters[key] = CorrectedLifter(phi, alpha=0.02)

        return result

    def lift_from(self, level_coarse: int, level_fine: int,
                  x_coarse: np.ndarray) -> np.ndarray:
        """Apply phi^-1 using CorrectedLifter."""
        key = (level_fine, level_coarse)
        lifter = self._lifters.get(key)
        if lifter is None:
            # Fall back to basic projection
            result = self._coarsen_cache.get(key)
            if result is None:
                raise ValueError(
                    f"No coarsening from L{level_fine}→L{level_coarse}. "
                    f"Call coarsen_to() first.")
            return result.projection @ x_coarse
        return lifter.lift(x_coarse)

    def update_lifter(self, level_fine: int, level_coarse: int,
                      x_coarse: np.ndarray, x_fine_actual: np.ndarray):
        """Update the lifter's correction matrix with observed data."""
        key = (level_fine, level_coarse)
        lifter = self._lifters.get(key)
        if lifter is not None:
            lifter.update(x_coarse, x_fine_actual)

    def eigenvalue_gap(self, level: int, t: float = -1) -> float:
        """Normalized eigenvalue gap: (lam1 - lam2) / lam1.

        Narrowing gap = phase transition imminent.
        """
        self._validate_level(level)
        G = self._get_G_at(level, t)
        if G is None:
            return 0.0

        n = self._get_active_size(level)
        if n < 2:
            return 1.0

        G_active = G[:n, :n]
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(G_active)))[::-1]
        if len(eigvals) < 2 or abs(eigvals[0]) < 1e-30:
            return 0.0
        return float((eigvals[0] - eigvals[1]) / eigvals[0])

    def free_energy_map(self, level: int, tau: float = 1.0,
                        t: float = -1) -> np.ndarray:
        """Compute F(node_i) for all nodes at given level/time."""
        self._validate_level(level)
        mna = self._mna.get(level)
        if mna is None:
            return np.zeros(self.max_nodes)

        x = self._state.get(level)
        if x is None:
            x = np.zeros(mna.n_total)

        if mna.n_total < 3:
            # Too small to coarsen — return zero free energies
            return np.zeros(self.max_nodes)

        # Need a coarsener — try with relaxed tolerance, gracefully degrade
        k = max(1, min(mna.n_total - 1, int(np.sqrt(mna.n_total))))
        try:
            phi = CoarseGrainingOperator(mna, k=k, tolerance=1.0)
            phi.coarsen()
        except ValueError:
            # Coarsening failed even with relaxed tolerance — skip
            return np.zeros(self.max_nodes)

        firing = compute_free_energy(mna, x, phi, tau=tau)
        result = np.zeros(self.max_nodes)
        result[:mna.n_total] = firing.free_energies
        return result

    def harmonic_signature(self, level: int, t: float = -1) -> HarmonicSignature:
        """Return cached HarmonicSignature for level at time t."""
        self._validate_level(level)
        t_idx = self._resolve_t_idx(level, t)
        sig = self._signatures[level].get(t_idx)
        if sig is not None:
            return sig
        # Recompute
        mna = self._mna.get(level)
        if mna is None:
            return HarmonicSignature(
                dominant_interval='unison', consonance_score=1.0,
                eigenvalue_ratios=np.array([1.0]),
                nearest_ratios=np.array([1.0]),
                tension_vector=np.zeros(1),
                predicted_resolution='unison', stability_verdict='stable',
            )
        sig = compute_harmonic_signature(mna.G, k=min(mna.n_total, 10))
        self._signatures[level][t_idx] = sig
        return sig

    def cross_level_resonance(self, level_a: int, level_b: int) -> float:
        """Measure harmonic alignment between two levels.

        resonance = 1 - |consonance_a - consonance_b| + interval_overlap
        Normalized to [0, 1].
        """
        self._validate_level(level_a)
        self._validate_level(level_b)

        sig_a = self.harmonic_signature(level_a)
        sig_b = self.harmonic_signature(level_b)

        # Consonance proximity
        cons_diff = abs(sig_a.consonance_score - sig_b.consonance_score)
        cons_proximity = 1.0 - cons_diff

        # Interval overlap: compare dominant intervals
        interval_match = 1.0 if sig_a.dominant_interval == sig_b.dominant_interval else 0.0

        # Eigenvalue ratio overlap (cosine similarity of ratio vectors)
        ra = sig_a.eigenvalue_ratios
        rb = sig_b.eigenvalue_ratios
        n_common = min(len(ra), len(rb))
        if n_common > 0:
            a_v = ra[:n_common]
            b_v = rb[:n_common]
            na = np.linalg.norm(a_v)
            nb = np.linalg.norm(b_v)
            if na > 1e-30 and nb > 1e-30:
                ratio_sim = float(np.dot(a_v, b_v) / (na * nb))
            else:
                ratio_sim = 0.0
        else:
            ratio_sim = 0.0

        # Weighted combination
        resonance = 0.4 * cons_proximity + 0.2 * interval_match + 0.4 * ratio_sim
        return float(np.clip(resonance, 0.0, 1.0))

    def phase_transition_risk(self, level: int) -> float:
        """Composite risk score 0-1.

        risk = w1*(1-gap) + w2*(1-consonance) + w3*regime_proximity
        """
        self._validate_level(level)
        mna = self._mna.get(level)
        if mna is None:
            return 0.0

        # Eigenvalue gap component
        gap = self.eigenvalue_gap(level)
        gap_risk = 1.0 - gap

        # Consonance component
        sig = self.harmonic_signature(level)
        cons_risk = 1.0 - sig.consonance_score

        # Regime proximity: how close eigenvalues are to degeneracy
        n = mna.n_total
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(mna.G)))[::-1]
        if len(eigvals) > 1 and eigvals[0] > 1e-30:
            # Min gap between adjacent eigenvalues, normalized
            gaps = np.diff(eigvals)
            min_adj_gap = np.min(np.abs(gaps)) / eigvals[0]
            regime_proximity = 1.0 / (1.0 + 10.0 * min_adj_gap)
        else:
            regime_proximity = 0.0

        # Weighted combination
        risk = 0.4 * gap_risk + 0.35 * cons_risk + 0.25 * regime_proximity
        return float(np.clip(risk, 0.0, 1.0))

    def tensor_snapshot(self, t: float = -1) -> dict:
        """Human-readable state of entire tensor at time t."""
        snapshot = {
            'max_nodes': self.max_nodes,
            'n_levels': self.n_levels,
            'levels': {},
        }
        for l in range(self.n_levels):
            mna = self._mna.get(l)
            if mna is None:
                snapshot['levels'][l] = {
                    'name': LEVEL_NAMES.get(l, f'L{l}'),
                    'populated': False,
                }
                continue

            sig = self.harmonic_signature(l, t)
            gap = self.eigenvalue_gap(l, t)
            risk = self.phase_transition_risk(l)

            # Active nodes (F < 0 as simple threshold)
            fe = self.free_energy_map(l, t=t)
            n_active = mna.n_total
            active_nodes = int(np.sum(fe[:n_active] < 0))

            level_info = {
                'name': LEVEL_NAMES.get(l, f'L{l}'),
                'populated': True,
                'n_nodes': mna.n_total,
                'harmonic_signature': {
                    'dominant_interval': sig.dominant_interval,
                    'consonance_score': round(sig.consonance_score, 4),
                    'stability_verdict': sig.stability_verdict,
                    'predicted_resolution': sig.predicted_resolution,
                },
                'eigenvalue_gap': round(gap, 4),
                'phase_transition_risk': round(risk, 4),
                'active_nodes': active_nodes,
            }

            # Cross-level resonance with adjacent levels
            resonances = {}
            for other in range(self.n_levels):
                if other != l and self._mna.get(other) is not None:
                    resonances[LEVEL_NAMES.get(other, f'L{other}')] = round(
                        self.cross_level_resonance(l, other), 4)
            level_info['cross_level_resonance'] = resonances

            snapshot['levels'][l] = level_info

        return snapshot

    def set_state(self, level: int, x: np.ndarray):
        """Set state vector at level."""
        self._validate_level(level)
        self._state[level] = x.copy()
