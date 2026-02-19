"""
Patch Graph — global topological map of HDVS patches.

Mathematical basis (LOGIC_FLOW.md, Section 0H):
  Nodes: LCA/non-abelian/chaotic patches
  Edges: transitions with curvature cost wᵢⱼ = Σₜ ρ(x(t))·Δt (holonomy integral)
  Shortest path: minimum total holonomy route between algebraically equivalent regions

Reference: LOGIC_FLOW.md Sections 0H, 0I
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Patch:
    """A classified operating region of the dynamical system."""
    id: int
    patch_type: str                     # 'lca' | 'nonabelian' | 'chaotic'
    operator_basis: np.ndarray          # r × n × n basis matrices
    spectrum: np.ndarray                # complex eigenvalues at centroid
    centroid: np.ndarray                # mean state in this region
    operator_rank: int = 1
    commutator_norm: float = 0.0
    curvature_ratio: float = 0.0
    spectral_gap: float = 0.0
    metadata: dict = field(default_factory=dict)

    def feature_vector(self) -> np.ndarray:
        """Numerical feature vector for neural embedding."""
        type_onehot = {
            'lca': [1, 0, 0],
            'nonabelian': [0, 1, 0],
            'chaotic': [0, 0, 1],
        }.get(self.patch_type, [0, 0, 0])

        spectrum_feats = np.concatenate([
            np.real(self.spectrum[:4]) if len(self.spectrum) >= 4
            else np.pad(np.real(self.spectrum), (0, 4 - len(self.spectrum))),
            np.imag(self.spectrum[:4]) if len(self.spectrum) >= 4
            else np.pad(np.imag(self.spectrum), (0, 4 - len(self.spectrum))),
        ])

        return np.array([
            self.operator_rank,
            self.commutator_norm,
            self.curvature_ratio,
            self.spectral_gap,
            *type_onehot,
            *spectrum_feats,
        ], dtype=np.float32)

    @classmethod
    def from_classification(cls, patch_id: int, classification) -> "Patch":
        """Create Patch from a PatchClassification result."""
        return cls(
            id=patch_id,
            patch_type=classification.patch_type,
            operator_basis=classification.basis_matrices,
            spectrum=classification.eigenvalues,
            centroid=classification.centroid,
            operator_rank=classification.operator_rank,
            commutator_norm=classification.commutator_norm,
            curvature_ratio=classification.curvature_ratio,
            spectral_gap=classification.spectral_gap,
        )


class PatchGraph:
    """
    Graph of HDVS patches connected by 3-component transition edges.

    Edge cost (whattodo.md): w(P,Q) = α·∫ρ(t)dt + β·interval_alignment + γ·Koopman_risk

    Enables shortest-path navigation: find the minimum holonomy path
    between two algebraically equivalent (abelian) regions.

    Usage:
        graph = PatchGraph()
        graph.add_patch(patch_a)
        graph.add_patch(patch_b)
        graph.add_transition(patch_a, patch_b, curvature_cost=0.3,
                             interval_cost=0.1, koopman_risk=0.2)

        path = graph.shortest_path(patch_a.id, patch_b.id)
        # [0, 1]  — directly connected
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5):
        """
        Args:
            alpha: weight for curvature cost (holonomy integral)
            beta:  weight for interval alignment cost (spectral dissonance)
            gamma: weight for Koopman risk (1 - koopman_trust)
        """
        self._patches: Dict[int, Patch] = {}
        # (i,j) → (curvature_cost, interval_cost, koopman_risk)
        self._edges: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        self._next_id: int = 0
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_patch(self, patch: Patch) -> None:
        """Add a patch node to the graph."""
        self._patches[patch.id] = patch

    def new_patch_id(self) -> int:
        """Get a fresh unique patch ID."""
        pid = self._next_id
        self._next_id += 1
        return pid

    def get_patch(self, patch_id: int) -> Optional[Patch]:
        return self._patches.get(patch_id)

    def all_patches(self) -> List[Patch]:
        return list(self._patches.values())

    def lca_patches(self) -> List[Patch]:
        return [p for p in self._patches.values() if p.patch_type == 'lca']

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    @staticmethod
    def combined_cost(
        costs: Tuple[float, float, float],
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
    ) -> float:
        """
        Compute combined edge cost from 3-component tuple.

        w = α·curvature + β·interval_alignment + γ·Koopman_risk
        (whattodo.md specification)
        """
        c, ia, kr = costs
        return alpha * c + beta * ia + gamma * kr

    def add_transition(
        self,
        patch1: Patch,
        patch2: Patch,
        curvature_cost: float,
        interval_cost: float = 0.0,
        koopman_risk: float = 0.0,
    ) -> None:
        """
        Add a directed transition edge between two patches.

        Args:
            curvature_cost:  α term — ∫ρ(x(t))dt holonomy integral
            interval_cost:   β term — spectral dissonance τ(ωᵢ,ωⱼ) between patches
            koopman_risk:    γ term — 1 - koopman_trust (risk of spectral hallucination)

        Lower combined cost = smoother, more trustworthy transition (preferred path).
        """
        key = (patch1.id, patch2.id)
        new_edge = (curvature_cost, interval_cost, koopman_risk)
        if key in self._edges:
            # Keep minimum combined cost edge
            old_total = self.combined_cost(self._edges[key], self._alpha, self._beta, self._gamma)
            new_total = self.combined_cost(new_edge, self._alpha, self._beta, self._gamma)
            if new_total < old_total:
                self._edges[key] = new_edge
        else:
            self._edges[key] = new_edge

        # Symmetric for undirected traversal
        key_rev = (patch2.id, patch1.id)
        if key_rev not in self._edges:
            self._edges[key_rev] = new_edge

    def get_neighbors(self, patch_id: int) -> List[Tuple[int, float]]:
        """Return [(neighbor_id, combined_cost), ...] for patch_id."""
        neighbors = []
        for (i, j), costs in self._edges.items():
            if i == patch_id:
                w = self.combined_cost(costs, self._alpha, self._beta, self._gamma)
                neighbors.append((j, w))
        return neighbors

    # ------------------------------------------------------------------
    # Shortest path (Dijkstra)
    # ------------------------------------------------------------------

    def shortest_path(self, start_id: int, end_id: int) -> List[int]:
        """
        Find minimum curvature-cost path from start to end.

        Returns list of patch IDs from start to end (inclusive).
        Returns empty list if no path exists.
        """
        if start_id == end_id:
            return [start_id]
        if start_id not in self._patches or end_id not in self._patches:
            return []

        # Dijkstra's algorithm
        import heapq
        dist = {pid: float('inf') for pid in self._patches}
        prev = {pid: None for pid in self._patches}
        dist[start_id] = 0.0
        heap = [(0.0, start_id)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == end_id:
                break
            for v, cost in self.get_neighbors(u):
                if v not in dist:
                    continue
                new_d = dist[u] + cost
                if new_d < dist[v]:
                    dist[v] = new_d
                    prev[v] = u
                    heapq.heappush(heap, (new_d, v))

        # Reconstruct path
        if prev[end_id] is None and start_id != end_id:
            return []

        path = []
        current = end_id
        while current is not None:
            path.append(current)
            current = prev[current]
        return list(reversed(path))

    def path_cost(
        self,
        path: List[int],
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> float:
        """Total combined cost along a path."""
        a = self._alpha if alpha is None else alpha
        b = self._beta if beta is None else beta
        g = self._gamma if gamma is None else gamma
        total = 0.0
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            costs = self._edges.get(key)
            if costs is None:
                return float('inf')
            total += self.combined_cost(costs, a, b, g)
        return total

    # ------------------------------------------------------------------
    # HDV embedding export
    # ------------------------------------------------------------------

    def export_hdv_embedding(self) -> dict:
        """
        Export patch feature vectors for integration with IntegratedHDVSystem.

        Returns dict mapping patch_id → feature_vector (numpy array).
        """
        return {
            pid: patch.feature_vector()
            for pid, patch in self._patches.items()
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        patches = list(self._patches.values())
        type_counts = {}
        for p in patches:
            type_counts[p.patch_type] = type_counts.get(p.patch_type, 0) + 1
        return {
            'n_patches': len(patches),
            'n_edges': len(self._edges) // 2,  # undirected
            'type_counts': type_counts,
            'lca_fraction': type_counts.get('lca', 0) / max(len(patches), 1),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"PatchGraph(patches={s['n_patches']}, edges={s['n_edges']}, "
                f"lca={s['type_counts'].get('lca', 0)}, "
                f"nonabelian={s['type_counts'].get('nonabelian', 0)}, "
                f"chaotic={s['type_counts'].get('chaotic', 0)})")
