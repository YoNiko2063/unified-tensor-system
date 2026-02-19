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
    Graph of HDVS patches connected by curvature-weighted transition edges.

    Enables shortest-path navigation: find the minimum holonomy path
    between two algebraically equivalent (abelian) regions.

    Usage:
        graph = PatchGraph()
        graph.add_patch(patch_a)
        graph.add_patch(patch_b)
        graph.add_transition(patch_a, patch_b, curvature_cost=0.3)

        path = graph.shortest_path(patch_a.id, patch_b.id)
        # [0, 1]  — directly connected
    """

    def __init__(self):
        self._patches: Dict[int, Patch] = {}
        self._edges: Dict[Tuple[int, int], float] = {}  # (i,j) → curvature cost
        self._next_id: int = 0

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

    def add_transition(
        self,
        patch1: Patch,
        patch2: Patch,
        curvature_cost: float,
    ) -> None:
        """
        Add a directed transition edge between two patches.

        Edge weight = curvature cost ≈ ∫ρ(x(t))dt along connecting trajectory.
        Lower weight = smoother transition (preferred path).
        """
        key = (patch1.id, patch2.id)
        # Keep minimum cost if edge already exists
        if key in self._edges:
            self._edges[key] = min(self._edges[key], curvature_cost)
        else:
            self._edges[key] = curvature_cost

        # Symmetric for undirected traversal
        key_rev = (patch2.id, patch1.id)
        if key_rev not in self._edges:
            self._edges[key_rev] = curvature_cost

    def get_neighbors(self, patch_id: int) -> List[Tuple[int, float]]:
        """Return [(neighbor_id, cost), ...] for patch_id."""
        neighbors = []
        for (i, j), cost in self._edges.items():
            if i == patch_id:
                neighbors.append((j, cost))
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

    def path_cost(self, path: List[int]) -> float:
        """Total curvature cost along a path."""
        total = 0.0
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            total += self._edges.get(key, float('inf'))
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
