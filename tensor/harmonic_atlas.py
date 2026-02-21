"""
Harmonic Atlas — automatic spectral chart builder for HDVS.

Mathematical basis (LOGIC_FLOW.md, Section 0I):
  A harmonic atlas A = {(Rᵢ, Φᵢ)} covers HDVS where Φᵢ is a spectral chart.
  Patches are automatically discovered, merged when similar, and organized
  into a navigable graph.

  Patch similarity metric:
    S(i,j) = α · ‖Λᵢ - Λⱼ‖/‖Λᵢ‖ + β · ‖[Aᵢ, Aⱼ]‖ + γ · |rᵢ - rⱼ|
  Low S(i,j) < threshold → same operator submanifold → merge.

Reference: LOGIC_FLOW.md Sections 0H, 0I
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tensor.lca_patch_detector import LCAPatchDetector, PatchClassification
from tensor.patch_graph import Patch, PatchGraph
from tensor.harmonic_closure import HarmonicClosureChecker


@dataclass
class AtlasStats:
    """Statistics about the current atlas state."""
    n_patches: int
    n_edges: int
    n_merges: int
    type_counts: Dict[str, int]
    lca_fraction: float


class HarmonicAtlas:
    """
    Automatically builds and maintains a spectral atlas of HDVS patches.

    Discovers patches from trajectories, merges redundant ones, and exports
    the result as a PatchGraph for shortest-path navigation.

    Usage:
        atlas = HarmonicAtlas(lca_detector)
        atlas.add_classification(classification)
        atlas.merge_similar(tol=0.15)
        graph = atlas.export_graph()
        patch = atlas.get_chart(x_new)
    """

    def __init__(
        self,
        lca_detector: Optional[LCAPatchDetector] = None,
        alpha: float = 0.5,   # weight for spectrum distance
        beta: float = 0.3,    # weight for commutator distance
        gamma: float = 0.2,   # weight for rank distance
        merge_tol: float = 0.15,
    ):
        """
        Args:
            lca_detector: LCAPatchDetector for classifying new states
            alpha, beta, gamma: weights for similarity metric
            merge_tol: similarity threshold for merging patches
        """
        self.detector = lca_detector
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.merge_tol = merge_tol

        self._patches: List[Patch] = []
        self._classifications: List[PatchClassification] = []
        self._next_id: int = 0
        self._n_merges: int = 0
        self._transition_history: List[Tuple[int, int, float]] = []  # (id1, id2, cost)

        # CRITICAL-3: algebraic closure checker — gates both auto_merge and merge_similar()
        self._closure_checker = HarmonicClosureChecker()

    # ------------------------------------------------------------------
    # Spectral proxy (internal)
    # ------------------------------------------------------------------

    @staticmethod
    def _cl_to_k_proxy(cl: PatchClassification) -> np.ndarray:
        """
        Diagonal spectral proxy from eigenvalues: diag(sorted |Re(λ)|, descending).

        Used by HarmonicClosureChecker in place of a full Koopman K matrix when
        only PatchClassification is available (K matrices live on EDMDKoopman,
        not on PatchClassification). The diagonal structure preserves spectral
        identity: two patches with the same eigenvalues produce identical proxies
        (r=0 → "redundant"), while patches with very different eigenvalues produce
        high projection residuals (r >> ε_closure → not redundant).
        """
        r = np.sort(np.abs(np.real(cl.eigenvalues)))[::-1]
        return np.diag(r).astype(float)

    # ------------------------------------------------------------------
    # Building the atlas
    # ------------------------------------------------------------------

    def add_classification(
        self,
        classification: PatchClassification,
        auto_merge: bool = True,
    ) -> Patch:
        """
        Add a new PatchClassification to the atlas.

        If auto_merge is True and a similar patch exists, the new classification
        is merged into the existing patch rather than creating a new one.

        Args:
            classification: from LCAPatchDetector.classify_region()
            auto_merge: whether to auto-merge with similar existing patches

        Returns:
            The Patch (new or existing after merge)
        """
        if auto_merge and self._patches:
            # CRITICAL-3: use HarmonicClosureChecker instead of raw similarity threshold.
            # Find most similar existing patch (by raw metric) as candidate, then
            # confirm with algebraic closure test — only merge if "redundant".
            best_idx, _ = self._find_most_similar(classification)
            K_new = self._cl_to_k_proxy(classification)
            K_existing = self._cl_to_k_proxy(self._classifications[best_idx])
            closure_result = self._closure_checker.check(
                K_new, [K_existing], trust_new=classification.koopman_trust
            )
            if closure_result == "redundant":
                # Merge: update centroid as running mean
                p = self._patches[best_idx]
                # Update centroid (running average)
                p.centroid = (p.centroid + classification.centroid) / 2.0
                p.curvature_ratio = (p.curvature_ratio + classification.curvature_ratio) / 2.0
                p.commutator_norm = (p.commutator_norm + classification.commutator_norm) / 2.0
                self._n_merges += 1
                return p

        # Create new patch
        pid = self._next_id
        self._next_id += 1
        patch = Patch.from_classification(pid, classification)
        self._patches.append(patch)
        self._classifications.append(classification)

        # Record transition from previous patch (for graph construction)
        if len(self._patches) > 1:
            prev_patch = self._patches[-2]
            # Curvature cost ≈ transition curvature ratio
            cost = classification.curvature_ratio
            self._transition_history.append((prev_patch.id, patch.id, cost))

        return patch

    def add_states(
        self,
        states: np.ndarray,
        window: int = 10,
        stride: Optional[int] = None,
    ) -> List[Patch]:
        """
        Classify a trajectory into patches and add them to the atlas.

        Args:
            states: (T, n) trajectory array
            window: samples per classification window
            stride: step between windows (default: window // 2)

        Returns:
            List of patches discovered from this trajectory
        """
        if self.detector is None:
            raise RuntimeError("HarmonicAtlas requires lca_detector to be set")

        stride = stride or (window // 2)
        T = len(states)
        added = []

        for start in range(0, T - window + 1, stride):
            window_samples = states[start:start + window]
            classification = self.detector.classify_region(window_samples)
            patch = self.add_classification(classification, auto_merge=True)
            added.append(patch)

        return added

    # ------------------------------------------------------------------
    # Similarity and merging
    # ------------------------------------------------------------------

    def similarity(
        self,
        cl1: PatchClassification,
        cl2: PatchClassification,
    ) -> float:
        """
        Compute patch similarity: S(i,j) = α·spectrum_dist + β·commutator_dist + γ·rank_dist

        Lower similarity → more similar patches.

        Returns:
            float in [0, ∞), lower means more similar
        """
        # Spectrum distance (normalized)
        lam1 = np.sort(np.abs(cl1.eigenvalues))[::-1]
        lam2 = np.sort(np.abs(cl2.eigenvalues))[::-1]

        # Pad to same length
        max_len = max(len(lam1), len(lam2))
        lam1 = np.pad(lam1, (0, max_len - len(lam1)))
        lam2 = np.pad(lam2, (0, max_len - len(lam2)))

        norm1 = np.linalg.norm(lam1)
        norm2 = np.linalg.norm(lam2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            spec_dist = 0.0 if (norm1 < 1e-10 and norm2 < 1e-10) else 1.0
        else:
            spec_dist = float(np.linalg.norm(lam1 / norm1 - lam2 / norm2))

        # Commutator distance (use stored commutator norms as proxy)
        comm_dist = abs(cl1.commutator_norm - cl2.commutator_norm) / (
            max(cl1.commutator_norm, cl2.commutator_norm) + 1e-10
        )

        # Rank distance (normalized)
        max_rank = max(cl1.operator_rank, cl2.operator_rank, 1)
        rank_dist = abs(cl1.operator_rank - cl2.operator_rank) / max_rank

        return self.alpha * spec_dist + self.beta * comm_dist + self.gamma * rank_dist

    def merge_similar(self, tol: Optional[float] = None) -> int:
        """
        Merge similar patches across the entire atlas.

        Iteratively merges the most similar pair until no pair is below tol.

        Args:
            tol: similarity threshold (default: self.merge_tol)

        Returns:
            Number of merges performed in this call
        """
        # tol parameter kept for backward compatibility but no longer drives the decision.
        # CRITICAL-3: merge decisions are now made exclusively by HarmonicClosureChecker.
        # Only pairs classified as "redundant" are merged.
        del tol  # explicitly unused after CRITICAL-3 patch
        n_merges = 0

        while True:
            redundant_pair = None

            for i in range(len(self._patches)):
                for j in range(i + 1, len(self._patches)):
                    K_i = self._cl_to_k_proxy(self._classifications[i])
                    K_j = self._cl_to_k_proxy(self._classifications[j])
                    result = self._closure_checker.check(
                        K_j, [K_i],
                        trust_new=self._classifications[j].koopman_trust,
                    )
                    if result == "redundant":
                        redundant_pair = (i, j)
                        break
                if redundant_pair is not None:
                    break

            if redundant_pair is None:
                break

            # Merge j into i
            i, j = redundant_pair
            pi = self._patches[i]
            pj = self._patches[j]
            cj = self._classifications[j]

            # Update centroid
            pi.centroid = (pi.centroid + pj.centroid) / 2.0
            pi.curvature_ratio = (pi.curvature_ratio + pj.curvature_ratio) / 2.0
            pi.commutator_norm = (pi.commutator_norm + pj.commutator_norm) / 2.0

            # Reroute transition history: replace pj.id with pi.id
            new_history = []
            for (a, b, cost) in self._transition_history:
                a2 = pi.id if a == pj.id else a
                b2 = pi.id if b == pj.id else b
                if a2 != b2:  # skip self-loops
                    new_history.append((a2, b2, cost))
            self._transition_history = new_history

            # Remove j
            self._patches.pop(j)
            self._classifications.pop(j)

            n_merges += 1
            self._n_merges += 1

        return n_merges

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_chart(self, x: np.ndarray) -> Optional[Patch]:
        """
        Find which atlas patch covers a given state x.

        Nearest-centroid lookup: returns patch with closest centroid to x.

        Args:
            x: state vector (n,)

        Returns:
            Patch or None if atlas is empty
        """
        if not self._patches:
            return None

        dists = [np.linalg.norm(p.centroid - x) for p in self._patches]
        best_idx = int(np.argmin(dists))
        return self._patches[best_idx]

    def all_patches(self) -> List[Patch]:
        """Return all patches in the atlas."""
        return list(self._patches)

    def lca_patches(self) -> List[Patch]:
        """Return only LCA-type patches."""
        return [p for p in self._patches if p.patch_type == 'lca']

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_graph(self) -> PatchGraph:
        """
        Build a PatchGraph from the atlas for shortest-path navigation.

        Returns:
            PatchGraph with all atlas patches as nodes and recorded
            transitions as weighted edges.
        """
        graph = PatchGraph()

        for patch in self._patches:
            graph.add_patch(patch)

        # Add edges from transition history
        patch_map = {p.id: p for p in self._patches}
        for (a_id, b_id, cost) in self._transition_history:
            pa = patch_map.get(a_id)
            pb = patch_map.get(b_id)
            if pa is not None and pb is not None:
                graph.add_transition(pa, pb, curvature_cost=cost)

        return graph

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> AtlasStats:
        """Return summary statistics about the atlas."""
        type_counts: Dict[str, int] = {}
        for p in self._patches:
            type_counts[p.patch_type] = type_counts.get(p.patch_type, 0) + 1

        n = len(self._patches)

        # Count unique edges
        edge_set = set()
        for (a, b, _) in self._transition_history:
            edge_set.add((min(a, b), max(a, b)))

        return AtlasStats(
            n_patches=n,
            n_edges=len(edge_set),
            n_merges=self._n_merges,
            type_counts=type_counts,
            lca_fraction=type_counts.get('lca', 0) / max(n, 1),
        )

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"HarmonicAtlas(patches={s.n_patches}, edges={s.n_edges}, "
            f"merges={s.n_merges}, lca={s.type_counts.get('lca', 0)}, "
            f"nonabelian={s.type_counts.get('nonabelian', 0)}, "
            f"chaotic={s.type_counts.get('chaotic', 0)})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_most_similar(
        self,
        classification: PatchClassification,
    ) -> Tuple[int, float]:
        """Find the index of the most similar existing patch. Returns (idx, similarity)."""
        best_idx = 0
        best_sim = float('inf')
        for i, cl in enumerate(self._classifications):
            sim = self.similarity(cl, classification)
            if sim < best_sim:
                best_sim = sim
                best_idx = i
        return best_idx, best_sim
