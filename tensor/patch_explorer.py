"""
Patch Exploration Scheduler — guides where to sample next in state space.

Mathematical basis (LOGIC_FLOW.md, Section 0K):
  Exploration priority = uncertainty = curvature_ratio / (koopman_trust + ε)

  High curvature + low trust = system is in a hard-to-model region.
  Next samples are drawn near the highest-uncertainty known patch centroids,
  perturbed by Gaussian noise scaled to region_std.

  This converts random simulation sweeps into directed geometric exploration.

Reference: LOGIC_FLOW.md Sections 0D, 0K
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

from tensor.lca_patch_detector import PatchClassification
from tensor.patch_graph import PatchGraph


class PatchExplorationScheduler:
    """
    Schedules which region of state space to explore next.

    Priority metric: uncertainty = curvature_ratio / (koopman_trust + ε)
      - High curvature → hard-to-model region
      - Low trust → Koopman can't reliably navigate it
      → Together they identify the frontier of geometric understanding.

    Usage:
        scheduler = PatchExplorationScheduler(graph, n_states=2)

        # After each classification:
        scheduler.record_patch(classification)

        # Get next region to sample:
        x_samples = scheduler.next_region(n_samples=20)
        # Pass to LCAPatchDetector.classify_region(x_samples)
    """

    def __init__(
        self,
        graph: PatchGraph,
        n_states: int,
        region_std: float = 0.1,
        max_history: int = 500,
        uncertainty_eps: float = 0.1,
    ):
        """
        Args:
            graph:           PatchGraph to query for topology context
            n_states:        dimension of state space
            region_std:      standard deviation of Gaussian perturbation around centroid
            max_history:     rolling window size for recorded patches
            uncertainty_eps: ε in curvature / (trust + ε) to prevent division by zero
        """
        self._graph = graph
        self._n = n_states
        self._region_std = region_std
        self._eps = uncertainty_eps
        self._history: deque = deque(maxlen=max_history)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_patch(self, classification: PatchClassification) -> None:
        """
        Record a newly classified patch for future exploration scheduling.

        Args:
            classification: result from LCAPatchDetector.classify_region()
        """
        self._history.append(classification)

    # ------------------------------------------------------------------
    # Uncertainty metric
    # ------------------------------------------------------------------

    def uncertainty(self, classification: PatchClassification) -> float:
        """
        Exploration priority metric for a single patch.

        uncertainty = curvature_ratio / (koopman_trust + ε)

        Range: [0, ∞) — higher means the patch is harder to model
        and therefore more valuable to explore further.
        """
        return classification.curvature_ratio / (classification.koopman_trust + self._eps)

    # ------------------------------------------------------------------
    # Next region selection
    # ------------------------------------------------------------------

    def next_region(self, n_samples: int = 20) -> np.ndarray:
        """
        Generate state samples for the next region to explore.

        Strategy:
          1. If no history, return samples near origin
          2. Otherwise: pick the highest-uncertainty recorded patch centroid,
             draw Gaussian samples around it
          3. Tie-breaking: prefer patches classified as 'chaotic' (unexplored geometry)

        Returns:
            (n_samples, n_states) array of sample points for classify_region()
        """
        if not self._history:
            # No history yet — explore near origin
            return np.random.randn(n_samples, self._n) * self._region_std

        # Find most uncertain patch
        top = max(self._history, key=self.uncertainty)
        centroid = top.centroid

        if centroid is None or len(centroid) != self._n:
            return np.random.randn(n_samples, self._n) * self._region_std

        # Draw samples around the centroid
        noise = np.random.randn(n_samples, self._n) * self._region_std
        return centroid[np.newaxis, :] + noise

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def top_uncertain(self, k: int = 3) -> List[PatchClassification]:
        """
        Return the k recorded patches with highest exploration priority.

        Args:
            k: number of patches to return (capped at history size)

        Returns:
            List sorted by uncertainty descending (highest first).
        """
        if not self._history:
            return []
        sorted_patches = sorted(self._history, key=self.uncertainty, reverse=True)
        return sorted_patches[:k]

    def summary(self) -> dict:
        """Return exploration stats: history size, mean uncertainty, top-1 uncertainty."""
        if not self._history:
            return {
                "n_recorded": 0,
                "mean_uncertainty": 0.0,
                "max_uncertainty": 0.0,
            }
        uncertainties = [self.uncertainty(c) for c in self._history]
        return {
            "n_recorded": len(self._history),
            "mean_uncertainty": float(np.mean(uncertainties)),
            "max_uncertainty": float(np.max(uncertainties)),
        }
