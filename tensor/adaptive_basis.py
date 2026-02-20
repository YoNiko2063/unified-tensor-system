"""
Adaptive Basis Controller — expand or prune Koopman observable basis in response to trust signals.

Mathematical basis (LOGIC_FLOW.md, Section 0G):
  When koopman_trust < τ_low: expand ψ degree (more observables, higher-order dynamics)
  When curvature low AND rank high: prune (redundant modes, overfitting risk)

This is the lift-and-reduce loop that converts the system from a fixed-structure
detector into an adaptive explorer.

Reference: LOGIC_FLOW.md Sections 0F, 0G
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from tensor.koopman_edmd import EDMDKoopman


class AdaptiveBasisController:
    """
    Controls the degree of the Koopman polynomial observable basis.

    Expansion policy:  if koopman_trust < expand_threshold → increase degree by 1
    Pruning policy:    if curvature_ratio < prune_threshold AND operator_rank is high
                       relative to state dimension → decrease degree by 1

    Usage:
        ctrl = AdaptiveBasisController()
        edmd = EDMDKoopman(observable_degree=2)
        edmd.fit(pairs)
        result = edmd.eigendecomposition()

        new_edmd, action = ctrl.adapt(
            edmd, pairs,
            trust=result.koopman_trust,
            curvature=classification.curvature_ratio,
            rank=classification.operator_rank,
            n_states=2,
        )
        # action ∈ {'expanded', 'pruned', 'unchanged'}
    """

    def __init__(
        self,
        min_degree: int = 1,
        max_degree: int = 4,
        expand_threshold: float = 0.3,
        prune_threshold: float = 0.05,
    ):
        """
        Args:
            min_degree:        minimum observable polynomial degree (floor)
            max_degree:        maximum observable polynomial degree (ceiling)
            expand_threshold:  koopman_trust below this → expand basis
            prune_threshold:   curvature_ratio below this → candidate for pruning
        """
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.expand_threshold = expand_threshold
        self.prune_threshold = prune_threshold

    # ------------------------------------------------------------------
    # Decision functions
    # ------------------------------------------------------------------

    def should_expand(self, koopman_trust: float) -> bool:
        """True when Koopman model has low trust → need more expressive basis."""
        return koopman_trust < self.expand_threshold

    def should_prune(
        self,
        curvature_ratio: float,
        operator_rank: int,
        n_states: int,
    ) -> bool:
        """
        True when dynamics are smooth (low curvature) but rank is high.

        High rank + smooth dynamics = the basis has redundant modes fitting noise.
        Reducing degree removes those redundant higher-order terms.
        """
        high_rank = operator_rank > max(n_states // 2, 1)
        return curvature_ratio < self.prune_threshold and high_rank

    # ------------------------------------------------------------------
    # Basis adjustment
    # ------------------------------------------------------------------

    def expand_basis(
        self,
        edmd: EDMDKoopman,
        pairs: list,
    ) -> EDMDKoopman:
        """
        Increase observable degree by 1 (capped at max_degree) and refit.

        Returns a new EDMDKoopman instance fitted on the same pairs.
        """
        new_degree = min(edmd.degree + 1, self.max_degree)
        new_edmd = EDMDKoopman(
            observable_degree=new_degree,
            spectral_gap_threshold=edmd.gap_threshold,
        )
        new_edmd.fit(pairs)
        return new_edmd

    def prune_basis(
        self,
        edmd: EDMDKoopman,
        pairs: list,
    ) -> EDMDKoopman:
        """
        Decrease observable degree by 1 (floored at min_degree) and refit.

        Returns a new EDMDKoopman instance fitted on the same pairs.
        """
        new_degree = max(edmd.degree - 1, self.min_degree)
        new_edmd = EDMDKoopman(
            observable_degree=new_degree,
            spectral_gap_threshold=edmd.gap_threshold,
        )
        new_edmd.fit(pairs)
        return new_edmd

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def adapt(
        self,
        edmd: EDMDKoopman,
        pairs: list,
        trust: float,
        curvature: float,
        rank: int,
        n_states: int,
    ) -> Tuple[EDMDKoopman, str]:
        """
        Decide whether to expand, prune, or keep the current basis.

        Expansion takes priority over pruning: low trust is more urgent
        than high-rank redundancy.

        Args:
            edmd:      current fitted EDMDKoopman
            pairs:     (x_k, x_{k+1}) trajectory pairs used to fit edmd
            trust:     koopman_trust score from eigendecomposition()
            curvature: mean curvature_ratio from LCAPatchDetector
            rank:      operator_rank from LCAPatchDetector
            n_states:  state space dimension

        Returns:
            (new_edmd, action) where action ∈ {'expanded', 'pruned', 'unchanged'}
        """
        if self.should_expand(trust) and edmd.degree < self.max_degree:
            return self.expand_basis(edmd, pairs), 'expanded'

        if self.should_prune(curvature, rank, n_states) and edmd.degree > self.min_degree:
            return self.prune_basis(edmd, pairs), 'pruned'

        return edmd, 'unchanged'
