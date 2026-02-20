"""Tests for tensor/adaptive_basis.py — AdaptiveBasisController."""

import numpy as np
import pytest

from tensor.koopman_edmd import EDMDKoopman
from tensor.adaptive_basis import AdaptiveBasisController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_pairs(n_points: int = 30, n_states: int = 2) -> list:
    """Simple linear system pairs: x_{k+1} = 0.9 x_k."""
    rng = np.random.default_rng(42)
    xs = rng.standard_normal((n_points, n_states))
    return [(xs[i], 0.9 * xs[i] + 0.01 * rng.standard_normal(n_states))
            for i in range(n_points - 1)]


def _fitted_edmd(degree: int = 2, n_states: int = 2) -> EDMDKoopman:
    """Return a fitted EDMDKoopman at the given degree."""
    edmd = EDMDKoopman(observable_degree=degree)
    edmd.fit(_linear_pairs(n_states=n_states))
    return edmd


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAdaptiveBasisControllerInit:
    def test_defaults(self):
        ctrl = AdaptiveBasisController()
        assert ctrl.min_degree == 1
        assert ctrl.max_degree == 4
        assert ctrl.expand_threshold == 0.3
        assert ctrl.prune_threshold == 0.05

    def test_custom(self):
        ctrl = AdaptiveBasisController(min_degree=2, max_degree=6,
                                       expand_threshold=0.5, prune_threshold=0.1)
        assert ctrl.min_degree == 2
        assert ctrl.max_degree == 6
        assert ctrl.expand_threshold == 0.5
        assert ctrl.prune_threshold == 0.1


# ---------------------------------------------------------------------------
# should_expand
# ---------------------------------------------------------------------------

class TestShouldExpand:
    def setup_method(self):
        self.ctrl = AdaptiveBasisController(expand_threshold=0.3)

    def test_low_trust_expands(self):
        assert self.ctrl.should_expand(0.1) is True
        assert self.ctrl.should_expand(0.0) is True

    def test_high_trust_no_expand(self):
        assert self.ctrl.should_expand(0.5) is False
        assert self.ctrl.should_expand(1.0) is False

    def test_boundary(self):
        # exactly at threshold: should NOT expand (< strict)
        assert self.ctrl.should_expand(0.3) is False


# ---------------------------------------------------------------------------
# should_prune
# ---------------------------------------------------------------------------

class TestShouldPrune:
    def setup_method(self):
        self.ctrl = AdaptiveBasisController(prune_threshold=0.05)

    def test_low_curvature_high_rank_prunes(self):
        # n_states=4, rank=3 → high (3 > 4//2=2)
        assert self.ctrl.should_prune(0.01, operator_rank=3, n_states=4) is True

    def test_high_curvature_no_prune(self):
        assert self.ctrl.should_prune(0.5, operator_rank=3, n_states=4) is False

    def test_low_rank_no_prune(self):
        # rank=1, n_states=4 → 1 ≤ 2 → NOT high rank
        assert self.ctrl.should_prune(0.01, operator_rank=1, n_states=4) is False

    def test_boundary_curvature(self):
        # exactly at threshold: should NOT prune (< strict)
        assert self.ctrl.should_prune(0.05, operator_rank=3, n_states=4) is False


# ---------------------------------------------------------------------------
# expand_basis
# ---------------------------------------------------------------------------

class TestExpandBasis:
    def setup_method(self):
        self.ctrl = AdaptiveBasisController(max_degree=4)
        self.pairs = _linear_pairs()

    def test_expand_increases_degree(self):
        edmd = _fitted_edmd(degree=2)
        new_edmd = self.ctrl.expand_basis(edmd, self.pairs)
        assert new_edmd.degree == 3

    def test_expand_capped_at_max(self):
        edmd = _fitted_edmd(degree=4)
        new_edmd = self.ctrl.expand_basis(edmd, self.pairs)
        assert new_edmd.degree == 4  # stays at max

    def test_expand_produces_fitted_edmd(self):
        edmd = _fitted_edmd(degree=2)
        new_edmd = self.ctrl.expand_basis(edmd, self.pairs)
        assert new_edmd._fitted
        result = new_edmd.eigendecomposition()
        assert len(result.eigenvalues) > 0


# ---------------------------------------------------------------------------
# prune_basis
# ---------------------------------------------------------------------------

class TestPruneBasis:
    def setup_method(self):
        self.ctrl = AdaptiveBasisController(min_degree=1)
        self.pairs = _linear_pairs()

    def test_prune_decreases_degree(self):
        edmd = _fitted_edmd(degree=3)
        new_edmd = self.ctrl.prune_basis(edmd, self.pairs)
        assert new_edmd.degree == 2

    def test_prune_floored_at_min(self):
        edmd = _fitted_edmd(degree=1)
        new_edmd = self.ctrl.prune_basis(edmd, self.pairs)
        assert new_edmd.degree == 1  # stays at min

    def test_prune_produces_fitted_edmd(self):
        edmd = _fitted_edmd(degree=3)
        new_edmd = self.ctrl.prune_basis(edmd, self.pairs)
        assert new_edmd._fitted
        result = new_edmd.eigendecomposition()
        assert len(result.eigenvalues) > 0


# ---------------------------------------------------------------------------
# adapt — main entry
# ---------------------------------------------------------------------------

class TestAdapt:
    def setup_method(self):
        self.ctrl = AdaptiveBasisController(
            min_degree=1, max_degree=4,
            expand_threshold=0.3, prune_threshold=0.05,
        )
        self.pairs = _linear_pairs()

    def test_expand_when_low_trust(self):
        edmd = _fitted_edmd(degree=2)
        new_edmd, action = self.ctrl.adapt(
            edmd, self.pairs,
            trust=0.1,           # low trust → expand
            curvature=0.5,
            rank=2,
            n_states=2,
        )
        assert action == 'expanded'
        assert new_edmd.degree == 3

    def test_prune_when_smooth_and_high_rank(self):
        edmd = _fitted_edmd(degree=3)
        new_edmd, action = self.ctrl.adapt(
            edmd, self.pairs,
            trust=0.9,           # high trust → no expand
            curvature=0.01,      # low curvature → prune candidate
            rank=2,              # n_states=2, rank=2 > 2//2=1 → high rank
            n_states=2,
        )
        assert action == 'pruned'
        assert new_edmd.degree == 2

    def test_unchanged_when_both_fine(self):
        edmd = _fitted_edmd(degree=2)
        new_edmd, action = self.ctrl.adapt(
            edmd, self.pairs,
            trust=0.8,           # high trust → no expand
            curvature=0.5,       # high curvature → no prune
            rank=1,
            n_states=2,
        )
        assert action == 'unchanged'
        assert new_edmd is edmd   # same object returned

    def test_expand_takes_priority_over_prune(self):
        """When trust is low AND curvature is also low: expand wins."""
        edmd = _fitted_edmd(degree=2)
        new_edmd, action = self.ctrl.adapt(
            edmd, self.pairs,
            trust=0.1,           # low trust → expand
            curvature=0.01,      # also low curvature → prune candidate
            rank=2,
            n_states=2,
        )
        assert action == 'expanded'

    def test_at_max_degree_no_expand(self):
        """At degree ceiling: skip expand even when trust is low, may prune or unchanged."""
        ctrl = AdaptiveBasisController(max_degree=2)
        edmd = _fitted_edmd(degree=2)
        new_edmd, action = ctrl.adapt(
            edmd, self.pairs,
            trust=0.1,           # low trust BUT already at max
            curvature=0.5,
            rank=1,
            n_states=2,
        )
        # Cannot expand → either prune (if conditions met) or unchanged
        assert action in ('pruned', 'unchanged')

    def test_at_min_degree_no_prune(self):
        """At degree floor: skip prune even when curvature is low."""
        ctrl = AdaptiveBasisController(min_degree=1)
        edmd = _fitted_edmd(degree=1)
        new_edmd, action = ctrl.adapt(
            edmd, self.pairs,
            trust=0.9,           # high trust → no expand
            curvature=0.01,      # low curvature but at min
            rank=2,
            n_states=2,
        )
        # Cannot prune → unchanged
        assert action == 'unchanged'
