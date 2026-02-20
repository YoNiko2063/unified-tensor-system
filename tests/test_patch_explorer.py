"""Tests for tensor/patch_explorer.py — PatchExplorationScheduler."""

import numpy as np
import pytest

from tensor.lca_patch_detector import PatchClassification
from tensor.patch_graph import Patch, PatchGraph
from tensor.patch_explorer import PatchExplorationScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph() -> PatchGraph:
    return PatchGraph()


def _classification(
    curvature: float,
    trust: float,
    centroid: np.ndarray,
    patch_type: str = 'lca',
    rank: int = 1,
) -> PatchClassification:
    n = len(centroid)
    basis = np.eye(n)[np.newaxis, :, :]
    return PatchClassification(
        patch_type=patch_type,
        operator_rank=rank,
        commutator_norm=0.0,
        curvature_ratio=curvature,
        spectral_gap=0.1,
        basis_matrices=basis,
        eigenvalues=np.array([-0.5 + 0.0j, -0.5 + 0.0j]),
        centroid=centroid,
        koopman_trust=trust,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        g = _make_graph()
        sched = PatchExplorationScheduler(g, n_states=2)
        assert sched._n == 2
        assert sched._region_std == 0.1
        assert sched._eps == 0.1

    def test_custom(self):
        g = _make_graph()
        sched = PatchExplorationScheduler(g, n_states=4, region_std=0.5, max_history=100)
        assert sched._n == 4
        assert sched._region_std == 0.5


# ---------------------------------------------------------------------------
# uncertainty()
# ---------------------------------------------------------------------------

class TestUncertainty:
    def setup_method(self):
        g = _make_graph()
        self.sched = PatchExplorationScheduler(g, n_states=2, uncertainty_eps=0.1)

    def test_high_curvature_high_uncertainty(self):
        c = _classification(curvature=1.0, trust=0.0, centroid=np.zeros(2))
        u = self.sched.uncertainty(c)
        # 1.0 / (0.0 + 0.1) = 10.0
        assert u == pytest.approx(10.0)

    def test_high_trust_lowers_uncertainty(self):
        c = _classification(curvature=1.0, trust=1.0, centroid=np.zeros(2))
        u = self.sched.uncertainty(c)
        # 1.0 / (1.0 + 0.1) ≈ 0.909
        assert u == pytest.approx(1.0 / 1.1, rel=1e-5)

    def test_zero_curvature_zero_uncertainty(self):
        c = _classification(curvature=0.0, trust=0.5, centroid=np.zeros(2))
        u = self.sched.uncertainty(c)
        assert u == pytest.approx(0.0)

    def test_nonnegative(self):
        c = _classification(curvature=0.3, trust=0.7, centroid=np.zeros(2))
        assert self.sched.uncertainty(c) >= 0.0


# ---------------------------------------------------------------------------
# record_patch / next_region
# ---------------------------------------------------------------------------

class TestRecordAndNextRegion:
    def setup_method(self):
        g = _make_graph()
        self.sched = PatchExplorationScheduler(g, n_states=2, region_std=0.05)

    def test_no_history_returns_near_origin(self):
        np.random.seed(0)
        samples = self.sched.next_region(n_samples=20)
        assert samples.shape == (20, 2)
        # Without history: samples near origin (std=0.05)
        assert np.abs(samples).mean() < 1.0  # rough sanity

    def test_returns_samples_near_most_uncertain_centroid(self):
        centroid_low = np.array([10.0, 0.0])
        centroid_high = np.array([0.0, 10.0])  # will be more uncertain

        c_low  = _classification(curvature=0.1, trust=0.9, centroid=centroid_low)
        c_high = _classification(curvature=2.0, trust=0.0, centroid=centroid_high)

        self.sched.record_patch(c_low)
        self.sched.record_patch(c_high)

        np.random.seed(42)
        samples = self.sched.next_region(n_samples=100)
        assert samples.shape == (100, 2)

        # Mean of samples should be near centroid_high = [0, 10]
        mean = samples.mean(axis=0)
        assert mean[1] > 5.0, f"Expected samples near [0,10], got mean {mean}"

    def test_shape_correct(self):
        c = _classification(curvature=0.5, trust=0.5, centroid=np.array([1.0, 2.0]))
        self.sched.record_patch(c)
        samples = self.sched.next_region(n_samples=15)
        assert samples.shape == (15, 2)


# ---------------------------------------------------------------------------
# top_uncertain
# ---------------------------------------------------------------------------

class TestTopUncertain:
    def setup_method(self):
        g = _make_graph()
        self.sched = PatchExplorationScheduler(g, n_states=2)

    def test_empty_history_returns_empty(self):
        result = self.sched.top_uncertain(k=3)
        assert result == []

    def test_returns_k_patches(self):
        for i in range(5):
            c = _classification(curvature=float(i) * 0.1, trust=0.5,
                                centroid=np.array([float(i), 0.0]))
            self.sched.record_patch(c)

        top = self.sched.top_uncertain(k=3)
        assert len(top) == 3

    def test_sorted_by_uncertainty_descending(self):
        c_low  = _classification(curvature=0.1, trust=0.9, centroid=np.zeros(2))
        c_mid  = _classification(curvature=0.5, trust=0.5, centroid=np.ones(2))
        c_high = _classification(curvature=2.0, trust=0.0, centroid=np.array([2.0, 0.0]))

        self.sched.record_patch(c_low)
        self.sched.record_patch(c_mid)
        self.sched.record_patch(c_high)

        top = self.sched.top_uncertain(k=3)
        uncertainties = [self.sched.uncertainty(c) for c in top]
        assert uncertainties[0] >= uncertainties[1] >= uncertainties[2]

    def test_most_uncertain_is_first(self):
        c_low  = _classification(curvature=0.1, trust=0.9, centroid=np.zeros(2))
        c_high = _classification(curvature=5.0, trust=0.0, centroid=np.array([1.0, 1.0]))

        self.sched.record_patch(c_low)
        self.sched.record_patch(c_high)

        top = self.sched.top_uncertain(k=2)
        assert top[0].curvature_ratio == pytest.approx(5.0)

    def test_k_larger_than_history(self):
        c = _classification(curvature=0.5, trust=0.5, centroid=np.zeros(2))
        self.sched.record_patch(c)
        top = self.sched.top_uncertain(k=10)
        assert len(top) == 1


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty(self):
        g = _make_graph()
        sched = PatchExplorationScheduler(g, n_states=2)
        s = sched.summary()
        assert s["n_recorded"] == 0
        assert s["mean_uncertainty"] == 0.0
        assert s["max_uncertainty"] == 0.0

    def test_with_records(self):
        g = _make_graph()
        sched = PatchExplorationScheduler(g, n_states=2)
        for i in range(3):
            c = _classification(curvature=float(i + 1) * 0.5, trust=0.1,
                                centroid=np.zeros(2))
            sched.record_patch(c)
        s = sched.summary()
        assert s["n_recorded"] == 3
        assert s["mean_uncertainty"] > 0
        assert s["max_uncertainty"] >= s["mean_uncertainty"]


# ---------------------------------------------------------------------------
# Rolling history cap
# ---------------------------------------------------------------------------

class TestMaxHistory:
    def test_history_capped(self):
        g = _make_graph()
        sched = PatchExplorationScheduler(g, n_states=2, max_history=5)
        for i in range(20):
            c = _classification(curvature=0.1, trust=0.5,
                                centroid=np.zeros(2))
            sched.record_patch(c)
        # deque maxlen=5 keeps only last 5
        assert sched.summary()["n_recorded"] == 5
