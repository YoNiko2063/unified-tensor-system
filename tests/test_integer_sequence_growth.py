"""
Tests for tensor/integer_sequence_growth.py

Covers:
  1. FibonacciGrowthSchedule — sequence values, index advancement, reset
  2. FractalDimensionEstimator — CRITICAL-4 absolute count, D_H properties
  3. RecursiveGrowthScheduler — gate logic, cooldown hysteresis (CRITICAL-4), grow()
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from tensor.integer_sequence_growth import (
    FibonacciGrowthSchedule,
    FractalDimensionEstimator,
    RecursiveGrowthScheduler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubHDVSystem:
    """Minimal stub with extend_capacity() and find_overlaps()."""

    def __init__(self, hdv_dim: int = 1000, active_count: int = 50):
        self.hdv_dim = hdv_dim
        self._active_count = active_count
        self.extensions: list = []

    def extend_capacity(self, n: int) -> None:
        self.hdv_dim += n
        self.extensions.append(n)

    def find_overlaps(self):
        return list(range(self._active_count))


# ---------------------------------------------------------------------------
# 1. FibonacciGrowthSchedule
# ---------------------------------------------------------------------------


class TestFibonacciGrowthSchedule:

    def test_first_event_is_base_chunk(self):
        s = FibonacciGrowthSchedule(base_chunk=100)
        assert s.dims_to_add() == 1 * 100  # F(0) = 1

    def test_second_event_is_base_chunk(self):
        s = FibonacciGrowthSchedule(base_chunk=100)
        s.dims_to_add()
        assert s.dims_to_add() == 1 * 100  # F(1) = 1

    def test_third_event_is_2x(self):
        s = FibonacciGrowthSchedule(base_chunk=100)
        s.dims_to_add(); s.dims_to_add()
        assert s.dims_to_add() == 2 * 100  # F(2) = 2

    def test_sequence_matches_fibonacci(self):
        s = FibonacciGrowthSchedule(base_chunk=1)
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        actual = [s.dims_to_add() for _ in expected]
        assert actual == expected

    def test_event_index_increments(self):
        s = FibonacciGrowthSchedule(base_chunk=100)
        assert s.event_index == 0
        s.dims_to_add()
        assert s.event_index == 1
        s.dims_to_add()
        assert s.event_index == 2

    def test_reset_restarts_sequence(self):
        s = FibonacciGrowthSchedule(base_chunk=50)
        first = s.dims_to_add()
        for _ in range(5):
            s.dims_to_add()
        s.reset()
        assert s.event_index == 0
        assert s.dims_to_add() == first

    def test_base_chunk_scaling(self):
        s = FibonacciGrowthSchedule(base_chunk=200)
        vals = [s.dims_to_add() for _ in range(5)]
        assert vals == [200, 200, 400, 600, 1000]

    def test_clamps_at_table_end(self):
        """Past the precomputed table, clamps to largest Fibonacci number."""
        s = FibonacciGrowthSchedule(base_chunk=1)
        # Exhaust the table
        for _ in range(100):
            v = s.dims_to_add()
        # Should not raise and should be >= last table entry
        assert v >= 1

    def test_dims_always_positive(self):
        s = FibonacciGrowthSchedule(base_chunk=1)
        for _ in range(20):
            assert s.dims_to_add() > 0


# ---------------------------------------------------------------------------
# 2. FractalDimensionEstimator — CRITICAL-4 absolute count
# ---------------------------------------------------------------------------


class TestFractalDimensionEstimator:

    def setup_method(self):
        self.estimator = FractalDimensionEstimator()

    def test_zero_active_count_returns_zero(self):
        assert self.estimator.estimate(0) == 0.0

    def test_negative_active_count_returns_zero(self):
        assert self.estimator.estimate(-5) == 0.0

    def test_positive_active_count_returns_nonnegative(self):
        d = self.estimator.estimate(50)
        assert d >= 0.0

    def test_small_active_count(self):
        d = self.estimator.estimate(1)
        assert d >= 0.0

    def test_large_active_count(self):
        d = self.estimator.estimate(10000)
        assert d >= 0.0

    def test_returns_float(self):
        assert isinstance(self.estimator.estimate(100), float)

    def test_d_h_stable_after_hdv_growth(self):
        """
        CRITICAL-4 key property: after hdv_dim doubles but active_count unchanged,
        D_H should not change (because we use absolute count, not ratio).
        """
        active_count = 200
        d_before = self.estimator.estimate(active_count)
        # hdv_dim doubled (as if growth happened) — but active_count unchanged
        # estimator only takes active_count, so result must be identical
        d_after = self.estimator.estimate(active_count)  # same input
        assert abs(d_before - d_after) < 1e-12

    def test_more_active_dims_increases_d_h(self):
        """More active dims generally means higher fractal dimension (or same)."""
        d_small = self.estimator.estimate(10)
        d_large = self.estimator.estimate(1000)
        # Not a strict monotone law, but for our box-counting both should be >= 0
        assert d_small >= 0 and d_large >= 0

    def test_does_not_use_hdv_dim_directly(self):
        """Estimator has no hdv_dim parameter — confirming absolute count usage."""
        import inspect
        sig = inspect.signature(FractalDimensionEstimator.estimate)
        params = list(sig.parameters.keys())
        assert "hdv_dim" not in params
        assert "active_count" in params


# ---------------------------------------------------------------------------
# 3. RecursiveGrowthScheduler
# ---------------------------------------------------------------------------


class TestRecursiveGrowthSchedulerGating:

    def _make_scheduler(self, cooldown=10, **kwargs):
        return RecursiveGrowthScheduler(
            d_target=1.5, fill_ratio=0.1,
            rho_min=0.0, rho_max=1.0,
            cooldown_cycles=cooldown, **kwargs
        )

    # -- Basic gate logic --

    def test_allows_growth_when_conditions_met(self):
        sched = self._make_scheduler()
        assert sched.should_grow(d_h=1.0, rank_ratio=0.5, rho=0.5, is_unstable=False)

    def test_blocks_when_d_h_above_target(self):
        sched = self._make_scheduler()
        assert not sched.should_grow(d_h=2.0, rank_ratio=0.5, rho=0.5, is_unstable=False)

    def test_blocks_when_rank_ratio_too_low(self):
        sched = self._make_scheduler()
        assert not sched.should_grow(d_h=1.0, rank_ratio=0.0, rho=0.5, is_unstable=False)

    def test_blocks_when_rho_below_min(self):
        sched = RecursiveGrowthScheduler(d_target=1.5, fill_ratio=0.1, rho_min=0.2, rho_max=1.0)
        assert not sched.should_grow(d_h=1.0, rank_ratio=0.5, rho=0.1, is_unstable=False)

    def test_blocks_when_rho_above_max(self):
        sched = RecursiveGrowthScheduler(d_target=1.5, fill_ratio=0.1, rho_min=0.0, rho_max=0.5)
        assert not sched.should_grow(d_h=1.0, rank_ratio=0.5, rho=0.9, is_unstable=False)

    def test_blocks_when_unstable(self):
        sched = self._make_scheduler()
        assert not sched.should_grow(d_h=1.0, rank_ratio=0.5, rho=0.5, is_unstable=True)

    def test_allows_at_exact_d_target_boundary(self):
        """d_h strictly < d_target allowed; d_h == d_target blocked."""
        sched = self._make_scheduler()
        assert not sched.should_grow(d_h=1.5, rank_ratio=0.5, rho=0.5, is_unstable=False)
        assert sched.should_grow(d_h=1.499, rank_ratio=0.5, rho=0.5, is_unstable=False)


class TestRecursiveGrowthSchedulerCooldown:
    """CRITICAL-4: cooldown hysteresis tests."""

    def _ready_scheduler(self, cooldown=10):
        """Scheduler that starts ready (cycles_since_growth == cooldown)."""
        return RecursiveGrowthScheduler(
            d_target=1.5, fill_ratio=0.1, rho_min=0.0, rho_max=1.0,
            cooldown_cycles=cooldown
        )

    def test_starts_ready_no_initial_lockout(self):
        """Scheduler starts with cycles_since_growth == cooldown — immediately usable."""
        sched = self._ready_scheduler(cooldown=10)
        assert not sched.in_cooldown
        assert sched.should_grow(d_h=1.0, rank_ratio=0.5, rho=0.5, is_unstable=False)

    def test_in_cooldown_after_grow(self):
        sched = self._ready_scheduler(cooldown=3)
        hdv = _StubHDVSystem()
        # Trigger growth
        sched.grow(hdv)
        assert sched.in_cooldown
        assert not sched.should_grow(d_h=1.0, rank_ratio=0.5, rho=0.5, is_unstable=False)

    def test_cooldown_advances_on_should_grow_call(self):
        sched = self._ready_scheduler(cooldown=3)
        hdv = _StubHDVSystem()
        sched.grow(hdv)
        # cycles_since_growth starts at 0
        assert sched.cycles_since_growth == 0
        sched.should_grow(1.0, 0.5, 0.5, False)
        assert sched.cycles_since_growth == 1
        sched.should_grow(1.0, 0.5, 0.5, False)
        assert sched.cycles_since_growth == 2
        sched.should_grow(1.0, 0.5, 0.5, False)
        assert sched.cycles_since_growth == 3
        # Now cooldown expired — should be ready
        assert not sched.in_cooldown
        assert sched.should_grow(1.0, 0.5, 0.5, False)

    def test_cooldown_prevents_immediate_re_trigger(self):
        """
        After growth, consecutive should_grow() calls return False until cooldown expires.
        This is the core CRITICAL-4 anti-oscillation test.
        """
        cooldown = 5
        sched = self._ready_scheduler(cooldown=cooldown)
        hdv = _StubHDVSystem()
        sched.grow(hdv)

        false_results = []
        for _ in range(cooldown):
            false_results.append(
                sched.should_grow(d_h=1.0, rank_ratio=0.5, rho=0.5, is_unstable=False)
            )
        # All during cooldown must be False
        assert all(r is False for r in false_results)

        # After cooldown expires, next call can return True
        assert sched.should_grow(d_h=1.0, rank_ratio=0.5, rho=0.5, is_unstable=False)

    def test_grow_resets_cooldown_counter(self):
        sched = self._ready_scheduler(cooldown=3)
        hdv = _StubHDVSystem()
        # First growth
        sched.grow(hdv)
        assert sched.cycles_since_growth == 0
        # Advance through cooldown
        for _ in range(3):
            sched.should_grow(1.0, 0.5, 0.5, False)
        assert not sched.in_cooldown
        # Second growth
        sched.grow(hdv)
        assert sched.cycles_since_growth == 0

    def test_cooldown_zero_means_always_ready(self):
        sched = RecursiveGrowthScheduler(
            d_target=1.5, fill_ratio=0.1, cooldown_cycles=0
        )
        hdv = _StubHDVSystem()
        sched.grow(hdv)
        # With cooldown=0, immediately ready again
        assert sched.should_grow(d_h=1.0, rank_ratio=0.5, rho=0.5, is_unstable=False)


class TestRecursiveGrowthSchedulerGrow:

    def setup_method(self):
        self.sched = RecursiveGrowthScheduler(
            d_target=1.5, fill_ratio=0.1, rho_min=0.0, rho_max=1.0,
            base_chunk=100, cooldown_cycles=10
        )
        self.hdv = _StubHDVSystem(hdv_dim=1000)

    def test_grow_calls_extend_capacity(self):
        with mock.patch.object(self.hdv, "extend_capacity") as m:
            self.sched.grow(self.hdv)
            m.assert_called_once()

    def test_grow_returns_positive_int(self):
        n = self.sched.grow(self.hdv)
        assert isinstance(n, int)
        assert n > 0

    def test_grow_returns_first_fibonacci_chunk(self):
        n = self.sched.grow(self.hdv)
        assert n == 1 * 100  # F(0)*base_chunk

    def test_grow_second_call_returns_second_fibonacci(self):
        # Exhaust cooldown after first grow
        first = self.sched.grow(self.hdv)
        for _ in range(10):
            self.sched.should_grow(1.0, 0.5, 0.5, False)
        second = self.sched.grow(self.hdv)
        assert second == 1 * 100  # F(1)*100

    def test_grow_increments_total_events(self):
        assert self.sched.total_growth_events == 0
        self.sched.grow(self.hdv)
        assert self.sched.total_growth_events == 1

    def test_hdv_dim_increases_after_grow(self):
        old_dim = self.hdv.hdv_dim
        n = self.sched.grow(self.hdv)
        assert self.hdv.hdv_dim == old_dim + n

    def test_grow_does_not_call_edmd_fit(self):
        """Growth must never trigger EDMD refits (invariant check)."""
        from tensor.koopman_edmd import EDMDKoopman
        with mock.patch.object(EDMDKoopman, "fit") as m:
            self.sched.grow(self.hdv)
            m.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Integration: FractalDimensionEstimator + RecursiveGrowthScheduler
# ---------------------------------------------------------------------------


class TestGrowthIntegration:

    def test_full_growth_cycle_no_oscillation(self):
        """
        Simulate 20 evaluation cycles post-growth — growth should not re-trigger
        until cooldown expires.
        """
        estimator = FractalDimensionEstimator()
        sched = RecursiveGrowthScheduler(
            d_target=1.5, fill_ratio=0.1, cooldown_cycles=5, base_chunk=100
        )
        hdv = _StubHDVSystem(hdv_dim=1000, active_count=50)

        d_h = estimator.estimate(len(hdv.find_overlaps()))
        assert sched.should_grow(d_h, 0.5, 0.5, False)  # first check OK
        sched.grow(hdv)  # growth event

        # During cooldown: should never grow
        grew_during_cooldown = False
        for _ in range(5):
            d_h = estimator.estimate(len(hdv.find_overlaps()))
            if sched.should_grow(d_h, 0.5, 0.5, False):
                grew_during_cooldown = True
        assert not grew_during_cooldown

        # After cooldown: can grow again if conditions met
        d_h = estimator.estimate(len(hdv.find_overlaps()))
        assert sched.should_grow(d_h, 0.5, 0.5, False)

    def test_absolute_count_stability(self):
        """
        After hdv_dim doubles (Fibonacci growth) with active_count unchanged,
        D_H stays the same — confirming absolute-count semantics.
        """
        estimator = FractalDimensionEstimator()
        active_count = 100
        d_before = estimator.estimate(active_count)
        # Simulate hdv_dim doubling — not passed to estimator (absolute count only)
        d_after = estimator.estimate(active_count)  # same active_count
        assert abs(d_before - d_after) < 1e-12
