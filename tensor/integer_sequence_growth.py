"""
Fractal Dimensional Growth — Fibonacci-scheduled HDV capacity expansion.

Mathematical basis (Plan §V, equations 5 and 6):

  Fibonacci growth schedule:
    dims_to_add at event n: F(n) × base_chunk, F = [1,1,2,3,5,8,...] (OEIS A000045)

  Box-counting fractal dimension (CRITICAL-4 — absolute active count):
    D_H ≈ log N(ε) / log(1/ε)  (log-linear fit over ε ∈ {0.1, 0.05, 0.01} × active_count)
    Active count = absolute number of active HDV dims (find_overlaps() or nonzero pattern count)
    NOT ratio active_dims/hdv_dim — using absolute count keeps D_H stable immediately post-growth.

  Growth gate (eq. 6):
    ALLOW_GROWTH iff D_H < D_target AND rank_ratio >= fill_ratio
                  AND rho_min <= rho <= rho_max AND NOT is_unstable

  Cooldown hysteresis (CRITICAL-4):
    After each growth event, cooldown_cycles evaluation cycles must pass before
    should_grow() can return True again.  This prevents oscillation where growth
    immediately re-satisfies growth conditions before new patterns fill capacity.
"""

from __future__ import annotations

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# FibonacciGrowthSchedule
# ---------------------------------------------------------------------------


class FibonacciGrowthSchedule:
    """
    Sub-exponential HDV dimension growth using Fibonacci sequence.

    dims_to_add(n) = F(n) * base_chunk where F = [1,1,2,3,5,8,13,21,34,...]

    F(n)/F(n-1) → φ ≈ 1.618 asymptotically — each step bounded.
    """

    _FIB: List[int] = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

    def __init__(self, base_chunk: int = 100) -> None:
        """
        Args:
            base_chunk: multiplier applied to each Fibonacci number (default 100)
        """
        self._base_chunk = base_chunk
        self._event_index: int = 0  # next Fibonacci index to use

    def dims_to_add(self) -> int:
        """
        Return the number of dims to add at the current growth event, then advance.

        Clips to the largest precomputed Fibonacci number once the table is exhausted.
        """
        idx = min(self._event_index, len(self._fib) - 1)
        dims = self._fib[idx] * self._base_chunk
        self._event_index += 1
        return dims

    @property
    def _fib(self) -> List[int]:
        return FibonacciGrowthSchedule._FIB

    @property
    def event_index(self) -> int:
        """Number of growth events that have occurred."""
        return self._event_index

    def reset(self) -> None:
        """Reset back to event 0 (F(0)=1)."""
        self._event_index = 0


# ---------------------------------------------------------------------------
# FractalDimensionEstimator
# ---------------------------------------------------------------------------


class FractalDimensionEstimator:
    """
    Box-counting estimate of fractal dimension of active HDV space.

    CRITICAL-4: uses ABSOLUTE active dim count, not ratio active/hdv_dim.

    D_H ≈ log N(ε) / log(1/ε)

    Three box sizes: ε ∈ {0.1, 0.05, 0.01} scaled by active_count.
    Log-linear regression on (log(1/ε), log N(ε)).
    """

    _EPSILON_FRACTIONS: List[float] = [0.1, 0.05, 0.01]

    def estimate(self, active_count: int) -> float:
        """
        Estimate D_H for an HDV space with `active_count` active dimensions.

        Args:
            active_count: absolute count of active dims (e.g. len(find_overlaps()))

        Returns:
            D_H estimate (float ≥ 0). Returns 0.0 if active_count ≤ 0.
        """
        if active_count <= 0:
            return 0.0

        # Box sizes in absolute dimension units
        epsilons = [f * active_count for f in self._EPSILON_FRACTIONS]

        # N(ε) = number of boxes of size ε needed to cover active_count dims
        # For a 1-D count: N(ε) = ceil(active_count / ε)
        log_inv_eps = []
        log_n = []
        for eps in epsilons:
            if eps <= 0:
                continue
            n = max(1, int(np.ceil(active_count / eps)))
            log_inv_eps.append(np.log(1.0 / (eps / active_count)))  # log(1/ε_fraction)
            log_n.append(np.log(float(n)))

        if len(log_inv_eps) < 2:
            # Not enough points for regression — return trivial estimate
            return 1.0

        # Log-linear regression: D_H = slope of log N vs log(1/ε)
        x = np.array(log_inv_eps)
        y = np.array(log_n)

        # Slope via least-squares (equivalent to polyfit degree 1)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)
        if den < 1e-12:
            return float(y_mean)
        slope = float(num / den)
        return max(0.0, slope)


# ---------------------------------------------------------------------------
# RecursiveGrowthScheduler
# ---------------------------------------------------------------------------


class RecursiveGrowthScheduler:
    """
    Gates and executes Fibonacci HDV capacity growth (eq. 6).

    CRITICAL-4 — cooldown_cycles: After each growth event, cooldown_cycles
    evaluation cycles pass before should_grow() can return True again.
    Prevents oscillation on fast growth re-trigger.

    Usage:
        scheduler = RecursiveGrowthScheduler(hdv_system)
        d_h = fractal_estimator.estimate(len(hdv_system.find_overlaps()))
        if scheduler.should_grow(d_h, rank_ratio, rho, is_unstable):
            n_added = scheduler.grow(hdv_system)
    """

    def __init__(
        self,
        d_target: float = 1.5,
        fill_ratio: float = 0.1,
        rho_min: float = 0.0,
        rho_max: float = 1.0,
        base_chunk: int = 100,
        cooldown_cycles: int = 10,
    ) -> None:
        """
        Args:
            d_target:       max D_H below which growth is permitted
            fill_ratio:     min rank(G)/d for growth to be permitted
            rho_min/max:    curvature-ratio window for growth
            base_chunk:     Fibonacci base multiplier (dims per event)
            cooldown_cycles: cycles to wait after each growth before re-enabling
        """
        self._d_target = d_target
        self._fill_ratio = fill_ratio
        self._rho_min = rho_min
        self._rho_max = rho_max
        self._schedule = FibonacciGrowthSchedule(base_chunk=base_chunk)
        self._cooldown: int = cooldown_cycles
        # Start ready: _cycles_since_growth = cooldown so first check is not blocked
        self._cycles_since_growth: int = cooldown_cycles
        self._total_growth_events: int = 0

    def should_grow(
        self,
        d_h: float,
        rank_ratio: float,
        rho: float,
        is_unstable: bool,
    ) -> bool:
        """
        Return True iff growth is permitted this cycle.

        Implements eq. (6) gating with CRITICAL-4 cooldown:
            (1) cooldown has expired
            (2) D_H < D_target           (space not yet saturated)
            (3) rank_ratio >= fill_ratio  (observable basis well-filled)
            (4) rho_min <= rho <= rho_max (within operating curvature window)
            (5) NOT is_unstable           (within success membrane S)

        Advances cooldown counter even when returning False.
        """
        if self._cycles_since_growth < self._cooldown:
            self._cycles_since_growth += 1
            return False

        return (
            d_h < self._d_target
            and rank_ratio >= self._fill_ratio
            and self._rho_min <= rho <= self._rho_max
            and not is_unstable
        )

    def grow(self, hdv_system) -> int:
        """
        Execute a growth event: extend hdv_system capacity by F(n)*base_chunk dims.

        Resets the cooldown counter.

        Args:
            hdv_system: object with extend_capacity(n_dims: int) method

        Returns:
            Number of dimensions added
        """
        n = self._schedule.dims_to_add()
        hdv_system.extend_capacity(n)
        self._cycles_since_growth = 0  # reset cooldown
        self._total_growth_events += 1
        return n

    @property
    def total_growth_events(self) -> int:
        return self._total_growth_events

    @property
    def cycles_since_growth(self) -> int:
        return self._cycles_since_growth

    @property
    def in_cooldown(self) -> bool:
        return self._cycles_since_growth < self._cooldown
