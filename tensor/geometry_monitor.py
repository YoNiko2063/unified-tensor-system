"""
Geometry Monitor — rolling stability monitor for adaptive basis expansion.

Mathematical basis (LOGIC_FLOW.md, Section 0M):
  Monitors 5 instability conditions on a rolling window of observations.
  When any condition fires: snapshot is returned for rollback, caller
  must restore previous basis degree and disable adaptive flags for one cycle.

Unstable conditions:
  1. Mean curvature monotonically increasing over full window (diverging)
  2. Mean koopman_trust < 0.2 (fitting noise, not dynamics)
  3. Observable degree jumped > +1 within window (runaway expansion)
  4. Patch count growth > 2× baseline rate (state space explosion)
  5. Equivalences per interval spike > 3σ above rolling mean (false positives)

Reference: LOGIC_FLOW.md Section 0M
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np


class GeometryMonitor:
    """
    Rolling stability monitor for adaptive Koopman basis expansion.

    Always instantiated in AutonomousLearningSystem, even when adaptive flags
    are False. When flags are False, is_unstable() logs baseline metrics but
    rollback() is a no-op (nothing adaptive is running to roll back).

    Usage:
        monitor = GeometryMonitor(window_size=100)

        # On each cycle:
        monitor.record(classification, edmd, patch_count, n_equivalences)
        monitor.snapshot_state(edmd.degree, graph.summary())

        if monitor.is_unstable():
            snap = monitor.rollback()
            edmd = EDMDKoopman(observable_degree=snap['degree'])
            edmd.fit(pairs)
    """

    def __init__(
        self,
        window_size: int = 100,
        max_degree_jump: int = 1,
        min_trust: float = 0.2,
        max_patch_growth_rate: float = 2.0,
        equiv_spike_sigma: float = 3.0,
    ):
        """
        Args:
            window_size:            rolling window length for all metrics
            max_degree_jump:        max allowed degree increase within window (default +1)
            min_trust:              mean trust below this triggers instability
            max_patch_growth_rate:  patch count growth above this × baseline triggers instability
            equiv_spike_sigma:      equivalence spike threshold in standard deviations
        """
        self._window = window_size
        self._max_degree_jump = max_degree_jump
        self._min_trust = min_trust
        self._max_growth_rate = max_patch_growth_rate
        self._spike_sigma = equiv_spike_sigma

        # Rolling deques
        self._curvatures: Deque[float] = deque(maxlen=window_size)
        self._trusts: Deque[float] = deque(maxlen=window_size)
        self._degrees: Deque[int] = deque(maxlen=window_size)
        self._patch_counts: Deque[int] = deque(maxlen=window_size)
        self._equiv_counts: Deque[int] = deque(maxlen=window_size)

        # Stable snapshot
        self._last_stable: Optional[Dict] = None
        self._last_rollback_time: Optional[float] = None
        self._rollback_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        classification,        # PatchClassification from lca_patch_detector
        edmd,                  # EDMDKoopman instance (reads .degree)
        patch_count: int,
        n_equivalences: int,
    ) -> None:
        """
        Record one observation cycle.

        Args:
            classification: result from LCAPatchDetector.classify_region()
            edmd:           current fitted EDMDKoopman
            patch_count:    total patches in PatchGraph
            n_equivalences: equivalences found in this interval
        """
        self._curvatures.append(float(classification.curvature_ratio))
        self._trusts.append(float(classification.koopman_trust))
        self._degrees.append(int(edmd.degree))
        self._patch_counts.append(int(patch_count))
        self._equiv_counts.append(int(n_equivalences))

    def snapshot_state(self, basis_degree: int, patch_summary: dict) -> None:
        """
        Save current stable state for potential rollback.

        Call this at the end of each cycle where the system is stable
        (i.e., is_unstable() returned False).

        Args:
            basis_degree:  current EDMDKoopman.degree
            patch_summary: dict from PatchGraph.summary()
        """
        self._last_stable = {
            'degree': basis_degree,
            'patch_summary': dict(patch_summary),
            'timestamp': time.time(),
        }

    # ------------------------------------------------------------------
    # Instability detection
    # ------------------------------------------------------------------

    def is_unstable(self) -> bool:
        """
        Check whether any instability condition fires.

        Returns True if ANY condition is triggered. Logs which conditions
        fired to facilitate debugging.

        Requires at least 3 observations to make a meaningful assessment;
        returns False if the window has fewer than 3 entries.
        """
        if len(self._curvatures) < 3:
            return False

        fired = self._fired_conditions()
        return len(fired) > 0

    def _fired_conditions(self) -> List[str]:
        """Return list of condition names that are currently violated."""
        fired = []

        # Condition 1: Monotonically increasing mean curvature
        if self._curvature_monotone_increasing():
            fired.append('curvature_monotone_increasing')

        # Condition 2: Mean trust below minimum
        if np.mean(self._trusts) < self._min_trust:
            fired.append('mean_trust_too_low')

        # Condition 3: Degree jumped > max_degree_jump within window
        degrees = list(self._degrees)
        if len(degrees) >= 2:
            degree_jump = max(degrees) - min(degrees)
            if degree_jump > self._max_degree_jump:
                fired.append(f'degree_jump_{degree_jump}')

        # Condition 4: Patch count growth > 2× baseline
        if self._patch_growth_excessive():
            fired.append('patch_growth_excessive')

        # Condition 5: Equivalence spike > 3σ
        if self._equivalence_spike():
            fired.append('equivalence_spike')

        return fired

    def _curvature_monotone_increasing(self) -> bool:
        """True if curvature is monotonically non-decreasing over full window."""
        curvs = list(self._curvatures)
        if len(curvs) < 3:
            return False
        # Use smoothed quartile windows to avoid noise
        half = len(curvs) // 2
        first_half_mean = float(np.mean(curvs[:half]))
        second_half_mean = float(np.mean(curvs[half:]))
        # Check every consecutive pair for non-decreasing (allow small tolerance)
        for i in range(len(curvs) - 1):
            if curvs[i] > curvs[i + 1] + 0.01 * abs(curvs[i]):
                return False
        # Also require second half is meaningfully higher than first half
        return second_half_mean > first_half_mean * 1.1

    def _patch_growth_excessive(self) -> bool:
        """True if patch count grew > max_growth_rate × first count within window."""
        counts = list(self._patch_counts)
        if len(counts) < 3 or counts[0] == 0:
            return False
        baseline = counts[0]
        current = counts[-1]
        return current > self._max_growth_rate * baseline

    def _equivalence_spike(self) -> bool:
        """True if most recent equivalence count is > spike_sigma SDs above mean."""
        equivs = list(self._equiv_counts)
        if len(equivs) < 3:
            return False
        mean = float(np.mean(equivs[:-1]))
        std = float(np.std(equivs[:-1]))
        if std < 1e-9:
            return False   # no variation, no spike
        return equivs[-1] > mean + self._spike_sigma * std

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self) -> Dict:
        """
        Return last stable snapshot for rollback.

        Caller responsibilities:
          1. Set edmd = EDMDKoopman(observable_degree=snapshot['degree'])
          2. Refit edmd on current pairs
          3. Temporarily disable adaptive flags for one cycle

        Returns:
            dict with keys: 'degree', 'patch_summary', 'timestamp'
            Returns {'degree': 1, 'patch_summary': {}, 'timestamp': 0.0}
            if no stable snapshot has been saved yet.
        """
        t = time.time()
        self._last_rollback_time = t
        conditions = self._fired_conditions()
        event = {
            'timestamp': t,
            'conditions': conditions,
            'snapshot': self._last_stable,
        }
        self._rollback_log.append(event)

        if self._last_stable is None:
            return {'degree': 1, 'patch_summary': {}, 'timestamp': 0.0}
        return dict(self._last_stable)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return current monitoring statistics."""
        mean_curv = float(np.mean(list(self._curvatures))) if self._curvatures else 0.0
        mean_trust = float(np.mean(list(self._trusts))) if self._trusts else 0.0
        cur_degree = int(self._degrees[-1]) if self._degrees else 0
        cur_patches = int(self._patch_counts[-1]) if self._patch_counts else 0
        fired = self._fired_conditions() if len(self._curvatures) >= 3 else []

        return {
            'n_observations': len(self._curvatures),
            'mean_curvature': mean_curv,
            'mean_trust': mean_trust,
            'current_degree': cur_degree,
            'current_patch_count': cur_patches,
            'is_unstable': len(fired) > 0,
            'fired_conditions': fired,
            'n_rollbacks': len(self._rollback_log),
            'last_rollback_time': self._last_rollback_time,
            'has_stable_snapshot': self._last_stable is not None,
        }
