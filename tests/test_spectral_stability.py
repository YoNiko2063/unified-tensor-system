"""
Phase 1 — Validate the Geometry.

KoopmanInvariantDescriptor must behave as a metric space that reflects
physical ordering.  Falsification criterion:

    Spearman(dist_ij, |f_i - f_j|) > 0.7

If this fails the invariant is noise and the memory idea collapses.

Targets: [500, 800, 1000, 1500, 2000] Hz
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import pytest
from scipy.stats import spearmanr

from optimization.koopman_memory import KoopmanExperienceMemory, _MemoryEntry
from optimization.rlc_evaluator import RLCEvaluator
from optimization.rlc_parameterization import RLCDesignMapper
from optimization.hdv_optimizer import ConstrainedHDVOptimizer


_TARGETS = [800.0, 1000.0, 1200.0, 1500.0, 2000.0]
_N_ITER  = 500   # enough accepted steps for Koopman fit
_MAX_OBJ = 0.05  # entries with objective > 5% are optimization failures; skip them


# ── helpers ───────────────────────────────────────────────────────────────────


def _run_and_get_entry(target_hz: float, n_iter: int = _N_ITER) -> _MemoryEntry | None:
    """
    Run one optimization and return the stored _MemoryEntry (or None).

    Returns None if Koopman fit failed OR if the optimizer failed to converge
    (objective > _MAX_OBJ), since a failed optimization produces a meaningless
    invariant that would corrupt the geometry test.
    """
    mapper    = RLCDesignMapper(hdv_dim=64, seed=42)
    evaluator = RLCEvaluator(max_Q=10.0, max_energy_loss=0.5)
    memory    = KoopmanExperienceMemory()
    opt = ConstrainedHDVOptimizer(mapper, evaluator, memory, n_iter=n_iter, seed=0)
    # pilot_steps=0 → pure cold start; invariant comes from the MAIN run only
    result = opt.optimize(target_hz, pilot_steps=0)
    if not memory._entries:
        return None
    if result.objective > _MAX_OBJ:
        return None   # optimizer failed to reach target; exclude from geometry test
    return memory._entries[0]


def _retrieval_distance(inv_a, inv_b) -> float:
    """L2 distance between to_retrieval_vector() representations."""
    va = inv_a.to_retrieval_vector()
    vb = inv_b.to_retrieval_vector()
    return float(np.linalg.norm(va - vb))


# ── sanity: entries are stored ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def entries():
    """Run all 5 targets once; cache results for the whole test module."""
    result = {}
    for f in _TARGETS:
        result[f] = _run_and_get_entry(f)
    return result


def test_entries_stored(entries):
    """Optimizer must store at least 4 of 5 entries (Koopman fit may fail rarely)."""
    stored = [f for f, e in entries.items() if e is not None]
    assert len(stored) >= 4, (
        f"Only {len(stored)} / {len(_TARGETS)} Koopman fits succeeded: {stored}"
    )


# ── Phase 1: spectral monotonicity ───────────────────────────────────────────


def test_spectral_monotonicity(entries):
    """
    Spearman rank correlation between pairwise descriptor distance and
    pairwise frequency difference must exceed 0.7.

    This is the hard falsification gate for the Koopman geometry claim.
    """
    valid = [(f, e) for f, e in entries.items() if e is not None]
    assert len(valid) >= 4, "Too few stored entries for correlation test."

    freqs = [f for f, _ in valid]
    vecs  = [e.invariant.to_retrieval_vector() for _, e in valid]

    dists = []
    diffs = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            dists.append(np.linalg.norm(vecs[i] - vecs[j]))
            diffs.append(abs(freqs[i] - freqs[j]))

    corr, pvalue = spearmanr(dists, diffs)

    # Report always (useful regardless of pass/fail)
    print(f"\n  Spearman ρ = {corr:.4f}  (p={pvalue:.4f})")
    print(f"  Pairs: {len(dists)}")
    print(f"  dist  range: [{min(dists):.4f}, {max(dists):.4f}]")
    print(f"  Δfreq range: [{min(diffs):.0f}, {max(diffs):.0f}] Hz")

    assert corr > 0.7, (
        f"Spectral monotonicity failed: Spearman ρ = {corr:.4f}  "
        f"(need > 0.7).  "
        f"The Koopman invariant does not reflect physical ordering — "
        f"redesign the invariant."
    )


def test_adjacent_targets_closer_than_extreme(entries):
    """
    Structural check: dist(800, 1000) < dist(800, 2000).
    Nearest frequencies must produce nearer invariants.
    """
    e800  = entries.get(800.0)
    e1000 = entries.get(1000.0)
    e2000 = entries.get(2000.0)
    if any(e is None for e in [e800, e1000, e2000]):
        pytest.skip("One of (800, 1000, 2000) Hz entries missing.")

    d_near = _retrieval_distance(e800.invariant, e1000.invariant)
    d_far  = _retrieval_distance(e800.invariant, e2000.invariant)

    print(f"\n  dist(800,1000)={d_near:.4f}  dist(800,2000)={d_far:.4f}")
    assert d_near < d_far, (
        f"Adjacent targets (800, 1000 Hz) produced a LARGER invariant distance "
        f"({d_near:.4f}) than extreme targets (800, 2000 Hz) ({d_far:.4f}).  "
        f"Geometry is wrong."
    )


def test_descriptor_distances_are_positive(entries):
    """Invariant distance between distinct targets must be > 0."""
    valid = [(f, e) for f, e in entries.items() if e is not None]
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            fi, ei = valid[i]
            fj, ej = valid[j]
            d = _retrieval_distance(ei.invariant, ej.invariant)
            assert d > 0, (
                f"Zero invariant distance between {fi} Hz and {fj} Hz — "
                f"entries are identical, invariant has collapsed."
            )


def test_self_distance_is_zero(entries):
    """Sanity: distance(inv, inv) == 0."""
    for f, e in entries.items():
        if e is None:
            continue
        v = e.invariant.to_retrieval_vector()
        assert np.linalg.norm(v - v) == pytest.approx(0.0)


# ── Spectral summary ──────────────────────────────────────────────────────────


def test_print_full_distance_matrix(entries, capsys):
    """Print the full distance matrix for inspection."""
    valid = sorted([(f, e) for f, e in entries.items() if e is not None])
    freqs = [f for f, _ in valid]
    vecs  = [e.invariant.to_retrieval_vector() for _, e in valid]
    n = len(valid)

    header = "     " + "".join(f"{int(f):>8}" for f in freqs)
    print(f"\n  Pairwise invariant L2 distances:")
    print(f"  {header}")
    for i in range(n):
        row = f"  {int(freqs[i]):>4}:"
        for j in range(n):
            d = np.linalg.norm(vecs[i] - vecs[j])
            row += f" {d:>7.4f}"
        print(row)
    # This test always passes — it's diagnostic
