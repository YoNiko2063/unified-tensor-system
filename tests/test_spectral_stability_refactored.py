"""
Phase 1 (refactored) — Validate Geometry with Domain-Invariant Invariant.

Same falsification criterion as test_spectral_stability.py:
    Spearman(dist_ij, |f_i - f_j|) > 0.7   (tightened to 0.8 here)

Now uses the refactored KoopmanInvariantDescriptor whose retrieval key is
the 3-D domain-invariant (log_ω₀_norm, log_Q_norm, ζ) — no raw log-param
coordinates.

Targets: [800, 1000, 1200, 1500, 2000] Hz  (all converge with 500 iterations)
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
_N_ITER  = 500
_MAX_OBJ = 0.05   # skip entries with > 5% error


# ── helpers ───────────────────────────────────────────────────────────────────


def _run_and_get_entry(target_hz: float, n_iter: int = _N_ITER) -> _MemoryEntry | None:
    mapper    = RLCDesignMapper(hdv_dim=64, seed=42)
    evaluator = RLCEvaluator(max_Q=10.0, max_energy_loss=0.5)
    memory    = KoopmanExperienceMemory()
    opt = ConstrainedHDVOptimizer(mapper, evaluator, memory, n_iter=n_iter, seed=0)
    result = opt.optimize(target_hz, pilot_steps=0)
    if not memory._entries:
        return None
    if result.objective > _MAX_OBJ:
        return None
    return memory._entries[0]


def _query_distance(inv_a, inv_b) -> float:
    """L2 on to_query_vector() — domain-invariant 3-D distance."""
    return float(np.linalg.norm(inv_a.to_query_vector() - inv_b.to_query_vector()))


def _retrieval_distance(inv_a, inv_b) -> float:
    """L2 on full to_retrieval_vector()."""
    return float(np.linalg.norm(inv_a.to_retrieval_vector() - inv_b.to_retrieval_vector()))


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def entries():
    result = {}
    for f in _TARGETS:
        result[f] = _run_and_get_entry(f)
    return result


# ── Sanity ────────────────────────────────────────────────────────────────────


def test_entries_stored(entries):
    stored = [f for f, e in entries.items() if e is not None]
    assert len(stored) >= 4, (
        f"Only {len(stored)} / {len(_TARGETS)} Koopman fits succeeded: {stored}"
    )


# ── Phase 1 gate: spectral monotonicity (query distance) ──────────────────────


def test_spectral_monotonicity_query_vector(entries):
    """
    Spearman ρ between pairwise to_query_vector() distance and pairwise
    frequency difference must exceed 0.8.

    The query vector is 3-D (log_ω₀_norm, log_Q_norm, ζ) — almost entirely
    determined by ω₀, so ρ should be high.
    """
    valid = [(f, e) for f, e in entries.items() if e is not None]
    assert len(valid) >= 4, "Too few stored entries."

    freqs = [f for f, _ in valid]
    qvecs = [e.invariant.to_query_vector() for _, e in valid]

    dists = []
    diffs = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            dists.append(np.linalg.norm(qvecs[i] - qvecs[j]))
            diffs.append(abs(freqs[i] - freqs[j]))

    corr, pvalue = spearmanr(dists, diffs)
    print(f"\n  [query_vector] Spearman ρ = {corr:.4f}  (p={pvalue:.4f})")

    assert corr > 0.8, (
        f"Query-vector spectral monotonicity failed: ρ = {corr:.4f}  (need > 0.8). "
        f"The domain-invariant (ω₀, Q, ζ) does not reflect frequency ordering."
    )


def test_spectral_monotonicity_retrieval_vector(entries):
    """
    Full to_retrieval_vector() distance should also correlate with frequency.
    Gate is 0.7 (same as original test_spectral_stability.py).
    """
    valid = [(f, e) for f, e in entries.items() if e is not None]
    assert len(valid) >= 4

    freqs = [f for f, _ in valid]
    rvecs = [e.invariant.to_retrieval_vector() for _, e in valid]

    dists = [np.linalg.norm(rvecs[i] - rvecs[j])
             for i in range(len(valid)) for j in range(i + 1, len(valid))]
    diffs = [abs(freqs[i] - freqs[j])
             for i in range(len(valid)) for j in range(i + 1, len(valid))]

    corr, pvalue = spearmanr(dists, diffs)
    print(f"  [retrieval_vector] Spearman ρ = {corr:.4f}  (p={pvalue:.4f})")

    assert corr > 0.7, (
        f"Retrieval-vector spectral monotonicity failed: ρ = {corr:.4f}  (need > 0.7)."
    )


# ── Structural checks ─────────────────────────────────────────────────────────


def test_adjacent_closer_than_extreme(entries):
    """dist(800, 1000) < dist(800, 2000) in query space."""
    e800  = entries.get(800.0)
    e1000 = entries.get(1000.0)
    e2000 = entries.get(2000.0)
    if any(e is None for e in [e800, e1000, e2000]):
        pytest.skip("Missing entries.")
    d_near = _query_distance(e800.invariant, e1000.invariant)
    d_far  = _query_distance(e800.invariant, e2000.invariant)
    print(f"\n  dist_q(800,1000)={d_near:.4f}  dist_q(800,2000)={d_far:.4f}")
    assert d_near < d_far, (
        f"Adjacent targets (800, 1000 Hz) query distance ({d_near:.4f}) "
        f"≥ extreme pair (800, 2000 Hz) ({d_far:.4f})."
    )


def test_log_omega0_norm_monotone(entries):
    """log_omega0_norm of stored invariants must increase with target frequency."""
    valid = sorted([(f, e) for f, e in entries.items() if e is not None])
    lnorms = [e.invariant.log_omega0_norm for _, e in valid]
    freqs  = [f for f, _ in valid]
    for i in range(len(lnorms) - 1):
        assert lnorms[i] < lnorms[i + 1], (
            f"log_omega0_norm not monotone: {freqs[i]} Hz → {lnorms[i]:.4f} "
            f">= {freqs[i+1]} Hz → {lnorms[i+1]:.4f}"
        )


def test_self_distance_is_zero(entries):
    """Sanity: to_query_vector() distance to itself is 0."""
    for f, e in entries.items():
        if e is None:
            continue
        qv = e.invariant.to_query_vector()
        assert np.linalg.norm(qv - qv) == pytest.approx(0.0)


def test_domain_stored_as_rlc(entries):
    """All stored experiences should have domain='rlc'."""
    for f, e in entries.items():
        if e is None:
            continue
        assert e.experience.domain == "rlc", (
            f"Entry for {f} Hz has domain='{e.experience.domain}', expected 'rlc'."
        )


# ── Diagnostic: full distance matrices ────────────────────────────────────────


def test_print_query_distance_matrix(entries, capsys):
    """Print the 3-D query distance matrix for inspection (always passes)."""
    valid = sorted([(f, e) for f, e in entries.items() if e is not None])
    freqs = [f for f, _ in valid]
    qvecs = [e.invariant.to_query_vector() for _, e in valid]
    n = len(valid)
    header = "     " + "".join(f"{int(f):>8}" for f in freqs)
    print(f"\n  Pairwise query-vector (3-D) L2 distances:")
    print(f"  {header}")
    for i in range(n):
        row = f"  {int(freqs[i]):>4}:"
        for j in range(n):
            d = np.linalg.norm(qvecs[i] - qvecs[j])
            row += f" {d:>7.4f}"
        print(row)

    print(f"\n  log_omega0_norm values:")
    for f, e in valid:
        print(f"    {int(f):>5} Hz → log_ω₀_norm={e.invariant.log_omega0_norm:.4f}  "
              f"log_Q_norm={e.invariant.log_Q_norm:.4f}  "
              f"ζ={e.invariant.damping_ratio:.4f}")
