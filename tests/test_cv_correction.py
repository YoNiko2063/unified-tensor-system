"""
Cross-Validation of Global Damping Correction — Test Suite.

Leave-2-out CV across C(10,2)=45 splits.  All computation is algebraic
on existing DampingSweepResult data — no new ODE evaluations.

Groups:
 1. CV mechanics (2 tests)
    — 45 splits produced, each with 8 train / 2 test generators
    — a_train > 0 for every split

 2. Stability gate (2 tests)
    — a range (max-min) / mean < 10%  (±5% criterion)
    — a_std / a_mean < 0.05  (coefficient of variation < 5%)

 3. Generalisation (2 tests)
    — max corrected error on held-out generators < 5% for every split
    — mean corrected error on held-out generators < 3% for every split

 4. Full CV table (1 test, always prints)
    — print all 45 splits + summary, assert stability gate
"""

from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from optimization.ieee39_benchmark import (
    IEEE39_GENERATORS,
    compute_cv_summary,
    cross_validate_correction,
    print_cv_table,
    run_damping_sweep,
)

# ── Shared cache ──────────────────────────────────────────────────────────────

_ZETA_VALUES = (0.01, 0.03, 0.05, 0.10, 0.20)

_SWEEP   = None
_CV      = None


def _get_sweep():
    global _SWEEP
    if _SWEEP is None:
        _SWEEP = run_damping_sweep(zeta_values=_ZETA_VALUES)
    return _SWEEP


def _get_cv():
    global _CV
    if _CV is None:
        _CV = cross_validate_correction(_get_sweep(), n_test=2)
    return _CV


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1 — CV mechanics
# ═══════════════════════════════════════════════════════════════════════════════


def test_cv_produces_45_splits():
    """C(10,2) = 45 splits, each holding out exactly 2 generators."""
    cv = _get_cv()
    assert len(cv) == 45, f"Expected 45 splits, got {len(cv)}"
    for split in cv:
        assert len(split["test_gens"]) == 2, (
            f"Test set has {len(split['test_gens'])} generators, expected 2"
        )


def test_a_train_positive_all_splits():
    """a_train > 0 for every split (correction always in the right direction)."""
    for i, split in enumerate(_get_cv(), 1):
        assert split["a_train"] > 0.0, (
            f"Split {i} (test={split['test_gens']}): a_train={split['a_train']:.4f} ≤ 0"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2 — Stability gate
# ═══════════════════════════════════════════════════════════════════════════════


def test_a_range_within_10pct():
    """
    (max(a) - min(a)) / mean(a) < 10%  — the ±5% stability criterion.

    If `a` swings more than ±5% across generator subsets, the correction
    is data-dependent and not globally valid.  Stability here means the
    invariant manifold structure (ω₀·CCT_EAC = const) genuinely explains
    the universality — it is not a fitting artefact.
    """
    summary = compute_cv_summary(_get_cv())
    print(f"\n  a: mean={summary['a_mean']:.4f}  "
          f"min={summary['a_min']:.4f}  max={summary['a_max']:.4f}  "
          f"range={summary['a_range_pct']:.2f}%")
    assert summary["is_stable"], (
        f"a range {summary['a_range_pct']:.1f}% ≥ 10% — correction is not robust"
    )


def test_a_cv_under_5pct():
    """Coefficient of variation std(a)/mean(a) < 5%."""
    summary = compute_cv_summary(_get_cv())
    cv_pct  = summary["a_std"] / summary["a_mean"] * 100.0
    print(f"\n  a CV = {cv_pct:.2f}%  (std={summary['a_std']:.4f}  "
          f"mean={summary['a_mean']:.4f})")
    assert cv_pct < 5.0, (
        f"a coefficient of variation {cv_pct:.2f}% ≥ 5%"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3 — Generalisation
# ═══════════════════════════════════════════════════════════════════════════════


def test_max_test_error_under_5pct_all_splits():
    """
    For every split: max |corrected error| on held-out generators < 5%.

    This is the key generalisation gate: a fitted on 8 generators must
    correct the 2 unseen generators to within the same 5% tolerance.
    """
    for i, split in enumerate(_get_cv(), 1):
        assert split["max_test_err"] < 5.0, (
            f"Split {i} (test={split['test_gens']}): "
            f"max test error {split['max_test_err']:.2f}% ≥ 5%"
        )


def test_mean_test_error_under_3pct_all_splits():
    """Mean |corrected error| on held-out generators < 3% for every split."""
    for i, split in enumerate(_get_cv(), 1):
        assert split["mean_test_err"] < 3.0, (
            f"Split {i} (test={split['test_gens']}): "
            f"mean test error {split['mean_test_err']:.2f}% ≥ 3%"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 4 — Full table
# ═══════════════════════════════════════════════════════════════════════════════


def test_cv_table():
    """
    Print all 45 splits + summary.  Assert stability gate passes.

    Stability criterion: a range < 10% (±5%).
    If this passes, the correction is externalisation-ready.
    """
    cv = _get_cv()
    print_cv_table(cv)

    summary = compute_cv_summary(cv)
    assert summary["is_stable"], (
        f"CV stability gate FAILED: a range = {summary['a_range_pct']:.1f}% ≥ 10%"
    )
    assert summary["max_test_err"] < 5.0, (
        f"CV generalisation gate FAILED: max test error {summary['max_test_err']:.2f}%"
    )
