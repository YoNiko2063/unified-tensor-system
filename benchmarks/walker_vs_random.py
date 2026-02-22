#!/usr/bin/env python3
"""
benchmarks/walker_vs_random.py — Product validation experiment.

Compares ParameterSpaceWalker against two baselines in 3D RLC parameter space.

Domain:
  Series RLC circuit — R (Ω), L (H), C (F)
  x = [v_C, i_L],  ẋ = [i_L/C, (-v_C - R·i_L)/L]

Target region (engineering design band):
  ζ ∈ [0.10, 0.30]          (underdamped with meaningful damping)
  f₀ ∈ [1 kHz, 100 kHz]    (natural frequency)
  Combined: a curved surface in 3D log-(R,L,C) space.

Experiment protocol:
  Phase 1 — Shared warmup: scan_random(N=200) over full param_bounds.
             Both random baselines and the walker observe the same landscape.
             Walker additionally trains on the warmup trajectory (50 epochs).
  Phase 2 — Benchmark: 100 independent trials, each starting from a
             random θ₀ outside the target region.
             Each method gets the same θ₀ and up to max_iter steps.
             Stop on first entry into target region.

Bias control:
  - θ₀ sampled outside target (reject sampling).
  - Walker trained on transitions (dissonance-based), not point labels.
  - All methods pay the same 200-point warmup cost.

Methods:
  random_search  — uniform log-space sample per step (no memory, no geometry)
  local_walk     — Gaussian step in log-space (locality, no spectral guidance)
  walker         — ParameterSpaceWalker (learned spectral manifold topology)

Metrics (over successful trials only):
  mean / median / p95 iterations, failure rate, speedup ratio.

Outputs:
  benchmarks/results/walker_vs_random.json
  benchmarks/results/walker_vs_random_cdf.png
  benchmarks/results/walker_vs_random_box.png

Usage:
  python benchmarks/walker_vs_random.py
  python benchmarks/walker_vs_random.py --n-trials 50 --max-iter 300 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tensor.harmonic_atlas import HarmonicAtlas
from tensor.eigenspace_mapper import EigenspaceMapper
from tensor.parameter_space_walker import ParameterSpaceWalker
from tensor.spectral_path import DissonanceMetric


# ── Parameter space ────────────────────────────────────────────────────────────

THETA_KEYS = ['R', 'L', 'C']

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'R': (1.0,   1_000.0),
    'L': (1e-6,  1e-2),
    'C': (1e-9,  1e-6),
}

# Engineering target
ZETA_LO,    ZETA_HI    = 0.10, 0.30
FREQ_LO_HZ, FREQ_HI_HZ = 1e3,  1e5


# ── Analytical helpers (no mapper overhead per evaluation step) ────────────────

def rlc_eigvals(theta: dict) -> np.ndarray:
    """Closed-form eigenvalues of series RLC at equilibrium (origin)."""
    R, L, C = theta['R'], theta['L'], theta['C']
    a = -R / (2.0 * L)
    disc = (R / (2.0 * L)) ** 2 - 1.0 / (L * C)
    if disc >= 0.0:
        b = float(np.sqrt(disc))
        return np.array([a + b, a - b], dtype=complex)
    b = float(np.sqrt(-disc))
    return np.array([a + 1j * b, a - 1j * b], dtype=complex)


def rlc_zeta(theta: dict) -> float:
    """Damping ratio ζ = (R/2) · √(C/L)."""
    R, L, C = theta['R'], theta['L'], theta['C']
    return (R / 2.0) * float(np.sqrt(C / L))


def rlc_natural_freq_hz(theta: dict) -> float:
    """Undamped natural frequency f₀ = 1 / (2π·√(LC))."""
    L, C = theta['L'], theta['C']
    return 1.0 / (2.0 * np.pi * float(np.sqrt(L * C)))


def in_target(theta: dict) -> bool:
    """True iff theta satisfies ζ ∈ [ZETA_LO, ZETA_HI] and f₀ ∈ [FREQ_LO_HZ, FREQ_HI_HZ]."""
    return (ZETA_LO <= rlc_zeta(theta) <= ZETA_HI
            and FREQ_LO_HZ <= rlc_natural_freq_hz(theta) <= FREQ_HI_HZ)


# ── Sampling utilities ─────────────────────────────────────────────────────────

def log_sample(bounds: Dict[str, Tuple[float, float]], rng: np.random.Generator) -> dict:
    """Uniform sample in log₁₀ space — gives equal coverage across decades."""
    return {
        k: float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
        for k, (lo, hi) in bounds.items()
    }


def clip_to_bounds(theta: dict, bounds: Dict[str, Tuple[float, float]]) -> dict:
    return {k: float(np.clip(v, bounds[k][0], bounds[k][1])) for k, v in theta.items()}


def sample_outside_target(rng: np.random.Generator, max_tries: int = 2000) -> dict:
    """Reject-sample a starting point that is strictly outside the target region."""
    for _ in range(max_tries):
        theta = log_sample(PARAM_BOUNDS, rng)
        if not in_target(theta):
            return theta
    # Fallback: underdamped high-frequency point (outside target by construction)
    return {'R': 1.0, 'L': 1e-6, 'C': 1e-9}


# ── Dissonance function (Walker training signal) ───────────────────────────────

_dissonance = DissonanceMetric(K=10)


def _tau(ev_a: np.ndarray, ev_b: np.ndarray) -> float:
    """τ between dominant imaginary frequencies of two eigenvalue arrays."""
    omega_a = float(np.max(np.abs(np.imag(ev_a)))) if len(ev_a) else 0.0
    omega_b = float(np.max(np.abs(np.imag(ev_b)))) if len(ev_b) else 0.0
    return _dissonance.compute(omega_a, omega_b)


# ── Search result ──────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    method:       str
    found:        bool
    n_iters:      int      # steps to hit target; max_iter if not found
    final_zeta:   float
    final_freq_hz: float
    tau_trajectory: List[float]   # per-step dissonance (diagnostic)


# ── Baseline 1: random search ──────────────────────────────────────────────────

def run_random_search(
    theta0: dict,
    max_iter: int,
    rng: np.random.Generator,
) -> SearchResult:
    """
    Pure random search: sample θ uniformly from full param_bounds each step.
    No memory. No spatial structure. Lower bound on search efficiency.
    """
    tau_traj: List[float] = []
    prev_ev = rlc_eigvals(theta0)
    theta = theta0

    for i in range(max_iter):
        theta = log_sample(PARAM_BOUNDS, rng)
        ev = rlc_eigvals(theta)
        tau_traj.append(_tau(prev_ev, ev))
        prev_ev = ev
        if in_target(theta):
            return SearchResult(
                'random_search', True, i + 1,
                rlc_zeta(theta), rlc_natural_freq_hz(theta), tau_traj,
            )

    return SearchResult(
        'random_search', False, max_iter,
        rlc_zeta(theta), rlc_natural_freq_hz(theta), tau_traj,
    )


# ── Baseline 2: local Gaussian walk in log-space ───────────────────────────────

def run_local_walk(
    theta0: dict,
    max_iter: int,
    rng: np.random.Generator,
    log_sigma: float = 0.12,
) -> SearchResult:
    """
    Local random walk: Gaussian perturbation in log₁₀ space.
    Explores the neighbourhood of the current point. No gradient, no spectral
    guidance. Upper bound for locality-only search.
    """
    theta = dict(theta0)
    prev_ev = rlc_eigvals(theta)
    tau_traj: List[float] = []

    for i in range(max_iter):
        theta_new = {}
        for k, (lo, hi) in PARAM_BOUNDS.items():
            log_v = np.log10(theta[k]) + rng.normal(0.0, log_sigma)
            log_v = float(np.clip(log_v, np.log10(lo), np.log10(hi)))
            theta_new[k] = float(10 ** log_v)
        ev = rlc_eigvals(theta_new)
        tau_traj.append(_tau(prev_ev, ev))
        theta = theta_new
        prev_ev = ev
        if in_target(theta):
            return SearchResult(
                'local_walk', True, i + 1,
                rlc_zeta(theta), rlc_natural_freq_hz(theta), tau_traj,
            )

    return SearchResult(
        'local_walk', False, max_iter,
        rlc_zeta(theta), rlc_natural_freq_hz(theta), tau_traj,
    )


# ── Method 3: ParameterSpaceWalker ────────────────────────────────────────────

def run_walker_search(
    theta0: dict,
    walker: ParameterSpaceWalker,
    max_iter: int,
    step_scale: float = 0.15,
) -> SearchResult:
    """
    Walker-guided search: at each step, feed current spectral state into the
    trained MLP to predict Δθ.  Step is scaled by step_scale and clipped to
    param_bounds.  Deterministic (no rng).
    """
    theta = dict(theta0)
    ev = rlc_eigvals(theta)
    prev_ev = ev.copy()
    tau_traj: List[float] = []

    for i in range(max_iter):
        delta_norm = walker.predict_step(theta, ev, regime='lca')
        delta_phys = walker._denormalize_delta(delta_norm) * step_scale
        theta_new = {
            k: theta[k] + delta_phys[j]
            for j, k in enumerate(THETA_KEYS)
        }
        theta_new = clip_to_bounds(theta_new, PARAM_BOUNDS)
        ev_new = rlc_eigvals(theta_new)
        tau_traj.append(_tau(prev_ev, ev_new))
        theta = theta_new
        ev = ev_new
        prev_ev = ev_new
        if in_target(theta):
            return SearchResult(
                'walker', True, i + 1,
                rlc_zeta(theta), rlc_natural_freq_hz(theta), tau_traj,
            )

    return SearchResult(
        'walker', False, max_iter,
        rlc_zeta(theta), rlc_natural_freq_hz(theta), tau_traj,
    )


# ── Discrete harmonic jumper ───────────────────────────────────────────────────

class RLCDiscreteJumper:
    """
    Generates physically grounded candidate (R, L, C) triples via two strategies.

    Strategy A — Rational ratio jumps  (harmonically grounded in atlas structure)
      Given ω₀ = |Im(λ)|, compute ω' = (p/q)·ω₀ for p,q ∈ {1..K}.
      Invert the RLC equations (anchoring current L):
        C' = 1 / (L · ω'²)
        R' = 2·ζ'·L·ω'
      This produces candidates whose frequencies are rationally related to the
      current operating point — the same p:q structure used by DomainCanonicalizer.

    Strategy B — Direct target-range candidates  (goal-directed)
      Enumerate f₀ ∈ [1kHz, 100kHz] log-spaced × ζ ∈ [0.10, 0.30] grid.
      Compute (R', C') analytically for each L' in log-spaced L grid.
      Used when the starting frequency is so far from the target band that
      no small rational ratio bridges the gap.

    Candidate selection: prefer candidates already in target; otherwise rank
    by combined normalised distance (ζ-axis + log-frequency-axis).
    """

    _ZETA_TARGETS = (0.10, 0.15, 0.20, 0.25, 0.30)
    # Log-spaced f₀ anchors inside the target band [1kHz, 100kHz]
    _F0_ANCHORS_HZ = (1e3, 3.16e3, 1e4, 3.16e4, 1e5)
    # Log-spaced L values to vary in Strategy B
    _L_ANCHORS = tuple(float(v) for v in
                       np.logspace(np.log10(PARAM_BOUNDS['L'][0]),
                                   np.log10(PARAM_BOUNDS['L'][1]), 5))

    def __init__(self, K: int = 8):
        self.K = K
        ratios: set = set()
        for p in range(1, K + 1):
            for q in range(1, K + 1):
                ratios.add(p / q)
        self._ratios = sorted(ratios)

    # ── Candidate generation ──────────────────────────────────────────────────

    def candidates(self, theta: dict, eigvals: np.ndarray) -> List[dict]:
        """All valid candidates from both strategies."""
        cands: List[dict] = []
        cands.extend(self._strategy_a(theta, eigvals))
        cands.extend(self._strategy_b())
        return cands

    def _strategy_a(self, theta: dict, eigvals: np.ndarray) -> List[dict]:
        """Rational-ratio jumps relative to current dominant frequency."""
        omega0 = float(np.max(np.abs(np.imag(eigvals))))
        if omega0 < 1.0:       # purely real spectrum — no oscillation to pivot from
            return []
        L = theta['L']
        out: List[dict] = []
        for ratio in self._ratios:
            omega_prime = ratio * omega0
            if omega_prime < 1.0:
                continue
            C_prime = 1.0 / (L * omega_prime ** 2)
            for zeta in self._ZETA_TARGETS:
                R_prime = 2.0 * zeta * L * omega_prime
                cand = {'R': R_prime, 'L': L, 'C': C_prime}
                if self._in_bounds(cand):
                    out.append(cand)
        return out

    def _strategy_b(self) -> List[dict]:
        """Direct target-range candidates from analytic inverse mapping."""
        out: List[dict] = []
        for f0_hz in self._F0_ANCHORS_HZ:
            omega_target = 2.0 * np.pi * f0_hz
            for L_prime in self._L_ANCHORS:
                C_prime = 1.0 / (L_prime * omega_target ** 2)
                for zeta in self._ZETA_TARGETS:
                    R_prime = 2.0 * zeta * L_prime * omega_target
                    cand = {'R': R_prime, 'L': L_prime, 'C': C_prime}
                    if self._in_bounds(cand):
                        out.append(cand)
        return out

    # ── Best-candidate selection ───────────────────────────────────────────────

    def best_candidate(
        self, theta: dict, eigvals: np.ndarray
    ) -> Optional[dict]:
        """
        Return the best candidate for a discrete jump.

        Priority:
          1. Candidates already satisfying in_target()  (random among them)
          2. Candidate minimising normalised distance to target centre.
        """
        cands = self.candidates(theta, eigvals)
        if not cands:
            return None
        in_tgt = [c for c in cands if in_target(c)]
        if in_tgt:
            # Multiple valid candidates → pick midpoint of ζ-target
            return min(in_tgt, key=lambda c: abs(rlc_zeta(c) - 0.20))
        # No in-target candidate: pick closest by normalised distance
        return min(cands, key=self._dist_to_target)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _in_bounds(cand: dict) -> bool:
        return all(
            PARAM_BOUNDS[k][0] <= cand[k] <= PARAM_BOUNDS[k][1]
            for k in THETA_KEYS
        )

    @staticmethod
    def _dist_to_target(cand: dict) -> float:
        zeta = rlc_zeta(cand)
        freq = rlc_natural_freq_hz(cand)
        z_dist = max(0.0, ZETA_LO - zeta) + max(0.0, zeta - ZETA_HI)
        # Log-distance for frequency (spans 2 decades)
        if freq < FREQ_LO_HZ:
            f_dist = np.log10(FREQ_LO_HZ / max(freq, 1.0))
        elif freq > FREQ_HI_HZ:
            f_dist = np.log10(freq / FREQ_HI_HZ)
        else:
            f_dist = 0.0
        # Normalise ζ-axis by band width
        return z_dist / (ZETA_HI - ZETA_LO) + f_dist


# ── Method 4: Hybrid walker + discrete fallback ────────────────────────────────

def run_hybrid_search(
    theta0: dict,
    walker: ParameterSpaceWalker,
    max_iter: int,
    step_scale: float = 0.15,
    stagnation_window: int = 12,
) -> SearchResult:
    """
    Hybrid search: continuous Walker with discrete harmonic fallback.

    Mode switching logic:
      - Count how many consecutive steps the walker has been clipping against
        a parameter boundary (a sign that it is stuck in a dead direction).
      - After `stagnation_window` boundary-hitting steps, perform one discrete
        jump to the best candidate from RLCDiscreteJumper.
      - After a discrete jump, reset the stagnation counter and continue in
        continuous mode from the new position.

    The discrete jumper uses:
      A) Rational-ratio frequency jumps (atlas-consistent)
      B) Direct target-range analytic inverse (goal-directed fallback)
    """
    jumper = RLCDiscreteJumper()
    theta = dict(theta0)
    ev = rlc_eigvals(theta)
    prev_ev = ev.copy()
    tau_traj: List[float] = []
    boundary_count = 0         # consecutive steps clipping boundary

    for i in range(max_iter):
        # ── Continuous walker step ─────────────────────────────────────────
        delta_norm = walker.predict_step(theta, ev, regime='lca')
        delta_phys = walker._denormalize_delta(delta_norm) * step_scale
        theta_raw = {k: theta[k] + delta_phys[j] for j, k in enumerate(THETA_KEYS)}
        theta_clipped = clip_to_bounds(theta_raw, PARAM_BOUNDS)

        # Detect boundary saturation: clipped value differs from raw
        clipped = any(
            abs(theta_clipped[k] - theta_raw[k]) > 1e-12 * abs(theta_raw[k] + 1e-15)
            for k in THETA_KEYS
        )
        boundary_count = boundary_count + 1 if clipped else max(0, boundary_count - 1)

        theta_new = theta_clipped
        ev_new = rlc_eigvals(theta_new)

        # ── Discrete fallback when stagnated ──────────────────────────────
        if boundary_count >= stagnation_window:
            cand = jumper.best_candidate(theta, ev)
            if cand is not None:
                theta_new = cand
                ev_new = rlc_eigvals(theta_new)
                boundary_count = 0           # reset after successful jump

        tau_traj.append(_tau(prev_ev, ev_new))
        theta = theta_new
        ev = ev_new
        prev_ev = ev_new

        if in_target(theta):
            return SearchResult(
                'hybrid', True, i + 1,
                rlc_zeta(theta), rlc_natural_freq_hz(theta), tau_traj,
            )

    return SearchResult(
        'hybrid', False, max_iter,
        rlc_zeta(theta), rlc_natural_freq_hz(theta), tau_traj,
    )


# ── Phase 1: warmup scan + walker training ─────────────────────────────────────

def warmup_and_train(n_warmup: int, seed: int) -> Tuple[ParameterSpaceWalker, dict]:
    """
    Shared warmup: grid scan over full param_bounds → train walker.

    Grid scan (rather than random scan) ensures consecutive result pairs are
    actual parameter-space neighbours — not random long-range jumps.  This
    gives the Walker coherent local-gradient training signal: the Δθ between
    two adjacent grid cells represents a meaningful directional example.

    n_warmup is mapped to the closest perfect cube: n_per_axis = round(n^(1/3)).
    E.g. n_warmup=200 → n_per_axis=6 → 216 grid points.
    """
    atlas = HarmonicAtlas()

    def _factory(theta):
        R, L, C = theta['R'], theta['L'], theta['C']
        def f(x):
            v_C, i_L = x
            return np.array([i_L / C, (-v_C - R * i_L) / L])
        return f

    mapper = EigenspaceMapper(
        system_factory=_factory,
        atlas=atlas,
        n_states=2,
        n_samples=20,
        sample_radius=0.005,
        rng_seed=seed,
    )

    # Grid in log-space: construct per-axis linspaces in log10, then exponentiate.
    # scan_grid accepts linear values, so we build a log-uniform grid manually
    # by sampling n_per_axis log-evenly spaced values per parameter.
    n_per_axis = max(2, round(n_warmup ** (1.0 / 3.0)))
    log_ranges: Dict[str, Tuple[float, float]] = {
        k: (float(np.log10(lo)), float(np.log10(hi)))
        for k, (lo, hi) in PARAM_BOUNDS.items()
    }
    import itertools
    grid_thetas = []
    axes = {k: np.logspace(log_ranges[k][0], log_ranges[k][1], n_per_axis)
            for k in THETA_KEYS}
    for vals in itertools.product(*[axes[k] for k in THETA_KEYS]):
        grid_thetas.append(dict(zip(THETA_KEYS, vals)))

    warmup_results = []
    for theta in grid_thetas:
        try:
            warmup_results.append(mapper.map_point(theta, 'rlc_warmup'))
        except Exception:
            pass

    report = mapper.scan_report(warmup_results)

    walker = ParameterSpaceWalker(
        theta_keys=THETA_KEYS,
        param_bounds=PARAM_BOUNDS,
        hidden=128,
        dissonance_quantile=0.25,
        seed=seed,
    )
    walker.record_from_scan(warmup_results, _tau)
    final_loss = walker.train(n_epochs=100, lr=1e-3, batch_size=32)

    # Serialise scan_report: tuples → lists, numpy → Python scalars
    def _clean(v):
        if isinstance(v, tuple):
            return list(v)
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    return walker, {
        'n_warmup_requested': n_warmup,
        'n_warmup_obtained': len(warmup_results),
        'scan_report': {k: _clean(v) for k, v in report.items()},
        'walker_final_loss': float(final_loss),
        'walker_buffer_size': walker.buffer_size(),
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def aggregate(results: List[SearchResult]) -> dict:
    iters = [r.n_iters for r in results if r.found]
    n_total = len(results)
    n_success = len(iters)
    if not iters:
        return {'n_success': 0, 'n_total': n_total, 'failure_rate': 1.0}
    return {
        'n_success':    n_success,
        'n_total':      n_total,
        'failure_rate': round(1.0 - n_success / n_total, 4),
        'mean_iters':   round(float(np.mean(iters)),          2),
        'median_iters': round(float(np.median(iters)),        2),
        'p75_iters':    round(float(np.percentile(iters, 75)), 2),
        'p95_iters':    round(float(np.percentile(iters, 95)), 2),
        'min_iters':    int(np.min(iters)),
        'max_iters':    int(np.max(iters)),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

_COLORS = {
    'random_search': '#e74c3c',
    'local_walk':    '#f39c12',
    'walker':        '#2ecc71',
    'hybrid':        '#2980b9',
}
_LABELS = {
    'random_search': 'Random Search',
    'local_walk':    'Local Walk',
    'walker':        'Walker',
    'hybrid':        'Hybrid (Walker + Discrete)',
}
_METHOD_ORDER = ['random_search', 'local_walk', 'walker', 'hybrid']


def plot_results(
    results_by_method: Dict[str, List[SearchResult]],
    out_dir: str,
    max_iter: int,
) -> List[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not available — skipping")
        return []

    saved: List[str] = []

    # ── CDF of iterations ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in _METHOD_ORDER:
        results = results_by_method.get(method, [])
        iters = sorted(r.n_iters for r in results if r.found)
        if not iters:
            continue
        n_total = len(results)
        xs = [0] + iters + [max_iter]
        ys = [0.0] + [i / n_total for i in range(1, len(iters) + 1)] + [len(iters) / n_total]
        ax.step(xs, ys, where='post',
                color=_COLORS.get(method, 'gray'),
                label=_LABELS.get(method, method),
                linewidth=2)

    ax.set_xlabel('Iterations to reach target', fontsize=12)
    ax.set_ylabel('Fraction of trials succeeded', fontsize=12)
    ax.set_title('Walker vs. Baselines — Iteration Efficiency (CDF)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_iter)
    ax.set_ylim(0.0, 1.05)
    cdf_path = os.path.join(out_dir, 'walker_vs_random_cdf.png')
    fig.tight_layout()
    fig.savefig(cdf_path, dpi=150)
    plt.close(fig)
    saved.append(cdf_path)

    # ── Box plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    data, tick_labels, colors_list = [], [], []
    for method in _METHOD_ORDER:
        iters = [r.n_iters for r in results_by_method.get(method, []) if r.found]
        if iters:
            data.append(iters)
            tick_labels.append(_LABELS.get(method, method))
            colors_list.append(_COLORS.get(method, 'gray'))

    if data:
        bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5)
        for patch, c in zip(bp['boxes'], colors_list):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticklabels(tick_labels, fontsize=11)

    ax.set_ylabel('Iterations to reach target', fontsize=12)
    ax.set_title('Search Effort Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    box_path = os.path.join(out_dir, 'walker_vs_random_box.png')
    fig.tight_layout()
    fig.savefig(box_path, dpi=150)
    plt.close(fig)
    saved.append(box_path)

    return saved


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--n-trials',    type=int,   default=100,
                        help='Number of benchmark trials (default: 100)')
    parser.add_argument('--max-iter',    type=int,   default=500,
                        help='Max iterations per trial (default: 500)')
    parser.add_argument('--warmup',      type=int,   default=200,
                        help='Warmup scan size (default: 200)')
    parser.add_argument('--seed',        type=int,   default=42,
                        help='Master RNG seed (default: 42)')
    parser.add_argument('--step-scale',  type=float, default=0.15,
                        help='Walker step scale (default: 0.15)')
    parser.add_argument('--log-sigma',   type=float, default=0.12,
                        help='Local walk log10 step σ (default: 0.12)')
    parser.add_argument('--out',         type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
                        help='Output directory for JSON + plots')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=== Walker vs. Random Benchmark (product validation) ===")
    print(f"  trials={args.n_trials}  max_iter={args.max_iter}  "
          f"warmup={args.warmup}  seed={args.seed}")
    print(f"  target: ζ ∈ [{ZETA_LO}, {ZETA_HI}]  "
          f"f₀ ∈ [{FREQ_LO_HZ/1e3:.0f}kHz, {FREQ_HI_HZ/1e3:.0f}kHz]")

    # ── Estimate target fraction ──────────────────────────────────────────────
    _est_rng = np.random.default_rng(args.seed + 9999)
    _est_n   = 5000
    target_frac = sum(1 for _ in range(_est_n)
                      if in_target(log_sample(PARAM_BOUNDS, _est_rng))) / _est_n
    print(f"  target region: ~{target_frac:.2%} of log-uniform param space")
    if target_frac < 0.002:
        print("  [warn] target very sparse — consider relaxing constraints for first run")

    # ── Phase 1: warmup + train ───────────────────────────────────────────────
    print(f"\n[Phase 1] Warmup scan ({args.warmup} pts) + Walker training (50 epochs)...")
    t0 = time.time()
    walker, warmup_stats = warmup_and_train(args.warmup, args.seed)
    t1 = time.time()
    print(f"  {warmup_stats['n_warmup_obtained']} points obtained in {t1-t0:.1f}s")
    print(f"  Walker training loss: {warmup_stats['walker_final_loss']:.6f}")
    print(f"  Buffer size (training transitions): {warmup_stats['walker_buffer_size']}")

    # ── Phase 2: benchmark trials ─────────────────────────────────────────────
    print(f"\n[Phase 2] {args.n_trials} trials × 3 methods...")
    results_by_method: Dict[str, List[SearchResult]] = {
        'random_search': [],
        'local_walk':    [],
        'walker':        [],
        'hybrid':        [],
    }
    all_trials = []

    for trial_idx in range(args.n_trials):
        t_seed = args.seed * 100_000 + trial_idx
        theta0 = sample_outside_target(np.random.default_rng(t_seed))

        r_rand = run_random_search(theta0, args.max_iter,
                                   np.random.default_rng(t_seed + 1))
        r_walk = run_local_walk   (theta0, args.max_iter,
                                   np.random.default_rng(t_seed + 2),
                                   log_sigma=args.log_sigma)
        r_wlkr = run_walker_search(theta0, walker, args.max_iter,
                                   step_scale=args.step_scale)
        r_hybr = run_hybrid_search(theta0, walker, args.max_iter,
                                   step_scale=args.step_scale,
                                   stagnation_window=5)

        for r in (r_rand, r_walk, r_wlkr, r_hybr):
            results_by_method[r.method].append(r)

        all_trials.append({
            'trial':   trial_idx,
            'theta_0': {k: float(v) for k, v in theta0.items()},
            'zeta_0':  round(rlc_zeta(theta0),           4),
            'freq_0':  round(rlc_natural_freq_hz(theta0), 2),
            'random_search': {'found': r_rand.found, 'n_iters': r_rand.n_iters,
                              'final_zeta': round(r_rand.final_zeta, 4),
                              'final_freq_hz': round(r_rand.final_freq_hz, 2)},
            'local_walk':    {'found': r_walk.found, 'n_iters': r_walk.n_iters,
                              'final_zeta': round(r_walk.final_zeta, 4),
                              'final_freq_hz': round(r_walk.final_freq_hz, 2)},
            'walker':        {'found': r_wlkr.found, 'n_iters': r_wlkr.n_iters,
                              'final_zeta': round(r_wlkr.final_zeta, 4),
                              'final_freq_hz': round(r_wlkr.final_freq_hz, 2)},
            'hybrid':        {'found': r_hybr.found, 'n_iters': r_hybr.n_iters,
                              'final_zeta': round(r_hybr.final_zeta, 4),
                              'final_freq_hz': round(r_hybr.final_freq_hz, 2)},
        })

        if (trial_idx + 1) % 10 == 0:
            print(f"  [{trial_idx+1:3d}/{args.n_trials}]  "
                  f"rand: {r_rand.n_iters:3d}({'✓' if r_rand.found else '✗'})  "
                  f"walk: {r_walk.n_iters:3d}({'✓' if r_walk.found else '✗'})  "
                  f"wlkr: {r_wlkr.n_iters:3d}({'✓' if r_wlkr.found else '✗'})  "
                  f"hybr: {r_hybr.n_iters:3d}({'✓' if r_hybr.found else '✗'})")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = {m: aggregate(results_by_method[m]) for m in _METHOD_ORDER}

    for method in ('walker', 'hybrid'):
        m_mean = summary[method].get('mean_iters')
        if m_mean and m_mean > 0:
            for base in ('random_search', 'local_walk'):
                base_mean = summary[base].get('mean_iters')
                if base_mean:
                    summary[method][f'speedup_vs_{base}'] = round(base_mean / m_mean, 2)

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    hdr = f"{'Method':<16}  {'Success':>8}  {'Mean':>7}  {'Median':>7}  {'p95':>7}  {'Fail%':>6}"
    print(hdr)
    print('-' * len(hdr))
    for m in _METHOD_ORDER:
        s = summary[m]
        ns = s.get('n_success', 0)
        nt = s.get('n_total', args.n_trials)
        print(f"{m:<16}  {ns:>4}/{nt:<4}  "
              f"{s.get('mean_iters',   float('nan')):>7.1f}  "
              f"{s.get('median_iters', float('nan')):>7.1f}  "
              f"{s.get('p95_iters',    float('nan')):>7.1f}  "
              f"{s.get('failure_rate', 1.0):>5.1%}")

    print()
    for method in ('walker', 'hybrid'):
        for base in ('random_search', 'local_walk'):
            key = f'speedup_vs_{base}'
            if key in summary[method]:
                print(f"Speedup ({method} vs {base}): {summary[method][key]}×")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        'config': {
            'n_trials':    args.n_trials,
            'max_iter':    args.max_iter,
            'warmup':      args.warmup,
            'seed':        args.seed,
            'step_scale':  args.step_scale,
            'log_sigma':   args.log_sigma,
            'target': {
                'zeta_lo':     ZETA_LO,
                'zeta_hi':     ZETA_HI,
                'freq_lo_hz':  FREQ_LO_HZ,
                'freq_hi_hz':  FREQ_HI_HZ,
            },
            'param_bounds': {k: list(v) for k, v in PARAM_BOUNDS.items()},
        },
        'target_fraction_estimate': round(target_frac, 4),
        'warmup_stats': warmup_stats,
        'summary':  summary,
        'trials':   all_trials,
    }

    def _default(obj):
        if isinstance(obj, (np.integer,)):      return int(obj)
        if isinstance(obj, (np.floating,)):     return float(obj)
        if isinstance(obj, np.ndarray):         return obj.tolist()
        if isinstance(obj, tuple):              return list(obj)
        return str(obj)

    json_path = os.path.join(args.out, 'walker_vs_random.json')
    with open(json_path, 'w') as fh:
        json.dump(output, fh, indent=2, default=_default)
    print(f"\nResults: {json_path}")

    for p in plot_results(results_by_method, args.out, args.max_iter):
        print(f"Plot:    {p}")


if __name__ == '__main__':
    main()
