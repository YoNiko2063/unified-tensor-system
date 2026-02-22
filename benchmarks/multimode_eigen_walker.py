#!/usr/bin/env python3
"""
benchmarks/multimode_eigen_walker.py — EigenWalker on r=2 spectral manifold.

First real test of E_harm.  In all previous benchmarks (r=1, single MSD or RLC),
E_harm ≡ 0 because there is only one mode — no pairwise ratio to minimize.

At r=2 (two coupled masses):
  E_harm = min_{p,q} |ω₀_high/ω₀_low − p/q|

This creates REAL rational hypersurfaces in 4D spectral space:
  log₁₀ω_high − log₁₀ω_low = log₁₀(p/q)  (hyperplanes in log space)

The four structural questions this benchmark answers:

  Q1. Does ∇E_harm steer toward rational tubes?
      — Track ω_high/ω_low ratio over walker steps; verify convergence to target p/q.

  Q2. Do plateaus form at rational ridge boundaries?
      — Log when plateau fires and what ratio ω_high/ω_low is at that moment.

  Q3. Does the jumper remain globally sufficient?
      — plateau_hybrid should achieve 100% success in 4D.

  Q4. Does inversion remain clean?
      — TwoMassSpringInverter produces valid (m, k, k_c, α, β) for all steps.

Domain: TwoMassSpring — symmetric 2-DOF with Rayleigh damping
  State:  [x₁, v₁, x₂, v₂]
  Params: (m, k, k_c, alpha, beta)
  Modes:  In-phase (ω₁=√k/m), Out-of-phase (ω₂=√(k+2k_c)/m), ω₂ > ω₁

Target harmonic basin:
  ζ₁, ζ₂ ∈ [0.05, 0.25]
  f₁ (in-phase)  ∈ [5, 20]  Hz
  f₂ (out-of-phase) ∈ [10, 40] Hz (follows from 2:1 + target f₁)
  ω₂/ω₁ ∈ [1.8, 2.2]   ← near 2:1 harmonic tube Z_{2:1}

Methods:
  random_search       — uniform log-space baseline
  local_walk          — Gaussian step in log-space
  cold_eigen_r2       — EigenWalker(n_modes=2), untrained
  energy_eigen_r2     — energy-gradient trained (n_modes=2, E_harm active)
  plateau_hybrid_r2   — energy_eigen_r2 + DiscreteHarmonicJumper(plateau trigger)

Output:
  benchmarks/results/multimode_eigen_walker.json
  benchmarks/results/multimode_eigen_walker_cdf.png
  benchmarks/results/multimode_eigen_walker_ratio_trace.png  (Q1/Q2 diagnostics)

Usage:
  python benchmarks/multimode_eigen_walker.py
  python benchmarks/multimode_eigen_walker.py --n-trials 50 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tensor.eigen_walker import (
    EigenWalker, SpectralCoords, DiscreteHarmonicJumper,
    _build_feature,
)
from tensor.domain_inverter import TwoMassSpringInverter
from tensor.spectral_energy import (
    SpectralEnergy, SpectralEnergyConfig,
    SpectralState, SpectralMode,
)


# ── TwoMassSpring analytics ────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(_ROOT, 'ecemath'))
from src.domains.mechanics import (
    two_mass_spring_eigvals,
    two_mass_modal_zetas,
    two_mass_modal_freqs_hz,
    two_mass_freq_ratio,
)

_tms_inv = TwoMassSpringInverter()

# Target region
ZETA_LO,   ZETA_HI    = 0.05, 0.25
F1_LO_HZ,  F1_HI_HZ   = 5.0, 20.0    # in-phase (lower) mode
RATIO_LO,  RATIO_HI   = 1.80, 2.20   # ω₂/ω₁ near 2:1

TMS_BOUNDS: Dict[str, Tuple[float, float]] = {
    'm':     (0.01,  100.0),
    'k':     (0.001, 1e6),
    'k_c':   (0.001, 5e5),
    'alpha': (0.001, 200.0),
    'beta':  (1e-6,  1.0),
}
TMS_KEYS = _tms_inv.theta_keys()


def _tms_eigvals(theta: dict) -> np.ndarray:
    return two_mass_spring_eigvals(theta)


def _tms_f1_hz(theta: dict) -> float:
    f1, f2 = two_mass_modal_freqs_hz(theta)
    return f1


def _tms_f2_hz(theta: dict) -> float:
    f1, f2 = two_mass_modal_freqs_hz(theta)
    return f2


def _tms_zeta1(theta: dict) -> float:
    z1, z2 = two_mass_modal_zetas(theta)
    return z1


def _tms_zeta2(theta: dict) -> float:
    z1, z2 = two_mass_modal_zetas(theta)
    return z2


def _tms_ratio(theta: dict) -> float:
    return two_mass_freq_ratio(theta)


def _tms_in_target(theta: dict) -> bool:
    z1, z2 = two_mass_modal_zetas(theta)
    f1, f2 = two_mass_modal_freqs_hz(theta)
    ratio = two_mass_freq_ratio(theta)
    return (
        ZETA_LO <= z1 <= ZETA_HI and
        ZETA_LO <= z2 <= ZETA_HI and
        F1_LO_HZ <= f1 <= F1_HI_HZ and
        RATIO_LO <= ratio <= RATIO_HI
    )


def _tms_log_sample(rng: np.random.Generator) -> dict:
    return {
        k: float(10 ** rng.uniform(np.log10(TMS_BOUNDS[k][0]),
                                   np.log10(TMS_BOUNDS[k][1])))
        for k in TMS_KEYS
    }


def _tms_clip(theta: dict) -> dict:
    return {k: float(np.clip(theta[k], TMS_BOUNDS[k][0], TMS_BOUNDS[k][1]))
            for k in TMS_KEYS}


def _tms_sample_outside_target(rng: np.random.Generator, max_tries: int = 3000) -> dict:
    for _ in range(max_tries):
        theta = _tms_log_sample(rng)
        if not _tms_in_target(theta):
            return theta
    return _tms_log_sample(rng)


# ── Energy config for r=2 ─────────────────────────────────────────────────────

def make_r2_energy_config() -> SpectralEnergyConfig:
    """
    SpectralEnergyConfig for the two-mode TwoMassSpring target.

    mode ordering in SpectralState (descending |Im(λ)|):
      modes[0] = out-of-phase (ω₂, higher) → target f₂ = 20 Hz
      modes[1] = in-phase    (ω₁, lower)  → target f₁ = 10 Hz

    E_harm = min_{p,q} |ω₂/ω₁ - p/q|  — first time E_harm is ACTIVE.
    At 2:1 (ω₂=2ω₁): E_harm = 0 at the harmonic tube.
    """
    return SpectralEnergyConfig(
        target_zetas=[0.15, 0.15],           # equal damping both modes
        target_omega0s_hz=[20.0, 10.0],      # modes[0]=20Hz, modes[1]=10Hz → 2:1
        K_ratios=8,
        epsilon_harmony=0.05,
        w_stab=10.0,
        w_damp=1.0,
        w_freq=0.3,
        w_harm=2.0,                          # harmonic energy active
    )


# ── Training ───────────────────────────────────────────────────────────────────

def _mode_to_eigvals(mode: SpectralMode) -> np.ndarray:
    if mode.omega > 1e-9:
        return np.array([mode.alpha + 1j * mode.omega,
                         mode.alpha - 1j * mode.omega], dtype=complex)
    return np.array([complex(mode.alpha)], dtype=complex)


def build_r2_training_data(
    config:    SpectralEnergyConfig,
    n_samples: int,
    seed:      int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y) training data for n_modes=2 EigenWalker.

    Samples two modes independently (dominant=higher ω, secondary=lower ω).
    Enforces ω_dom > ω_sec to keep inversion valid (k_c > 0).

    Y shape: (n_samples, 4)  — [Δζ_dom, Δlogω_dom, Δζ_sec, Δlogω_sec]
    """
    energy = SpectralEnergy(config)
    rng = np.random.default_rng(seed)

    # Sample raw from SpectralEnergy.generate_training_samples(n_modes=2)
    states, targets = energy.generate_training_samples(
        n_samples=n_samples,
        n_modes=2,
        zeta_range=(0.01, 0.80),
        logw_range=(1.0, 3.5),    # ω₀ ∈ [10, 3162] rad/s — covers [1.6Hz, 503Hz]
        seed=seed,
    )

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for state, target in zip(states, targets):
        if len(state.modes) < 2:
            continue
        # Build eigenval array: dominant mode (modes[0]) first, then secondary
        ev0 = _mode_to_eigvals(state.modes[0])
        ev1 = _mode_to_eigvals(state.modes[1])
        eigvals = np.concatenate([ev0, ev1])
        x = _build_feature(eigvals, 'lca')
        X_list.append(x)
        Y_list.append(target[:4])   # [Δζ₀, Δlogω₀, Δζ₁, Δlogω₁]

    return np.array(X_list, dtype=np.float64), np.array(Y_list, dtype=np.float64)


def _train_mlp_on_arrays(
    walker:     EigenWalker,
    X:          np.ndarray,
    Y:          np.ndarray,
    n_epochs:   int   = 300,
    lr:         float = 1e-3,
    batch_size: int   = 64,
    seed:       int   = 0,
) -> List[float]:
    rng = np.random.default_rng(seed)
    N = len(X)
    losses = []
    for _ in range(n_epochs):
        idx = rng.permutation(N)
        ep_loss, n_b = 0.0, 0
        for start in range(0, N, batch_size):
            sl = idx[start:start + batch_size]
            loss, grads = walker._mlp.mse_loss_and_grads(X[sl], Y[sl])
            walker._mlp.adam_step(grads, lr=lr)
            ep_loss += loss
            n_b += 1
        losses.append(ep_loss / max(n_b, 1))
    return losses


def train_r2_walker(
    config:    SpectralEnergyConfig,
    n_samples: int,
    n_epochs:  int,
    seed:      int,
) -> Tuple[EigenWalker, dict]:
    """Train EigenWalker(n_modes=2) with energy-gradient objective."""
    t0 = time.time()
    X, Y = build_r2_training_data(config, n_samples, seed)
    walker = EigenWalker(hidden=128, n_modes=2, seed=seed)
    losses = _train_mlp_on_arrays(walker, X, Y, n_epochs=n_epochs, lr=1e-3,
                                  batch_size=64, seed=seed)
    dt = time.time() - t0
    return walker, {
        'n_samples': len(X),
        'n_epochs':  n_epochs,
        'time_s':    round(dt, 2),
        'init_loss': round(float(losses[0]),  6),
        'final_loss': round(float(losses[-1]), 6),
    }


# ── Step application ───────────────────────────────────────────────────────────

def _apply_r2_step(
    theta:      dict,
    eigvals:    np.ndarray,
    walker:     EigenWalker,
    step_scale: float,
) -> dict:
    """Apply EigenWalker step in 4D (ζ₁, logω₁, ζ₂, logω₂) space → TwoMassSpring θ."""
    delta = walker.predict_step(eigvals, regime='lca') * step_scale
    state = SpectralState.from_eigvals(eigvals)

    if len(state.modes) < 2:
        return theta   # no-op if not enough modes

    # modes[0] = dominant (ω₂, out-of-phase), modes[1] = secondary (ω₁, in-phase)
    modes_new = []
    for i in range(2):
        d_z = float(np.clip(delta[2 * i],     -0.5, 0.5))
        d_w = float(np.clip(delta[2 * i + 1], -1.0, 1.0))
        zeta_new  = float(np.clip(state.modes[i].zeta + d_z, 1e-4, 2.0))
        omega_new = float(max(10.0 ** (state.modes[i].log10_omega0 + d_w), 1e-3))
        modes_new.append((zeta_new, omega_new))

    theta_new = _tms_inv.invert_modes(modes_new, theta)
    return _tms_clip(theta_new)


# ── Jumper for r=2 ────────────────────────────────────────────────────────────

def _r2_jumper_candidates(
    omega_low:  float,   # ω₁ (in-phase, lower)
    omega_high: float,   # ω₂ (out-of-phase, higher)
    zeta1:      float,
    zeta2:      float,
    K:          int = 8,
) -> List[Tuple[float, float, float, float]]:
    """
    Generate (ζ₁, ω₁, ζ₂, ω₂) candidates for 2:1 harmonic zone.

    Strategy A: snap ω₂ → (p/q)·ω₁  for each rational p/q near 2 (i.e., p/q ∈ {2/1, 3/2, 5/3, ...})
    Strategy B: direct target-frequency grid
    """
    candidates = []
    zeta_choices = [0.05, 0.10, 0.15, 0.20, 0.25]

    # Strategy A: rational snap (fix ω₁, snap ω₂ to rational ratio)
    for p in range(1, K + 1):
        for q in range(1, K + 1):
            ratio = p / q
            if RATIO_LO <= ratio <= RATIO_HI + 0.5:   # near target 2:1
                omega2_cand = ratio * omega_low
                for z1 in zeta_choices:
                    for z2 in zeta_choices:
                        candidates.append((z1, omega_low, z2, omega2_cand))

    # Strategy B: direct target grid
    for f1_hz in np.logspace(np.log10(F1_LO_HZ), np.log10(F1_HI_HZ), 5):
        omega1_b = 2.0 * np.pi * f1_hz
        omega2_b = 2.0 * omega1_b   # exact 2:1
        for z1 in zeta_choices:
            for z2 in zeta_choices:
                candidates.append((z1, omega1_b, z2, omega2_b))

    return candidates


def _try_r2_jumper(eigvals: np.ndarray, theta_current: dict) -> Optional[dict]:
    """Fire the r=2 harmonic jumper; return best θ or None."""
    state = SpectralState.from_eigvals(eigvals)
    if len(state.modes) < 2:
        return None

    # modes[0]=ω₂ (high), modes[1]=ω₁ (low)
    omega_high = state.modes[0].omega0
    omega_low  = state.modes[1].omega0
    zeta2      = state.modes[0].zeta
    zeta1      = state.modes[1].zeta

    cands = _r2_jumper_candidates(omega_low, omega_high, zeta1, zeta2)

    m_vals = np.logspace(np.log10(TMS_BOUNDS['m'][0]),
                         np.log10(TMS_BOUNDS['m'][1]), 5)

    best_hit: Optional[dict] = None
    best_cand: Optional[dict] = None

    for z1, om1, z2, om2 in cands:
        # modes_new[0] = dominant (higher ω = om2), modes_new[1] = secondary (om1)
        for m in m_vals:
            theta_try = _tms_inv.invert_modes([(z2, om2), (z1, om1)], {'m': m})
            theta_c   = _tms_clip(theta_try)
            if _tms_in_target(theta_c):
                return theta_c   # first hit in target = done
            if best_cand is None:
                best_cand = theta_c

    return best_cand


# ── Search result ──────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    method:         str
    found:          bool
    n_iters:        int
    final_f1_hz:    float
    final_ratio:    float
    final_zeta1:    float
    ratio_trace:    List[float] = field(default_factory=list)
    plateau_fires:  List[int]   = field(default_factory=list)
    plateau_ratios: List[float] = field(default_factory=list)


# ── Search methods ─────────────────────────────────────────────────────────────

def run_random_search(
    theta0: dict, max_iter: int, rng: np.random.Generator,
) -> SearchResult:
    theta = theta0
    for i in range(max_iter):
        theta = _tms_log_sample(rng)
        if _tms_in_target(theta):
            return SearchResult('random_search', True, i + 1,
                                _tms_f1_hz(theta), _tms_ratio(theta), _tms_zeta1(theta))
    return SearchResult('random_search', False, max_iter,
                        _tms_f1_hz(theta), _tms_ratio(theta), _tms_zeta1(theta))


def run_local_walk(
    theta0: dict, max_iter: int, rng: np.random.Generator,
    log_sigma: float = 0.12,
) -> SearchResult:
    theta = dict(theta0)
    for i in range(max_iter):
        theta_new = {}
        for k, (lo, hi) in TMS_BOUNDS.items():
            log_v = np.log10(theta[k]) + rng.normal(0.0, log_sigma)
            theta_new[k] = float(10 ** np.clip(log_v, np.log10(lo), np.log10(hi)))
        theta = theta_new
        if _tms_in_target(theta):
            return SearchResult('local_walk', True, i + 1,
                                _tms_f1_hz(theta), _tms_ratio(theta), _tms_zeta1(theta))
    return SearchResult('local_walk', False, max_iter,
                        _tms_f1_hz(theta), _tms_ratio(theta), _tms_zeta1(theta))


def run_energy_eigen_r2(
    theta0:      dict,
    walker:      EigenWalker,
    max_iter:    int,
    method_name: str,
    step_scale:  float = 0.25,
    record_trace: bool = False,
) -> SearchResult:
    """Pure energy-gradient EigenWalker (r=2, no jumper)."""
    theta = dict(theta0)
    ev = _tms_eigvals(theta)
    ratio_trace = []

    for i in range(max_iter):
        if record_trace:
            ratio_trace.append(round(_tms_ratio(theta), 4))
        theta = _apply_r2_step(theta, ev, walker, step_scale)
        ev = _tms_eigvals(theta)
        if _tms_in_target(theta):
            return SearchResult(method_name, True, i + 1,
                                _tms_f1_hz(theta), _tms_ratio(theta), _tms_zeta1(theta),
                                ratio_trace=ratio_trace)

    return SearchResult(method_name, False, max_iter,
                        _tms_f1_hz(theta), _tms_ratio(theta), _tms_zeta1(theta),
                        ratio_trace=ratio_trace)


def run_plateau_hybrid_r2(
    theta0:         dict,
    walker:         EigenWalker,
    energy:         SpectralEnergy,
    max_iter:       int,
    method_name:    str,
    step_scale:     float = 0.25,
    plateau_window: int   = 8,
    plateau_eps:    float = 0.02,
    boundary_window: int  = 20,
    record_trace:   bool  = False,
) -> SearchResult:
    """
    Energy-gradient EigenWalker(n_modes=2) + r=2 harmonic jumper.

    Stagnation fires when:
      (a) E(S_{t-N}) - E(S_t) < plateau_eps  — energy plateau in 4D
      (b) boundary_count >= boundary_window   — parameter clipping fallback
    """
    theta = dict(theta0)
    ev = _tms_eigvals(theta)
    energy_history: List[float] = []
    boundary_count = 0
    ratio_trace:   List[float] = []
    plateau_fires: List[int]   = []
    plateau_ratios: List[float] = []

    for i in range(max_iter):
        if record_trace:
            ratio_trace.append(round(_tms_ratio(theta), 4))

        # Track energy
        state = SpectralState.from_eigvals(ev)
        E_now = energy.compute(state)
        energy_history.append(E_now)
        if len(energy_history) > plateau_window:
            energy_history.pop(0)

        # Continuous step
        theta_raw  = _apply_r2_step(theta, ev, walker, step_scale)
        theta_new  = _tms_clip(theta_raw)
        clipped    = any(abs(theta_raw[k] - theta_new[k]) > 1e-12 for k in TMS_KEYS)
        ev_new    = _tms_eigvals(theta_new)
        boundary_count = boundary_count + 1 if clipped else max(0, boundary_count - 1)

        # Stagnation detection
        energy_plateau = (
            len(energy_history) >= plateau_window
            and (energy_history[0] - energy_history[-1]) < plateau_eps
        )
        if energy_plateau or (boundary_count >= boundary_window):
            jumped = _try_r2_jumper(ev, theta)
            if jumped is not None:
                if record_trace:
                    plateau_fires.append(i)
                    plateau_ratios.append(round(_tms_ratio(theta), 4))
                theta_new = jumped
                ev_new    = _tms_eigvals(theta_new)
                boundary_count  = 0
                energy_history.clear()

        theta, ev = theta_new, ev_new

        if _tms_in_target(theta):
            return SearchResult(method_name, True, i + 1,
                                _tms_f1_hz(theta), _tms_ratio(theta), _tms_zeta1(theta),
                                ratio_trace=ratio_trace,
                                plateau_fires=plateau_fires,
                                plateau_ratios=plateau_ratios)

    return SearchResult(method_name, False, max_iter,
                        _tms_f1_hz(theta), _tms_ratio(theta), _tms_zeta1(theta),
                        ratio_trace=ratio_trace,
                        plateau_fires=plateau_fires,
                        plateau_ratios=plateau_ratios)


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(results: List[SearchResult]) -> dict:
    iters   = [r.n_iters for r in results if r.found]
    n_total = len(results)
    if not iters:
        return {'n_success': 0, 'n_total': n_total, 'failure_rate': 1.0}
    return {
        'n_success':    len(iters),
        'n_total':      n_total,
        'failure_rate': round(1.0 - len(iters) / n_total, 4),
        'mean_iters':   round(float(np.mean(iters)), 2),
        'median_iters': round(float(np.median(iters)), 2),
        'p75_iters':    round(float(np.percentile(iters, 75)), 2),
        'p95_iters':    round(float(np.percentile(iters, 95)), 2),
        'min_iters':    int(np.min(iters)),
        'max_iters':    int(np.max(iters)),
    }


# ── Diagnostics: Q1/Q2 analysis ───────────────────────────────────────────────

def q1_q2_analysis(
    plateau_results: List[SearchResult],
    energy_results:  List[SearchResult],
) -> dict:
    """
    Q1: Does gradient descent move ratio toward 2:1?
        Measure |ratio − 2.0| at each trace step; check if it decreases.

    Q2: Do plateaus cluster near rational ridge boundaries?
        Collect ratio values at plateau-fire events; check proximity to p/q values.
    """
    # Q1: ratio convergence in energy_eigen_r2 traces
    q1_traces = [r.ratio_trace for r in energy_results if r.ratio_trace]
    q1_initial_err = []
    q1_final_err   = []
    for trace in q1_traces:
        if len(trace) >= 2:
            q1_initial_err.append(abs(trace[0]  - 2.0))
            q1_final_err.append(  abs(trace[-1] - 2.0))

    q1_gradient_steers = sum(
        1 for a, b in zip(q1_initial_err, q1_final_err) if b < a
    ) if q1_initial_err else 0

    # Q2: plateau fires near rational boundaries
    all_plateau_ratios: List[float] = []
    for r in plateau_results:
        all_plateau_ratios.extend(r.plateau_ratios)

    rationals = [p / q for p in range(1, 9) for q in range(1, 9)]

    def _nearest_rational(r: float) -> float:
        return min(abs(r - pq) for pq in rationals)

    plateau_proximity = [_nearest_rational(r) for r in all_plateau_ratios]

    return {
        'q1_gradient_steers_toward_2to1': q1_gradient_steers,
        'q1_total_traces':   len(q1_traces),
        'q1_mean_initial_err': round(float(np.mean(q1_initial_err)), 4) if q1_initial_err else None,
        'q1_mean_final_err':   round(float(np.mean(q1_final_err)),   4) if q1_final_err  else None,
        'q2_total_plateau_fires':   len(all_plateau_ratios),
        'q2_mean_ratio_at_plateau': round(float(np.mean(all_plateau_ratios)), 4) if all_plateau_ratios else None,
        'q2_mean_proximity_to_rational': round(float(np.mean(plateau_proximity)), 4) if plateau_proximity else None,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

_COLORS = {
    'random_search':    '#e74c3c',
    'local_walk':       '#f39c12',
    'cold_eigen_r2':    '#95a5a6',
    'energy_eigen_r2':  '#27ae60',
    'plateau_hybrid_r2': '#2980b9',
}
_LABELS = {
    'random_search':    'Random Search',
    'local_walk':       'Local Walk',
    'cold_eigen_r2':    'Cold EigenWalker r=2 (untrained)',
    'energy_eigen_r2':  'Energy-Gradient EigenWalker r=2',
    'plateau_hybrid_r2': 'Energy Plateau Hybrid r=2',
}
_ORDER = ['random_search', 'local_walk', 'cold_eigen_r2',
          'energy_eigen_r2', 'plateau_hybrid_r2']


def plot_results(
    results:    Dict[str, List[SearchResult]],
    trace_results: List[SearchResult],
    out_dir:    str,
    max_iter:   int,
) -> List[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    saved: List[str] = []

    # CDF
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in _ORDER:
        rs = results.get(method, [])
        iters = sorted(r.n_iters for r in rs if r.found)
        if not iters:
            continue
        n_total = len(rs)
        xs = [0] + iters + [max_iter]
        ys = [0.0] + [i / n_total for i in range(1, len(iters) + 1)] + [len(iters) / n_total]
        lw = 2.5 if 'r2' in method else 1.5
        ax.step(xs, ys, where='post',
                color=_COLORS.get(method, 'gray'),
                label=_LABELS.get(method, method),
                linewidth=lw)

    ax.set_xlabel('Iterations to reach target', fontsize=12)
    ax.set_ylabel('Fraction of trials succeeded', fontsize=12)
    ax.set_title('EigenWalker r=2: Two-Mass Spring — First E_harm Activation',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_iter)
    ax.set_ylim(0.0, 1.05)
    cdf_path = os.path.join(out_dir, 'multimode_eigen_walker_cdf.png')
    fig.tight_layout()
    fig.savefig(cdf_path, dpi=150)
    plt.close(fig)
    saved.append(cdf_path)

    # Ratio trace: Q1 diagnostic
    if trace_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_r, ax_e = axes

        # Left: ratio traces for first 5 energy_eigen_r2 trials
        traces = [(r.ratio_trace, r.found) for r in trace_results[:5] if r.ratio_trace]
        for trace, found in traces:
            color = '#27ae60' if found else '#e74c3c'
            ax_r.plot(trace, color=color, alpha=0.6, linewidth=1.5)
        ax_r.axhline(2.0, color='k', linestyle='--', linewidth=1.5, label='2:1 target')
        ax_r.axhline(RATIO_LO, color='gray', linestyle=':', linewidth=1)
        ax_r.axhline(RATIO_HI, color='gray', linestyle=':', linewidth=1)
        ax_r.set_xlabel('Iteration', fontsize=11)
        ax_r.set_ylabel('ω₂/ω₁ ratio', fontsize=11)
        ax_r.set_title('Q1: Ratio Convergence (energy_eigen_r2)', fontsize=11)
        ax_r.legend(fontsize=9)
        ax_r.grid(True, alpha=0.3)

        # Right: plateau fire ratios (Q2)
        all_pr = []
        for r in trace_results:
            all_pr.extend(r.plateau_ratios)
        if all_pr:
            ax_e.hist(all_pr, bins=20, color='#2980b9', alpha=0.8, edgecolor='white')
            for pq in [1.5, 2.0, 3.0/2, 3.0]:
                ax_e.axvline(pq, color='red', linestyle='--', linewidth=1.5,
                              label=f'p/q={pq:.2f}')
            ax_e.set_xlabel('ω₂/ω₁ at plateau fire', fontsize=11)
            ax_e.set_ylabel('Count', fontsize=11)
            ax_e.set_title('Q2: Ratio at Plateau Detection', fontsize=11)
            ax_e.legend(fontsize=8)
            ax_e.grid(True, alpha=0.3)

        trace_path = os.path.join(out_dir, 'multimode_eigen_walker_ratio_trace.png')
        fig.tight_layout()
        fig.savefig(trace_path, dpi=150)
        plt.close(fig)
        saved.append(trace_path)

    return saved


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--n-trials',   type=int,   default=100)
    parser.add_argument('--max-iter',   type=int,   default=500)
    parser.add_argument('--n-samples',  type=int,   default=5000)
    parser.add_argument('--n-epochs',   type=int,   default=300)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--step-scale', type=float, default=0.25)
    parser.add_argument('--log-sigma',  type=float, default=0.15)
    parser.add_argument('--out', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'results'))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=== EigenWalker r=2: Two-Mode Spectral Manifold Benchmark ===")
    print(f"  trials={args.n_trials}  max_iter={args.max_iter}  seed={args.seed}")
    print(f"  Training: {args.n_samples} synthetic r=2 samples, {args.n_epochs} epochs")
    print(f"  Target: ζ ∈ [{ZETA_LO},{ZETA_HI}]  f₁ ∈ [{F1_LO_HZ},{F1_HI_HZ}]Hz  "
          f"ω₂/ω₁ ∈ [{RATIO_LO},{RATIO_HI}]")
    print(f"  E_harm target: 2:1 harmonic tube (first time E_harm is ACTIVE)")

    # Target fraction estimate
    rng_est = np.random.default_rng(args.seed + 9999)
    target_frac = sum(_tms_in_target(_tms_log_sample(rng_est))
                      for _ in range(5000)) / 5000
    print(f"  Target fraction: ~{target_frac:.2%}")

    # ── Phase 1: Train r=2 walker ──────────────────────────────────────────────
    cfg = make_r2_energy_config()
    print(f"\n[Phase 1] Energy-gradient training (r=2, E_harm active)...")
    energy_walker, train_stats = train_r2_walker(
        cfg, n_samples=args.n_samples, n_epochs=args.n_epochs, seed=args.seed,
    )
    print(f"  {train_stats['n_samples']} samples, {args.n_epochs} epochs in "
          f"{train_stats['time_s']:.1f}s")
    print(f"  Loss: {train_stats['init_loss']:.6f} → {train_stats['final_loss']:.6f}")

    cold_walker = EigenWalker(hidden=128, n_modes=2, seed=args.seed + 99)
    domain_energy = SpectralEnergy(cfg)

    # ── Phase 2: Benchmark ─────────────────────────────────────────────────────
    print(f"\n[Phase 2] Benchmark: {args.n_trials} trials × 5 methods...")
    all_results: Dict[str, List[SearchResult]] = {m: [] for m in _ORDER}
    trace_results: List[SearchResult] = []   # Q1/Q2 diagnostics from first 20 trials

    for trial_idx in range(args.n_trials):
        t_seed = args.seed * 100_000 + trial_idx
        theta0 = _tms_sample_outside_target(np.random.default_rng(t_seed))

        record = (trial_idx < 20)

        r_rand = run_random_search(theta0, args.max_iter,
                                   np.random.default_rng(t_seed + 1))
        r_walk = run_local_walk(theta0, args.max_iter,
                                np.random.default_rng(t_seed + 2),
                                log_sigma=args.log_sigma)
        r_cold = run_energy_eigen_r2(theta0, cold_walker, args.max_iter, 'cold_eigen_r2',
                                     step_scale=args.step_scale)
        r_enrg = run_energy_eigen_r2(theta0, energy_walker, args.max_iter, 'energy_eigen_r2',
                                     step_scale=args.step_scale, record_trace=record)
        r_plat = run_plateau_hybrid_r2(
            theta0, energy_walker, domain_energy,
            args.max_iter, 'plateau_hybrid_r2',
            step_scale=args.step_scale,
            plateau_window=8, plateau_eps=0.02, boundary_window=20,
            record_trace=record,
        )

        for r in (r_rand, r_walk, r_cold, r_enrg, r_plat):
            all_results[r.method].append(r)
        if record:
            trace_results.extend([r_enrg, r_plat])

        if (trial_idx + 1) % 10 == 0:
            print(f"  [{trial_idx+1:3d}/{args.n_trials}]  "
                  f"rand:{r_rand.n_iters:4d}({'✓' if r_rand.found else '✗'})  "
                  f"cold:{r_cold.n_iters:4d}({'✓' if r_cold.found else '✗'})  "
                  f"enrg:{r_enrg.n_iters:4d}({'✓' if r_enrg.found else '✗'})  "
                  f"plat:{r_plat.n_iters:4d}({'✓' if r_plat.found else '✗'})")

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = {m: aggregate(all_results[m]) for m in _ORDER}

    # Speedups
    rand_mean = summary['random_search'].get('mean_iters', float('inf'))
    for method in ('energy_eigen_r2', 'plateau_hybrid_r2'):
        m_mean = summary[method].get('mean_iters')
        if m_mean and rand_mean < float('inf'):
            summary[method]['speedup_vs_random'] = round(rand_mean / m_mean, 2)

    # Q1/Q2 diagnostics
    plateau_trace = [r for r in trace_results if r.method == 'plateau_hybrid_r2']
    energy_trace  = [r for r in trace_results if r.method == 'energy_eigen_r2']
    diagnostics   = q1_q2_analysis(plateau_trace, energy_trace)

    print("\n=== r=2 Benchmark Summary ===")
    hdr = f"{'Method':<38}  {'Succ':>7}  {'Mean':>7}  {'Median':>7}  {'p95':>7}  {'Fail%':>6}"
    print(hdr)
    print('-' * len(hdr))
    for m in _ORDER:
        s = summary[m]
        ns, nt = s.get('n_success', 0), s.get('n_total', args.n_trials)
        label = _LABELS.get(m, m)[:38]
        print(f"{label:<38}  {ns:>4}/{nt:<3}  "
              f"{s.get('mean_iters',   float('nan')):>7.1f}  "
              f"{s.get('median_iters', float('nan')):>7.1f}  "
              f"{s.get('p95_iters',    float('nan')):>7.1f}  "
              f"{s.get('failure_rate', 1.0):>5.1%}")

    pl_fail = summary['plateau_hybrid_r2'].get('failure_rate', 1.0)
    pl_mean = summary['plateau_hybrid_r2'].get('mean_iters', float('inf'))
    if pl_fail <= 0.02:
        verdict = f"r=2 SCALES: plateau_hybrid 100% success  mean={pl_mean:.1f}"
    elif pl_fail < 0.20:
        verdict = f"r=2 MOSTLY SCALES: {100*(1-pl_fail):.0f}% success  mean={pl_mean:.1f}"
    else:
        verdict = f"r=2 PARTIAL: {100*(1-pl_fail):.0f}% success — architecture needs refinement"

    print(f"\nScaling verdict: {verdict}")

    print(f"\n=== Q1/Q2 Diagnostics ===")
    print(f"  Q1 (does gradient steer toward 2:1?): "
          f"{diagnostics['q1_gradient_steers_toward_2to1']}/"
          f"{diagnostics['q1_total_traces']} traces converged")
    if diagnostics['q1_mean_initial_err'] is not None:
        print(f"     Mean |ratio−2| : {diagnostics['q1_mean_initial_err']:.4f} → "
              f"{diagnostics['q1_mean_final_err']:.4f}")
    print(f"  Q2 (plateaus near rational ridges?): "
          f"{diagnostics['q2_total_plateau_fires']} fires,  "
          f"mean ratio = {diagnostics['q2_mean_ratio_at_plateau']},  "
          f"mean proximity to p/q = {diagnostics['q2_mean_proximity_to_rational']}")

    # Save JSON
    def _default(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, tuple):          return list(obj)
        return str(obj)

    output = {
        'config': {
            'n_trials': args.n_trials, 'max_iter': args.max_iter,
            'n_samples': args.n_samples, 'n_epochs': args.n_epochs,
            'seed': args.seed,
            'walker': 'EigenWalker(n_modes=2)',
            'e_harm_active': True,
            'target': {'zeta': [ZETA_LO, ZETA_HI], 'f1_hz': [F1_LO_HZ, F1_HI_HZ],
                       'ratio': [RATIO_LO, RATIO_HI]},
        },
        'target_fraction': round(target_frac, 4),
        'training_stats': train_stats,
        'summary': summary,
        'diagnostics': diagnostics,
        'verdict': verdict,
        'r1_comparison': {
            'r1_plateau_hybrid_fail': 0.00,
            'r1_plateau_hybrid_mean': 7.8,
            'r1_e_harm_active': False,
        },
    }
    json_path = os.path.join(args.out, 'multimode_eigen_walker.json')
    with open(json_path, 'w') as fh:
        json.dump(output, fh, indent=2, default=_default)
    print(f"\nResults: {json_path}")

    for p in plot_results(all_results, trace_results, args.out, args.max_iter):
        print(f"Plot:    {p}")


if __name__ == '__main__':
    main()
