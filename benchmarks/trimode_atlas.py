"""
benchmarks/trimode_atlas.py — r=3 spectral atlas: 6D space, 2D Farey product graph

Extends rational_atlas (r=2, 1D ratio space) to r=3 (6D spectral state, 2D ratio space).

Physical model: Three decoupled oscillators with shared Rayleigh damping.
  θ = (m, k_hi, k_mid, k_lo, alpha, beta)
  ω_i = sqrt(k_i/m),  ζ_i = alpha/(2ω_i) + beta·ω_i/2

Ratio space (2D consecutive ratios):
  ρ₁ = modes[0].omega0 / modes[1].omega0  (high/mid frequency ratio)
  ρ₂ = modes[1].omega0 / modes[2].omega0  (mid/low frequency ratio)

2D Farey product graph (Cartesian product):
  Nodes: ((p1,q1),(p2,q2)) — pair of reduced fractions
  Edges: exactly ONE ratio changes by Farey step  |pb−qa|=1
  Distance: d_2d = d_1d(ρ₁→ρ₁_target) + d_1d(ρ₂→ρ₂_target)  [Manhattan, separable]

Start:  forced near Z_{(3:1, 2:1)}: ρ₁∈[2.8,3.2], ρ₂∈[1.85,2.15]
Target: Z_{(3:2, 4:3)}: ρ₁∈[1.40,1.60], ρ₂∈[1.20,1.45]
2D Farey distance: d1(3:1→3:2) + d2(2:1→4:3) = 2 + 2 = 4

Hypotheses:
  H1_2D: All transitions are Farey-adjacent in exactly one dimension (product structure)
  H2_2D: Diagonal transitions (both ratios change) are absent (decoupled system)
  H5_2D: Farey-distance routing is monotone (d_total = d1+d2 never increases)
  H5b_2D: 100% success maintained at r=3
"""

import os
import sys
import time
import json
import math
import argparse
from collections import defaultdict, deque
from math import gcd
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tensor.spectral_energy import SpectralEnergy, SpectralEnergyConfig, SpectralState, SpectralMode
from tensor.eigen_walker    import EigenWalker, _build_feature

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(_ROOT, 'benchmarks', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 2D target tube
TARGET_TUBE_2D  = ((3, 2), (4, 3))
# ρ₁ = modes[0]/modes[1] (high/mid)
TARGET_RATIO1_LO, TARGET_RATIO1_HI = 1.40, 1.60    # 3:2 zone (3/2=1.5)
# ρ₂ = modes[1]/modes[2] (mid/low)
TARGET_RATIO2_LO, TARGET_RATIO2_HI = 1.20, 1.45    # 4:3 zone (4/3≈1.333)
TARGET_F_LO,  TARGET_F_HI  = 4.0, 15.0             # Hz, lowest mode
TARGET_ZETA_LO, TARGET_ZETA_HI = 0.05, 0.30

# Start region: forced near Z_{(3:1, 2:1)}
START_RATIO1_LO, START_RATIO1_HI = 2.80, 3.20      # ρ₁ ≈ 3:1
START_RATIO2_LO, START_RATIO2_HI = 1.85, 2.15      # ρ₂ ≈ 2:1

# Jumper parameters — uniform zeta sweeps + omega_lo anchors
ZETA_TARGETS = [0.10, 0.15, 0.20]                  # applied uniformly to all 3 modes
OMEGA_LO_ANCHORS_RADS = [2.0 * math.pi * f for f in [5.0, 7.0, 9.0, 12.0, 15.0]]

# Parameter bounds
TM3_BOUNDS = {
    'm':     (0.01,  100.0),
    'k_hi':  (1e-4,  1e6),
    'k_mid': (1e-4,  1e6),
    'k_lo':  (1e-4,  1e6),
    'alpha': (1e-4,  200.0),
    'beta':  (1e-8,  1.0),
}
TM3_KEYS = ['m', 'k_hi', 'k_mid', 'k_lo', 'alpha', 'beta']


# ── Three-mode decoupled physics ───────────────────────────────────────────────

def _3mode_eigvals(theta: dict) -> np.ndarray:
    """6 eigenvalues: k_hi → modes[0], k_mid → modes[1], k_lo → modes[2]."""
    m, alpha, beta = theta['m'], theta['alpha'], theta['beta']
    ev = []
    for k_key in ['k_hi', 'k_mid', 'k_lo']:
        k     = theta[k_key]
        omega = math.sqrt(max(k / m, 1e-12))
        zeta  = alpha / (2.0 * omega) + beta * omega / 2.0
        zeta  = max(0.0, min(zeta, 0.9999))
        wd    = omega * math.sqrt(max(1.0 - zeta**2, 0.0))
        if wd > 1e-12:
            ev.extend([complex(-zeta*omega,  wd), complex(-zeta*omega, -wd)])
        else:
            ev.extend([complex(-zeta*omega),      complex(-zeta*omega)])
    return np.array(ev, dtype=complex)


def _3mode_modal_freqs_hz(theta: dict) -> Tuple[float, float, float]:
    """(f_hi, f_mid, f_lo) Hz, descending."""
    m, alpha, beta = theta['m'], theta['alpha'], theta['beta']
    vals = []
    for k_key in ['k_hi', 'k_mid', 'k_lo']:
        k    = theta[k_key]
        omega = math.sqrt(max(k / m, 1e-12))
        zeta  = max(0.0, min(alpha / (2.0*omega) + beta*omega/2.0, 0.9999))
        wd    = omega * math.sqrt(max(1.0 - zeta**2, 0.0))
        vals.append(wd / (2.0 * math.pi))
    vals.sort(reverse=True)
    return tuple(vals)


def _3mode_modal_zetas(theta: dict) -> List[float]:
    """Zeta values sorted descending by frequency."""
    m, alpha, beta = theta['m'], theta['alpha'], theta['beta']
    pairs = []
    for k_key in ['k_hi', 'k_mid', 'k_lo']:
        k    = theta[k_key]
        omega = math.sqrt(max(k / m, 1e-12))
        zeta  = max(0.0, min(alpha / (2.0*omega) + beta*omega/2.0, 0.9999))
        pairs.append((omega, zeta))
    pairs.sort(key=lambda x: -x[0])
    return [p[1] for p in pairs]


def _invert_modes_3mode(
    modes:         List[Tuple[float, float]],   # [(ζ,ω) for hi, mid, lo] descending ω
    theta_current: dict,
) -> dict:
    """
    Analytic inversion for decoupled model.
    kᵢ = m·ωᵢ² (exact).  [α,β] from least-squares over three ζ constraints.
    """
    m = float(theta_current.get('m', 1.0))
    if len(modes) < 3:
        return dict(theta_current)

    (z_hi, w_hi), (z_mid, w_mid), (z_lo, w_lo) = modes[0], modes[1], modes[2]

    k_hi  = m * w_hi**2
    k_mid = m * w_mid**2
    k_lo  = m * w_lo**2

    # Least-squares: [α,β] from ζᵢ = α/(2ωᵢ) + β·ωᵢ/2
    A = np.array([
        [1.0 / (2.0 * w_hi),  w_hi  / 2.0],
        [1.0 / (2.0 * w_mid), w_mid / 2.0],
        [1.0 / (2.0 * w_lo),  w_lo  / 2.0],
    ])
    b    = np.array([z_hi, z_mid, z_lo])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    alpha = max(float(coef[0]), 1e-4)
    beta  = max(float(coef[1]), 1e-8)

    return {
        'm':     m,
        'k_hi':  max(k_hi,  1e-4),
        'k_mid': max(k_mid, 1e-4),
        'k_lo':  max(k_lo,  1e-4),
        'alpha': alpha,
        'beta':  beta,
    }


def _tm3_clip(theta: dict) -> dict:
    return {k: float(np.clip(theta[k], TM3_BOUNDS[k][0], TM3_BOUNDS[k][1]))
            for k in TM3_KEYS}


def _in_target(theta: dict) -> bool:
    try:
        f_hi, f_mid, f_lo = _3mode_modal_freqs_hz(theta)
        r1   = f_hi  / max(f_mid, 1e-12)
        r2   = f_mid / max(f_lo,  1e-12)
        zs   = _3mode_modal_zetas(theta)
        return (
            TARGET_RATIO1_LO <= r1  <= TARGET_RATIO1_HI
            and TARGET_RATIO2_LO <= r2  <= TARGET_RATIO2_HI
            and TARGET_F_LO <= f_lo <= TARGET_F_HI
            and all(TARGET_ZETA_LO <= z <= TARGET_ZETA_HI for z in zs)
        )
    except Exception:
        return False


# ── 1D Farey graph ─────────────────────────────────────────────────────────────

def build_farey_graph(K: int, ratio_lo: float = 0.9, ratio_hi: float = 5.0):
    nodes = sorted({(p, q)
                    for p in range(1, K+1) for q in range(1, K+1)
                    if gcd(p, q) == 1 and ratio_lo <= p/q <= ratio_hi})
    adj = {n: [] for n in nodes}
    for (p, q) in nodes:
        for (a, b) in nodes:
            if (p, q) != (a, b) and abs(p*b - q*a) == 1:
                adj[(p, q)].append((a, b))
    return adj, nodes


def farey_bfs(source: tuple, adj: dict) -> dict:
    dist = {source: 0}
    q = deque([source])
    while q:
        node = q.popleft()
        for nb in adj.get(node, []):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist


def is_farey_adjacent_1d(p: int, q: int, a: int, b: int) -> bool:
    return abs(p*b - q*a) == 1


# ── Ratio extraction ───────────────────────────────────────────────────────────

def _nearest_rational(ratio: float, max_p: int = 12, max_q: int = 12
                      ) -> Tuple[int, int, float]:
    best_p, best_q, best_d = 1, 1, abs(ratio - 1.0)
    for p in range(1, max_p+1):
        for q in range(1, max_q+1):
            d = abs(ratio - p/q)
            if d < best_d:
                best_p, best_q, best_d = p, q, d
    return best_p, best_q, best_d


def _nearest_node(ratio: float, nodes_1d: List[tuple]) -> tuple:
    """Snap ratio to nearest node IN the Farey graph (prevents off-graph tubes)."""
    return min(nodes_1d, key=lambda n: abs(ratio - n[0] / n[1]))


def _get_ratios_from_ev(ev: np.ndarray) -> Tuple[float, float]:
    try:
        state = SpectralState.from_eigvals(ev)
        if len(state.modes) < 3:
            return 1.0, 1.0
        r1 = state.modes[0].omega0 / max(state.modes[1].omega0, 1e-10)
        r2 = state.modes[1].omega0 / max(state.modes[2].omega0, 1e-10)
        return r1, r2
    except Exception:
        return 1.0, 1.0


# nodes_1d is set at startup and shared as a module-level variable
_FAREY_NODES_1D: List[tuple] = []


def _get_current_tube(ev: np.ndarray) -> Tuple[tuple, tuple]:
    rho1, rho2 = _get_ratios_from_ev(ev)
    r1 = _nearest_node(rho1, _FAREY_NODES_1D)
    r2 = _nearest_node(rho2, _FAREY_NODES_1D)
    return r1, r2


# ── Sampling ───────────────────────────────────────────────────────────────────

def _sample_near_start(rng: np.random.Generator) -> dict:
    """Sample near Z_{(3:1, 2:1)}: ρ₁≈3.0, ρ₂≈2.0."""
    m    = float(rng.uniform(0.1, 2.0))
    f_lo = float(rng.uniform(5.0, 9.0))        # Hz, lowest mode
    omega_lo = 2.0 * math.pi * f_lo

    rho2     = float(rng.uniform(START_RATIO2_LO, START_RATIO2_HI))  # ω_mid/ω_lo
    rho1     = float(rng.uniform(START_RATIO1_LO, START_RATIO1_HI))  # ω_hi/ω_mid
    omega_mid = omega_lo  * rho2
    omega_hi  = omega_mid * rho1

    # Direct mode parameterization avoids Rayleigh overdamping (β·ω_hi/2 > 1)
    z_hi  = float(rng.uniform(0.05, 0.25))
    z_mid = float(rng.uniform(0.05, 0.25))
    z_lo  = float(rng.uniform(0.05, 0.25))

    return _tm3_clip(_invert_modes_3mode(
        [(z_hi, omega_hi), (z_mid, omega_mid), (z_lo, omega_lo)], {'m': m}
    ))


# ── Training ───────────────────────────────────────────────────────────────────

def make_target_config() -> SpectralEnergyConfig:
    # 12/8 = 3:2, 8/6 = 4:3 — matching TARGET_TUBE_2D
    return SpectralEnergyConfig(
        target_zetas=[0.15, 0.15, 0.15],
        target_omega0s_hz=[12.0, 8.0, 6.0],
        K_ratios=8,
        w_stab=10.0,
        w_damp=1.0,
        w_freq=0.5,
        w_harm=2.0,
    )


def _mode_to_eigvals(mode: SpectralMode) -> np.ndarray:
    if mode.omega > 1e-12:
        return np.array([mode.alpha + 1j*mode.omega,
                         mode.alpha - 1j*mode.omega], dtype=complex)
    return np.array([complex(mode.alpha), complex(mode.alpha)], dtype=complex)


def build_training_data(
    config:   SpectralEnergyConfig,
    n_samples: int,
    seed:     int,
) -> Tuple[np.ndarray, np.ndarray]:
    energy = SpectralEnergy(config)
    states, targets = energy.generate_training_samples(
        n_samples=n_samples, n_modes=3,
        zeta_range=(0.01, 0.50),
        logw_range=(1.0, 3.2),
        seed=seed,
    )
    X, Y = [], []
    for state, target in zip(states, targets):
        if len(state.modes) < 3:
            continue
        ev = np.concatenate([_mode_to_eigvals(state.modes[i]) for i in range(3)])
        X.append(_build_feature(ev, 'lca'))
        Y.append(target[:6])    # [Δζ₁,Δlw₁, Δζ₂,Δlw₂, Δζ₃,Δlw₃]
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)


def _train_mlp_on_arrays(
    walker:     EigenWalker,
    X:          np.ndarray,
    Y:          np.ndarray,
    n_epochs:   int  = 300,
    lr:         float = 1e-3,
    batch_size: int  = 64,
    seed:       int  = 0,
) -> List[float]:
    rng    = np.random.default_rng(seed)
    losses = []
    idx    = np.arange(len(X))
    for epoch in range(n_epochs):
        rng.shuffle(idx)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, len(X), batch_size):
            batch = idx[start:start + batch_size]
            Xb, Yb = X[batch], Y[batch]
            loss, grads = walker._mlp.mse_loss_and_grads(Xb, Yb)
            walker._mlp.adam_step(grads, lr=lr)
            epoch_loss += loss
            n_batches  += 1
        losses.append(epoch_loss / max(n_batches, 1))
    return losses


# ── Walker step ────────────────────────────────────────────────────────────────

def _apply_r3_step(
    theta:      dict,
    ev:         np.ndarray,
    walker:     EigenWalker,
    step_scale: float,
) -> dict:
    """Apply EigenWalker(n_modes=3) step to theta via analytic inversion."""
    try:
        state = SpectralState.from_eigvals(ev)
        if len(state.modes) < 3:
            return dict(theta)
        ev_in = np.concatenate([_mode_to_eigvals(state.modes[i]) for i in range(3)])
        delta = walker.predict_step(ev_in, 'lca')   # shape (6,)

        modes_new = []
        for i in range(3):
            mode    = state.modes[i]
            zeta_i  = mode.damping_ratio
            omega_i = mode.omega0
            if omega_i < 1e-12:
                modes_new.append((0.1, 1.0))
                continue
            dz = float(delta[2*i])     * step_scale
            dw = float(delta[2*i + 1]) * step_scale
            zeta_new  = float(np.clip(zeta_i + dz, 0.01, 0.95))
            omega_new = omega_i * math.exp(dw)
            modes_new.append((zeta_new, omega_new))

        return _tm3_clip(_invert_modes_3mode(modes_new, theta))
    except Exception:
        return dict(theta)


# ── 2D Strategy-A candidates ───────────────────────────────────────────────────

def _strategy_a_candidates_2d(
    ev:               np.ndarray,
    theta:            dict,
    adj_1d:           dict,
    omega_lo_anchors: Optional[List[float]] = None,
) -> Tuple[List[Tuple[dict, tuple]], tuple]:
    """
    Generate candidates from 2D Farey neighbors (Cartesian product):
    exactly ONE ratio changes by a Farey step.

    Returns (candidates, current_tube_2d).
    candidates: [(theta_cand, to_tube_2d), ...]
    """
    rho1, rho2 = _get_ratios_from_ev(ev)
    r1 = _nearest_node(rho1, _FAREY_NODES_1D)
    r2 = _nearest_node(rho2, _FAREY_NODES_1D)
    current_tube = (r1, r2)

    candidates = []
    try:
        state = SpectralState.from_eigvals(ev)
        if len(state.modes) < 3:
            return candidates, current_tube
        omega_lo_curr = state.modes[2].omega0   # lowest freq (rad/s)
    except Exception:
        return candidates, current_tube

    omega_lo_tries = [omega_lo_curr] + (omega_lo_anchors or [])

    # ── Dimension 1: change r1 (ω_hi/ω_mid), keep r2 (ω_mid/ω_lo) fixed ───────
    for (a1, b1) in adj_1d.get(r1, []):
        for omega_lo in omega_lo_tries:
            omega_mid = omega_lo  * (r2[0] / r2[1])       # keep ρ₂ = r2
            omega_hi  = omega_mid * (a1    / b1   )        # new  ρ₁ = a1:b1
            for z in ZETA_TARGETS:
                try:
                    raw = _invert_modes_3mode(
                        [(z, omega_hi), (z, omega_mid), (z, omega_lo)], theta
                    )
                    candidates.append((_tm3_clip(raw), (tuple([a1, b1]), r2)))
                except Exception:
                    pass

    # ── Dimension 2: change r2 (ω_mid/ω_lo), keep r1 (ω_hi/ω_mid) fixed ───────
    for (a2, b2) in adj_1d.get(r2, []):
        for omega_lo in omega_lo_tries:
            omega_mid = omega_lo  * (a2    / b2   )        # new  ρ₂ = a2:b2
            omega_hi  = omega_mid * (r1[0] / r1[1])        # keep ρ₁ = r1
            for z in ZETA_TARGETS:
                try:
                    raw = _invert_modes_3mode(
                        [(z, omega_hi), (z, omega_mid), (z, omega_lo)], theta
                    )
                    candidates.append((_tm3_clip(raw), (r1, tuple([a2, b2]))))
                except Exception:
                    pass

    return candidates, current_tube


def _dist_2d(tube: tuple, d1: dict, d2: dict) -> float:
    """2D Manhattan distance to target (separable on Cartesian product)."""
    return float(d1.get(tube[0], 999)) + float(d2.get(tube[1], 999))


# ── Routing selectors ──────────────────────────────────────────────────────────

def _try_farey_routed_2d(
    ev:     np.ndarray,
    theta:  dict,
    adj_1d: dict,
    d1:     dict,
    d2:     dict,
) -> Tuple[Optional[dict], tuple, Optional[tuple]]:
    """
    Select 2D Farey neighbor by minimum Manhattan distance to target.
    Two-pass: prefer _in_target candidates (last-mile aware) over pure d_min.
    """
    candidates, from_tube = _strategy_a_candidates_2d(
        ev, theta, adj_1d, omega_lo_anchors=OMEGA_LO_ANCHORS_RADS
    )

    best_theta, best_tube, best_d = None, None, float('inf')
    target_theta, target_tube    = None, None   # pass-1: _in_target candidates

    for (cand, to_tube) in candidates:
        d = _dist_2d(to_tube, d1, d2)
        # Validate eigenvalues
        try:
            ev_c  = _3mode_eigvals(cand)
            state = SpectralState.from_eigvals(ev_c)
            if len(state.modes) < 3:
                continue
        except Exception:
            continue

        # Pass 1: prefer candidates that immediately satisfy _in_target
        if d == 0 and target_theta is None and _in_target(cand):
            target_theta, target_tube = cand, to_tube

        # Pass 2: argmin d
        if d < best_d:
            best_d, best_theta, best_tube = d, cand, to_tube

    if target_theta is not None:
        return target_theta, from_tube, target_tube
    return best_theta, from_tube, best_tube


def _try_energy_routed_2d(
    ev:     np.ndarray,
    theta:  dict,
    adj_1d: dict,
    energy: SpectralEnergy,
) -> Tuple[Optional[dict], tuple, Optional[tuple]]:
    """Select 2D Farey neighbor by minimum energy (causes cycles — baseline)."""
    candidates, from_tube = _strategy_a_candidates_2d(ev, theta, adj_1d)

    best_theta, best_tube, best_e = None, None, float('inf')
    for (cand, to_tube) in candidates:
        try:
            ev_c  = _3mode_eigvals(cand)
            state = SpectralState.from_eigvals(ev_c)
            if len(state.modes) < 3:
                continue
            E = energy.compute(state)
        except Exception:
            continue
        if E < best_e:
            best_e, best_theta, best_tube = E, cand, to_tube

    return best_theta, from_tube, best_tube


# ── Run functions ──────────────────────────────────────────────────────────────

def _is_transition_product(from_tube: tuple, to_tube: tuple) -> Tuple[bool, bool]:
    """
    Returns (is_product, is_diagonal).
    product: exactly one ratio changes (and the change is Farey)
    diagonal: both ratios change simultaneously
    """
    r1_changed = from_tube[0] != to_tube[0]
    r2_changed = from_tube[1] != to_tube[1]
    if r1_changed and not r2_changed:
        adj = is_farey_adjacent_1d(from_tube[0][0], from_tube[0][1],
                                   to_tube[0][0],   to_tube[0][1])
        return adj, False
    if r2_changed and not r1_changed:
        adj = is_farey_adjacent_1d(from_tube[1][0], from_tube[1][1],
                                   to_tube[1][0],   to_tube[1][1])
        return adj, False
    if r1_changed and r2_changed:
        return False, True  # diagonal jump
    return False, False     # no change (self-loop)


def run_farey_routing_3mode(
    theta0:     dict,
    walker:     EigenWalker,
    energy:     SpectralEnergy,
    max_iter:   int,
    adj_1d:     dict,
    d1:         dict,    # BFS dist from target r1
    d2:         dict,    # BFS dist from target r2
    step_scale: float = 0.30,
    plateau_window: int = 8,
    plateau_eps:    float = 0.02,
    boundary_window: int = 20,
) -> Tuple[int, bool, List[dict]]:
    """Farey-distance routed plateau hybrid — 2D version."""
    theta = dict(theta0)
    try:
        ev = _3mode_eigvals(theta)
    except Exception:
        return max_iter, False, []

    energy_history: List[float] = []
    boundary_count  = 0
    hop_sequence:   List[dict]  = []

    for i in range(max_iter):
        if _in_target(theta):
            return i + 1, True, hop_sequence

        try:
            state = SpectralState.from_eigvals(ev)
            E_now = energy.compute(state)
        except Exception:
            E_now = float('inf')
        energy_history.append(E_now)
        if len(energy_history) > plateau_window:
            energy_history.pop(0)

        try:
            theta_raw = _apply_r3_step(theta, ev, walker, step_scale)
            theta_new = _tm3_clip(theta_raw)
            clipped   = any(abs(theta_raw[k] - theta_new[k]) > 1e-12 for k in TM3_KEYS)
        except Exception:
            theta_new = dict(theta)
            clipped   = False

        boundary_count = boundary_count + 1 if clipped else max(0, boundary_count - 1)

        energy_plateau = (
            len(energy_history) >= plateau_window
            and (energy_history[0] - energy_history[-1]) < plateau_eps
        )

        if energy_plateau or (boundary_count >= boundary_window):
            rho1_b, rho2_b = _get_ratios_from_ev(ev)

            # Suppress jumper when already in target 2D tube
            cur_tube = _get_current_tube(ev)
            d_curr = _dist_2d(cur_tube, d1, d2)
            if d_curr == 0:
                energy_history.clear()
                boundary_count = 0
                continue

            jumped, from_tube, to_tube = _try_farey_routed_2d(
                ev, theta, adj_1d, d1, d2
            )

            if jumped is not None and to_tube is not None:
                d_before = _dist_2d(from_tube, d1, d2)
                d_after  = _dist_2d(to_tube,   d1, d2)
                is_prod, is_diag = _is_transition_product(from_tube, to_tube)

                try:
                    ev_j = _3mode_eigvals(jumped)
                    rho1_a, rho2_a = _get_ratios_from_ev(ev_j)
                    ev = ev_j
                except Exception:
                    rho1_a = rho2_a = 0.0

                hop_sequence.append({
                    'step':     i,
                    'from':     from_tube,
                    'to':       to_tube,
                    'product':  is_prod,
                    'diagonal': is_diag,
                    'd_before': d_before,
                    'd_after':  d_after,
                    'rho1_before': rho1_b,
                    'rho2_before': rho2_b,
                    'rho1_after':  rho1_a,
                    'rho2_after':  rho2_a,
                })
                theta = jumped

            energy_history.clear()
            boundary_count = 0
            continue

        try:
            ev    = _3mode_eigvals(theta_new)
            theta = theta_new
        except Exception:
            pass

    return max_iter, False, hop_sequence


def run_energy_routing_3mode(
    theta0:     dict,
    walker:     EigenWalker,
    energy:     SpectralEnergy,
    max_iter:   int,
    adj_1d:     dict,
    step_scale: float = 0.30,
    plateau_window: int = 8,
    plateau_eps:    float = 0.02,
    boundary_window: int = 20,
) -> Tuple[int, bool, List[dict]]:
    """Energy-routed plateau hybrid (baseline — may cycle)."""
    theta = dict(theta0)
    try:
        ev = _3mode_eigvals(theta)
    except Exception:
        return max_iter, False, []

    energy_history: List[float] = []
    boundary_count = 0
    hop_sequence:   List[dict]  = []

    for i in range(max_iter):
        if _in_target(theta):
            return i + 1, True, hop_sequence
        try:
            state = SpectralState.from_eigvals(ev)
            E_now = energy.compute(state)
        except Exception:
            E_now = float('inf')
        energy_history.append(E_now)
        if len(energy_history) > plateau_window:
            energy_history.pop(0)

        try:
            theta_raw = _apply_r3_step(theta, ev, walker, step_scale)
            theta_new = _tm3_clip(theta_raw)
            clipped   = any(abs(theta_raw[k] - theta_new[k]) > 1e-12 for k in TM3_KEYS)
        except Exception:
            theta_new = dict(theta)
            clipped   = False

        boundary_count = boundary_count + 1 if clipped else max(0, boundary_count - 1)

        energy_plateau = (
            len(energy_history) >= plateau_window
            and (energy_history[0] - energy_history[-1]) < plateau_eps
        )

        if energy_plateau or (boundary_count >= boundary_window):
            jumped, from_tube, to_tube = _try_energy_routed_2d(ev, theta, adj_1d, energy)

            if jumped is not None and to_tube is not None:
                is_prod, is_diag = _is_transition_product(from_tube, to_tube)
                try:
                    ev_j = _3mode_eigvals(jumped)
                    ev   = ev_j
                except Exception:
                    pass
                hop_sequence.append({
                    'step':     i,
                    'from':     from_tube,
                    'to':       to_tube,
                    'product':  is_prod,
                    'diagonal': is_diag,
                })
                theta = jumped

            energy_history.clear()
            boundary_count = 0
            continue

        try:
            ev    = _3mode_eigvals(theta_new)
            theta = theta_new
        except Exception:
            pass

    return max_iter, False, hop_sequence


def run_walker_only_3mode(
    theta0:     dict,
    walker:     EigenWalker,
    energy:     SpectralEnergy,
    max_iter:   int,
    step_scale: float = 0.30,
) -> Tuple[int, bool]:
    theta = dict(theta0)
    try:
        ev = _3mode_eigvals(theta)
    except Exception:
        return max_iter, False

    for i in range(max_iter):
        if _in_target(theta):
            return i + 1, True
        try:
            theta_new = _tm3_clip(_apply_r3_step(theta, ev, walker, step_scale))
            ev        = _3mode_eigvals(theta_new)
            theta     = theta_new
        except Exception:
            pass

    return max_iter, False


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_hop_distribution(hop_counts: List[int], path: str, title: str = '') -> None:
    if not hop_counts:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    max_h = max(hop_counts) + 1
    ax.bar(range(max_h + 1),
           [hop_counts.count(h) for h in range(max_h + 1)],
           color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Hops')
    ax.set_ylabel('Count')
    ax.set_title(f'Hop Distribution — r=3 {title}')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def _plot_2d_atlas(
    all_transitions: Dict,
    adj_1d:          dict,
    target_tube:     tuple,
    d1:              dict,
    d2:              dict,
    path:            str,
    title:           str = '',
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, (ax, ratio_dim, adj_dim, d_dim, label) in enumerate(zip(
        axes,
        [lambda t: t[0], lambda t: t[1]],
        [lambda t: adj_1d.get(t[0], []), lambda t: adj_1d.get(t[1], [])],
        [d1, d2],
        ['ρ₁ (high/mid)', 'ρ₂ (mid/low)'],
    )):
        # Aggregate 1D transitions for this dimension
        counts_1d: Dict[tuple, int] = defaultdict(int)
        for (from_t, to_t), cnt in all_transitions.items():
            if from_t[0] != to_t[0]:   # dim 1 changed
                if ax_idx == 0:
                    counts_1d[(from_t[0], to_t[0])] += cnt
            else:                        # dim 2 changed
                if ax_idx == 1:
                    counts_1d[(from_t[1], to_t[1])] += cnt

        nodes = sorted(adj_1d.keys(), key=lambda n: n[0]/n[1])
        x = {n: i for i, n in enumerate(nodes)}
        y = {n: d_dim.get(n, -1) for n in nodes}

        # Draw edges with width proportional to count
        max_cnt = max(counts_1d.values(), default=1)
        for (fn, tn), cnt in counts_1d.items():
            if fn in x and tn in x:
                lw = 0.5 + 4.0 * cnt / max_cnt
                ax.plot([x[fn], x[tn]], [y[fn], y[tn]], '-o',
                        lw=lw, alpha=0.6, color='steelblue', ms=4)

        target_1d = target_tube[ax_idx]
        for n in nodes:
            c = 'gold' if n == target_1d else 'tomato'
            ax.plot(x[n], y[n], 'o', ms=10, color=c, zorder=5)
            ax.text(x[n], y[n] + 0.05, f'{n[0]}:{n[1]}', ha='center', fontsize=7)

        ax.set_xlabel(f'Node index in {label}')
        ax.set_ylabel('BFS distance to target')
        ax.set_title(f'{label} transitions {title}')

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials',  type=int,   default=100)
    parser.add_argument('--max-iter',  type=int,   default=600)
    parser.add_argument('--n-samples', type=int,   default=5000)
    parser.add_argument('--n-epochs',  type=int,   default=300)
    parser.add_argument('--K',         type=int,   default=8)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print('=== Trimode Atlas: r=3 Spectral Atlas (2D Farey Product Graph) ===')
    print(f'  trials={args.n_trials}  max_iter={args.max_iter}  K={args.K}  seed={args.seed}')
    print(f'  Start:  Z_{{(3:1, 2:1)}} (ρ₁∈[{START_RATIO1_LO},{START_RATIO1_HI}], '
          f'ρ₂∈[{START_RATIO2_LO},{START_RATIO2_HI}])')
    print(f'  Target: Z_{{(3:2, 4:3)}} (ρ₁∈[{TARGET_RATIO1_LO},{TARGET_RATIO1_HI}], '
          f'ρ₂∈[{TARGET_RATIO2_LO},{TARGET_RATIO2_HI}])')
    print(f'  Model:  Three decoupled oscillators, shared Rayleigh damping')
    print()

    # ── Build 1D Farey graph and 2D distances ──────────────────────────────────
    print('[Setup] 1D Farey graph F_K (for product construction)...')
    adj_1d, nodes_1d = build_farey_graph(args.K, ratio_lo=0.9, ratio_hi=5.0)
    n_nodes_1d = len(nodes_1d)
    _FAREY_NODES_1D.clear()
    _FAREY_NODES_1D.extend(nodes_1d)

    target_r1, target_r2 = TARGET_TUBE_2D
    d1 = farey_bfs(target_r1, adj_1d)   # distances to ρ₁ target
    d2 = farey_bfs(target_r2, adj_1d)   # distances to ρ₂ target

    # Number of 2D product nodes and edges (for reporting)
    n_nodes_2d = n_nodes_1d ** 2
    n_edges_2d = sum(
        len(adj_1d[r1]) + len(adj_1d[r2])
        for r1 in nodes_1d for r2 in nodes_1d
    )

    d_21_to_32 = d1.get((2,1), -1)   # Farey dist 2:1→3:2 in dim 1
    d_31_to_32 = d1.get((3,1), -1)   # Farey dist 3:1→3:2 in dim 1
    d_21_to_43 = d2.get((2,1), -1)   # Farey dist 2:1→4:3 in dim 2
    d_total    = int(d1.get((3,1),-1)) + int(d2.get((2,1),-1))

    print(f'  1D Farey F_{args.K}: {n_nodes_1d} nodes')
    print(f'  2D Product graph:   {n_nodes_2d} nodes, ~{n_edges_2d} edges')
    print(f'  2D Manhattan distance (3:1,2:1)→(3:2,4:3):')
    print(f'    d1(3:1→3:2) = {d_31_to_32}  +  d2(2:1→4:3) = {d_21_to_43}  = {d_total}')
    print(f'  Neighbors of Z_{{3:1}} in 1D: {sorted(adj_1d.get((3,1),[]))}')
    print(f'  Neighbors of Z_{{2:1}} in 1D: {sorted(adj_1d.get((2,1),[]))}')
    print(f'  Neighbors of Z_{{3:2}} in 1D: {sorted(adj_1d.get((3,2),[]))}')
    print(f'  Neighbors of Z_{{4:3}} in 1D: {sorted(adj_1d.get((4,3),[]))}')
    print()

    # ── Phase 1: Train EigenWalker(n_modes=3) ─────────────────────────────────
    print('[Phase 1] Training EigenWalker(n_modes=3) on (3:2,4:3) energy gradient...')
    config = make_target_config()
    t0 = time.time()
    X, Y = build_training_data(config, args.n_samples, seed=args.seed)
    walker = EigenWalker(hidden=128, n_modes=3, seed=args.seed)
    losses = _train_mlp_on_arrays(
        walker, X, Y, n_epochs=args.n_epochs, lr=1e-3, batch_size=64, seed=args.seed
    )
    print(f'  {len(X)} samples, {args.n_epochs} epochs in {time.time()-t0:.1f}s')
    print(f'  Loss: {losses[0]:.6f} → {losses[-1]:.6f}')
    print()

    energy = SpectralEnergy(config)

    # ── Phase 2: Benchmark ─────────────────────────────────────────────────────
    print(f'[Phase 2] Benchmark: {args.n_trials} trials × 3 methods...')

    results_fr = []   # Farey-distance routing
    results_er = []   # Energy routing (baseline)
    results_w  = []   # Walker-only
    all_seqs_fr: List[List[dict]] = []
    all_seqs_er: List[List[dict]] = []

    for trial in range(args.n_trials):
        theta0 = _sample_near_start(rng)

        iters_fr, suc_fr, seq_fr = run_farey_routing_3mode(
            theta0, walker, energy, args.max_iter, adj_1d, d1, d2
        )
        results_fr.append({'iters': iters_fr, 'success': suc_fr, 'hops': len(seq_fr)})
        all_seqs_fr.append(seq_fr)

        iters_er, suc_er, seq_er = run_energy_routing_3mode(
            theta0, walker, energy, args.max_iter, adj_1d
        )
        results_er.append({'iters': iters_er, 'success': suc_er, 'hops': len(seq_er)})
        all_seqs_er.append(seq_er)

        iters_w, suc_w = run_walker_only_3mode(theta0, walker, energy, args.max_iter)
        results_w.append({'iters': iters_w, 'success': suc_w})

        if (trial + 1) % 10 == 0:
            print(f'  [{trial+1:3d}/{args.n_trials}]  '
                  f'farey_routed: {iters_fr:3d}({"✓" if suc_fr else "✗"})  '
                  f'energy_routed: {iters_er:3d}({"✓" if suc_er else "✗"})  '
                  f'walker_only: {iters_w:3d}({"✓" if suc_w else "✗"})')

    print()

    # ── Phase 3: Analysis ──────────────────────────────────────────────────────

    def _fmt(res, name):
        succ  = [r for r in res if r['success']]
        n     = len(res)
        iters = [r['iters'] for r in succ]
        mean_i = float(np.mean(iters))    if iters else float('nan')
        med_i  = float(np.median(iters))  if iters else float('nan')
        p95_i  = float(np.percentile(iters, 95)) if iters else float('nan')
        fail_p = 100.0 * (n - len(succ)) / n
        print(f'{name:<42} {len(succ):>3}/{n:<3} '
              f'{mean_i:>8.1f} {med_i:>8.1f} {p95_i:>8.1f} {fail_p:>6.1f}%')

    print('=== Results ===')
    print(f'{"Method":<42} {"Succ":>6} {"Mean":>8} {"Median":>8} {"p95":>8} {"Fail%":>7}')
    print('-' * 78)
    _fmt(results_fr, 'Farey-Distance Routed (r=3)')
    _fmt(results_er, 'Energy-Routed (r=3)')
    _fmt(results_w,  'Pure Walker (no jumper, r=3)')
    print()

    # ── Farey topology analysis ────────────────────────────────────────────────

    # Count product vs diagonal transitions for Farey routing
    total_hops_fr = sum(len(s) for s in all_seqs_fr)
    prod_fr    = sum(1 for s in all_seqs_fr for h in s if h['product'])
    diag_fr    = sum(1 for s in all_seqs_fr for h in s if h['diagonal'])
    d_before_fr = [h['d_before'] for s in all_seqs_fr for h in s if h.get('d_before',0) >= 0]
    d_after_fr  = [h['d_after']  for s in all_seqs_fr for h in s if h.get('d_after', 0) >= 0]
    mono_fr     = sum(1 for b,a in zip(d_before_fr, d_after_fr) if a <= b)
    retro_fr    = sum(1 for b,a in zip(d_before_fr, d_after_fr) if a > b)
    opt_fr      = sum(1 for b,a in zip(d_before_fr, d_after_fr) if a == b - 1)

    total_hops_er = sum(len(s) for s in all_seqs_er)
    prod_er    = sum(1 for s in all_seqs_er for h in s if h['product'])
    diag_er    = sum(1 for s in all_seqs_er for h in s if h['diagonal'])

    hop_counts_fr = [r['hops'] for r in results_fr]
    hop_counts_er = [r['hops'] for r in results_er]

    print('=== 2D Farey Topology Analysis ===')
    print(f'  Product transitions (dim 1 only OR dim 2 only):')
    print(f'    Farey-routed:  {prod_fr}/{total_hops_fr} = '
          f'{100*prod_fr/max(total_hops_fr,1):.1f}%')
    print(f'    Energy-routed: {prod_er}/{total_hops_er} = '
          f'{100*prod_er/max(total_hops_er,1):.1f}%')
    print(f'  Diagonal transitions (both ratios change simultaneously):')
    print(f'    Farey-routed:  {diag_fr}/{total_hops_fr} = '
          f'{100*diag_fr/max(total_hops_fr,1):.1f}%')
    print(f'    Energy-routed: {diag_er}/{total_hops_er} = '
          f'{100*diag_er/max(total_hops_er,1):.1f}%')
    print()

    print('=== Farey-Routing Monotonicity (2D Manhattan) ===')
    print(f'  Monotonic (d_after ≤ d_before): '
          f'{mono_fr}/{len(d_before_fr)} = '
          f'{100*mono_fr/max(len(d_before_fr),1):.1f}%  [target: 100%]')
    print(f'  Optimal (d decreases by 1):     '
          f'{opt_fr}/{len(d_before_fr)} = '
          f'{100*opt_fr/max(len(d_before_fr),1):.1f}%')
    print(f'  Retrograde (d increases):       '
          f'{retro_fr}/{len(d_before_fr)} = '
          f'{100*retro_fr/max(len(d_before_fr),1):.1f}%  [target: 0%]')
    print(f'  Max hops (Farey-routed):         {max(hop_counts_fr, default=0)}'
          f'  [Expected ≤ 4×diameter ≈ {4*d_total}]')
    print()

    # Hop distribution comparison
    print('=== Hop Distribution ===')
    max_hop = max(max(hop_counts_fr, default=0), max(hop_counts_er, default=0))
    print(f'  {"Hops":<6}  {"Farey-Rtd":>12}  {"Energy-Rtd":>12}')
    for h in range(0, max_hop + 2):
        cnt_fr = sum(1 for x in hop_counts_fr if x == h)
        cnt_er = sum(1 for x in hop_counts_er if x == h)
        if cnt_fr > 0 or cnt_er > 0:
            bar_fr = '▒' * cnt_fr
            bar_er = '█' * cnt_er
            print(f'  {h} hops:  {cnt_fr:>3}/{args.n_trials} {bar_fr:<30}  '
                  f'{cnt_er:>3}/{args.n_trials} {bar_er}')

    # Multi-hop path examples (Farey-routed)
    multi_fr = [(i, seq) for i, seq in enumerate(all_seqs_fr) if len(seq) > 1]
    if multi_fr:
        print()
        print(f'=== Multi-Hop Examples — Farey-Routed ({len(multi_fr)} trials) ===')
        for i, seq in multi_fr[:8]:
            parts = [f'({h["from"][0][0]}:{h["from"][0][1]},{h["from"][1][0]}:{h["from"][1][1]})'
                     for h in seq]
            if seq:
                last = seq[-1]
                parts.append(f'({last["to"][0][0]}:{last["to"][0][1]},'
                              f'{last["to"][1][0]}:{last["to"][1][1]})')
            path = ' → '.join(parts)
            d_trace = ' '.join(f'd={h["d_before"]:.0f}→{h["d_after"]:.0f}' for h in seq)
            print(f'  Trial {i:3d}: {path}   [{d_trace}]   '
                  f'{"✓" if results_fr[i]["success"] else "✗"}')

    # Hypotheses
    print()
    print('=== Hypothesis Check ===')
    h1_2d_pct_fr = 100 * prod_fr / max(total_hops_fr, 1)
    h2_2d_diag   = 100 * diag_fr / max(total_hops_fr, 1)
    h5_2d_mono   = 100 * mono_fr / max(len(d_before_fr), 1)
    h5_2d_retro  = 100 * retro_fr / max(len(d_before_fr), 1)
    succ_fr = 100 * sum(1 for r in results_fr if r['success']) / len(results_fr)
    succ_er = 100 * sum(1 for r in results_er if r['success']) / len(results_er)

    print(f'  H1_2D (product-Farey transitions):    '
          f'{"✓ CONFIRMED" if h1_2d_pct_fr >= 99.0 else f"✗ {h1_2d_pct_fr:.1f}%"}  '
          f'({prod_fr}/{total_hops_fr})')
    print(f'  H2_2D (no diagonal transitions):       '
          f'{"✓ CONFIRMED" if h2_2d_diag < 1.0 else f"✗ {h2_2d_diag:.1f}% diagonal"}')
    print(f'  H5_2D (monotone routing):              '
          f'{"✓ CONFIRMED" if h5_2d_retro < 1.0 else f"✗ {h5_2d_retro:.1f}% retrograde"}  '
          f'({h5_2d_mono:.0f}% monotone)')
    print(f'  H5b_2D (100% success at r=3):          '
          f'{"✓ CONFIRMED" if succ_fr == 100 else "✗"} ({succ_fr:.0f}%)')
    print(f'  [energy-routing success for reference: {succ_er:.0f}%]')

    # Atlas of 2D transitions (Farey routing)
    atlas_fr: Dict[tuple, int] = defaultdict(int)
    for seq in all_seqs_fr:
        for hop in seq:
            atlas_fr[(hop['from'], hop['to'])] += 1

    if atlas_fr:
        print()
        print('=== Top 2D Atlas Transitions (Farey-Routed) ===')
        for (from_t, to_t), cnt in sorted(atlas_fr.items(), key=lambda x: -x[1])[:12]:
            is_prod, is_diag = _is_transition_product(from_t, to_t)
            d_f = _dist_2d(from_t, d1, d2)
            d_t = _dist_2d(to_t, d1, d2)
            tgt = '← TARGET' if to_t == TARGET_TUBE_2D else ''
            dim = 'dim1' if from_t[1]==to_t[1] else ('dim2' if from_t[0]==to_t[0] else 'diag')
            print(f'  ({from_t[0][0]}:{from_t[0][1]},{from_t[1][0]}:{from_t[1][1]}) → '
                  f'({to_t[0][0]}:{to_t[0][1]},{to_t[1][0]}:{to_t[1][1]})  '
                  f'cnt={cnt:3d}  {dim}  d:{d_f:.0f}→{d_t:.0f}  {tgt}')

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out = {
        'args': vars(args),
        'farey_graph': {
            'n_nodes_1d':    n_nodes_1d,
            'n_nodes_2d':    n_nodes_2d,
            'n_edges_2d':    n_edges_2d,
            'd_31_to_32':    int(d_31_to_32),
            'd_21_to_43':    int(d_21_to_43),
            'd_total_start': d_total,
        },
        'results_fr': [{'iters': r['iters'], 'success': r['success'], 'hops': r['hops']}
                       for r in results_fr],
        'results_er': [{'iters': r['iters'], 'success': r['success'], 'hops': r['hops']}
                       for r in results_er],
        'results_w':  [{'iters': r['iters'], 'success': r['success']}
                       for r in results_w],
        'topology': {
            'farey_routing': {
                'total_hops':       total_hops_fr,
                'product_pct':      round(h1_2d_pct_fr, 1),
                'diagonal_pct':     round(h2_2d_diag, 1),
                'monotone_pct':     round(h5_2d_mono, 1),
                'retrograde_pct':   round(h5_2d_retro, 1),
                'max_hops':         max(hop_counts_fr, default=0),
                'success_pct':      round(succ_fr, 1),
            },
            'energy_routing': {
                'total_hops':       total_hops_er,
                'product_pct':      round(100*prod_er/max(total_hops_er,1), 1),
                'diagonal_pct':     round(100*diag_er/max(total_hops_er,1), 1),
                'success_pct':      round(succ_er, 1),
            },
        },
        'atlas_fr': {
            f'({f[0][0]}:{f[0][1]},{f[1][0]}:{f[1][1]})→({t[0][0]}:{t[0][1]},{t[1][0]}:{t[1][1]})': cnt
            for (f, t), cnt in atlas_fr.items()
        },
    }
    json_path = os.path.join(RESULTS_DIR, 'trimode_atlas.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults: {json_path}')

    # Plots
    _plot_hop_distribution(
        hop_counts_fr,
        os.path.join(RESULTS_DIR, 'trimode_hops_farey.png'),
        title='Farey-Routed',
    )
    _plot_hop_distribution(
        hop_counts_er,
        os.path.join(RESULTS_DIR, 'trimode_hops_energy.png'),
        title='Energy-Routed',
    )
    _plot_2d_atlas(
        dict(atlas_fr), adj_1d, TARGET_TUBE_2D, d1, d2,
        os.path.join(RESULTS_DIR, 'trimode_atlas_farey.png'),
        title='(Farey-Routed)',
    )
    print(f'Plots: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
