"""
benchmarks/coupled_trimode_atlas.py — Coupled r=3: where separability breaks

Stress-tests the product-Farey routing architecture by introducing weak spring
coupling between the three previously decoupled oscillators.

Physical model — three masses with coupling springs:
  m·ẍ₁ + (k₁+k_c12)·x₁ - k_c12·x₂              + c·ẋ₁ = 0
  m·ẍ₂ - k_c12·x₁ + (k₂+k_c12+k_c23)·x₂ - k_c23·x₃ + c·ẋ₂ = 0
  m·ẍ₃              - k_c23·x₂ + (k₃+k_c23)·x₃  + c·ẋ₃ = 0

Stiffness matrix K:
  K = diag(k₁,k₂,k₃) + K_coupling
  K_coupling = [[k_c12, -k_c12, 0], [-k_c12, k_c12+k_c23, -k_c23], [0, -k_c23, k_c23]]

Damping: proportional C = γ·M (mass-proportional only; avoids freq-dependent β)
  so ζᵢ = γ/(2ωᵢ) — identical to alpha-Rayleigh, simpler for inversion

θ = (m, k₁, k₂, k₃, gamma, k_c12, k_c23)

Coupling parameter: ε = k_cij / kᵢ  (fraction of diagonal stiffness)
  ε = 0     → decoupled → exact product Farey (trimode_atlas baseline)
  ε = 0.01  → weak coupling → test graceful degradation
  ε = 0.05  → moderate coupling
  ε = 0.10  → strong coupling → expected topology breakdown

Sweep: run Farey routing at ε ∈ {0, 0.01, 0.02, 0.05, 0.10, 0.20}

Measurements per ε level:
  • Success rate
  • % monotone hops (H5_2D degradation)
  • % product transitions (H1_2D degradation)
  • Mean hops to target
  • Ratio deviation: |ρ_actual - ρ_Farey_node| after each jump
  • Effective graph distortion: d_eff = ‖λ_actual - λ_node‖ / ‖Δλ_hop‖

Hypotheses:
  H6:   Success rate degrades monotonically with ε (routing still works but slower)
  H7:   Monotonicity survives weak coupling (ε < ε_crit) — graceful degradation
  H8:   ε_crit ≈ 0.05–0.10 (from structural perturbation theory estimates)
  H9:   Diagonal transitions appear above ε_crit (coupling cross-pollinates dims)
  H10:  Ratio snap error ∝ ε (linear perturbation regime at small ε)
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

# Same target as trimode_atlas: Z_{(3:2, 4:3)}
TARGET_TUBE_2D  = ((3, 2), (4, 3))
TARGET_RATIO1_LO, TARGET_RATIO1_HI = 1.40, 1.60   # ρ₁ = ω_hi/ω_mid
TARGET_RATIO2_LO, TARGET_RATIO2_HI = 1.20, 1.45   # ρ₂ = ω_mid/ω_lo
TARGET_F_LO,  TARGET_F_HI  = 4.0, 15.0            # Hz, lowest mode
TARGET_ZETA_LO, TARGET_ZETA_HI = 0.05, 0.30

START_RATIO1_LO, START_RATIO1_HI = 2.80, 3.20
START_RATIO2_LO, START_RATIO2_HI = 1.85, 2.15

ZETA_TARGETS = [0.10, 0.15, 0.20]
OMEGA_LO_ANCHORS_RADS = [2.0 * math.pi * f for f in [5.0, 7.0, 9.0, 12.0, 15.0]]

# Coupling levels to sweep
EPSILON_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

# Parameter bounds (extended for coupling)
TM3C_BOUNDS = {
    'm':     (0.01,  100.0),
    'k1':    (1e-4,  1e6),
    'k2':    (1e-4,  1e6),
    'k3':    (1e-4,  1e6),
    'gamma': (1e-4,  200.0),
    'k_c12': (0.0,   1e5),
    'k_c23': (0.0,   1e5),
}
TM3C_KEYS = ['m', 'k1', 'k2', 'k3', 'gamma', 'k_c12', 'k_c23']

# Module-level Farey nodes (set at startup)
_FAREY_NODES_1D: List[tuple] = []


# ── Coupled three-mass-spring physics ─────────────────────────────────────────

def _coupled_eigvals(theta: dict) -> np.ndarray:
    """
    Eigenvalues of coupled three-mass-spring with mass-proportional damping.

    Equations of motion:
      M·ẍ + C·ẋ + K·x = 0
      M = m·I
      C = gamma·M  (mass-proportional → ζᵢ = gamma/(2ωᵢ))
      K = diag(k1,k2,k3) + K_coupling

    State-space: ẏ = A·y, y = [x; ẋ]
      A = [[0, I], [-M⁻¹K, -M⁻¹C]]

    Returns 6 eigenvalues (3 complex conjugate pairs if underdamped).
    """
    m      = theta['m']
    k1, k2, k3 = theta['k1'], theta['k2'], theta['k3']
    gamma  = theta['gamma']
    k_c12  = theta.get('k_c12', 0.0)
    k_c23  = theta.get('k_c23', 0.0)

    # Stiffness matrix
    K = np.array([
        [k1 + k_c12,          -k_c12,                0.0],
        [-k_c12,   k2 + k_c12 + k_c23,           -k_c23],
        [0.0,                 -k_c23,      k3 + k_c23],
    ])

    # Mass and damping matrices
    M     = m * np.eye(3)
    C     = gamma * M
    Mi    = (1.0 / m) * np.eye(3)

    # Build 6×6 state-space A
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = -Mi @ K
    A[3:, 3:] = -Mi @ C

    vals = np.linalg.eigvals(A)
    # Sort: most negative Re first (highest freq), descending Im to break ties
    vals = vals[np.argsort(-np.abs(vals.imag))]
    return vals.astype(complex)


def _coupled_modal_freqs_hz(theta: dict) -> Tuple[float, float, float]:
    """Damped natural frequencies (Hz) of the coupled system, descending."""
    ev   = _coupled_eigvals(theta)
    # Extract positive imaginary parts (damped frequencies in rad/s)
    wd   = sorted([v.imag for v in ev if v.imag > 1e-6], reverse=True)
    while len(wd) < 3:
        wd.append(1e-6)
    return tuple(w / (2.0 * math.pi) for w in wd[:3])


def _coupled_modal_zetas(theta: dict) -> List[float]:
    """Damping ratios of the coupled system, sorted by descending frequency."""
    ev    = _coupled_eigvals(theta)
    modes = []
    for v in ev:
        if v.imag > 1e-6:
            omega = abs(v)
            zeta  = max(0.0, min(-v.real / max(omega, 1e-12), 0.9999))
            wd    = v.imag
            modes.append((wd, zeta))
    modes.sort(key=lambda x: -x[0])
    return [m[1] for m in modes[:3]] if modes else [0.0, 0.0, 0.0]


def _tm3c_clip(theta: dict) -> dict:
    return {k: float(np.clip(theta.get(k, TM3C_BOUNDS[k][0]),
                             TM3C_BOUNDS[k][0], TM3C_BOUNDS[k][1]))
            for k in TM3C_KEYS}


def _in_target_coupled(theta: dict) -> bool:
    try:
        freqs = _coupled_modal_freqs_hz(theta)
        r1 = freqs[0] / max(freqs[1], 1e-12)
        r2 = freqs[1] / max(freqs[2], 1e-12)
        zs = _coupled_modal_zetas(theta)
        return (
            TARGET_RATIO1_LO <= r1 <= TARGET_RATIO1_HI
            and TARGET_RATIO2_LO <= r2 <= TARGET_RATIO2_HI
            and TARGET_F_LO <= freqs[2] <= TARGET_F_HI
            and all(TARGET_ZETA_LO <= z <= TARGET_ZETA_HI for z in zs)
        )
    except Exception:
        return False


# ── Inversion: coupled system → θ ─────────────────────────────────────────────

def _invert_coupled(
    omega_hi: float,
    omega_mid: float,
    omega_lo: float,
    zeta_target: float,
    theta_current: dict,
    epsilon: float,
) -> dict:
    """
    Given target modal frequencies and a coupling level ε,
    find θ that approximately achieves them.

    Approach:
      1. Compute decoupled k values (exact for ε=0)
      2. Subtract coupling stiffness correction from diagonal (first-order)
      3. gamma from ζ = gamma/(2·ω_lo)  [lowest mode, least sensitive]
      4. k_cij = ε × k_i
    """
    m     = float(theta_current.get('m', 1.0))

    # First-order coupling correction:
    # In the decoupled limit, k_i_eff ≈ k_i + k_cij (neighbor coupling adds to diagonal)
    # To achieve target ωᵢ, need k_i = m·ωᵢ² - coupling_correction
    k_c12 = epsilon * m * omega_hi**2   # ε × k1_decoupled
    k_c23 = epsilon * m * omega_mid**2  # ε × k2_decoupled

    # Corrected diagonal stiffnesses
    k1 = max(m * omega_hi**2  - k_c12,             1e-4)
    k2 = max(m * omega_mid**2 - k_c12 - k_c23,     1e-4)
    k3 = max(m * omega_lo**2  - k_c23,             1e-4)

    # gamma from lowest mode ζ = gamma/(2·ω_lo)
    gamma = max(2.0 * zeta_target * omega_lo, 1e-4)

    return {
        'm':     m,
        'k1':    k1,
        'k2':    k2,
        'k3':    k3,
        'gamma': gamma,
        'k_c12': k_c12,
        'k_c23': k_c23,
    }


# ── Farey graph (same as trimode_atlas) ───────────────────────────────────────

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


def _nearest_node(ratio: float, nodes_1d: List[tuple]) -> tuple:
    return min(nodes_1d, key=lambda n: abs(ratio - n[0] / n[1]))


def _nearest_node_err(ratio: float, nodes_1d: List[tuple]) -> Tuple[tuple, float]:
    """Return (nearest_node, absolute_error)."""
    best = min(nodes_1d, key=lambda n: abs(ratio - n[0] / n[1]))
    return best, abs(ratio - best[0] / best[1])


# ── Ratio extraction (from eigenvalues) ───────────────────────────────────────

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


def _get_current_tube(ev: np.ndarray) -> Tuple[tuple, tuple]:
    rho1, rho2 = _get_ratios_from_ev(ev)
    r1 = _nearest_node(rho1, _FAREY_NODES_1D)
    r2 = _nearest_node(rho2, _FAREY_NODES_1D)
    return r1, r2


def _snap_errors(ev: np.ndarray) -> Tuple[float, float]:
    """How far the current eigenvalue ratios are from their Farey snap nodes."""
    rho1, rho2 = _get_ratios_from_ev(ev)
    _, e1 = _nearest_node_err(rho1, _FAREY_NODES_1D)
    _, e2 = _nearest_node_err(rho2, _FAREY_NODES_1D)
    return e1, e2


def _dist_2d(tube: tuple, d1: dict, d2: dict) -> float:
    return float(d1.get(tube[0], 999)) + float(d2.get(tube[1], 999))


def _is_transition_product(from_tube: tuple, to_tube: tuple) -> Tuple[bool, bool]:
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
        return False, True
    return False, False


# ── Sampling ───────────────────────────────────────────────────────────────────

def _sample_near_start(rng: np.random.Generator, epsilon: float) -> dict:
    """Sample near Z_{(3:1, 2:1)} with given coupling level."""
    m    = float(rng.uniform(0.1, 2.0))
    f_lo = float(rng.uniform(5.0, 9.0))
    omega_lo = 2.0 * math.pi * f_lo

    rho2     = float(rng.uniform(START_RATIO2_LO, START_RATIO2_HI))
    rho1     = float(rng.uniform(START_RATIO1_LO, START_RATIO1_HI))
    omega_mid = omega_lo  * rho2
    omega_hi  = omega_mid * rho1

    z = float(rng.uniform(0.08, 0.22))   # uniform zeta all modes

    raw = _invert_coupled(omega_hi, omega_mid, omega_lo, z, {'m': m}, epsilon)
    raw['m'] = m
    return _tm3c_clip(raw)


# ── Strategy-A candidates (2D product jumper) ─────────────────────────────────

def _strategy_a_candidates_2d_coupled(
    ev:               np.ndarray,
    theta:            dict,
    adj_1d:           dict,
    epsilon:          float,
    omega_lo_anchors: Optional[List[float]] = None,
) -> Tuple[List[Tuple[dict, tuple]], tuple]:
    """
    Same product-Farey structure as trimode_atlas, but using coupled inversion.
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
        omega_lo_curr = state.modes[2].omega0
    except Exception:
        return candidates, current_tube

    omega_lo_tries = [omega_lo_curr] + (omega_lo_anchors or [])

    # Dim 1: change r1, keep r2 fixed
    for (a1, b1) in adj_1d.get(r1, []):
        for omega_lo in omega_lo_tries:
            omega_mid = omega_lo  * (r2[0] / r2[1])
            omega_hi  = omega_mid * (a1    / b1   )
            for z in ZETA_TARGETS:
                try:
                    raw = _invert_coupled(omega_hi, omega_mid, omega_lo, z, theta, epsilon)
                    raw['m'] = theta['m']
                    candidates.append((_tm3c_clip(raw), (tuple([a1, b1]), r2)))
                except Exception:
                    pass

    # Dim 2: change r2, keep r1 fixed
    for (a2, b2) in adj_1d.get(r2, []):
        for omega_lo in omega_lo_tries:
            omega_mid = omega_lo  * (a2    / b2   )
            omega_hi  = omega_mid * (r1[0] / r1[1])
            for z in ZETA_TARGETS:
                try:
                    raw = _invert_coupled(omega_hi, omega_mid, omega_lo, z, theta, epsilon)
                    raw['m'] = theta['m']
                    candidates.append((_tm3c_clip(raw), (r1, tuple([a2, b2]))))
                except Exception:
                    pass

    return candidates, current_tube


# ── Walker step ────────────────────────────────────────────────────────────────

def _mode_to_eigvals(mode: SpectralMode) -> np.ndarray:
    if mode.omega > 1e-12:
        return np.array([mode.alpha + 1j*mode.omega,
                         mode.alpha - 1j*mode.omega], dtype=complex)
    return np.array([complex(mode.alpha), complex(mode.alpha)], dtype=complex)


def _apply_r3_step_coupled(
    theta:      dict,
    ev:         np.ndarray,
    walker:     EigenWalker,
    step_scale: float,
    epsilon:    float,
) -> dict:
    try:
        state = SpectralState.from_eigvals(ev)
        if len(state.modes) < 3:
            return dict(theta)
        ev_in = np.concatenate([_mode_to_eigvals(state.modes[i]) for i in range(3)])
        delta = walker.predict_step(ev_in, 'lca')

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

        # Use coupled inversion
        (z_hi, w_hi), (z_mid, w_mid), (z_lo, w_lo) = modes_new[0], modes_new[1], modes_new[2]
        raw = _invert_coupled(w_hi, w_mid, w_lo, float(np.mean([z_hi, z_mid, z_lo])),
                              theta, epsilon)
        raw['m'] = theta['m']
        return _tm3c_clip(raw)
    except Exception:
        return dict(theta)


# ── Farey routing selector ────────────────────────────────────────────────────

def _try_farey_routed_2d_coupled(
    ev:       np.ndarray,
    theta:    dict,
    adj_1d:   dict,
    d1:       dict,
    d2:       dict,
    epsilon:  float,
) -> Tuple[Optional[dict], tuple, Optional[tuple], List[float]]:
    """
    Returns (jumped_theta, from_tube, to_tube, snap_errors_after).
    snap_errors_after: [err_rho1, err_rho2] — how far post-jump ratios are from node.
    """
    candidates, from_tube = _strategy_a_candidates_2d_coupled(
        ev, theta, adj_1d, epsilon, omega_lo_anchors=OMEGA_LO_ANCHORS_RADS
    )

    best_theta, best_tube, best_d = None, None, float('inf')
    target_theta, target_tube     = None, None

    for (cand, to_tube) in candidates:
        d = _dist_2d(to_tube, d1, d2)
        try:
            ev_c  = _coupled_eigvals(cand)
            state = SpectralState.from_eigvals(ev_c)
            if len(state.modes) < 3:
                continue
        except Exception:
            continue

        if d == 0 and target_theta is None and _in_target_coupled(cand):
            target_theta, target_tube = cand, to_tube

        if d < best_d:
            best_d, best_theta, best_tube = d, cand, to_tube

    winner_theta = target_theta if target_theta is not None else best_theta
    winner_tube  = target_tube  if target_theta is not None else best_tube

    snap_errs = [float('nan'), float('nan')]
    if winner_theta is not None:
        try:
            ev_w = _coupled_eigvals(winner_theta)
            e1, e2 = _snap_errors(ev_w)
            snap_errs = [e1, e2]
        except Exception:
            pass

    return winner_theta, from_tube, winner_tube, snap_errs


# ── Run function ──────────────────────────────────────────────────────────────

def run_farey_coupled(
    theta0:         dict,
    walker:         EigenWalker,
    energy:         SpectralEnergy,
    max_iter:       int,
    adj_1d:         dict,
    d1:             dict,
    d2:             dict,
    epsilon:        float,
    step_scale:     float = 0.30,
    plateau_window: int   = 8,
    plateau_eps:    float = 0.02,
    boundary_window: int  = 20,
) -> Tuple[int, bool, List[dict]]:
    """Farey-distance routed plateau hybrid for coupled system."""
    theta = dict(theta0)
    try:
        ev = _coupled_eigvals(theta)
    except Exception:
        return max_iter, False, []

    energy_history: List[float] = []
    boundary_count = 0
    hop_sequence:   List[dict]  = []

    for i in range(max_iter):
        if _in_target_coupled(theta):
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
            theta_raw = _apply_r3_step_coupled(theta, ev, walker, step_scale, epsilon)
            theta_new = _tm3c_clip(theta_raw)
            clipped   = any(abs(theta_raw.get(k, 0) - theta_new[k]) > 1e-12 for k in TM3C_KEYS)
        except Exception:
            theta_new = dict(theta)
            clipped   = False

        boundary_count = boundary_count + 1 if clipped else max(0, boundary_count - 1)

        energy_plateau = (
            len(energy_history) >= plateau_window
            and (energy_history[0] - energy_history[-1]) < plateau_eps
        )

        if energy_plateau or (boundary_count >= boundary_window):
            cur_tube = _get_current_tube(ev)
            d_curr = _dist_2d(cur_tube, d1, d2)

            if d_curr == 0:
                energy_history.clear()
                boundary_count = 0
                continue

            jumped, from_tube, to_tube, snap_errs = _try_farey_routed_2d_coupled(
                ev, theta, adj_1d, d1, d2, epsilon
            )

            if jumped is not None and to_tube is not None:
                d_before = _dist_2d(from_tube, d1, d2)
                d_after  = _dist_2d(to_tube,   d1, d2)
                is_prod, is_diag = _is_transition_product(from_tube, to_tube)

                try:
                    ev_j = _coupled_eigvals(jumped)
                    rho1_a, rho2_a = _get_ratios_from_ev(ev_j)
                    ev = ev_j
                except Exception:
                    rho1_a = rho2_a = 0.0

                rho1_b, rho2_b = _get_ratios_from_ev(ev)

                hop_sequence.append({
                    'step':      i,
                    'from':      from_tube,
                    'to':        to_tube,
                    'product':   is_prod,
                    'diagonal':  is_diag,
                    'd_before':  d_before,
                    'd_after':   d_after,
                    'snap_err1': snap_errs[0],
                    'snap_err2': snap_errs[1],
                })
                theta = jumped

            energy_history.clear()
            boundary_count = 0
            continue

        try:
            ev    = _coupled_eigvals(theta_new)
            theta = theta_new
        except Exception:
            pass

    return max_iter, False, hop_sequence


# ── Training (reuse trimode architecture) ─────────────────────────────────────

def make_target_config() -> SpectralEnergyConfig:
    return SpectralEnergyConfig(
        target_zetas=[0.15, 0.15, 0.15],
        target_omega0s_hz=[12.0, 8.0, 6.0],
        K_ratios=8,
        w_stab=10.0,
        w_damp=1.0,
        w_freq=0.5,
        w_harm=2.0,
    )


def build_training_data(
    config:    SpectralEnergyConfig,
    n_samples: int,
    seed:      int,
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
        Y.append(target[:6])
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)


def _train_mlp_on_arrays(
    walker:     EigenWalker,
    X:          np.ndarray,
    Y:          np.ndarray,
    n_epochs:   int   = 300,
    lr:         float = 1e-3,
    batch_size: int   = 64,
    seed:       int   = 0,
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


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_coupling_sweep(epsilon_levels, metrics, path):
    """
    Multi-panel coupling sweep plot:
    - Success rate vs ε
    - Monotonicity % vs ε
    - Product % (H1_2D) vs ε
    - Mean snap error vs ε
    """
    epsilons = [m['epsilon'] for m in metrics]
    success  = [m['success_pct']   for m in metrics]
    mono     = [m['monotone_pct']  for m in metrics]
    product  = [m['product_pct']   for m in metrics]
    diagonal = [m['diagonal_pct']  for m in metrics]
    snap_err = [m['mean_snap_err'] for m in metrics]
    mean_hop = [m['mean_hops'] if not math.isnan(m['mean_hops']) else 0 for m in metrics]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Coupling Sweep: r=3 Farey Routing Degradation', fontsize=13)

    def _scatter(ax, x, y, ylabel, color='steelblue', hline=None):
        ax.plot(x, y, 'o-', color=color, lw=2, ms=7)
        ax.set_xlabel('Coupling ε')
        ax.set_ylabel(ylabel)
        ax.set_xscale('symlog', linthresh=0.005)
        ax.grid(True, alpha=0.3)
        if hline is not None:
            ax.axhline(hline, ls='--', color='red', alpha=0.4, label=f'baseline={hline}')
            ax.legend(fontsize=7)

    _scatter(axes[0, 0], epsilons, success,  'Success rate (%)', hline=100.0)
    _scatter(axes[0, 1], epsilons, mono,     'Monotone hops (%)', hline=100.0)
    _scatter(axes[0, 2], epsilons, product,  'Product-Farey hops (%)', hline=100.0)
    _scatter(axes[1, 0], epsilons, diagonal, 'Diagonal hops (%)', color='tomato', hline=0.0)
    _scatter(axes[1, 1], epsilons, snap_err, 'Mean ratio snap error')
    _scatter(axes[1, 2], epsilons, mean_hop, 'Mean hops (success only)')

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def _plot_snap_error_distribution(all_snap_errs_by_eps, epsilons, path):
    """Box plot of snap errors per coupling level."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data   = [all_snap_errs_by_eps.get(eps, [0.0]) for eps in epsilons]
    labels = [f'ε={eps}' for eps in epsilons]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel('Ratio snap error |ρ_actual - ρ_node|')
    ax.set_title('Ratio snap error vs coupling strength')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials',  type=int,   default=100)
    parser.add_argument('--max-iter',  type=int,   default=800)
    parser.add_argument('--n-samples', type=int,   default=5000)
    parser.add_argument('--n-epochs',  type=int,   default=300)
    parser.add_argument('--K',         type=int,   default=8)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print('=== Coupled Trimode Atlas: Separability Stress Test ===')
    print(f'  trials={args.n_trials}  max_iter={args.max_iter}  K={args.K}  seed={args.seed}')
    print(f'  ε sweep: {EPSILON_LEVELS}')
    print(f'  Model: three coupled masses, mass-proportional damping')
    print(f'  Coupling: ε = k_cij/k_i  (ε=0 → decoupled baseline)')
    print()

    # ── Build Farey graph ──────────────────────────────────────────────────────
    print('[Setup] Building 1D Farey graph F_K...')
    adj_1d, nodes_1d = build_farey_graph(args.K, ratio_lo=0.9, ratio_hi=5.0)
    _FAREY_NODES_1D.clear()
    _FAREY_NODES_1D.extend(nodes_1d)
    n_nodes_1d = len(nodes_1d)

    target_r1, target_r2 = TARGET_TUBE_2D
    d1 = farey_bfs(target_r1, adj_1d)
    d2 = farey_bfs(target_r2, adj_1d)

    d_31_to_32 = d1.get((3, 1), -1)
    d_21_to_43 = d2.get((2, 1), -1)
    d_total    = int(d_31_to_32) + int(d_21_to_43)

    print(f'  F_{args.K}: {n_nodes_1d} nodes')
    print(f'  2D Manhattan distance (3:1,2:1)→(3:2,4:3): d1={d_31_to_32} + d2={d_21_to_43} = {d_total}')
    print()

    # ── Train walker (same as trimode_atlas, ε-independent) ───────────────────
    print('[Phase 1] Training EigenWalker(n_modes=3)...')
    config = make_target_config()
    t0 = time.time()
    X, Y = build_training_data(config, args.n_samples, seed=args.seed)
    walker = EigenWalker(hidden=128, n_modes=3, seed=args.seed)
    _train_mlp_on_arrays(walker, X, Y, n_epochs=args.n_epochs, lr=1e-3,
                         batch_size=64, seed=args.seed)
    print(f'  {len(X)} samples, {args.n_epochs} epochs in {time.time()-t0:.1f}s')
    print()

    energy = SpectralEnergy(config)

    # ── Coupling sweep ─────────────────────────────────────────────────────────
    print(f'[Phase 2] Coupling sweep: {len(EPSILON_LEVELS)} ε levels × {args.n_trials} trials')
    print()

    all_metrics  = []
    all_snap_errs_by_eps: Dict[float, List[float]] = {}
    all_results_by_eps = {}

    for eps in EPSILON_LEVELS:
        print(f'  ── ε = {eps:.3f} ──────────────────────────────')
        rng_eps = np.random.default_rng(args.seed + int(eps * 1000))

        results_eps: List[dict] = []
        all_seqs_eps: List[List[dict]] = []

        for trial in range(args.n_trials):
            theta0 = _sample_near_start(rng_eps, eps)
            iters, suc, seq = run_farey_coupled(
                theta0, walker, energy, args.max_iter, adj_1d, d1, d2, eps
            )
            results_eps.append({'iters': iters, 'success': suc, 'hops': len(seq)})
            all_seqs_eps.append(seq)

        # Compute metrics for this ε
        succ_trials   = [r for r in results_eps if r['success']]
        total_hops    = sum(len(s) for s in all_seqs_eps)
        prod_hops     = sum(1 for s in all_seqs_eps for h in s if h['product'])
        diag_hops     = sum(1 for s in all_seqs_eps for h in s if h['diagonal'])
        d_befores     = [h['d_before'] for s in all_seqs_eps for h in s]
        d_afters      = [h['d_after']  for s in all_seqs_eps for h in s]
        mono          = sum(1 for b, a in zip(d_befores, d_afters) if a <= b)
        retro         = sum(1 for b, a in zip(d_befores, d_afters) if a > b)
        snap_errs     = [h['snap_err1'] for s in all_seqs_eps for h in s
                         if not math.isnan(h.get('snap_err1', float('nan')))]
        snap_errs    += [h['snap_err2'] for s in all_seqs_eps for h in s
                         if not math.isnan(h.get('snap_err2', float('nan')))]
        all_snap_errs_by_eps[eps] = snap_errs

        success_pct  = 100.0 * len(succ_trials) / args.n_trials
        monotone_pct = 100.0 * mono / max(len(d_befores), 1)
        product_pct  = 100.0 * prod_hops / max(total_hops, 1)
        diagonal_pct = 100.0 * diag_hops / max(total_hops, 1)
        retro_pct    = 100.0 * retro / max(len(d_befores), 1)
        mean_snap    = float(np.mean(snap_errs)) if snap_errs else float('nan')
        mean_hops    = float(np.mean([r['iters'] for r in succ_trials])) if succ_trials else float('nan')
        max_hops     = max((r['hops'] for r in results_eps), default=0)

        m = {
            'epsilon':       eps,
            'success_pct':   success_pct,
            'monotone_pct':  monotone_pct,
            'product_pct':   product_pct,
            'diagonal_pct':  diagonal_pct,
            'retrograde_pct': retro_pct,
            'mean_snap_err': mean_snap,
            'mean_hops':     mean_hops,
            'max_hops':      max_hops,
            'total_hops':    total_hops,
        }
        all_metrics.append(m)
        all_results_by_eps[str(eps)] = [
            {'iters': r['iters'], 'success': r['success'], 'hops': r['hops']}
            for r in results_eps
        ]

        print(f'    success={success_pct:.0f}%  mono={monotone_pct:.1f}%  '
              f'product={product_pct:.1f}%  diag={diagonal_pct:.1f}%  '
              f'retro={retro_pct:.1f}%')
        print(f'    snap_err={mean_snap:.4f}  mean_hops={mean_hops:.1f}  '
              f'max_hops={max_hops}  total_hops={total_hops}')
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print('=== Coupling Sweep Summary ===')
    print(f'{"ε":>7}  {"Succ%":>6}  {"Mono%":>6}  {"Prod%":>6}  '
          f'{"Diag%":>6}  {"SnapErr":>8}  {"MeanHop":>8}')
    print('-' * 60)
    for m in all_metrics:
        print(f'{m["epsilon"]:>7.3f}  {m["success_pct"]:>6.1f}  '
              f'{m["monotone_pct"]:>6.1f}  {m["product_pct"]:>6.1f}  '
              f'{m["diagonal_pct"]:>6.1f}  {m["mean_snap_err"]:>8.4f}  '
              f'{m["mean_hops"]:>8.1f}')
    print()

    # ── Hypothesis checks ─────────────────────────────────────────────────────
    print('=== Hypothesis Check ===')
    success_vals = [m['success_pct']  for m in all_metrics]
    mono_vals    = [m['monotone_pct'] for m in all_metrics]
    diag_vals    = [m['diagonal_pct'] for m in all_metrics]
    snap_vals    = [m['mean_snap_err'] for m in all_metrics]

    h6_monotone_success = all(
        success_vals[i] >= success_vals[i+1] - 5.0  # allow 5% noise
        for i in range(len(success_vals)-1)
    )

    # Find ε_crit: first ε where success drops below 95% or mono drops below 90%
    eps_crit_success = None
    eps_crit_mono    = None
    for m in all_metrics:
        if eps_crit_success is None and m['success_pct'] < 95.0:
            eps_crit_success = m['epsilon']
        if eps_crit_mono is None and m['monotone_pct'] < 90.0:
            eps_crit_mono = m['epsilon']

    h7_weak_coupling = all_metrics[1]['monotone_pct'] >= 90.0 if len(all_metrics) > 1 else False
    h8_crit = eps_crit_success is not None and 0.02 <= eps_crit_success <= 0.15
    h9_diag = any(m['diagonal_pct'] > 1.0 for m in all_metrics[2:])

    # H10: linear snap error at small ε
    if len(snap_vals) >= 3 and all(not math.isnan(v) for v in snap_vals[:3]):
        # Check if snap_err[1]/snap_err[0] ≈ EPSILON_LEVELS[1]/EPSILON_LEVELS[0]
        # (with ε=0 giving near-zero error)
        eps_nonzero = [(EPSILON_LEVELS[i], snap_vals[i]) for i in range(1, min(4, len(snap_vals)))]
        h10_linear = (len(eps_nonzero) >= 2 and
                      eps_nonzero[-1][1] / max(eps_nonzero[0][1], 1e-8) <
                      eps_nonzero[-1][0] / max(eps_nonzero[0][0], 1e-8) * 3.0)
    else:
        h10_linear = False

    print(f'  H6 (monotone success degradation): '
          f'{"✓ CONFIRMED" if h6_monotone_success else "✗ non-monotone"}')
    print(f'  H7 (weak coupling ε=0.01 preserves mono≥90%): '
          f'{"✓ CONFIRMED" if h7_weak_coupling else "✗"}  '
          f'(mono={all_metrics[1]["monotone_pct"] if len(all_metrics)>1 else "?"}%)')
    print(f'  H8 (ε_crit ∈ [0.02, 0.15]):  '
          f'{"✓ CONFIRMED" if h8_crit else "✗"}  '
          f'ε_crit_success={eps_crit_success}  ε_crit_mono={eps_crit_mono}')
    print(f'  H9 (diagonal transitions above ε_crit): '
          f'{"✓ CONFIRMED" if h9_diag else "✗ no diagonals"}')
    print(f'  H10 (snap error ∝ ε, linear perturbation): '
          f'{"✓ CONFIRMED" if h10_linear else "✗"}')
    print()

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out = {
        'args':          vars(args),
        'epsilon_levels': EPSILON_LEVELS,
        'farey_graph': {
            'n_nodes_1d':    n_nodes_1d,
            'd_total_start': d_total,
        },
        'metrics':       all_metrics,
        'results_by_eps': all_results_by_eps,
        'hypotheses': {
            'H6_monotone_success': h6_monotone_success,
            'H7_weak_coupling':    h7_weak_coupling,
            'H8_eps_crit_success': eps_crit_success,
            'H8_eps_crit_mono':    eps_crit_mono,
            'H9_diagonal_appear':  h9_diag,
            'H10_linear_snap':     h10_linear,
        },
    }
    json_path = os.path.join(RESULTS_DIR, 'coupled_trimode_atlas.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Results: {json_path}')

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_coupling_sweep(
        EPSILON_LEVELS, all_metrics,
        os.path.join(RESULTS_DIR, 'coupled_trimode_sweep.png'),
    )
    _plot_snap_error_distribution(
        all_snap_errs_by_eps, EPSILON_LEVELS,
        os.path.join(RESULTS_DIR, 'coupled_trimode_snap_err.png'),
    )
    print(f'Plots: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
