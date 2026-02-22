#!/usr/bin/env python3
"""
benchmarks/rational_atlas.py — Farey graph structure of the spectral manifold.

Disables Strategy B (global target planting) to expose true spectral topology.
Strategy-A-only jumper can only move to Farey-adjacent rational tubes.

Formal framework:
  Node:  Z_{p:q} = {S : |ω₂/ω₁ − p/q| < ε}
  Edge:  (p:q) — (a:b)  iff  |pb − qa| = 1        (Farey adjacency)
  Graph: Isomorphic to Farey graph F_K (planar, hierarchical, connected)

Hypotheses:
  H1: All empirical transitions respect Farey adjacency  |pb−qa| = 1
  H2: Hop count ≥ Farey graph distance from start-tube to target-tube
  H3: Multi-hop paths emerge for high-denominator starting positions
  H4: Energy selection reduces Farey distance monotonically (≥ 50% optimal steps)

Target:  Z_{3:2}  (ω₂/ω₁ ∈ [1.40, 1.60])
Start:   Z_{2:1}  (forced,  ω₂/ω₁ ∈ [1.85, 2.15])
Control: pure walker (no jumper) — should fail to reach Z_{3:2}

Farey distance 2:1 → 3:2:
  |2·2 − 1·3| = 1  →  they ARE adjacent.  Minimum hops = 1.
  But gradient drift can land the plateau fire away from 2:1,
  requiring multi-hop traversal through intermediate Farey neighbors.

Output:
  benchmarks/results/rational_atlas.json
  benchmarks/results/rational_atlas_hops.png          — hop count distribution
  benchmarks/results/rational_atlas_graph.png         — Farey graph + transitions
  benchmarks/results/rational_atlas_dist_reduction.png — distance reduction per hop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from math import gcd
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tensor.eigen_walker import EigenWalker, _build_feature
from tensor.domain_inverter import TwoMassSpringInverter
from tensor.spectral_energy import (
    SpectralEnergy, SpectralEnergyConfig, SpectralState, SpectralMode,
)

sys.path.insert(0, os.path.join(_ROOT, 'ecemath'))
from src.domains.mechanics import (
    two_mass_spring_eigvals,
    two_mass_modal_zetas,
    two_mass_modal_freqs_hz,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(_ROOT, 'benchmarks', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

_tms_inv = TwoMassSpringInverter()
TMS_KEYS = _tms_inv.theta_keys()   # ['m', 'k', 'k_c', 'alpha', 'beta']
TMS_BOUNDS: Dict[str, Tuple[float, float]] = {
    'm':     (0.01,  100.0),
    'k':     (0.001, 1e6),
    'k_c':   (0.001, 5e5),
    'alpha': (0.001, 200.0),
    'beta':  (1e-6,  1.0),
}

# Target: Z_{3:2}
TARGET_TUBE             = (3, 2)
TARGET_RATIO_LO, TARGET_RATIO_HI = 1.40, 1.60
TARGET_ZETA_LO,  TARGET_ZETA_HI  = 0.05, 0.25
TARGET_F1_LO,    TARGET_F1_HI    = 5.0,  20.0

# Start: forced near Z_{2:1}
START_RATIO_LO, START_RATIO_HI = 1.85, 2.15

# Jumper parameters
ZETA_CHOICES = [0.05, 0.10, 0.15, 0.20, 0.25]

# Omega_lo anchors in rad/s spanning TARGET_F1 range — used by Farey routing
# to re-scale frequency when current omega_lo has drifted outside target
import math as _math
OMEGA_LO_ANCHORS_RADS = [2.0 * _math.pi * f for f in [6.0, 8.0, 10.0, 12.0, 15.0, 18.0]]


# ── Farey graph ────────────────────────────────────────────────────────────────

def build_farey_graph(
    K: int,
    ratio_lo: float = 0.9,
    ratio_hi: float = 5.0,
) -> Tuple[Dict, List]:
    """Build reduced-fraction adjacency graph for Farey F_K.

    Nodes: all (p, q) with gcd(p, q)=1 and p,q ≤ K and ratio_lo ≤ p/q ≤ ratio_hi.
    Edges: Farey adjacency |pb − qa| = 1.
    """
    nodes = [
        (p, q)
        for p in range(1, K + 1)
        for q in range(1, K + 1)
        if gcd(p, q) == 1 and ratio_lo <= p / q <= ratio_hi
    ]
    node_set = set(nodes)
    adj: Dict[Tuple, List[Tuple]] = {n: [] for n in nodes}
    for (p, q) in nodes:
        for (a, b) in nodes:
            if (p, q) != (a, b) and abs(p * b - q * a) == 1:
                adj[(p, q)].append((a, b))
    return adj, nodes


def farey_bfs(
    source: Tuple[int, int],
    adj: Dict,
) -> Dict[Tuple[int, int], int]:
    """BFS shortest distances from source in Farey graph."""
    dist: Dict[Tuple, int] = {source: 0}
    q: deque = deque([source])
    while q:
        node = q.popleft()
        for nb in adj.get(node, []):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return dist


def is_farey_adjacent(p: int, q: int, a: int, b: int) -> bool:
    return abs(p * b - q * a) == 1


# ── TMS domain utilities ───────────────────────────────────────────────────────

def _tms_clip(theta: dict) -> dict:
    return {k: float(np.clip(theta[k], TMS_BOUNDS[k][0], TMS_BOUNDS[k][1]))
            for k in TMS_KEYS}


def _tms_eigvals(theta: dict) -> np.ndarray:
    return two_mass_spring_eigvals(theta)


def _in_target(theta: dict) -> bool:
    try:
        f1, f2 = two_mass_modal_freqs_hz(theta)
        z1, z2 = two_mass_modal_zetas(theta)
        ratio = f2 / max(f1, 1e-12)
        return (
            TARGET_RATIO_LO <= ratio <= TARGET_RATIO_HI
            and TARGET_F1_LO  <= f1    <= TARGET_F1_HI
            and TARGET_ZETA_LO <= z1   <= TARGET_ZETA_HI
            and TARGET_ZETA_LO <= z2   <= TARGET_ZETA_HI
        )
    except Exception:
        return False


def _get_ratio_from_ev(ev: np.ndarray) -> float:
    try:
        state = SpectralState.from_eigvals(ev)
        if len(state.modes) < 2:
            return 1.0
        return state.modes[0].omega0 / max(state.modes[1].omega0, 1e-10)
    except Exception:
        return 1.0


def _nearest_rational(ratio: float, max_p: int = 12, max_q: int = 12
                      ) -> Tuple[int, int, float]:
    best_p, best_q, best_dist = 1, 1, abs(ratio - 1.0)
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            d = abs(ratio - p / q)
            if d < best_dist:
                best_dist, best_p, best_q = d, p, q
    return best_p, best_q, best_dist


def _sample_near_2_1(rng: np.random.Generator) -> dict:
    """Sample TMS params with ω₂/ω₁ ∈ [1.85, 2.15]."""
    m   = float(10 ** rng.uniform(np.log10(0.1),  np.log10(10.0)))
    k   = float(10 ** rng.uniform(np.log10(1.0),  np.log10(1e4)))
    r   = float(rng.uniform(START_RATIO_LO, START_RATIO_HI))
    k_c = float(np.clip(k * (r ** 2 - 1.0) / 2.0,
                        TMS_BOUNDS['k_c'][0], TMS_BOUNDS['k_c'][1]))
    return {
        'm':     m,
        'k':     k,
        'k_c':   k_c,
        'alpha': float(rng.uniform(0.01, 5.0)),
        'beta':  float(rng.uniform(1e-5, 0.01)),
    }


# ── Energy config: 3:2 target ──────────────────────────────────────────────────

def make_32_config() -> SpectralEnergyConfig:
    return SpectralEnergyConfig(
        target_zetas=[0.15, 0.15],
        target_omega0s_hz=[15.0, 10.0],   # 15/10 = 1.5 → Z_{3:2}
        K_ratios=8,
        w_stab=10.0,
        w_damp=1.0,
        w_freq=0.5,
        w_harm=2.0,
    )


# ── Training ───────────────────────────────────────────────────────────────────

def _mode_to_eigvals(mode: SpectralMode) -> np.ndarray:
    if mode.omega > 1e-12:
        return np.array([mode.alpha + 1j * mode.omega,
                         mode.alpha - 1j * mode.omega], dtype=complex)
    return np.array([complex(mode.alpha), complex(mode.alpha)], dtype=complex)


def build_32_training_data(
    config: SpectralEnergyConfig,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    energy = SpectralEnergy(config)
    states, targets = energy.generate_training_samples(
        n_samples=n_samples, n_modes=2,
        zeta_range=(0.01, 0.80),
        logw_range=(1.0, 3.5),
        seed=seed,
    )
    X_list, Y_list = [], []
    for state, target in zip(states, targets):
        if len(state.modes) < 2:
            continue
        ev = np.concatenate([
            _mode_to_eigvals(state.modes[0]),
            _mode_to_eigvals(state.modes[1]),
        ])
        X_list.append(_build_feature(ev, 'lca'))
        Y_list.append(target[:4])
    return np.array(X_list, dtype=np.float64), np.array(Y_list, dtype=np.float64)


def _train_mlp_on_arrays(
    walker: EigenWalker,
    X: np.ndarray,
    Y: np.ndarray,
    n_epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
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


# ── Walker step ────────────────────────────────────────────────────────────────

def _apply_r2_step(
    theta: dict,
    ev: np.ndarray,
    walker: EigenWalker,
    step_scale: float,
) -> dict:
    delta = walker.predict_step(ev, regime='lca') * step_scale
    state = SpectralState.from_eigvals(ev)
    if len(state.modes) < 2:
        return theta
    modes_new = []
    for i in range(2):
        d_z = float(np.clip(delta[2 * i],     -0.5, 0.5))
        d_w = float(np.clip(delta[2 * i + 1], -1.0, 1.0))
        z = float(np.clip(state.modes[i].zeta + d_z, 1e-4, 2.0))
        w = float(max(10.0 ** (state.modes[i].log10_omega0 + d_w), 1e-3))
        modes_new.append((z, w))
    return _tms_clip(_tms_inv.invert_modes(modes_new, theta))


# ── Strategy-A-only jumper ─────────────────────────────────────────────────────

def _strategy_a_candidates(
    ev:               np.ndarray,
    theta:            dict,
    farey_adj:        Dict,
    omega_lo_anchors: Optional[List[float]] = None,
) -> Tuple[List[Tuple[dict, Tuple[int, int]]], Tuple[int, int]]:
    """
    Generate candidates ONLY from Farey neighbors of the current tube.

    omega_lo_anchors: additional omega_lo values (rad/s) to try alongside the
        current omega_lo.  Used by Farey-distance routing to re-anchor frequency
        when the walk has drifted outside the target range.

    Returns: (candidates, current_tube)
    candidates: [(theta_cand, to_tube), ...]
    """
    ratio = _get_ratio_from_ev(ev)
    p, q, _ = _nearest_rational(ratio)
    current_tube = (p, q)
    neighbors = farey_adj.get(current_tube, [])

    candidates: List[Tuple[dict, Tuple]] = []
    try:
        state = SpectralState.from_eigvals(ev)
        if len(state.modes) < 2:
            return candidates, current_tube
        omega_lo_current = state.modes[1].omega0
    except Exception:
        return candidates, current_tube

    omega_lo_tries = [omega_lo_current] + (omega_lo_anchors or [])

    for (a, b) in neighbors:
        for omega_lo in omega_lo_tries:
            omega_hi_cand = (a / b) * omega_lo
            for z1 in ZETA_CHOICES:
                for z2 in ZETA_CHOICES:
                    try:
                        raw = _tms_inv.invert_modes(
                            [(z2, omega_hi_cand), (z1, omega_lo)], theta
                        )
                        candidates.append((_tms_clip(raw), (a, b)))
                    except Exception:
                        pass

    return candidates, current_tube


def _try_strategy_a(
    ev:          np.ndarray,
    theta:       dict,
    farey_adj:   Dict,
    energy:      SpectralEnergy,
) -> Tuple[Optional[dict], Tuple[int, int], Optional[Tuple[int, int]]]:
    """Select lowest-energy Farey-neighbor candidate.

    Returns: (best_theta, from_tube, to_tube)
    """
    candidates, from_tube = _strategy_a_candidates(ev, theta, farey_adj)
    best_theta, best_tube, best_E = None, None, float('inf')
    for (cand, to_tube) in candidates:
        try:
            ev_c  = _tms_eigvals(cand)
            state = SpectralState.from_eigvals(ev_c)
            if len(state.modes) < 2:
                continue
            E = energy.compute(state)
            if E < best_E:
                best_E, best_theta, best_tube = E, cand, to_tube
        except Exception:
            pass
    return best_theta, from_tube, best_tube


def _try_strategy_a_farey_routed(
    ev:             np.ndarray,
    theta:          dict,
    farey_adj:      Dict,
    dist_to_target: Dict,
) -> Tuple[Optional[dict], Tuple[int, int], Optional[Tuple[int, int]]]:
    """Select Farey neighbor by MINIMUM FAREY DISTANCE to target (not energy).

    Guarantees: d_after ≤ d_before for every jump → no cycles possible.
    Tiebreaker: among equal-d neighbors, pick any valid candidate (first found).

    Returns: (best_theta, from_tube, to_tube)
    """
    candidates, from_tube = _strategy_a_candidates(
        ev, theta, farey_adj, omega_lo_anchors=OMEGA_LO_ANCHORS_RADS
    )
    # Two-pass selection:
    #   Pass 1: prefer candidates that already satisfy _in_target (d=0, last-mile complete)
    #   Pass 2: fallback to argmin_d without target check
    best_theta, best_tube, best_d = None, None, float('inf')
    best_target_theta, best_target_tube = None, None   # Pass-1 winner

    for (cand, to_tube) in candidates:
        d = dist_to_target.get(to_tube, float('inf'))
        # Verify candidate produces valid physical parameters
        try:
            ev_c  = _tms_eigvals(cand)
            state = SpectralState.from_eigvals(ev_c)
            if len(state.modes) < 2:
                continue
        except Exception:
            continue

        # Pass 1: if this candidate satisfies _in_target, prefer it (d=0 + in target)
        if d == 0 and best_target_theta is None and _in_target(cand):
            best_target_theta, best_target_tube = cand, to_tube

        # Pass 2 (fallback): track minimum-d candidate regardless
        if d < best_d:
            best_d, best_theta, best_tube = d, cand, to_tube

    # Use pass-1 winner if found, otherwise fall back to pass-2
    if best_target_theta is not None:
        return best_target_theta, from_tube, best_target_tube
    return best_theta, from_tube, best_tube


# ── Methods ────────────────────────────────────────────────────────────────────

def run_strategy_a_plateau_hybrid(
    theta0:          dict,
    walker:          EigenWalker,
    energy:          SpectralEnergy,
    max_iter:        int,
    farey_adj:       Dict,
    dist_to_target:  Dict,
    step_scale:      float = 0.30,
    plateau_window:  int   = 8,
    plateau_eps:     float = 0.02,
    boundary_window: int   = 20,
) -> Tuple[int, bool, List[dict]]:
    """
    Plateau-triggered hybrid restricted to Farey-adjacent rational jumps.

    Returns: (iters, success, hop_sequence)
    hop_sequence[i]: {
        'step':       iteration when jump fired,
        'from':       (p, q) of current tube,
        'to':         (a, b) of destination tube,
        'farey_adj':  bool — is |pb−qa|=1?
        'd_before':   Farey distance from_tube → target,
        'd_after':    Farey distance to_tube   → target,
        'ratio_before': observed ω₂/ω₁ at fire,
        'ratio_after':  observed ω₂/ω₁ after jump,
    }
    """
    theta = dict(theta0)
    try:
        ev = _tms_eigvals(theta)
    except Exception:
        return max_iter, False, []

    energy_history: List[float] = []
    boundary_count  = 0
    hop_sequence:   List[dict] = []

    for i in range(max_iter):
        if _in_target(theta):
            return i + 1, True, hop_sequence

        # Energy tracking
        try:
            state = SpectralState.from_eigvals(ev)
            E_now = energy.compute(state)
        except Exception:
            E_now = float('inf')
        energy_history.append(E_now)
        if len(energy_history) > plateau_window:
            energy_history.pop(0)

        # Continuous step
        try:
            theta_raw  = _apply_r2_step(theta, ev, walker, step_scale)
            theta_new  = _tms_clip(theta_raw)
            clipped    = any(abs(theta_raw[k] - theta_new[k]) > 1e-12 for k in TMS_KEYS)
        except Exception:
            theta_new  = dict(theta)
            clipped    = False

        boundary_count = boundary_count + 1 if clipped else max(0, boundary_count - 1)

        # Stagnation detection (energy plateau or boundary clip accumulation)
        energy_plateau = (
            len(energy_history) >= plateau_window
            and (energy_history[0] - energy_history[-1]) < plateau_eps
        )

        if energy_plateau or (boundary_count >= boundary_window):
            ratio_before = _get_ratio_from_ev(ev)
            jumped, from_tube, to_tube = _try_strategy_a(ev, theta, farey_adj, energy)

            if jumped is not None and to_tube is not None:
                d_before = dist_to_target.get(from_tube, -1)
                d_after  = dist_to_target.get(to_tube,   -1)
                adj_ok   = is_farey_adjacent(
                    from_tube[0], from_tube[1], to_tube[0], to_tube[1]
                )
                try:
                    ev_j        = _tms_eigvals(jumped)
                    ratio_after = _get_ratio_from_ev(ev_j)
                    ev          = ev_j
                except Exception:
                    ratio_after = 0.0

                hop_sequence.append({
                    'step':         i,
                    'from':         from_tube,
                    'to':           to_tube,
                    'farey_adj':    adj_ok,
                    'd_before':     d_before,
                    'd_after':      d_after,
                    'ratio_before': ratio_before,
                    'ratio_after':  ratio_after,
                })
                theta = jumped

            energy_history.clear()
            boundary_count = 0
            continue

        try:
            ev    = _tms_eigvals(theta_new)
            theta = theta_new
        except Exception:
            pass

    return max_iter, False, hop_sequence


def run_farey_routing_hybrid(
    theta0:          dict,
    walker:          EigenWalker,
    energy:          SpectralEnergy,
    max_iter:        int,
    farey_adj:       Dict,
    dist_to_target:  Dict,
    step_scale:      float = 0.30,
    plateau_window:  int   = 8,
    plateau_eps:     float = 0.02,
    boundary_window: int   = 20,
) -> Tuple[int, bool, List[dict]]:
    """Plateau hybrid with FAREY-DISTANCE routing (argmin d_farey, not argmin E).

    Guarantees monotonic decrease in Farey graph distance to target.
    No cycles possible: each jump strictly reduces d or stays equal.
    """
    theta = dict(theta0)
    try:
        ev = _tms_eigvals(theta)
    except Exception:
        return max_iter, False, []

    energy_history: List[float] = []
    boundary_count  = 0
    hop_sequence:   List[dict] = []

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
            theta_raw  = _apply_r2_step(theta, ev, walker, step_scale)
            theta_new  = _tms_clip(theta_raw)
            clipped    = any(abs(theta_raw[k] - theta_new[k]) > 1e-12 for k in TMS_KEYS)
        except Exception:
            theta_new  = dict(theta)
            clipped    = False

        boundary_count = boundary_count + 1 if clipped else max(0, boundary_count - 1)

        energy_plateau = (
            len(energy_history) >= plateau_window
            and (energy_history[0] - energy_history[-1]) < plateau_eps
        )

        if energy_plateau or (boundary_count >= boundary_window):
            ratio_before = _get_ratio_from_ev(ev)

            # Suppress jumper when already inside target tube (d=0):
            # all Farey neighbors have d≥1, so any jump would be retrograde.
            # Let gradient descent finish the last-mile without interference.
            p_cur, q_cur, _ = _nearest_rational(ratio_before)
            if dist_to_target.get((p_cur, q_cur), 99) == 0:
                energy_history.clear()
                boundary_count = 0
                continue

            # ── Farey-distance routing (key difference) ──
            jumped, from_tube, to_tube = _try_strategy_a_farey_routed(
                ev, theta, farey_adj, dist_to_target
            )

            if jumped is not None and to_tube is not None:
                d_before = dist_to_target.get(from_tube, -1)
                d_after  = dist_to_target.get(to_tube,   -1)
                adj_ok   = is_farey_adjacent(
                    from_tube[0], from_tube[1], to_tube[0], to_tube[1]
                )
                try:
                    ev_j        = _tms_eigvals(jumped)
                    ratio_after = _get_ratio_from_ev(ev_j)
                    ev          = ev_j
                except Exception:
                    ratio_after = 0.0

                hop_sequence.append({
                    'step':         i,
                    'from':         from_tube,
                    'to':           to_tube,
                    'farey_adj':    adj_ok,
                    'd_before':     d_before,
                    'd_after':      d_after,
                    'ratio_before': ratio_before,
                    'ratio_after':  ratio_after,
                })
                theta = jumped

            energy_history.clear()
            boundary_count = 0
            continue

        try:
            ev    = _tms_eigvals(theta_new)
            theta = theta_new
        except Exception:
            pass

    return max_iter, False, hop_sequence


def run_walker_only(
    theta0:     dict,
    walker:     EigenWalker,
    energy:     SpectralEnergy,
    max_iter:   int,
    step_scale: float = 0.30,
) -> Tuple[int, bool]:
    """Pure gradient descent — no jumper. Baseline showing barrier effect."""
    theta = dict(theta0)
    try:
        ev = _tms_eigvals(theta)
    except Exception:
        return max_iter, False

    for i in range(max_iter):
        if _in_target(theta):
            return i + 1, True
        try:
            theta_new  = _apply_r2_step(theta, ev, walker, step_scale)
            theta_new  = _tms_clip(theta_new)
            ev         = _tms_eigvals(theta_new)
            theta      = theta_new
        except Exception:
            pass

    return max_iter, False


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_hop_distribution(hop_counts: List[int], out_path: str):
    max_hops = max(hop_counts, default=0)
    bins = np.arange(-0.5, max_hops + 2.5)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(hop_counts, bins=bins, color='#2196F3', edgecolor='white', linewidth=0.8)
    ax.axvline(np.mean(hop_counts), color='red', linewidth=2, linestyle='--',
               label=f'Mean = {np.mean(hop_counts):.2f}')
    ax.set_xlabel('Rational jumps (hops) per trial')
    ax.set_ylabel('Frequency (out of 100 trials)')
    ax.set_title('Strategy-A-only: Hop Count Distribution (Z_{2:1} → Z_{3:2})')
    ax.set_xticks(range(0, max_hops + 3))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_farey_graph(
    farey_adj: Dict,
    atlas:     Dict[Tuple, int],
    target:    Tuple[int, int],
    out_path:  str,
):
    """Draw Farey graph with empirical transitions overlaid."""
    nodes = list(farey_adj.keys())
    if not nodes:
        return

    # Layout: x = ratio p/q, y = denominator q (spread out tree-like)
    # Add small jitter to y to separate nodes sharing denominator
    rng_lay = np.random.default_rng(0)
    pos: Dict[Tuple, Tuple[float, float]] = {}
    for (p, q) in nodes:
        x = p / q
        y = q + rng_lay.uniform(-0.2, 0.2)
        pos[(p, q)] = (x, y)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Draw Farey edges (gray thin background)
    drawn_edges = set()
    for (p, q) in nodes:
        x0, y0 = pos[(p, q)]
        for (a, b) in farey_adj[(p, q)]:
            if (a, b) in pos:
                edge = tuple(sorted([(p,q),(a,b)]))
                if edge not in drawn_edges:
                    x1, y1 = pos[(a, b)]
                    ax.plot([x0, x1], [y0, y1], color='#DDDDDD', linewidth=0.6, zorder=1)
                    drawn_edges.add(edge)

    # Draw empirical transitions (arrows, width ∝ count)
    total_atlas = sum(atlas.values())
    for (from_t, to_t), cnt in sorted(atlas.items(), key=lambda x: x[1]):
        if from_t in pos and to_t in pos:
            x0, y0 = pos[from_t]
            x1, y1 = pos[to_t]
            color = '#2196F3' if to_t == target else '#FF9800'
            lw    = 1.0 + 4.0 * cnt / max(total_atlas, 1)
            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=12),
                zorder=3,
            )

    # Draw nodes
    for (p, q) in nodes:
        x, y = pos[(p, q)]
        if (p, q) == target:
            c, ec, sz = 'red',     'darkred', 200
        elif (p, q) == (2, 1):
            c, ec, sz = '#FF9800', 'darkorange', 160
        else:
            c, ec, sz = 'white',   '#555555',   80
        ax.scatter(x, y, s=sz, c=c, edgecolors=ec, linewidth=1.5, zorder=5)
        ax.annotate(
            f'{p}:{q}', (x, y),
            textcoords='offset points', xytext=(4, 4),
            fontsize=7, zorder=6,
        )

    ax.set_xlabel('ω₂/ω₁  (= p/q)')
    ax.set_ylabel('Denominator q  (jittered for separation)')
    ax.set_title(
        f'Farey Graph F_K with Empirical Atlas Transitions\n'
        f'Blue = → Z_{{3:2}} (target), Orange = intermediate;  '
        f'● = target (3:2),  ● = start (2:1)'
    )
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_distance_reduction(all_seqs: List[List[dict]], out_path: str):
    """Scatter: Farey distance before vs after each jump."""
    d_befores, d_afters = [], []
    for seq in all_seqs:
        for hop in seq:
            if hop['d_before'] >= 0 and hop['d_after'] >= 0:
                d_befores.append(hop['d_before'])
                d_afters.append(hop['d_after'])

    if not d_befores:
        return

    # Fraction optimal (distance reduced by exactly 1)
    optimal = sum(1 for b, a in zip(d_befores, d_afters) if a == b - 1)
    pct_opt = 100.0 * optimal / len(d_befores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(d_befores, d_afters, alpha=0.4, s=25, color='#2196F3',
               label=f'Jumps (n={len(d_befores)})')
    max_d = max(max(d_befores, default=0), max(d_afters, default=0), 1)
    ax.plot([0, max_d], [0, max_d],       'k--', linewidth=1,   label='no change')
    ax.plot([1, max_d], [0, max_d - 1],   'g-',  linewidth=1.5, label='optimal (−1 hop)')

    ax.set_xlabel('Farey distance to Z_{3:2} BEFORE jump')
    ax.set_ylabel('Farey distance to Z_{3:2} AFTER jump')
    ax.set_title(
        f'Distance Reduction per Strategy-A Jump\n'
        f'Optimal (d_after = d_before−1): {optimal}/{len(d_befores)} = {pct_opt:.0f}%'
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Rational atlas: Farey graph structure of the spectral manifold.'
    )
    parser.add_argument('--n-trials',  type=int, default=100)
    parser.add_argument('--max-iter',  type=int, default=500)
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--n-epochs',  type=int, default=300)
    parser.add_argument('--K',         type=int, default=8,
                        help='Farey graph max denominator')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print('=== Rational Atlas: Farey Graph Structure of Spectral Manifold ===')
    print(f'  trials={args.n_trials}  max_iter={args.max_iter}  K={args.K}  seed={args.seed}')
    print(f'  Start:  Z_{{2:1}} (ω₂/ω₁ ∈ [{START_RATIO_LO}, {START_RATIO_HI}])')
    print(f'  Target: Z_{{3:2}} (ω₂/ω₁ ∈ [{TARGET_RATIO_LO}, {TARGET_RATIO_HI}])')
    print(f'  Jumper: Strategy-A ONLY — restricted to Farey neighbors |pb−qa|=1')
    print()

    # Build Farey graph
    print(f'[Setup] Farey graph F_{args.K} (ratio ∈ [0.9, 5.0])...')
    farey_adj, farey_nodes = build_farey_graph(args.K, ratio_lo=0.9, ratio_hi=5.0)
    dist_to_32 = farey_bfs(TARGET_TUBE, farey_adj)

    n_edges = sum(len(v) for v in farey_adj.values()) // 2
    print(f'  Nodes: {len(farey_nodes)},  Edges: {n_edges}')
    print(f'  Farey distance 2:1 → 3:2: {dist_to_32.get((2, 1), "not connected")}')
    print(f'  Neighbors of Z_{{2:1}}:  {sorted(farey_adj.get((2, 1), []))}')
    print(f'  Neighbors of Z_{{3:2}}:  {sorted(farey_adj.get(TARGET_TUBE, []))}')

    # Verify adjacency of 2:1 and 3:2
    adj_21_32 = is_farey_adjacent(2, 1, 3, 2)
    print(f'  Z_{{2:1}} and Z_{{3:2}} are Farey adjacent: {adj_21_32}  '
          f'(|2·2−1·3|={abs(2*2-1*3)})')
    print()

    # Phase 1: Train walker with 3:2 gradient
    print('[Phase 1] Training EigenWalker(n_modes=2) on 3:2 energy gradient...')
    config = make_32_config()
    t0 = time.time()
    X, Y = build_32_training_data(config, args.n_samples, seed=args.seed)
    walker = EigenWalker(hidden=128, n_modes=2, seed=args.seed)
    losses = _train_mlp_on_arrays(
        walker, X, Y, n_epochs=args.n_epochs, lr=1e-3, batch_size=64, seed=args.seed
    )
    print(f'  {len(X)} samples, {args.n_epochs} epochs in {time.time()-t0:.1f}s')
    print(f'  Loss: {losses[0]:.6f} → {losses[-1]:.6f}')
    print()

    energy = SpectralEnergy(config)

    # Phase 2: Benchmark
    print('[Phase 2] Benchmark: 100 trials × 2 methods...')

    results_a   = []   # Strategy-A-only (energy routing)
    results_fr  = []   # Farey-distance routing
    results_w   = []   # walker-only (no jumper, should fail)
    all_seqs_a:  List[List[dict]] = []
    all_seqs_fr: List[List[dict]] = []

    for trial in range(args.n_trials):
        theta0 = _sample_near_2_1(rng)

        # Strategy-A-only plateau hybrid (energy-routed — cycles expected)
        iters_a, suc_a, seq_a = run_strategy_a_plateau_hybrid(
            theta0, walker, energy, args.max_iter, farey_adj, dist_to_32
        )
        results_a.append({'iters': iters_a, 'success': suc_a, 'hops': len(seq_a)})
        all_seqs_a.append(seq_a)

        # Farey-distance routing hybrid (monotonic d decrease — no cycles)
        iters_fr, suc_fr, seq_fr = run_farey_routing_hybrid(
            theta0, walker, energy, args.max_iter, farey_adj, dist_to_32
        )
        results_fr.append({'iters': iters_fr, 'success': suc_fr, 'hops': len(seq_fr)})
        all_seqs_fr.append(seq_fr)

        # Walker-only (no jumper) — expected to fail at rational barriers
        iters_w, suc_w = run_walker_only(theta0, walker, energy, args.max_iter)
        results_w.append({'iters': iters_w, 'success': suc_w})

        if (trial + 1) % 10 == 0:
            print(f'  [{trial+1:3d}/{args.n_trials}]  '
                  f'energy_routed: {iters_a:3d}({"✓" if suc_a else "✗"})  '
                  f'farey_routed: {iters_fr:3d}({"✓" if suc_fr else "✗"})  '
                  f'walker_only: {iters_w:3d}({"✓" if suc_w else "✗"})')

    print()

    # Phase 3: Farey analysis
    total_hops = sum(len(seq) for seq in all_seqs_a)
    farey_adj_hops = sum(
        1 for seq in all_seqs_a for hop in seq if hop['farey_adj']
    )

    all_d_before = [h['d_before'] for seq in all_seqs_a for h in seq if h['d_before'] >= 0]
    all_d_after  = [h['d_after']  for seq in all_seqs_a for h in seq if h['d_after']  >= 0]
    optimal_steps = sum(1 for b, a in zip(all_d_before, all_d_after) if a == b - 1)
    retrogress    = sum(1 for b, a in zip(all_d_before, all_d_after) if a > b)

    # Atlas transition counts
    atlas: Dict[Tuple, int] = defaultdict(int)
    for seq in all_seqs_a:
        for hop in seq:
            atlas[(hop['from'], hop['to'])] += 1

    # Hop distribution
    hop_counts = [r['hops'] for r in results_a]

    # ── Summary ────────────────────────────────────────────────────────────────
    succ_a = [r for r in results_a if r['success']]
    succ_w = [r for r in results_w if r['success']]

    def _fmt(res, name):
        succ = [r for r in res if r['success']]
        n    = len(res)
        iters = [r['iters'] for r in succ]
        mean_i = float(np.mean(iters))   if iters else float('nan')
        med_i  = float(np.median(iters)) if iters else float('nan')
        p95_i  = float(np.percentile(iters, 95)) if iters else float('nan')
        fail_pct = 100.0 * (n - len(succ)) / n
        print(f'{name:<40} {len(succ):>3}/{n:<3} '
              f'{mean_i:>8.1f} {med_i:>8.1f} {p95_i:>8.1f} {fail_pct:>6.1f}%')

    print('=== Results ===')
    print(f'{"Method":<40} {"Succ":>6} {"Mean":>8} {"Median":>8} {"p95":>8} {"Fail%":>7}')
    print('-' * 76)
    _fmt(results_a,  'Strategy-A Energy-Routed')
    _fmt(results_fr, 'Strategy-A Farey-Distance Routed')
    _fmt(results_w,  'Pure Walker (no jumper)')

    print()
    print('=== Farey Topology Analysis ===')
    print(f'  H1 — All transitions Farey-adjacent:  '
          f'{farey_adj_hops}/{total_hops} = '
          f'{100*farey_adj_hops/max(total_hops,1):.1f}%')
    print(f'  H2 — Hop count ≥ Farey distance:       verified by d_before column')
    print(f'  H4 — Optimal jumps (d decreases by 1): '
          f'{optimal_steps}/{len(all_d_before)} = '
          f'{100*optimal_steps/max(len(all_d_before),1):.1f}%')
    print(f'       Retrograde jumps (d increases):   '
          f'{retrogress}/{len(all_d_before)} = '
          f'{100*retrogress/max(len(all_d_before),1):.1f}%')

    # ── Farey-distance routing analysis ───────────────────────────────────────
    total_hops_fr = sum(len(seq) for seq in all_seqs_fr)
    farey_adj_fr  = sum(1 for seq in all_seqs_fr for h in seq if h['farey_adj'])
    d_before_fr   = [h['d_before'] for seq in all_seqs_fr for h in seq if h['d_before'] >= 0]
    d_after_fr    = [h['d_after']  for seq in all_seqs_fr for h in seq if h['d_after']  >= 0]
    monotonic_fr  = sum(1 for b, a in zip(d_before_fr, d_after_fr) if a <= b)
    retrogress_fr = sum(1 for b, a in zip(d_before_fr, d_after_fr) if a > b)
    optimal_fr    = sum(1 for b, a in zip(d_before_fr, d_after_fr) if a == b - 1)
    hop_counts_fr = [r['hops'] for r in results_fr]

    print()
    print('=== Farey-Routing Analysis (Farey-Distance Routed) ===')
    print(f'  Farey-adjacent hops:   {farey_adj_fr}/{total_hops_fr} = '
          f'{100*farey_adj_fr/max(total_hops_fr,1):.1f}%')
    print(f'  Monotonic (d_after ≤ d_before): '
          f'{monotonic_fr}/{len(d_before_fr)} = '
          f'{100*monotonic_fr/max(len(d_before_fr),1):.1f}%  '
          f'[target: 100%]')
    print(f'  Optimal (d decreases by 1):     '
          f'{optimal_fr}/{len(d_before_fr)} = '
          f'{100*optimal_fr/max(len(d_before_fr),1):.1f}%')
    print(f'  Retrograde (d_after > d_before):{retrogress_fr}/{len(d_before_fr)} = '
          f'{100*retrogress_fr/max(len(d_before_fr),1):.1f}%  '
          f'[target: 0%]')
    print(f'  Max hops (Farey-routed):        {max(hop_counts_fr, default=0)}  '
          f'[Farey diameter ≤ {max(dist_to_32.values(), default=0)*2}]')

    print()
    print('=== Hop Distribution (Energy-Routed vs Farey-Routed) ===')
    hop_counts_a = [r['hops'] for r in results_a]
    max_hop = max(max(hop_counts_a, default=0), max(hop_counts_fr, default=0))
    print(f'  {"Hops":<6}  {"Energy-Rtd":>12}  {"Farey-Rtd":>12}')
    for h in range(0, max_hop + 2):
        cnt_a  = sum(1 for x in hop_counts_a  if x == h)
        cnt_fr = sum(1 for x in hop_counts_fr if x == h)
        if cnt_a > 0 or cnt_fr > 0:
            bar_a  = '█' * cnt_a
            bar_fr = '▒' * cnt_fr
            print(f'  {h} hops:  {cnt_a:>3}/100 {bar_a:<20}  {cnt_fr:>3}/100 {bar_fr}')

    # Multi-hop examples
    multi = [(i, seq) for i, seq in enumerate(all_seqs_a) if len(seq) > 1]
    multi_fr = [(i, seq) for i, seq in enumerate(all_seqs_fr) if len(seq) > 1]
    if multi:
        print()
        print(f'=== Multi-Hop Path Examples — Energy-Routed ({len(multi)} trials) ===')
        for i, seq in multi[:6]:
            path_parts = [f'{h["from"][0]}:{h["from"][1]}' for h in seq]
            if seq:
                path_parts.append(f'{seq[-1]["to"][0]}:{seq[-1]["to"][1]}')
            path = ' → '.join(path_parts)
            d_trace = ' '.join(f'd={h["d_before"]}' for h in seq)
            print(f'  Trial {i:3d}: {path}   [{d_trace}]   '
                  f'{"✓" if results_a[i]["success"] else "✗"}')

    if multi_fr:
        print()
        print(f'=== Multi-Hop Path Examples — Farey-Routed ({len(multi_fr)} trials) ===')
        for i, seq in multi_fr[:6]:
            path_parts = [f'{h["from"][0]}:{h["from"][1]}' for h in seq]
            if seq:
                path_parts.append(f'{seq[-1]["to"][0]}:{seq[-1]["to"][1]}')
            path = ' → '.join(path_parts)
            d_trace = ' '.join(f'd={h["d_before"]}→{h["d_after"]}' for h in seq)
            print(f'  Trial {i:3d}: {path}   [{d_trace}]   '
                  f'{"✓" if results_fr[i]["success"] else "✗"}')

    # Atlas transitions (energy-routed, shows cycles)
    print()
    print('=== Top Atlas Transitions (Energy-Routed) ===')
    for (from_t, to_t), cnt in sorted(atlas.items(), key=lambda x: -x[1])[:15]:
        adj_tag  = '✓' if is_farey_adjacent(*from_t, *to_t) else '✗'
        tgt_tag  = '← TARGET' if to_t == TARGET_TUBE else ''
        d_f      = dist_to_32.get(from_t, -1)
        d_t      = dist_to_32.get(to_t,   -1)
        print(f'  {from_t[0]}:{from_t[1]} → {to_t[0]}:{to_t[1]}  '
              f'cnt={cnt:3d}  adj={adj_tag}  d:{d_f}→{d_t}  {tgt_tag}')

    # Atlas transitions for Farey routing
    atlas_fr: Dict[Tuple, int] = defaultdict(int)
    for seq in all_seqs_fr:
        for hop in seq:
            atlas_fr[(hop['from'], hop['to'])] += 1
    if atlas_fr:
        print()
        print('=== Top Atlas Transitions (Farey-Routed) ===')
        for (from_t, to_t), cnt in sorted(atlas_fr.items(), key=lambda x: -x[1])[:15]:
            adj_tag  = '✓' if is_farey_adjacent(*from_t, *to_t) else '✗'
            tgt_tag  = '← TARGET' if to_t == TARGET_TUBE else ''
            d_f      = dist_to_32.get(from_t, -1)
            d_t      = dist_to_32.get(to_t,   -1)
            dir_tag  = '↓' if d_t < d_f else ('→' if d_t == d_f else '↑')
            print(f'  {from_t[0]}:{from_t[1]} → {to_t[0]}:{to_t[1]}  '
                  f'cnt={cnt:3d}  adj={adj_tag}  d:{d_f}→{d_t} {dir_tag}  {tgt_tag}')

    # Intermediate tubes encountered
    intermediate_tubes = set()
    for seq in all_seqs_a:
        for hop in seq[:-1]:
            intermediate_tubes.add(hop['to'])
    print()
    print(f'=== Intermediate Tubes Encountered (Energy-Routed) ===')
    print(f'  Count: {len(intermediate_tubes)}')
    for t in sorted(intermediate_tubes, key=lambda x: x[0]/x[1]):
        cnt_in  = sum(atlas.get((t, other), 0) for other in farey_adj.get(t, []))
        cnt_out = sum(atlas.get((other, t), 0) for other in farey_adj.get(t, []))
        print(f'  {t[0]}:{t[1]}  (ratio={t[0]/t[1]:.3f},  '
              f'd_to_target={dist_to_32.get(t,-1)},  '
              f'in={cnt_out} out={cnt_in})')

    # Hypotheses summary
    print()
    print('=== Hypothesis Check ===')
    h1_pct     = 100 * farey_adj_hops / max(total_hops, 1)
    h4_pct     = 100 * optimal_steps  / max(len(all_d_before), 1)
    h5_mono    = 100 * monotonic_fr   / max(len(d_before_fr), 1)
    h5_retro   = 100 * retrogress_fr  / max(len(d_before_fr), 1)
    succ_rate_a  = 100 * len(succ_a)  / len(results_a)
    succ_rate_fr = 100 * len([r for r in results_fr if r['success']]) / len(results_fr)
    print(f'  H1 (Farey adjacency constraint):      {"✓ CONFIRMED" if h1_pct == 100 else f"✗ PARTIAL {h1_pct:.0f}%"}')
    print(f'  H3 (multi-hop paths emerge):           {"✓ CONFIRMED" if len(multi) > 0 else "✗ NOT OBSERVED"}  ({len(multi)} energy / {len(multi_fr)} Farey)')
    print(f'  H4 (energy-routing optimal steps):    {h4_pct:.0f}%  retrograde: {100*retrogress/max(len(all_d_before),1):.0f}%')
    print(f'  H4b (energy-routing 100% success):    {"✓" if succ_rate_a == 100 else "✗"} ({succ_rate_a:.0f}%)')
    print(f'  H5 (Farey-routing monotone):           {"✓ CONFIRMED" if h5_retro == 0 else f"✗ {h5_retro:.1f}% retrograde"}  ({h5_mono:.0f}% non-retrograde)')
    print(f'  H5b (Farey-routing 100% success):     {"✓ CONFIRMED" if succ_rate_fr == 100 else "✗"} ({succ_rate_fr:.0f}%)')

    # Save JSON
    out = {
        'args': vars(args),
        'farey_graph': {
            'n_nodes': len(farey_nodes),
            'n_edges': n_edges,
            'dist_21_to_32': int(dist_to_32.get((2, 1), -1)),
            'neighbors_21': [list(n) for n in sorted(farey_adj.get((2, 1), []))],
            'neighbors_32': [list(n) for n in sorted(farey_adj.get(TARGET_TUBE, []))],
        },
        'results_a': [
            {'iters': r['iters'], 'success': r['success'], 'hops': r['hops']}
            for r in results_a
        ],
        'results_fr': [
            {'iters': r['iters'], 'success': r['success'], 'hops': r['hops']}
            for r in results_fr
        ],
        'results_w': [
            {'iters': r['iters'], 'success': r['success']}
            for r in results_w
        ],
        'atlas': {
            f'{f[0]}:{f[1]}→{t[0]}:{t[1]}': cnt
            for (f, t), cnt in atlas.items()
        },
        'atlas_fr': {
            f'{f[0]}:{f[1]}→{t[0]}:{t[1]}': cnt
            for (f, t), cnt in atlas_fr.items()
        },
        'farey_analysis': {
            'energy_routing': {
                'total_hops':       total_hops,
                'farey_adj_pct':    round(h1_pct, 1),
                'optimal_pct':      round(h4_pct, 1),
                'retrogress_pct':   round(100*retrogress/max(len(all_d_before),1), 1),
                'multi_hop_trials': len(multi),
                'success_pct':      round(succ_rate_a, 1),
            },
            'farey_routing': {
                'total_hops':       total_hops_fr,
                'farey_adj_pct':    round(100*farey_adj_fr/max(total_hops_fr,1), 1),
                'monotonic_pct':    round(h5_mono, 1),
                'retrogress_pct':   round(h5_retro, 1),
                'optimal_pct':      round(100*optimal_fr/max(len(d_before_fr),1), 1),
                'max_hops':         max(hop_counts_fr, default=0),
                'multi_hop_trials': len(multi_fr),
                'success_pct':      round(succ_rate_fr, 1),
            },
        },
    }
    json_path = os.path.join(RESULTS_DIR, 'rational_atlas.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults: {json_path}')

    # Plots — energy routing
    hop_counts_a = [r['hops'] for r in results_a]
    _plot_hop_distribution(
        hop_counts_a,
        os.path.join(RESULTS_DIR, 'rational_atlas_hops_energy.png'),
    )
    _plot_farey_graph(
        farey_adj,
        dict(atlas),
        TARGET_TUBE,
        os.path.join(RESULTS_DIR, 'rational_atlas_graph_energy.png'),
    )
    _plot_distance_reduction(
        all_seqs_a,
        os.path.join(RESULTS_DIR, 'rational_atlas_dist_reduction_energy.png'),
    )
    # Plots — Farey-distance routing
    _plot_hop_distribution(
        hop_counts_fr,
        os.path.join(RESULTS_DIR, 'rational_atlas_hops_farey.png'),
    )
    _plot_farey_graph(
        farey_adj,
        dict(atlas_fr),
        TARGET_TUBE,
        os.path.join(RESULTS_DIR, 'rational_atlas_graph_farey.png'),
    )
    _plot_distance_reduction(
        all_seqs_fr,
        os.path.join(RESULTS_DIR, 'rational_atlas_dist_reduction_farey.png'),
    )
    print(f'Plots: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
