#!/usr/bin/env python3
"""
benchmarks/competing_surfaces.py — Inter-tube navigation: Z_{2:1} → Z_{3:2}.

First test of multi-hop traversal across competing rational hypersurfaces.

START:  θ₀ forced near Z_{2:1}  (ω₂/ω₁ ∈ [1.85, 2.15])
TARGET: Z_{3:2} basin           (ω₂/ω₁ ∈ [1.40, 1.60], ζ∈[0.05,0.25], f₁∈[5,20]Hz)

Structural questions:
  Q1: Does a single rational jump (2:1 → 3:2) usually suffice?
  Q2: Which intermediate rational tubes appear? (empirical atlas graph)
  Q3: Does 100% success hold under forced cross-basin initialization?
  Q4: What is the hop-count cost of cross-basin vs same-basin navigation?

Architecture: Continuous descent → plateau detect → rational projection (repeat)

Domain: TwoMassSpring (symmetric 2-DOF, Rayleigh damping)
  ω₁ = √(k/m) [in-phase], ω₂ = √((k+2k_c)/m) [out-of-phase]
  ζₙ = α/(2ωₙ) + β·ωₙ/2   (Rayleigh modal damping)

Output:
  benchmarks/results/competing_surfaces.json
  benchmarks/results/competing_surfaces_cdf.png
  benchmarks/results/competing_surfaces_atlas.png
  benchmarks/results/competing_surfaces_ratio_traces.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tensor.eigen_walker import EigenWalker, _build_feature
from tensor.domain_inverter import TwoMassSpringInverter
from tensor.spectral_energy import (
    SpectralEnergy, SpectralEnergyConfig,
    SpectralState, SpectralMode,
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
import matplotlib.patches as mpatches

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

# Target basin: Z_{3:2}
TARGET_RATIO_LO, TARGET_RATIO_HI = 1.40, 1.60
TARGET_ZETA_LO,  TARGET_ZETA_HI  = 0.05, 0.25
TARGET_F1_LO,    TARGET_F1_HI    = 5.0,  20.0   # in-phase mode [Hz]

# Start region: forced near Z_{2:1}
START_RATIO_LO, START_RATIO_HI = 1.85, 2.15


# ── TMS utilities ──────────────────────────────────────────────────────────────

def _tms_clip(theta: dict) -> dict:
    return {k: float(np.clip(theta[k], TMS_BOUNDS[k][0], TMS_BOUNDS[k][1]))
            for k in TMS_KEYS}


def _tms_eigvals(theta: dict) -> np.ndarray:
    return two_mass_spring_eigvals(theta)


def _tms_ratio(theta: dict) -> float:
    """ω₂/ω₁ frequency ratio."""
    try:
        f1, f2 = two_mass_modal_freqs_hz(theta)
        return f2 / f1 if f1 > 1e-12 else 999.0
    except Exception:
        return 999.0


def _get_ratio_from_ev(ev: np.ndarray) -> float:
    """ω_dominant / ω_secondary from eigenvalue array."""
    try:
        state = SpectralState.from_eigvals(ev)
        if len(state.modes) < 2:
            return 1.0
        return state.modes[0].omega0 / max(state.modes[1].omega0, 1e-10)
    except Exception:
        return 1.0


def _in_target(theta: dict) -> bool:
    """Is θ in the Z_{3:2} target basin?"""
    try:
        f1, f2 = two_mass_modal_freqs_hz(theta)
        z1, z2 = two_mass_modal_zetas(theta)
        ratio = f2 / max(f1, 1e-12)
        return (
            TARGET_RATIO_LO <= ratio  <= TARGET_RATIO_HI
            and TARGET_F1_LO  <= f1    <= TARGET_F1_HI
            and TARGET_ZETA_LO <= z1   <= TARGET_ZETA_HI
            and TARGET_ZETA_LO <= z2   <= TARGET_ZETA_HI
        )
    except Exception:
        return False


def _sample_near_2_1(rng: np.random.Generator) -> dict:
    """Sample TMS params with ω₂/ω₁ ∈ [1.85, 2.15] (forced near Z_{2:1}).

    ω₂/ω₁ = √(1 + 2·k_c/k)  →  k_c/k = (r²−1)/2
    """
    m   = float(10 ** rng.uniform(np.log10(0.1),  np.log10(10.0)))
    k   = float(10 ** rng.uniform(np.log10(1.0),  np.log10(1e4)))
    r   = float(rng.uniform(START_RATIO_LO, START_RATIO_HI))
    k_c = float(np.clip(k * (r**2 - 1.0) / 2.0,
                        TMS_BOUNDS['k_c'][0], TMS_BOUNDS['k_c'][1]))
    alpha = float(rng.uniform(0.01, 5.0))
    beta  = float(rng.uniform(1e-5, 0.01))
    return {'m': m, 'k': k, 'k_c': k_c, 'alpha': alpha, 'beta': beta}


def _sample_random_tms(rng: np.random.Generator) -> dict:
    """Uniform log-space sample over TMS bounds (no ratio constraint)."""
    return {
        k: float(10 ** rng.uniform(np.log10(TMS_BOUNDS[k][0]),
                                   np.log10(TMS_BOUNDS[k][1])))
        for k in TMS_KEYS
    }


# ── Nearest-rational lookup ────────────────────────────────────────────────────

def _nearest_rational(ratio: float, max_p: int = 12, max_q: int = 12
                      ) -> Tuple[int, int, float]:
    """Return (p, q, dist) for the nearest p/q fraction to ratio."""
    best_p, best_q, best_dist = 1, 1, abs(ratio - 1.0)
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            d = abs(ratio - p / q)
            if d < best_dist:
                best_dist, best_p, best_q = d, p, q
    return best_p, best_q, best_dist


# ── Energy config: 3:2 target ─────────────────────────────────────────────────

def make_32_config() -> SpectralEnergyConfig:
    """SpectralEnergyConfig targeting Z_{3:2}: f_dom=15 Hz, f_sec=10 Hz → ratio=1.5."""
    return SpectralEnergyConfig(
        target_zetas=[0.15, 0.15],
        target_omega0s_hz=[15.0, 10.0],   # modes[0]=15Hz, modes[1]=10Hz → 3:2
        K_ratios=8,
        w_stab=10.0,
        w_damp=1.0,
        w_freq=0.5,
        w_harm=2.0,
    )


# ── Training data builder ──────────────────────────────────────────────────────

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
    """Build (X, Y) energy-gradient pairs for n_modes=2 with 3:2 target."""
    energy = SpectralEnergy(config)
    states, targets = energy.generate_training_samples(
        n_samples=n_samples,
        n_modes=2,
        zeta_range=(0.01, 0.80),
        logw_range=(1.0, 3.5),   # ω₀ ∈ [10, 3162] rad/s
        seed=seed,
    )
    X_list, Y_list = [], []
    for state, target in zip(states, targets):
        if len(state.modes) < 2:
            continue
        ev0 = _mode_to_eigvals(state.modes[0])
        ev1 = _mode_to_eigvals(state.modes[1])
        eigvals = np.concatenate([ev0, ev1])
        X_list.append(_build_feature(eigvals, 'lca'))
        Y_list.append(target[:4])   # [Δζ_dom, Δlogω_dom, Δζ_sec, Δlogω_sec]
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


# ── Walker step application ────────────────────────────────────────────────────

def _apply_r2_step(theta: dict, ev: np.ndarray,
                   walker: EigenWalker, step_scale: float) -> dict:
    """Apply EigenWalker step in 4D spectral space, invert to θ."""
    delta = walker.predict_step(ev, regime='lca') * step_scale
    state = SpectralState.from_eigvals(ev)
    if len(state.modes) < 2:
        return theta

    modes_new = []
    for i in range(2):
        d_z = float(np.clip(delta[2 * i],     -0.5, 0.5))
        d_w = float(np.clip(delta[2 * i + 1], -1.0, 1.0))
        zeta_new  = float(np.clip(state.modes[i].zeta + d_z, 1e-4, 2.0))
        omega_new = float(max(10.0 ** (state.modes[i].log10_omega0 + d_w), 1e-3))
        modes_new.append((zeta_new, omega_new))

    theta_new = _tms_inv.invert_modes(modes_new, theta)
    return _tms_clip(theta_new)


# ── Jumper: rational projection toward Z_{3:2} ────────────────────────────────

def _jumper_candidates(ev: np.ndarray, theta: dict, K: int = 8) -> List[dict]:
    """
    Generate θ candidates for cross-tube navigation toward Z_{3:2}.

    Strategy A: rational snap of ω_high around current ω_low
                (explores all nearby rational tubes)
    Strategy B: direct target grid in Z_{3:2}
                (f₁∈[5..20]Hz, ratio∈[1.40..1.60], ζ combinations)
    """
    candidates = []
    try:
        state = SpectralState.from_eigvals(ev)
        if len(state.modes) < 2:
            return candidates
        omega_lo  = state.modes[1].omega0
        omega_hi  = state.modes[0].omega0
        z_lo      = state.modes[1].zeta
        z_hi      = state.modes[0].zeta
    except Exception:
        return candidates

    zeta_choices = [0.05, 0.10, 0.15, 0.20, 0.25]

    # Strategy A: rational snap — ω_hi → (p/q)·ω_lo across a range of ratios
    for p in range(1, K + 1):
        for q in range(1, K + 1):
            ratio_cand = p / q
            if 1.0 <= ratio_cand <= 3.5:
                omega_hi_cand = ratio_cand * omega_lo
                for z1 in zeta_choices:
                    for z2 in zeta_choices:
                        try:
                            raw = _tms_inv.invert_modes(
                                [(z2, omega_hi_cand), (z1, omega_lo)], theta
                            )
                            candidates.append(_tms_clip(raw))
                        except Exception:
                            pass

    # Strategy B: direct target grid in Z_{3:2}
    for f1_hz in [5.0, 7.5, 10.0, 12.5, 15.0, 20.0]:
        omega_lo_t = 2.0 * np.pi * f1_hz
        for ratio_t in [1.40, 1.45, 1.50, 1.55, 1.60]:
            omega_hi_t = ratio_t * omega_lo_t
            for z1 in zeta_choices:
                for z2 in zeta_choices:
                    try:
                        raw = _tms_inv.invert_modes(
                            [(z2, omega_hi_t), (z1, omega_lo_t)], theta
                        )
                        candidates.append(_tms_clip(raw))
                    except Exception:
                        pass

    return candidates


def _try_jumper(ev: np.ndarray, theta: dict,
                energy: SpectralEnergy) -> Optional[dict]:
    """Evaluate all candidates, return lowest-energy valid θ."""
    candidates = _jumper_candidates(ev, theta)
    best_theta, best_E = None, float('inf')
    for cand in candidates:
        try:
            ev_c  = _tms_eigvals(cand)
            state = SpectralState.from_eigvals(ev_c)
            if len(state.modes) < 2:
                continue
            E = energy.compute(state)
            if E < best_E:
                best_E    = E
                best_theta = cand
        except Exception:
            pass
    return best_theta


# ── Methods ────────────────────────────────────────────────────────────────────

def run_random_search(theta0: dict, max_iter: int, rng: np.random.Generator,
                      forced_start: bool = True) -> Tuple[int, bool, List]:
    """Random restart: each step samples a fresh θ from the start region."""
    sampler = _sample_near_2_1 if forced_start else _sample_random_tms
    for i in range(max_iter):
        theta = sampler(rng)
        if _in_target(theta):
            return i + 1, True, []
    return max_iter, False, []


def run_plateau_hybrid(
    theta0:         dict,
    walker:         EigenWalker,
    energy:         SpectralEnergy,
    max_iter:       int,
    step_scale:     float = 0.30,
    plateau_window: int   = 8,
    plateau_eps:    float = 0.02,
    boundary_window: int  = 20,
    track_ratios:   bool  = False,
) -> Tuple[int, bool, List[Tuple], Optional[List]]:
    """
    Plateau-triggered hybrid with tube-sequence tracking.

    Returns:
        (iters, success, tube_sequence, ratio_trace_or_None)

    tube_sequence entries: (step, from_tube_str, ratio_before, to_tube_str, ratio_after)
    ratio_trace: [(step, ratio), ...] at every plateau fire (if track_ratios)
    """
    theta = dict(theta0)
    try:
        ev = _tms_eigvals(theta)
    except Exception:
        return max_iter, False, [], None

    energy_history: List[float] = []
    boundary_count  = 0
    tube_sequence:  List[Tuple] = []
    ratio_trace:    Optional[List] = [] if track_ratios else None

    for i in range(max_iter):
        if _in_target(theta):
            return i + 1, True, tube_sequence, ratio_trace

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
            clipped    = any(abs(theta_raw[k] - theta_new[k]) > 1e-12
                             for k in TMS_KEYS)
        except Exception:
            theta_new = dict(theta)
            clipped   = False

        boundary_count = boundary_count + 1 if clipped else max(0, boundary_count - 1)

        # Stagnation detection
        energy_plateau = (
            len(energy_history) >= plateau_window
            and (energy_history[0] - energy_history[-1]) < plateau_eps
        )

        if energy_plateau or (boundary_count >= boundary_window):
            # Record pre-jump tube
            ratio_before = _get_ratio_from_ev(ev)
            p_b, q_b, _  = _nearest_rational(ratio_before)
            from_tube     = f'{p_b}:{q_b}'

            if track_ratios and ratio_trace is not None:
                ratio_trace.append((i, ratio_before))

            jumped = _try_jumper(ev, theta, energy)
            if jumped is not None:
                theta = jumped
                try:
                    ev_j         = _tms_eigvals(theta)
                    ratio_after  = _get_ratio_from_ev(ev_j)
                    p_a, q_a, _  = _nearest_rational(ratio_after)
                    to_tube       = f'{p_a}:{q_a}'
                    tube_sequence.append(
                        (i, from_tube, ratio_before, to_tube, ratio_after)
                    )
                    ev = ev_j
                except Exception:
                    ev = _tms_eigvals(theta)
            energy_history.clear()
            boundary_count = 0
            continue

        try:
            ev    = _tms_eigvals(theta_new)
            theta = theta_new
        except Exception:
            pass

    return max_iter, False, tube_sequence, ratio_trace


# ── Atlas graph construction ───────────────────────────────────────────────────

def build_atlas(tube_sequences: List[List[Tuple]]) -> Dict[Tuple, int]:
    """Count (from_tube, to_tube) transitions across all trials."""
    graph: Dict[Tuple, int] = defaultdict(int)
    for seq in tube_sequences:
        for (step, from_t, from_r, to_t, to_r) in seq:
            if from_t != to_t:
                graph[(from_t, to_t)] += 1
    return dict(graph)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_cdf(results: Dict, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {
        'random_forced':     ('Random (forced 2:1 start)',  '#888888', '-'),
        'plateau_forced':    ('Plateau Hybrid (forced 2:1→3:2)', '#2196F3', '-'),
        'plateau_random':    ('Plateau Hybrid (random start→3:2)', '#4CAF50', '--'),
    }
    for key, res in results.items():
        label, color, ls = style.get(key, (key, 'k', '-'))
        succ_iters = sorted(r['iters'] for r in res if r['success'])
        if not succ_iters:
            continue
        n_total = len(res)
        ys = np.arange(1, len(succ_iters) + 1) / n_total
        ax.plot(succ_iters, ys, color=color, linestyle=ls, linewidth=2, label=label)
    ax.set_xlabel('Iterations to Success')
    ax.set_ylabel('CDF (fraction of all trials)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_atlas(atlas: Dict[Tuple, int], out_path: str):
    if not atlas:
        return
    sorted_items = sorted(atlas.items(), key=lambda x: -x[1])[:20]
    labels = [f'{f}→{t}' for (f, t), _ in sorted_items]
    counts = [c for _, c in sorted_items]
    target_hit = [('3:2' in t) for (_, t), _ in sorted_items]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 5))
    colors_bar = ['#2196F3' if h else '#FF9800' for h in target_hit]
    ax.bar(range(len(labels)), counts, color=colors_bar)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Transition Count')
    ax.set_title('Empirical Atlas: Rational Tube Transitions (Z_{2:1} → Z_{3:2})')
    patches = [
        mpatches.Patch(color='#2196F3', label='→ 3:2 (target tube)'),
        mpatches.Patch(color='#FF9800', label='intermediate / other'),
    ]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_ratio_traces(traces: List[Tuple[List, List]], out_path: str):
    """Plot ω₂/ω₁ ratio at each plateau-fire step for sample trials."""
    if not traces:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for steps, ratios in traces[:30]:
        if steps and ratios:
            ax.scatter(steps, ratios, s=18, alpha=0.5)
    ax.axhline(1.5, color='#2196F3', linewidth=2, linestyle='--', label='3:2 target (1.50)')
    ax.axhline(2.0, color='#FF9800', linewidth=2, linestyle='--', label='2:1 start (2.00)')
    ax.axhspan(TARGET_RATIO_LO, TARGET_RATIO_HI, alpha=0.12, color='#2196F3',
               label='target band')
    ax.axhspan(START_RATIO_LO, START_RATIO_HI, alpha=0.08, color='#FF9800',
               label='start band')
    ax.set_xlabel('Plateau-Fire Step')
    ax.set_ylabel('ω₂/ω₁ at plateau fire')
    ax.set_title('Ratio at plateau fires: cross-basin navigation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Competing rational surfaces: Z_{2:1} → Z_{3:2} navigation.'
    )
    parser.add_argument('--n-trials',  type=int, default=100)
    parser.add_argument('--max-iter',  type=int, default=500)
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--n-epochs',  type=int, default=300)
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print('=== Competing Rational Surfaces: Z_{2:1} → Z_{3:2} ===')
    print(f'  trials={args.n_trials}  max_iter={args.max_iter}  seed={args.seed}')
    print(f'  Start region: ω₂/ω₁ ∈ [{START_RATIO_LO}, {START_RATIO_HI}] (forced Z_{{2:1}})')
    print(f'  Target basin: ω₂/ω₁ ∈ [{TARGET_RATIO_LO}, {TARGET_RATIO_HI}] (Z_{{3:2}})')
    print()

    # Phase 1: Train walker with 3:2 energy-gradient
    print('[Phase 1] Training EigenWalker(n_modes=2) on 3:2 energy gradient...')
    config = make_32_config()
    t0 = time.time()
    X, Y = build_32_training_data(config, args.n_samples, seed=args.seed)
    walker = EigenWalker(hidden=128, n_modes=2, seed=args.seed)
    losses = _train_mlp_on_arrays(walker, X, Y, n_epochs=args.n_epochs,
                                  lr=1e-3, batch_size=64, seed=args.seed)
    print(f'  {len(X)} samples, {args.n_epochs} epochs in {time.time()-t0:.1f}s')
    print(f'  Loss: {losses[0]:.6f} → {losses[-1]:.6f}')
    print()

    energy = SpectralEnergy(config)

    # Phase 2: Benchmark
    print('[Phase 2] Benchmark: 100 trials × 3 methods...')

    results = {
        'random_forced':  [],
        'plateau_forced': [],
        'plateau_random': [],
    }
    all_seqs: List[List[Tuple]] = []
    ratio_traces: List[Tuple[List, List]] = []

    for trial in range(args.n_trials):
        theta0_forced = _sample_near_2_1(rng)
        theta0_random = _sample_random_tms(rng)

        # Random search (forced Z_{2:1} restarts)
        rng_r = np.random.default_rng(args.seed + trial * 37 + 3)
        iters_r, suc_r, _ = run_random_search(
            theta0_forced, args.max_iter, rng_r, forced_start=True
        )
        results['random_forced'].append({'iters': iters_r, 'success': suc_r})

        # Plateau hybrid — forced Z_{2:1} start → Z_{3:2} target
        collect_trace = (trial < 30)
        iters_f, suc_f, seq_f, rtrace = run_plateau_hybrid(
            theta0_forced, walker, energy, args.max_iter,
            track_ratios=collect_trace,
        )
        results['plateau_forced'].append({
            'iters': iters_f, 'success': suc_f, 'hops': len(seq_f)
        })
        all_seqs.append(seq_f)
        if collect_trace and rtrace:
            steps  = [s for s, _ in rtrace]
            ratios = [r for _, r in rtrace]
            ratio_traces.append((steps, ratios))

        # Plateau hybrid — random start → Z_{3:2} target
        iters_ra, suc_ra, _, _ = run_plateau_hybrid(
            theta0_random, walker, energy, args.max_iter,
        )
        results['plateau_random'].append({'iters': iters_ra, 'success': suc_ra})

        if (trial + 1) % 10 == 0:
            print(f'  [{trial+1:3d}/{args.n_trials}]  '
                  f'rand: {iters_r}({"✓" if suc_r else "✗"})  '
                  f'forced: {iters_f}({"✓" if suc_f else "✗"})  '
                  f'random: {iters_ra}({"✓" if suc_ra else "✗"})')

    print()

    # Phase 3: Atlas graph
    atlas = build_atlas(all_seqs)

    # Phase 4: Summary
    print('=== Results ===')
    label_map = {
        'random_forced':  'Random Search (forced Z_{2:1} start)',
        'plateau_forced': 'Plateau Hybrid (forced Z_{2:1} → Z_{3:2})',
        'plateau_random': 'Plateau Hybrid (random start → Z_{3:2})',
    }
    print(f'{"Method":<48} {"Succ":>6} {"Mean":>8} {"Median":>8} {"p95":>8} {"Fail%":>7}')
    print('-' * 88)
    for key, res in results.items():
        succ = [r for r in res if r['success']]
        n    = len(res)
        if succ:
            iters  = [r['iters'] for r in succ]
            mean_i = float(np.mean(iters))
            med_i  = float(np.median(iters))
            p95_i  = float(np.percentile(iters, 95))
        else:
            mean_i = med_i = p95_i = float('nan')
        fail_pct = 100.0 * (n - len(succ)) / n
        print(f'{label_map[key]:<48} {len(succ):>3}/{n:<3} '
              f'{mean_i:>8.1f} {med_i:>8.1f} {p95_i:>8.1f} {fail_pct:>6.1f}%')

    # Hop statistics
    hops_forced = [r['hops'] for r in results['plateau_forced']]
    hops_succ   = [r['hops'] for r in results['plateau_forced'] if r['success']]
    zero_hop_succ = sum(
        1 for r in results['plateau_forced'] if r['success'] and r['hops'] == 0
    )
    print()
    print('=== Hop Statistics (plateau_forced) ===')
    print(f'  Mean hops per trial:    {np.mean(hops_forced):.2f}')
    print(f'  Mean hops (successes):  {np.mean(hops_succ):.2f}' if hops_succ else '')
    print(f'  Max hops:               {max(hops_forced) if hops_forced else 0}')
    print(f'  Zero-hop successes:     {zero_hop_succ}')
    one_hop = sum(1 for r, seq in zip(results['plateau_forced'], all_seqs)
                  if r['success'] and len(seq) == 1)
    multi_hop = sum(1 for r, seq in zip(results['plateau_forced'], all_seqs)
                    if r['success'] and len(seq) > 1)
    print(f'  Single-hop successes:   {one_hop}')
    print(f'  Multi-hop successes:    {multi_hop}')

    # Atlas
    print()
    print('=== Empirical Atlas: Top Transitions ===')
    sorted_atlas = sorted(atlas.items(), key=lambda x: -x[1])
    for (from_t, to_t), cnt in sorted_atlas[:15]:
        tag = '← TARGET' if to_t == '3:2' else ''
        print(f'  {from_t:8s} → {to_t:8s}  : {cnt:4d}  {tag}')

    # Structural questions
    print()
    print('=== Structural Questions ===')
    n_forced = len(results['plateau_forced'])
    n_succ_f = sum(1 for r in results['plateau_forced'] if r['success'])
    n_succ_r = sum(1 for r in results['plateau_random'] if r['success'])
    print(f'  Q1 (single-hop sufficient?):    {one_hop}/{n_succ_f} successes use 1 hop')
    print(f'  Q2 (intermediate tubes?):       {len(set(f for (f,t) in atlas if t!="3:2"))} '
          f'distinct intermediate tubes appear in atlas')
    print(f'  Q3 (100% success maintained?):  {n_succ_f}/{n_forced} forced, '
          f'{n_succ_r}/{n_forced} random')
    if hops_succ and [r for r in results['plateau_random'] if r['success']]:
        mean_r_iters = np.mean([r['iters'] for r in results['plateau_random'] if r['success']])
        mean_f_iters = np.mean([r['iters'] for r in results['plateau_forced'] if r['success']])
        print(f'  Q4 (cross-basin cost):          '
              f'forced={mean_f_iters:.1f} vs random={mean_r_iters:.1f} iters')

    # Save JSON
    out = {
        'args': vars(args),
        'results': {
            k: [{'iters': r['iters'], 'success': r['success']}
                for r in res]
            for k, res in results.items()
        },
        'atlas': {f'{f}→{t}': cnt for (f, t), cnt in atlas.items()},
        'hop_stats': {
            'mean_hops_per_trial': float(np.mean(hops_forced)) if hops_forced else 0.0,
            'mean_hops_success':   float(np.mean(hops_succ)) if hops_succ else 0.0,
            'max_hops':            int(max(hops_forced)) if hops_forced else 0,
            'zero_hop_success':    zero_hop_succ,
            'single_hop_success':  one_hop,
            'multi_hop_success':   multi_hop,
        },
    }
    json_path = os.path.join(RESULTS_DIR, 'competing_surfaces.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults: {json_path}')

    # Plots
    _plot_cdf(
        {k: results[k] for k in ('random_forced', 'plateau_forced', 'plateau_random')},
        'Z_{2:1} → Z_{3:2}: Cross-Basin Navigation',
        os.path.join(RESULTS_DIR, 'competing_surfaces_cdf.png'),
    )
    _plot_atlas(atlas, os.path.join(RESULTS_DIR, 'competing_surfaces_atlas.png'))
    _plot_ratio_traces(ratio_traces,
                       os.path.join(RESULTS_DIR, 'competing_surfaces_ratio_traces.png'))
    print(f'Plots: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
