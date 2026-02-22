#!/usr/bin/env python3
"""
benchmarks/instability_lead_time.py — Spectral vs. time-domain detection.

System: Stuart–Landau normal form (canonical Hopf bifurcation)
  ẋ₁ = μ·x₁ − x₂ − x₁·(x₁² + x₂²)
  ẋ₂ = x₁   + μ·x₂ − x₂·(x₁² + x₂²)

Linearised at origin:  A(μ) = [[μ, −1], [1, μ]]
Eigenvalues:           λ = μ ± j
Bifurcation:           μ = 0  →  Re(λ) crosses zero exactly

This gives a clean, analytically verifiable test case:
  - Spectral detection is exact (Re(λ) = μ > 0 the moment μ > 0)
  - Time-domain detection requires finite integration to observe growth

Experiment:
  Scan μ from mu_lo to mu_hi in n_mu steps.
  At each μ:
    Spectral:    Re(λ) > SPECTRAL_THRESH?  (exact, O(1) computation)
    Time-domain: integrate from x₀ with RK4, T steps, dt each.
                 ‖x(T)‖₂ > TD_THRESH?

  Report:
    μ_spectral        — first μ where spectral flags (should be ≈ 0.000)
    μ_td[T]           — first μ where time-domain flags at horizon T
    Δμ[T] = μ_td[T] − μ_spectral  (lead time; positive → spectral detects first)

Outputs:
  benchmarks/results/instability_lead_time.json
  benchmarks/results/instability_lead_time.png

Usage:
  python benchmarks/instability_lead_time.py
  python benchmarks/instability_lead_time.py --mu-lo -0.3 --mu-hi 0.3 --n-mu 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


# ── System definition ──────────────────────────────────────────────────────────

def hopf_eigvals(mu: float) -> np.ndarray:
    """Exact eigenvalues of linearised Stuart–Landau at origin."""
    return np.array([mu + 1j, mu - 1j])


def hopf_rhs(mu: float, x: np.ndarray) -> np.ndarray:
    """Full nonlinear vector field."""
    x1, x2 = x
    r2 = x1 ** 2 + x2 ** 2
    return np.array([mu * x1 - x2 - x1 * r2,
                     x1 + mu * x2 - x2 * r2])


def rk4_final(mu: float, x0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """RK4 integration — returns final state only."""
    x = x0.copy()
    for _ in range(n_steps):
        k1 = hopf_rhs(mu, x)
        k2 = hopf_rhs(mu, x + 0.5 * dt * k1)
        k3 = hopf_rhs(mu, x + 0.5 * dt * k2)
        k4 = hopf_rhs(mu, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# ── Detection thresholds ───────────────────────────────────────────────────────

SPECTRAL_THRESH = 1e-10      # Re(λ) > this → spectral flags instability
TD_THRESH       = 0.5        # ‖x(T)‖₂ > this → time-domain flags instability
X0              = np.array([0.01, 0.0])   # small perturbation around origin
DT              = 0.02       # RK4 step ≈ 1/50 of one oscillation period


# ── Scan ───────────────────────────────────────────────────────────────────────

def run_scan(
    mu_lo: float,
    mu_hi: float,
    n_mu: int,
    td_horizons: List[int],
) -> dict:
    """
    Scan μ from mu_lo to mu_hi.

    Returns a results dict with per-μ records and aggregate detection statistics.
    """
    mus = np.linspace(mu_lo, mu_hi, n_mu)

    # Detection thresholds: first μ where each method flags
    spectral_flag_mu: Optional[float] = None
    td_flag_mus: Dict[int, Optional[float]] = {T: None for T in td_horizons}

    records: List[dict] = []

    for mu_val in mus:
        mu = float(mu_val)
        ev = hopf_eigvals(mu)
        hurwitz = float(np.max(np.real(ev)))

        spec_flag = hurwitz > SPECTRAL_THRESH
        if spec_flag and spectral_flag_mu is None:
            spectral_flag_mu = mu

        td_norms: Dict[int, float] = {}
        td_flags: Dict[int, bool]  = {}
        for T in td_horizons:
            xf = rk4_final(mu, X0, DT, T)
            norm = float(np.linalg.norm(xf))
            td_norms[T] = norm
            td_flags[T] = norm > TD_THRESH
            if td_flags[T] and td_flag_mus[T] is None:
                td_flag_mus[T] = mu

        records.append({
            'mu':             round(mu, 8),
            'hurwitz_margin': round(hurwitz, 8),
            'spectral_flag':  spec_flag,
            'td_flags': {str(T): td_flags[T]    for T in td_horizons},
            'td_norms': {str(T): round(td_norms[T], 6) for T in td_horizons},
        })

    # Lead times: Δμ = μ_td − μ_spectral
    lead_times: Dict[str, Optional[float]] = {}
    for T in td_horizons:
        if td_flag_mus[T] is not None and spectral_flag_mu is not None:
            lead_times[str(T)] = round(float(td_flag_mus[T]) - spectral_flag_mu, 8)
        else:
            lead_times[str(T)] = None

    return {
        'mu_range':        [mu_lo, mu_hi],
        'n_mu':            n_mu,
        'mu_step':         round((mu_hi - mu_lo) / n_mu, 8),
        'td_horizons':     td_horizons,
        'dt':              DT,
        'spectral_thresh': SPECTRAL_THRESH,
        'td_thresh':       TD_THRESH,
        'x0':              X0.tolist(),
        'spectral_flag_mu':    spectral_flag_mu,
        'td_flag_mus':         {str(T): td_flag_mus[T]   for T in td_horizons},
        'lead_times_delta_mu': lead_times,
        'records':             records,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(result: dict, out_dir: str) -> List[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not available — skipping")
        return []

    horizons = result['td_horizons']
    records  = result['records']
    mus      = [r['mu']             for r in records]
    margins  = [r['hurwitz_margin'] for r in records]

    td_colors = ['#c0392b', '#e67e22', '#d4ac0d', '#8e44ad']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # ── Top: Hurwitz margin ────────────────────────────────────────────────────
    ax1.plot(mus, margins, color='#2980b9', linewidth=2, label='Re(λ)  [spectral]')
    ax1.axhline(0.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.7,
                label='True bifurcation (μ = 0)')

    mu_spec = result['spectral_flag_mu']
    if mu_spec is not None:
        ax1.axvline(mu_spec, color='#2980b9', linestyle='-.', linewidth=1.5,
                    label=f'Spectral flag  μ = {mu_spec:.5f}')

    ax1.set_ylabel('Re(λ)', fontsize=11)
    ax1.set_title(
        'Spectral vs. Time-Domain Instability Detection\n'
        'Stuart–Landau system  (Hopf bifurcation at μ = 0)',
        fontsize=12, fontweight='bold',
    )
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ── Bottom: time-domain norms ──────────────────────────────────────────────
    for i, T in enumerate(horizons):
        norms = [r['td_norms'][str(T)] for r in records]
        # Clip to avoid log(0)
        norms_safe = [max(v, 1e-12) for v in norms]
        ax2.semilogy(mus, norms_safe,
                     color=td_colors[i % len(td_colors)],
                     linewidth=1.5,
                     label=f'‖x(T)‖  T={T} steps')

        mu_td = result['td_flag_mus'].get(str(T))
        if mu_td is not None:
            ax2.axvline(mu_td, color=td_colors[i % len(td_colors)],
                        linestyle=':', linewidth=1.2, alpha=0.8)

    ax2.axhline(result['td_thresh'], color='gray', linestyle='--', linewidth=1,
                label=f'threshold = {result["td_thresh"]}')
    ax2.axvline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax2.set_xlabel('μ  (bifurcation parameter)', fontsize=11)
    ax2.set_ylabel('‖x(T)‖₂  (log scale)', fontsize=11)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'instability_lead_time.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return [out_path]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mu-lo',  type=float, default=-0.5)
    parser.add_argument('--mu-hi',  type=float, default=+0.5)
    parser.add_argument('--n-mu',   type=int,   default=200)
    parser.add_argument('--out',    type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    TD_HORIZONS = [100, 300, 500, 1000]

    print("=== Instability Lead Time Benchmark ===")
    print(f"  System:   Stuart–Landau (Hopf at μ = 0)")
    print(f"  Scan:     μ ∈ [{args.mu_lo}, {args.mu_hi}], n={args.n_mu}")
    print(f"  Horizons: {TD_HORIZONS} RK4 steps  (dt={DT})")
    print(f"  TD thresh: ‖x(T)‖ > {TD_THRESH}")

    t0 = time.time()
    result = run_scan(args.mu_lo, args.mu_hi, args.n_mu, TD_HORIZONS)
    elapsed = time.time() - t0

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\nScan complete ({elapsed:.2f}s) — {args.n_mu} parameter points")
    print(f"\n  Spectral detection:  μ = {result['spectral_flag_mu']}  "
          f"(exact answer: 0.000)")

    mu_step = result['mu_step']
    print(f"\n  Time-domain detections (Δμ = parameter-space lead time):")
    print(f"  {'Horizon':>8}  {'μ_td':>10}  {'Δμ (lead)':>12}  {'steps ahead':>12}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}")

    mu_spec = result['spectral_flag_mu']
    for T in TD_HORIZONS:
        mu_td = result['td_flag_mus'].get(str(T))
        lead  = result['lead_times_delta_mu'].get(str(T))
        steps_ahead = round(lead / mu_step) if lead is not None else 'N/A'
        print(f"  T={T:>6}   "
              f"{(mu_td if mu_td is not None else 'NOT FOUND'):>10}  "
              f"{(f'{lead:+.5f}' if lead is not None else 'N/A'):>12}  "
              f"{str(steps_ahead):>12}")

    print(f"\n  [Interpretation]")
    lead_100 = result['lead_times_delta_mu'].get(str(TD_HORIZONS[0]))
    if lead_100 is not None and lead_100 > 0:
        steps = round(lead_100 / mu_step)
        print(f"  Spectral flags instability {lead_100:.4f} μ-units ({steps} scan steps) "
              f"before T={TD_HORIZONS[0]} time-domain.")
    else:
        print(f"  No lead time recorded for T={TD_HORIZONS[0]}.")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = os.path.join(args.out, 'instability_lead_time.json')
    with open(json_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults: {json_path}")

    for p in plot_results(result, args.out):
        print(f"Plot:    {p}")


if __name__ == '__main__':
    main()
