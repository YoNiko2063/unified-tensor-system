"""
Whitepaper figure generation — all publication-quality plots.

Usage:
    python whitepaper/figures.py

Outputs PDF figures to whitepaper/figures/
"""

import os
import sys

# Project root for imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# --------------------------------------------------------------------------
# Common style
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ==========================================================================
# Figure 1: Complex Plane Eigenvalue Plot (10 MHz RLC)
# ==========================================================================
def fig1_complex_plane():
    f0 = 10e6
    zeta = 0.2
    omega0 = 2 * np.pi * f0

    sigma = -zeta * omega0
    omega_d = omega0 * np.sqrt(1 - zeta**2)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Imaginary axis (stability boundary)
    ax.axvline(0, color="k", linewidth=0.8, zorder=1)
    ax.axhline(0, color="k", linewidth=0.8, zorder=1)

    # Stability region shading
    xlims = (sigma * 2.5, -sigma * 0.5)
    ax.axvspan(xlims[0], 0, alpha=0.06, color="green", label="Stable half-plane")

    # Eigenvalue pair
    ax.plot(sigma, omega_d, "go", markersize=10, zorder=5, label=r"$\lambda_1$")
    ax.plot(sigma, -omega_d, "go", markersize=10, zorder=5, label=r"$\lambda_2$")

    # Target markers
    ax.plot(sigma, omega_d, "kx", markersize=12, markeredgewidth=2, zorder=6)
    ax.plot(sigma, -omega_d, "kx", markersize=12, markeredgewidth=2, zorder=6)

    # Annotations
    ax.annotate(
        rf"$\lambda = {sigma/1e6:.2f} \pm j{omega_d/1e6:.2f}$ (MHz)",
        xy=(sigma, omega_d), xytext=(sigma * 0.4, omega_d * 1.15),
        fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    # Damping ratio line
    theta = np.arctan2(omega_d, -sigma)
    r = np.sqrt(sigma**2 + omega_d**2)
    line_x = np.array([0, sigma * 1.3])
    line_y = np.array([0, omega_d * 1.3])
    ax.plot(line_x, line_y, "b--", linewidth=0.8, alpha=0.5)
    ax.text(
        sigma * 0.5, omega_d * 0.65,
        rf"$\zeta = {zeta}$",
        fontsize=9, color="blue", rotation=np.degrees(theta),
    )

    ax.set_xlabel(r"Re$(\lambda)$ [rad/s]")
    ax.set_ylabel(r"Im$(\lambda)$ [rad/s]")
    ax.set_title("Optimized Eigenvalues: 10 MHz Series RLC")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(FIGDIR, "fig1_complex_plane.pdf"))
    plt.close(fig)
    print("  fig1_complex_plane.pdf")


# ==========================================================================
# Figure 2: Frequency Response (10 MHz RLC)
# ==========================================================================
def fig2_frequency_response():
    R = 798.0
    L = 10e-6
    C = 2.53e-12
    f0 = 10e6

    w = np.logspace(6, 8.5, 2000)
    # Series RLC transfer function: H(jw) = jwRC / (1 - w^2 LC + jwRC)
    s = 1j * w
    H = (s * R * C) / (1 + s * R * C + (s**2) * L * C)
    H_mag = 20 * np.log10(np.abs(H) + 1e-30)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogx(w / (2 * np.pi), H_mag, "b-", linewidth=1.2)
    ax.axvline(f0, linestyle="--", color="r", linewidth=0.8, label=rf"$f_0 = {f0/1e6:.0f}$ MHz")

    # Q factor annotation
    Q = 1 / (2 * 0.2)
    bw = f0 / Q
    ax.axvspan(f0 - bw / 2, f0 + bw / 2, alpha=0.08, color="orange", label=rf"$-3$ dB BW ($Q={Q:.1f}$)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Bandpass Response: Optimized 10 MHz RLC")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(-40, 5)
    ax.set_xlim(1e6, 1e8)

    fig.savefig(os.path.join(FIGDIR, "fig2_frequency_response.pdf"))
    plt.close(fig)
    print("  fig2_frequency_response.pdf")


# ==========================================================================
# Figure 3: Pareto Front (from actual optimizer)
# ==========================================================================
def fig3_pareto_front():
    from optimization.circuit_optimizer import CircuitSpec, CircuitOptimizer

    spec = CircuitSpec(center_freq_hz=1000.0, Q_target=5.0)
    opt = CircuitOptimizer(spec)
    pareto = opt.optimize()

    errors = [c.eigenvalue_error for c in pareto.all_candidates]
    gaps = [c.spectral_gap for c in pareto.all_candidates]
    costs = [c.cost for c in pareto.all_candidates]

    # Pareto front members
    front_errors = [c.eigenvalue_error for c in pareto.pareto_front]
    front_gaps = [c.spectral_gap for c in pareto.pareto_front]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    sc = ax.scatter(errors, gaps, c=costs, cmap="viridis", s=50, alpha=0.7,
                    edgecolors="gray", linewidths=0.5, label="Candidates")
    ax.scatter(front_errors, front_gaps, s=120, facecolors="none",
               edgecolors="red", linewidths=2, label="Pareto front", zorder=5)

    ax.set_xlabel(r"Eigenvalue Error $\|\lambda - \lambda^*\|$")
    ax.set_ylabel("Spectral Gap")
    ax.set_title("Multi-Objective Pareto Front")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Cost $J$", shrink=0.85)

    fig.savefig(os.path.join(FIGDIR, "fig3_pareto_front.pdf"))
    plt.close(fig)
    print("  fig3_pareto_front.pdf")


# ==========================================================================
# Figure 4: Stability Basin Heatmap (from actual Monte Carlo)
# ==========================================================================
def fig4_stability_basin():
    from optimization.circuit_optimizer import (
        CircuitSpec, EigenvalueMapper, CircuitOptimizer,
        MonteCarloStabilityAnalyzer,
    )

    # Run optimizer to get nominal values
    spec = CircuitSpec(
        center_freq_hz=1000.0, Q_target=5.0,
        component_tolerances={"R": 0.20, "L": 0.20, "C": 0.20},
    )
    opt = CircuitOptimizer(spec)
    pareto = opt.optimize()
    result = pareto.best_eigenvalue
    mapper = EigenvalueMapper()

    # Dense grid scan
    n_grid = 80
    R_nom, C_nom, L = result.R, result.C, result.L
    R_vals = np.linspace(R_nom * 0.7, R_nom * 1.3, n_grid)
    C_vals = np.linspace(C_nom * 0.7, C_nom * 1.3, n_grid)

    basin = np.zeros((n_grid, n_grid))
    for i, Rv in enumerate(R_vals):
        for j, Cv in enumerate(C_vals):
            eigs = mapper.eigenvalues(Rv, L, Cv)
            # LCA: both eigenvalues have Re < 0
            if np.all(np.real(eigs) < 0):
                gap = abs(abs(eigs[0]) - abs(eigs[1]))
                basin[i, j] = gap
            else:
                basin[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        basin, origin="lower", aspect="auto",
        extent=[C_vals[0] * 1e6, C_vals[-1] * 1e6, R_vals[0], R_vals[-1]],
        cmap="RdYlGn",
    )
    ax.plot(C_nom * 1e6, R_nom, "w+", markersize=14, markeredgewidth=2.5, zorder=5)
    ax.set_xlabel(r"Capacitance ($\mu$F)")
    ax.set_ylabel(r"Resistance ($\Omega$)")
    ax.set_title(r"Stability Basin: Spectral Gap under $\pm 30\%$ Tolerance")
    fig.colorbar(im, ax=ax, label="Spectral Gap", shrink=0.85)

    fig.savefig(os.path.join(FIGDIR, "fig4_stability_basin.pdf"))
    plt.close(fig)
    print("  fig4_stability_basin.pdf")


# ==========================================================================
# Figure 5: Observer 10K-Step Energy Trajectory
# ==========================================================================
def fig5_observer_energy():
    from tensor.semantic_observer import ObserverConfig, SemanticObserver, semantic_energy

    config = ObserverConfig(
        state_dim=32, input_dim=16, dt=0.01,
        energy_cap=10.0, gamma_damp=0.1, lambda_max=2.0,
    )
    obs = SemanticObserver(config)
    rng = np.random.default_rng(42)

    N = 10000
    energies = np.zeros(N)
    norms = np.zeros(N)

    for step in range(N):
        u = rng.standard_normal(16)
        u = u / (np.linalg.norm(u) + 1e-12)
        obs.step(u)
        E = semantic_energy(obs.x, np.zeros_like(obs.x), obs.P)
        energies[step] = E
        norms[step] = np.linalg.norm(obs.x)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    ax1.plot(energies, linewidth=0.3, color="steelblue", alpha=0.7)
    ax1.axhline(config.energy_cap, color="r", linestyle="--", linewidth=0.8,
                label=rf"Energy cap = {config.energy_cap}")
    ax1.axhline(config.energy_cap * 5, color="darkred", linestyle=":", linewidth=0.8,
                label=r"$5\times$ cap (violation threshold)")
    ax1.set_ylabel(r"$E_s = x^T P x$")
    ax1.set_title("Semantic Observer: 10,000-Step Energy Trajectory")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(norms, linewidth=0.3, color="darkorange", alpha=0.7)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel(r"$\|x\|_2$")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_observer_energy.pdf"))
    plt.close(fig)
    print("  fig5_observer_energy.pdf")


# ==========================================================================
# Figure 6: HDV Orthogonality Cross-Contamination Matrix
# ==========================================================================
def fig6_hdv_orthogonality():
    from tensor.semantic_observer import HDVOrthogonalizer

    hdv_dim = 2000
    orth = HDVOrthogonalizer(hdv_dim=hdv_dim)
    rng = np.random.default_rng(42)
    domains = ["circuit", "semantic", "market", "code"]
    n_trials = 200

    # Cross-contamination matrix
    contam = np.zeros((4, 4))
    for _ in range(n_trials):
        vec = rng.standard_normal(hdv_dim)
        projections = [orth.project(vec, d) for d in domains]
        for i in range(4):
            for j in range(4):
                contam[i, j] += abs(np.dot(projections[i], projections[j]))
    contam /= n_trials

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(contam, cmap="Blues", vmin=0)
    for i in range(4):
        for j in range(4):
            val = contam[i, j]
            color = "white" if val > contam.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(4))
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(domains)
    ax.set_title(r"HDV Cross-Contamination: $\langle H_i v, H_j v \rangle$")
    fig.colorbar(im, ax=ax, shrink=0.85)

    fig.savefig(os.path.join(FIGDIR, "fig6_hdv_orthogonality.pdf"))
    plt.close(fig)
    print("  fig6_hdv_orthogonality.pdf")


# ==========================================================================
# Figure 7: Architecture Diagram (programmatic)
# ==========================================================================
def fig7_architecture():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    # Boxes
    boxes = [
        # (x, y, w, h, label, color)
        (0.0, 4.0, 2.2, 1.0, "Circuit Optimizer\n$J = \\sum w_k J_k$", "#E3F2FD"),
        (2.7, 4.0, 2.2, 1.0, "Semantic Observer\n$\\dot{x} = Ax + g(x) + Bu$", "#E8F5E9"),
        (5.4, 4.0, 2.0, 1.0, "Koopman EDMD\n$K = G^+ A$", "#FFF3E0"),
        (0.0, 2.2, 2.2, 1.0, "Bifurcation\nDetector", "#FCE4EC"),
        (2.7, 2.2, 2.2, 1.0, "Spectral Path\n$\\tau(\\omega_i, \\omega_j)$", "#F3E5F5"),
        (5.4, 2.2, 2.0, 1.0, "Calendar Regime\n$\\theta_k, a_k$", "#E0F7FA"),
        (0.5, 0.3, 2.8, 1.0, "Cross-Timescale\nLifting Operators\n$\\Phi_{S{\\to}M}, \\Phi_{M{\\to}L}$", "#FFFDE7"),
        (4.2, 0.3, 2.8, 1.0, "Multi-Horizon Mixer\n$w_k = \\mathrm{softmax}(\\cdot)$", "#EFEBE9"),
    ]

    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                              linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7.5, zorder=3, linespacing=1.3)

    # Arrows
    arrows = [
        (1.1, 4.0, 1.1, 3.2),   # CircuitOpt → Bifurcation
        (3.8, 4.0, 3.8, 3.2),   # Observer → SpectralPath
        (6.4, 4.0, 6.4, 3.2),   # Koopman → Calendar
        (1.1, 2.2, 1.9, 1.3),   # Bifurcation → Lifting
        (3.8, 2.2, 3.0, 1.3),   # SpectralPath → Lifting
        (6.4, 2.2, 5.6, 1.3),   # Calendar → Mixer
        (3.3, 0.3, 4.2, 0.8),   # Lifting → Mixer
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="gray", linewidth=1.2))

    ax.set_title("Unified Tensor System: Module Architecture", fontsize=13, fontweight="bold", pad=10)

    fig.savefig(os.path.join(FIGDIR, "fig7_architecture.pdf"))
    plt.close(fig)
    print("  fig7_architecture.pdf")


# ==========================================================================
# Figure 8: Basin Stability Across Q Values
# ==========================================================================
def fig8_basin_vs_Q():
    from optimization.circuit_optimizer import (
        CircuitSpec, EigenvalueMapper, MonteCarloStabilityAnalyzer, OptimizationResult,
    )

    mapper = EigenvalueMapper()
    Q_values = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    lca_fracs = []
    spreads = []

    for Q in Q_values:
        spec = CircuitSpec(center_freq_hz=1000.0, Q_target=Q,
                           component_tolerances={"R": 0.10, "L": 0.10, "C": 0.10})
        inv = mapper.inverse_map(spec.target_eigenvalues)
        eigs = mapper.eigenvalues(inv["R"], inv["L"], inv["C"])
        sigma = float(np.real(eigs[0]))
        omega_d = abs(float(np.imag(eigs[0])))
        omega0 = float(np.sqrt(sigma**2 + omega_d**2))
        zeta = abs(sigma) / omega0 if omega0 > 0 else 0.0
        Qv = 1.0 / (2.0 * max(zeta, 1e-12))
        gap = abs(abs(eigs[0]) - abs(eigs[1]))
        result = OptimizationResult(
            R=inv["R"], L=inv["L"], C=inv["C"],
            achieved_eigenvalues=eigs, eigenvalue_error=0.0, cost=0.0,
            regime_type="lca", spectral_gap=float(gap),
            omega0_achieved=omega0, Q_achieved=Qv, converged=True,
        )
        analyzer = MonteCarloStabilityAnalyzer(n_samples=300, seed=42)
        basin = analyzer.analyze(result, spec)
        lca_fracs.append(basin.lca_fraction)
        spreads.append(basin.mean_eigenvalue_spread)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    ax1.plot(Q_values, lca_fracs, "bo-", markersize=5, linewidth=1.2)
    ax1.set_xscale("log")
    ax1.set_xlabel("Quality Factor $Q$")
    ax1.set_ylabel("LCA Fraction")
    ax1.set_title("Regime Stability vs. $Q$")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.9, color="g", linestyle="--", linewidth=0.7, alpha=0.5, label="90% threshold")
    ax1.legend(fontsize=8)

    ax2.plot(Q_values, spreads, "rs-", markersize=5, linewidth=1.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Quality Factor $Q$")
    ax2.set_ylabel("Mean Eigenvalue Spread")
    ax2.set_title("Spectral Spread vs. $Q$")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig8_basin_vs_Q.pdf"))
    plt.close(fig)
    print("  fig8_basin_vs_Q.pdf")


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    print("Generating whitepaper figures...")
    fig1_complex_plane()
    fig2_frequency_response()
    fig3_pareto_front()
    fig4_stability_basin()
    fig5_observer_energy()
    fig6_hdv_orthogonality()
    fig7_architecture()
    fig8_basin_vs_Q()
    print("Done.")
