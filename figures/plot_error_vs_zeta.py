"""
Figure 1 — IEEE 39-Bus CCT Error vs Damping Ratio.

Generates:  figures/error_vs_zeta.pdf   (print quality, for brief)
            figures/error_vs_zeta.png   (screen quality, for deck)

Usage:
    conda run -n tensor python figures/plot_error_vs_zeta.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from optimization.ieee39_benchmark import (
    compute_corrected_errors,
    fit_damping_correction,
    run_damping_sweep,
)

# ── Data ─────────────────────────────────────────────────────────────────────

ZETA_VALUES = (0.01, 0.03, 0.05, 0.10, 0.20)

print("Running damping sweep …", flush=True)
sweep     = run_damping_sweep(zeta_values=ZETA_VALUES)
a         = fit_damping_correction(sweep)
corrected = compute_corrected_errors(sweep, a)
print(f"  a = {a:.4f}")

zetas = sorted(set(r.zeta for r in sweep))

raw_mean, raw_lo, raw_hi     = [], [], []
corr_mean, corr_lo, corr_hi  = [], [], []

for z in zetas:
    re = [abs(r.cct_error_pct)  for r in sweep     if r.zeta == z]
    ce = [abs(c["corr_err_pct"]) for c in corrected if c["zeta"] == z]

    raw_mean.append(np.mean(re));  raw_lo.append(np.min(re));  raw_hi.append(np.max(re))
    corr_mean.append(np.mean(ce)); corr_lo.append(np.min(ce)); corr_hi.append(np.max(ce))

z  = np.array(zetas)
rm = np.array(raw_mean);  rl = np.array(raw_lo);  rh = np.array(raw_hi)
cm = np.array(corr_mean); cl = np.array(corr_lo); ch = np.array(corr_hi)

# ── Style ─────────────────────────────────────────────────────────────────────

RED  = "#c0392b"
BLUE = "#2980b9"
GREY = "#555555"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.35,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
})

# ── Figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7.5, 4.8))

# ── Valid envelope shade (ζ ≤ 0.20) ──────────────────────────────────────────
ax.axvspan(0.0, 0.205, color=BLUE, alpha=0.05, zorder=0)

# ── 5% gate ───────────────────────────────────────────────────────────────────
ax.axhline(5.0, color=GREY, linewidth=1.0, linestyle="--", zorder=1,
           label="5% accuracy gate")

# ── Raw EAC band + line ───────────────────────────────────────────────────────
ax.fill_between(z, rl, rh, alpha=0.18, color=RED, zorder=2)
ax.plot(z, rm, "o-", color=RED, linewidth=2.0, markersize=6, zorder=3,
        label="EAC  (D = 0 formula)")

# ── Corrected EAC band + line ─────────────────────────────────────────────────
ax.fill_between(z, cl, ch, alpha=0.20, color=BLUE, zorder=4)
ax.plot(z, cm, "s-", color=BLUE, linewidth=2.0, markersize=6, zorder=5,
        label=f"EAC + correction  (a = {a:.2f})")

# ── ζ* markers ────────────────────────────────────────────────────────────────
ax.axvline(0.01, color=RED,  linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
ax.axvline(0.20, color=BLUE, linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)

# ── Annotations ───────────────────────────────────────────────────────────────
ax.annotate(
    "ζ* = 0.01\n(raw EAC)",
    xy=(0.01, 5.0), xytext=(0.035, 18.5),
    color=RED, fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color=RED, lw=0.9),
)
ax.annotate(
    "ζ* = 0.20  (corrected)\nmax error 2.73%  (Q ≥ 2)",
    xy=(0.20, 2.73), xytext=(0.105, 9.5),
    color=BLUE, fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9),
)

# Shaded band label
ax.text(
    0.196, 33.5,
    "generator\nspread",
    ha="right", va="top", fontsize=7.5, color=GREY, style="italic",
)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.004, 0.213)
ax.set_ylim(-0.8, 35.5)
ax.set_xticks(list(zetas))
ax.set_xticklabels([f"{v:.2f}" for v in zetas])
ax.set_xlabel("Damping ratio  ζ = D / (2 M ω₀)", fontsize=11)
ax.set_ylabel("|CCT error|  [%]", fontsize=11)
ax.set_title(
    "IEEE 39-Bus New England System — CCT Error vs Damping Ratio\n"
    "10 generators · shaded band = min/max across machines",
    fontsize=10.5, pad=10,
)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, axis="y")

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(out_dir, f"error_vs_zeta.{ext}")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved → {path}")

plt.close()
print("Done.")
