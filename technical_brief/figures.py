"""
Generate figures for technical_brief.tex.

Produces:
  figures/error_vs_zeta.pdf  — CCT error vs damping ratio (raw EAC vs corrected)

Self-contained: no optimization/ imports required.
"""

from __future__ import annotations
import math, os, sys
from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

OMEGA_S = 2.0 * math.pi * 60.0
A_CORR  = 1.51   # universal damping correction

@dataclass
class Generator:
    name: str; H: float; P_m: float; P_e: float

GENERATORS = [
    Generator("G1",  500.0,  2.50,  5.00),
    Generator("G2",   30.3,  5.73, 11.46),
    Generator("G3",   35.8,  6.50, 13.00),
    Generator("G4",   28.6,  6.32, 12.64),
    Generator("G5",   26.0,  5.08, 10.16),
    Generator("G6",   34.8,  6.50, 13.00),
    Generator("G7",   26.4,  5.60, 11.20),
    Generator("G8",   24.3,  5.40, 10.80),
    Generator("G9",   34.5,  8.30, 16.60),
    Generator("G10",  42.0, 10.04, 20.08),
]

def eac_cct(gen):
    M = 2.0 * gen.H / OMEGA_S
    ds = math.asin(gen.P_m / gen.P_e)
    cdc = gen.P_m * (math.pi - 2*ds) / gen.P_e - math.cos(ds)
    dc = math.acos(max(-1.0, min(1.0, cdc)))
    return math.sqrt(2.0 * M * (dc - ds) / gen.P_m)

def _rk4(f, t, y, dt):
    k1 = f(t, y); k2 = f(t+dt/2, y+dt/2*k1)
    k3 = f(t+dt/2, y+dt/2*k2); k4 = f(t+dt, y+dt*k3)
    return y + dt/6*(k1+2*k2+2*k3+k4)

def is_stable(gen, D, t_fault, dt=0.01, n=3000):
    M = 2.0*gen.H/OMEGA_S
    ds = math.asin(gen.P_m/gen.P_e); du = math.pi-ds
    ff = lambda t,y: np.array([y[1], (gen.P_m - D*y[1])/M])
    fp = lambda t,y: np.array([y[1], (gen.P_m - gen.P_e*math.sin(y[0]) - D*y[1])/M])
    y, t = np.array([ds, 0.0]), 0.0
    for _ in range(max(1, round(t_fault/dt))): y=_rk4(ff,t,y,dt); t+=dt
    for _ in range(n):
        if y[0]>=du: return False
        y=_rk4(fp,t,y,dt); t+=dt
    return True

def ref_cct(gen, D, tol=1e-3, dt=0.01):
    lo, hi = 0.0, 5.0
    while hi-lo > tol:
        mid=(lo+hi)/2
        if is_stable(gen,D,mid,dt): lo=mid
        else: hi=mid
    return (lo+hi)/2

# ── Run damping sweep ─────────────────────────────────────────────────────────

ZETAS = [0.01, 0.03, 0.05, 0.10, 0.20]

print("Generating error_vs_zeta figure (running damping sweep ~60s)...")

raw_mean, raw_min, raw_max   = [], [], []
cor_mean, cor_min, cor_max   = [], [], []

for zeta in ZETAS:
    raw_errs, cor_errs = [], []
    for gen in GENERATORS:
        M  = 2.0*gen.H/OMEGA_S
        ds = math.asin(gen.P_m/gen.P_e)
        omega0 = math.sqrt(gen.P_e*math.cos(ds)/M)
        D = 2.0*M*omega0*zeta

        cct_e = eac_cct(gen)
        cct_r = ref_cct(gen, D)

        raw_err = (cct_e - cct_r)/cct_r*100
        cor_cct = cct_e/(1.0 - A_CORR*zeta)
        cor_err = (cor_cct - cct_r)/cct_r*100

        raw_errs.append(abs(raw_err))
        cor_errs.append(abs(cor_err))
        print(f"  ζ={zeta:.2f}  {gen.name}: raw={raw_err:+.2f}%  corr={cor_err:+.2f}%")

    raw_mean.append(np.mean(raw_errs))
    raw_min.append(np.min(raw_errs))
    raw_max.append(np.max(raw_errs))
    cor_mean.append(np.mean(cor_errs))
    cor_min.append(np.min(cor_errs))
    cor_max.append(np.max(cor_errs))

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 3.8))

raw_mean = np.array(raw_mean); raw_min = np.array(raw_min); raw_max = np.array(raw_max)
cor_mean = np.array(cor_mean); cor_min = np.array(cor_min); cor_max = np.array(cor_max)

ax.plot(ZETAS, raw_mean, 'o-', color='#c0392b', lw=1.8, ms=5, label='Raw EAC ($D=0$ formula)')
ax.fill_between(ZETAS, raw_min, raw_max, color='#c0392b', alpha=0.15)

ax.plot(ZETAS, cor_mean, 's-', color='#1a5276', lw=1.8, ms=5, label=r'Corrected EAC ($a=1.51$)')
ax.fill_between(ZETAS, cor_min, cor_max, color='#1a5276', alpha=0.15)

ax.axhline(5.0, color='gray', lw=1, ls='--', label='5% accuracy gate')
ax.axhspan(0, cor_max[-1]+0.3, xmin=0, xmax=1, color='#1a5276', alpha=0.04)

ax.set_xlabel(r'Damping ratio $\zeta$', fontsize=11)
ax.set_ylabel('|CCT error| [%]', fontsize=11)
ax.set_title(r'CCT Error vs Damping — Raw EAC vs Corrected ($a=1.51$)', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(0, 0.22); ax.set_ylim(0, max(raw_max)*1.15)
ax.set_xticks(ZETAS)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'error_vs_zeta.pdf'), bbox_inches='tight')
print(f"Saved: figures/error_vs_zeta.pdf")
