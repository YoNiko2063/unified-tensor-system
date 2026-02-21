"""
phase4_self_optimize.py — Phase 4: Self-Optimization Loop (Controlled Experiment)

ChatGPT-recommended structure:
  Step 1  Rank Python functions by E_total = E_python + λ₂·E_borrow_est
  Step 2  Identify top-decile bottlenecks
  Step 3  Dynamic profile the already-ported function (Duffing RK4)
  Step 4  Recompute K_before (Python) and K_after (Rust)
  Step 5  Compare predicted ΔE_total vs observed ΔE_total
  Step 6  Nominate next 2 candidates for Rust porting

If predicted ΔE_total matches observed within ±0.20 log₁₀ decades:
  → The manifold IS prescriptive.
  → Proceed to Step 4 (Rust HTML parser) to further build evidence.

If it doesn't match:
  → The model needs recalibration.
  → Collect more data before claiming prescriptive power.

Calibrated constants (from Phase 3):
  λ₂ = 0.30  (E_borrow weight, Spearman ρ=0.833 cross-domain correlation)
  Δlog_ω₀ = log₁₀(speedup) per Rust substitution
  D_sep = 0.43  (BorrowVector compile boundary)

Usage:
    cd unified-tensor-system
    conda run -n tensor python optimization/phase4_self_optimize.py

Requires: numpy, scipy, sklearn  (tensor env)
          rust_physics_kernel     (built in Step 1 via setup.sh)
"""

from __future__ import annotations

import ast
import inspect
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from optimization.code_profiler import (
    ASTMathClassifier,
    DynamicProfiler,
    MathPattern,
    _COMPLEXITY_TABLE,
    _HW_OMEGA0_REF,
    _LOG_OMEGA0_SCALE,
    pattern_to_invariants,
)

# ── Phase 3 constants ─────────────────────────────────────────────────────────
LAMBDA2       = 0.30   # calibrated from Step 3 Spearman ρ=0.833
DSEP          = 0.43   # empirical BorrowVector constraint boundary
VERIFY_TOL    = 0.20   # |ΔE_predicted - ΔE_observed| < 0.20 → prescriptive

# E_borrow estimates per complexity class (canonical Rust idiom)
_EBORROW_BY_CLASS = {
    "reduction":    0.000,   # fn sum(x: &[f64]) → f64  (immutable, no mut)
    "element_wise": 0.060,   # fn scale(x: &mut [f64], a: f64)  (1 &mut)
    "sort":         0.080,   # fn sort(x: &mut [f64])  (&mut in-place)
    "fft":          0.145,   # complex in-place + scratch buffers
    "matvec":       0.130,   # fn matvec(a:&[f64], x:&[f64], y:&mut [f64])
    "matmul":       0.230,   # fn matmul with tile blocking  (high B1,B4)
    "stencil":      0.100,   # 2-pass stencil with &mut output
    "conv_direct":  0.150,   # kernel sliding window with &mut
    "unknown":      0.050,   # conservative low estimate
}

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FunctionRecord:
    filename: str
    fn_name:  str
    complexity_class: str
    alpha:    float   # complexity exponent (O(N^alpha))
    Q:        float   # arithmetic intensity
    log_w:    float   # log_omega0_norm (Python static)
    log_q:    float   # log_Q_norm
    zeta:     float
    e_python: float   # max(0, -log_w) — computational cost
    e_borrow: float   # estimated E_borrow if Rust-ported
    e_total:  float   # e_python + λ₂ * e_borrow
    confidence: float


# ── Step 1: Static function inventory ────────────────────────────────────────

def scan_optimization_dir(opt_dir: str) -> list[FunctionRecord]:
    """
    Walk optimization/*.py, extract all function defs, classify each.
    Skip test files, __init__.py, and functions < 4 lines.
    """
    classifier = ASTMathClassifier()
    records: list[FunctionRecord] = []

    skip_prefixes = ("test_", "__")
    skip_files    = {"__init__.py", "cross_domain_borrow_analysis.py",
                     "phase3_predictor.py", "phase4_self_optimize.py"}

    for fname in sorted(os.listdir(opt_dir)):
        if not fname.endswith(".py") or fname in skip_files:
            continue
        fpath = os.path.join(opt_dir, fname)
        try:
            src = open(fpath).read()
            tree = ast.parse(src)
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            fn_name = node.name
            if any(fn_name.startswith(p) for p in skip_prefixes):
                continue
            if len(node.body) < 2:
                continue

            # Extract source lines for this function
            try:
                fn_lines = ast.get_source_segment(src, node)
            except Exception:
                fn_lines = None
            if not fn_lines or len(fn_lines.splitlines()) < 3:
                continue

            pattern = classifier.classify(fn_lines)
            log_w, log_q, zeta = pattern_to_invariants(pattern)
            e_python = max(0.0, -log_w)
            e_borrow = _EBORROW_BY_CLASS.get(pattern.complexity_class, 0.05)
            e_total  = e_python + LAMBDA2 * e_borrow

            records.append(FunctionRecord(
                filename=fname,
                fn_name=fn_name,
                complexity_class=pattern.complexity_class,
                alpha=pattern.complexity_exponent,
                Q=pattern.arithmetic_intensity,
                log_w=log_w,
                log_q=log_q,
                zeta=zeta,
                e_python=e_python,
                e_borrow=e_borrow,
                e_total=e_total,
                confidence=pattern.confidence,
            ))

    return sorted(records, key=lambda r: r.e_total, reverse=True)


# ── Step 2: Dynamic profiling — Python Duffing RK4 ───────────────────────────

def duffing_rhs_py(state, t, alpha, beta, delta, f_drive, omega):
    x, v = state
    return [v, -delta*v - alpha*x - beta*x**3 + f_drive*math.cos(omega*t)]

def rk4_step_py(state, t, dt, alpha, beta, delta, f_drive, omega):
    def rhs(s): return duffing_rhs_py(s, t, alpha, beta, delta, f_drive, omega)
    def rhs2(s, t2): return duffing_rhs_py(s, t2, alpha, beta, delta, f_drive, omega)
    k1 = rhs(state)
    k2 = rhs2([state[0]+k1[0]*dt/2, state[1]+k1[1]*dt/2], t+dt/2)
    k3 = rhs2([state[0]+k2[0]*dt/2, state[1]+k2[1]*dt/2], t+dt/2)
    k4 = rhs2([state[0]+k3[0]*dt,   state[1]+k3[1]*dt],   t+dt)
    return [state[0]+dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
            state[1]+dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])]

def gen_traj_py(n_steps, x0=1.0, v0=0.0, dt=1e-3,
                alpha=1.0, beta=0.1, delta=0.3, f_drive=0.5, omega=1.2):
    state = [x0, v0]
    t = 0.0
    traj = []
    for _ in range(n_steps):
        traj.append(state[:])
        state = rk4_step_py(state, t, dt, alpha, beta, delta, f_drive, omega)
        t += dt
    return traj


def time_fn(fn, n_reps=3):
    """Wall-clock time, best of n_reps."""
    best = float("inf")
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def profile_duffing(sizes=(100, 500, 1000, 5000, 10_000)):
    """
    Time Python and (optionally) Rust Duffing RK4 at multiple n_steps.
    Returns (py_rows, rust_rows) where each row is (n_steps, time_sec).
    """
    py_rows = []
    for n in sizes:
        t = time_fn(lambda: gen_traj_py(n))
        py_rows.append((n, t))

    # Try Rust kernel
    rust_rows = []
    try:
        import rust_physics_kernel as rk
        for n in sizes:
            t = time_fn(lambda: rk.py_generate_trajectory(
                1.0, 0.0, 1e-3, n, 1.0, 0.1, 0.3, 0.5, 1.2))
            rust_rows.append((n, t))
    except ImportError:
        pass

    return py_rows, rust_rows


def empirical_invariants_from_timing(rows, n_ref=1_000):
    """
    Fit log-linear scaling to timing data and compute (log_w, log_q, zeta).
    rows: list of (n_steps, time_sec)
    For Duffing RK4: flops ≈ 40·n_steps, mem ≈ 32·n_steps bytes
    """
    ns   = np.array([r[0] for r in rows], dtype=float)
    ts   = np.array([r[1] for r in rows], dtype=float)

    # Fit log(t) = α·log(n) + β
    valid = ts > 0
    log_n = np.log(ns[valid])
    log_t = np.log(ts[valid])
    A = np.stack([log_n, np.ones_like(log_n)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, log_t, rcond=None)
    alpha_fit, beta_fit = float(coeffs[0]), float(coeffs[1])

    # Interpolate time at n_ref
    t_ref = math.exp(alpha_fit * math.log(n_ref) + beta_fit)
    if t_ref <= 0:
        t_ref = 1e-6

    # Duffing flop/mem estimates at n_ref
    flops_ref = 40.0 * n_ref   # 4 stages × 10 ops/stage
    mem_ref   = 32.0 * n_ref   # 2 floats × 8 bytes × 2 (read+write)

    omega0 = flops_ref / t_ref
    Q      = flops_ref / max(mem_ref, 1.0)
    log_w  = float(np.clip((math.log(max(omega0, 1e-30)) - _HW_OMEGA0_REF)
                            / _LOG_OMEGA0_SCALE, -3.0, 3.0))
    log_q  = float(np.clip(math.log(max(Q, 1e-30)) / _LOG_OMEGA0_SCALE, -3.0, 3.0))
    zeta   = 1.0 / (1.0 + Q)
    return log_w, log_q, float(np.clip(zeta, 0.0, 1.0)), alpha_fit


# ── Step 3 / 5: Prediction and verification ───────────────────────────────────

def predict_delta_e_total(log_w_before, e_borrow_before, log_w_after, e_borrow_after):
    """
    ΔE_total = (E_python_after + λ₂·E_borrow_after) - (E_python_before + λ₂·E_borrow_before)
    where E_python = max(0, -log_w)
    """
    e_py_before = max(0.0, -log_w_before)
    e_py_after  = max(0.0, -log_w_after)
    return ((e_py_after  + LAMBDA2 * e_borrow_after) -
            (e_py_before + LAMBDA2 * e_borrow_before))


# ── Phase 3 predictor (re-trained inline) ────────────────────────────────────

def load_phase3_model():
    """Re-train the Phase 3 LogReg on training data (fast, <0.1s)."""
    import json
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_path = "/home/nyoo/projects/rust-constraint-manifold/metrics.jsonl"
    if not os.path.exists(train_path):
        return None, None

    samples = [json.loads(l) for l in open(train_path) if l.strip()]
    X = np.array([[s["b1"],s["b2"],s["b3"],s["b4"],s["b5"],s["e_borrow"]]
                  for s in samples])
    y = np.array([1 if s["compile_success"] else 0 for s in samples])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(Xs, y)
    return model, scaler


def predict_compile_success(bv_tuple, model, scaler):
    """P(compile_success) for a BorrowVector tuple (b1..b5)."""
    if model is None:
        return float("nan")
    b1, b2, b3, b4, b5 = bv_tuple
    e = 0.25*b1 + 0.20*b2 + 0.15*b3 + 0.20*b4 + 0.20*b5
    X = np.array([[b1, b2, b3, b4, b5, e]])
    Xs = scaler.transform(X)
    return float(model.predict_proba(Xs)[0, 1])


# ── BorrowVector for known functions ─────────────────────────────────────────

# Known BorrowVectors from Step 1 (Duffing RK4 actual) and Step 3 estimates
_KNOWN_BV = {
    "duffing_rk4":  (0.10, 0.00, 0.00, 0.00, 0.00),   # actual: E_borrow=0.025
    "reduction":    (0.00, 0.00, 0.00, 0.00, 0.00),
    "element_wise": (0.00, 0.00, 0.00, 0.30, 0.00),
    "fft":          (0.20, 0.10, 0.10, 0.30, 0.00),
    "matvec":       (0.20, 0.00, 0.00, 0.40, 0.00),
    "matmul":       (0.30, 0.20, 0.10, 0.50, 0.00),
    "stencil":      (0.10, 0.00, 0.00, 0.25, 0.00),
    "conv_direct":  (0.20, 0.10, 0.05, 0.35, 0.00),
    "unknown":      (0.10, 0.00, 0.00, 0.10, 0.00),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    opt_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 72)
    print("Phase 4: Self-Optimization Loop (Controlled Experiment)")
    print("=" * 72)
    print(f"\nCalibration:  λ₂={LAMBDA2}  D_sep={DSEP}  verify_tol=±{VERIFY_TOL}")
    print(f"Ported so far:  duffing_rk4  (rust-physics-kernel, Step 1)")

    # ── Load Phase 3 model ────────────────────────────────────────────────
    p3_model, p3_scaler = load_phase3_model()
    if p3_model is None:
        print("WARNING: Phase 3 training data not found; P(compile_success) will be nan")

    # ── Step 1: Static inventory ──────────────────────────────────────────
    print("\n" + "─" * 72)
    print("Step 1 — Static Function Inventory (optimization/*.py)")
    print("─" * 72)

    records = scan_optimization_dir(opt_dir)
    print(f"  Scanned {len(records)} functions across optimization/\n")

    # Header
    print(f"  {'Rank':>4}  {'File':<32}  {'Function':<28}  "
          f"{'Class':<12}  {'E_py':>6}  {'E_bw':>6}  {'E_tot':>6}")
    print(f"  {'----':>4}  {'-'*32}  {'-'*28}  {'-'*12}  "
          f"{'------':>6}  {'------':>6}  {'------':>6}")

    n_show = min(len(records), 20)
    for i, r in enumerate(records[:n_show]):
        port_flag = " ★" if r.fn_name in ("_run_duffing_rk4", "generate_trajectory",
                                           "rk4_step", "_trim_trajectory") else ""
        print(f"  {i+1:>4}  {r.filename:<32}  {r.fn_name:<28}  "
              f"{r.complexity_class:<12}  {r.e_python:>6.3f}  "
              f"{r.e_borrow:>6.3f}  {r.e_total:>6.3f}{port_flag}")

    if len(records) > n_show:
        print(f"  ... ({len(records) - n_show} more functions below threshold)")

    # Top decile
    n_decile = max(1, len(records) // 10)
    top_decile = records[:n_decile]
    print(f"\n  Top decile ({n_decile} functions, E_total ≥ {top_decile[-1].e_total:.3f}):")
    for r in top_decile:
        print(f"    {r.filename}:{r.fn_name}  "
              f"[{r.complexity_class}]  E_total={r.e_total:.3f}")

    # ── Step 2: Identify bottlenecks (top decile, unported) ──────────────
    already_ported = {"duffing_rk4", "_run_duffing_rk4", "rk4_step",
                      "generate_trajectory", "_trim_trajectory"}
    candidates = [r for r in top_decile if r.fn_name not in already_ported]

    print(f"\n  Unported top-decile candidates: {len(candidates)}")
    for r in candidates[:5]:
        print(f"    → {r.filename}:{r.fn_name}  [{r.complexity_class}]  "
              f"E_total={r.e_total:.3f}")

    # ── Step 3: Dynamic profiling — Python and Rust Duffing RK4 ──────────
    print("\n" + "─" * 72)
    print("Step 3 — Dynamic Profiling: Python vs Rust Duffing RK4")
    print("─" * 72)

    SIZES = (100, 500, 1000, 5000, 10_000)
    print(f"  Timing at n_steps ∈ {SIZES} …", flush=True)

    py_rows, rust_rows = profile_duffing(SIZES)

    has_rust = len(rust_rows) > 0

    print(f"\n  {'n_steps':>8}  {'Python (ms)':>12}  "
          + (f"{'Rust (ms)':>12}  {'Speedup':>8}" if has_rust else ""))
    print(f"  {'-------':>8}  {'----------':>12}  "
          + (f"{'--------':>12}  {'-------':>8}" if has_rust else ""))

    speedups = []
    for i, (n, t_py) in enumerate(py_rows):
        row = f"  {n:>8}  {t_py*1e3:>12.2f}"
        if has_rust and i < len(rust_rows):
            t_rs = rust_rows[i][1]
            sp = t_py / t_rs
            speedups.append(sp)
            row += f"  {t_rs*1e3:>12.2f}  {sp:>8.1f}×"
        print(row)

    # ── Step 4: Compute K_before and K_after ─────────────────────────────
    print("\n" + "─" * 72)
    print("Step 4 — Recompute: K_before (Python) and K_after (Rust)")
    print("─" * 72)

    log_w_py, log_q_py, zeta_py, alpha_py = empirical_invariants_from_timing(py_rows)
    e_python = max(0.0, -log_w_py)
    e_borrow_python = 0.0   # Python has no borrow complexity
    e_total_python  = e_python + LAMBDA2 * e_borrow_python

    print(f"\n  K_before (Python Duffing RK4, empirical):")
    print(f"    log_ω₀_norm = {log_w_py:+.3f}   log_Q_norm = {log_q_py:+.3f}"
          f"   ζ = {zeta_py:.3f}   α_fit = {alpha_py:.3f}")
    print(f"    E_python    = {e_python:.4f}   E_borrow = {e_borrow_python:.4f}"
          f"   E_total = {e_total_python:.4f}")

    if has_rust:
        log_w_rs, log_q_rs, zeta_rs, alpha_rs = empirical_invariants_from_timing(rust_rows)
        e_borrow_rust = _EBORROW_BY_CLASS["element_wise"]   # Duffing RK4 ≈ 0.025 actual
        # Use actual measured BorrowVector: E_borrow=0.025 (from lib.rs annotation)
        e_borrow_rust_actual = 0.025
        e_python_rust = max(0.0, -log_w_rs)
        e_total_rust  = e_python_rust + LAMBDA2 * e_borrow_rust_actual

        print(f"\n  K_after  (Rust  Duffing RK4, empirical):")
        print(f"    log_ω₀_norm = {log_w_rs:+.3f}   log_Q_norm = {log_q_rs:+.3f}"
              f"   ζ = {zeta_rs:.3f}   α_fit = {alpha_rs:.3f}")
        print(f"    E_python    = {e_python_rust:.4f}   E_borrow = {e_borrow_rust_actual:.4f}"
              f"   E_total = {e_total_rust:.4f}")

        measured_speedup = np.median(speedups) if speedups else float("nan")
        delta_logw_obs = log_w_rs - log_w_py
        delta_e_total_obs = e_total_rust - e_total_python

        print(f"\n  Observed:")
        print(f"    Speedup (median):         {measured_speedup:.1f}×")
        print(f"    Δlog_ω₀_norm (observed):  {delta_logw_obs:+.3f}")
        print(f"    ΔE_total (observed):       {delta_e_total_obs:+.4f}")

    # ── Step 5: Prediction vs Observation ────────────────────────────────
    print("\n" + "─" * 72)
    print("Step 5 — Compare Predicted ΔE_total vs Observed")
    print("─" * 72)

    # Predicted from Phase 3 calibration:
    # Rust speedup from Step 1 benchmark: 9.7×  → Δlog_ω₀ = log10(9.7) = +0.987
    # E_borrow_rust = 0.025 (actual from lib.rs)
    # E_borrow_python = 0 (Python has no borrow complexity)
    predicted_speedup = 9.7   # from Step 1 benchmark.py
    delta_logw_pred   = math.log10(predicted_speedup)
    log_w_py_pred     = log_w_py            # start from measured Python baseline
    log_w_rs_pred     = min(3.0, log_w_py_pred + delta_logw_pred)
    e_python_pred     = max(0.0, -log_w_py_pred)
    e_python_rust_pred= max(0.0, -log_w_rs_pred)
    bv_rust           = _KNOWN_BV["duffing_rk4"]
    e_borrow_rust_pred = 0.25*bv_rust[0] + 0.20*bv_rust[1] + 0.15*bv_rust[2] \
                       + 0.20*bv_rust[3] + 0.20*bv_rust[4]   # = 0.025

    delta_e_total_pred = ((e_python_rust_pred + LAMBDA2 * e_borrow_rust_pred) -
                          (e_python_pred       + LAMBDA2 * e_borrow_python))

    # P(compile_success) for Duffing RK4 BorrowVector
    p_compile = predict_compile_success(bv_rust, p3_model, p3_scaler)

    print(f"\n  Prediction (Phase 3 calibration + Step 1 benchmark):")
    print(f"    Predicted speedup:              {predicted_speedup:.1f}×")
    print(f"    Predicted Δlog_ω₀:              {delta_logw_pred:+.3f}")
    print(f"    E_borrow_rust (BorrowVector):   {e_borrow_rust_pred:.4f}")
    print(f"    ΔE_total (predicted):           {delta_e_total_pred:+.4f}")
    print(f"    P(compile_success) [Phase 3]:   {p_compile:.4f}")
    print(f"    Safety margin to D_sep:         "
          f"{DSEP - e_borrow_rust_pred:.3f} "
          f"(E_borrow={e_borrow_rust_pred:.3f} << D_sep={DSEP})")

    if has_rust:
        error = abs(delta_e_total_pred - delta_e_total_obs)
        prescriptive = error < VERIFY_TOL
        print(f"\n  Verification:")
        print(f"    ΔE_total predicted:  {delta_e_total_pred:+.4f}")
        print(f"    ΔE_total observed:   {delta_e_total_obs:+.4f}")
        print(f"    |error|:             {error:.4f}   "
              f"(threshold = {VERIFY_TOL})  "
              f"→  {'PRESCRIPTIVE ✓' if prescriptive else 'NOT YET PRESCRIPTIVE ✗'}")
    else:
        prescriptive = None
        print(f"\n  rust_physics_kernel not importable — install via:")
        print(f"    cd /home/nyoo/projects/rust-physics-kernel && ./setup.sh")
        print(f"    source ~/.cargo/env && maturin develop --release")

    # ── Step 6: Nominate next 2 candidates ───────────────────────────────
    print("\n" + "─" * 72)
    print("Step 6 — Nominate Next 2 Rust Porting Candidates")
    print("─" * 72)

    nominees = candidates[:2] if len(candidates) >= 2 else candidates

    for rank, r in enumerate(nominees, 1):
        bv = _KNOWN_BV.get(r.complexity_class, _KNOWN_BV["unknown"])
        e_bw = sum(w*b for w,b in zip((0.25,0.20,0.15,0.20,0.20), bv))
        e_py_cand = r.e_python
        e_tot_after = max(0.0, e_py_cand - delta_logw_pred) + LAMBDA2 * e_bw
        delta_e = e_tot_after - r.e_total
        p_comp = predict_compile_success(bv, p3_model, p3_scaler)
        safety = DSEP - e_bw

        print(f"\n  Nominee {rank}: {r.filename}:{r.fn_name}")
        print(f"    Complexity class:       {r.complexity_class}  (α={r.alpha:.2f}, Q={r.Q:.1f})")
        print(f"    Current E_total:        {r.e_total:.4f}  (E_python={r.e_python:.4f})")
        print(f"    Predicted BorrowVector: {bv}")
        print(f"    Predicted E_borrow:     {e_bw:.4f}")
        print(f"    Predicted ΔE_total:     {delta_e:+.4f}  (if Rust-ported at ~9.7× speedup)")
        print(f"    P(compile_success):     {p_comp:.4f}")
        print(f"    Safety margin to D_sep: {safety:.3f}  "
              f"({'✓ safe' if safety > 0.05 else '⚠ near boundary'})")

    # ── Final verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 4 Verdict")
    print("=" * 72)

    if prescriptive is True:
        print("""
  ✓ MANIFOLD IS PRESCRIPTIVE

  Predicted ΔE_total matched observed within ±{:.2f} log₁₀ decades.

  Evidence chain:
    Step 1:  Rust kernel built, 9.7× speedup measured
    Step 2:  BorrowVector constraint manifold measured (250 samples)
    Step 3:  Cross-domain ΔK=+0.987, ρ=0.833 confirmed
    Phase 3: LogReg CV AUC=0.918, hold-out AUC=0.951, no overfit
    Phase 4: Predicted ΔE_total ≈ Observed ΔE_total  ← THIS IS THE KEY

  The invariant manifold correctly predicted the improvement from
  Python → Rust substitution of the Duffing RK4 integrator.

  Next: Proceed with Step 4 (Rust HTML parser) to add a second
        porting data point and further validate prescriptive power.

  Longer term: Gather >5 ported function pairs, then re-fit λ₂
               empirically rather than using calibrated 0.30.
""".format(VERIFY_TOL))

    elif prescriptive is False:
        print(f"""
  ✗ NOT YET PRESCRIPTIVE (|error| = {error:.4f} > {VERIFY_TOL})

  The predicted and observed ΔE_total disagree beyond tolerance.

  Likely causes:
    A) λ₂={LAMBDA2} needs recalibration (more function pairs needed)
    B) Duffing RK4 speedup at small N differs from large N
       (pyo3 serialization overhead dominates at small n_steps)
    C) The E_python reference point differs between static and dynamic

  Recommended: Profile at n_steps ≥ 10_000 only (avoids pyo3 overhead)
               then recompute ΔE_total.
""")

    else:
        print("""
  [PENDING] — Rust kernel not available for comparison.
  Run ./setup.sh in rust-physics-kernel then re-run this script.
""")

    print("=" * 72)

    # Return exit code
    if prescriptive is True:
        return 0
    elif prescriptive is False:
        return 2    # "not yet" — not a fatal failure
    return 1        # rust not available


if __name__ == "__main__":
    sys.exit(main())
