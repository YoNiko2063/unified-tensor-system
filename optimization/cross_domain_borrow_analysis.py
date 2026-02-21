"""
cross_domain_borrow_analysis.py — Step 3: Cross-Domain Comparison
Location: unified-tensor-system/optimization/ (analysis only, no new modules)

Maps the same behavioral signatures across Python and Rust:
  K_python = (log_ω₀_norm, log_Q_norm, ζ) from code_profiler static analysis
  K_rust   = K_python + ΔK  (Δlog_ω₀_norm from measured Rust speedup)
  E_borrow = BorrowVector energy for each Rust implementation (design-level estimate)
  E_python = log-computational-intensity proxy (−log_ω₀_norm_python)

Goal: measure Correlation(E_borrow, E_python) across 6 function pairs,
and compare D_sep(BorrowVector)=0.43 to the Duffing separatrix analogy.

Validation gate (Step 2 → Step 3):
  Correlation(E_borrow, E_python | same behavior) measured (any value is informative).
  Gate: 5+ function pairs measured.

Usage:
    cd unified-tensor-system
    conda run -n tensor python optimization/cross_domain_borrow_analysis.py
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
from scipy.stats import spearmanr

# ── Project path ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from optimization.code_profiler import pattern_to_invariants, MathPattern, _COMPLEXITY_TABLE

# ── Constants (from koopman_signature.py) ────────────────────────────────────
_LOG_OMEGA0_SCALE = math.log(10.0)
_HW_OMEGA0_REF = math.log(1e6)          # log(1 MFLOPS/sec reference)

# ── Measured Rust speedup from benchmark.py (Step 1 result) ──────────────────
# Python Duffing RK4 at n_steps=10000: 10.24 ms
# Rust   Duffing RK4 at n_steps=10000:  1.06 ms
# Speedup: 10.24 / 1.06 ≈ 9.7×
RUST_SPEEDUP_MEASURED = 9.7  # at n_steps=10_000, from benchmark.py

# Δlog_ω₀_norm from Python → Rust (one log₁₀ decade per 10× speedup)
DELTA_LOG_OMEGA0 = math.log10(RUST_SPEEDUP_MEASURED)   # ≈ +0.987

# ── Function pair definitions ─────────────────────────────────────────────────
#
# Each entry:
#   name        : human-readable label
#   complexity  : key into _COMPLEXITY_TABLE (determines K_python)
#   description : "Rust implementation": canonical Rust idiom for this function
#   borrow_vec  : (B1, B2, B3, B4, B5) — design-level BorrowVector estimate
#                 based on the canonical Rust implementation pattern
#
# BorrowVector interpretation:
#   B1 = cross_module reference density (fn calls with & across mod boundaries)
#   B2 = lifetime annotation density (explicit 'a on structs/fns)
#   B3 = clone density (copying vs borrowing — protective, increases E_borrow less)
#   B4 = mutable reference density (&mut params, in-place mutation)
#   B5 = interior mutability density (RefCell, Mutex)
#
# E_borrow = 0.25·B1 + 0.20·B2 + 0.15·B3 + 0.20·B4 + 0.20·B5

FUNCTION_PAIRS = [
    {
        "name": "reduction",
        "complexity": "reduction",
        "description": "fn sum(x: &[f64]) -> f64  [immutable slice, fold]",
        "borrow_vec": (0.0, 0.0, 0.0, 0.0, 0.0),
        # No &mut, no cross-module refs, no lifetimes needed — purely functional
    },
    {
        "name": "duffing_rk4",
        "complexity": "element_wise",    # O(n_steps), Q≈2 arithmetic per step
        "description": "fn rk4_step(state: &[f64;2], …) -> [f64;2]  [Copy types, stack-only]",
        "borrow_vec": (0.1, 0.0, 0.0, 0.0, 0.0),
        # Measured actual BorrowVector from src/lib.rs annotation: E_borrow=0.025
        # Only B1=0.1 from params reference across call boundary
    },
    {
        "name": "element_wise",
        "complexity": "element_wise",
        "description": "fn scale(x: &mut [f64], a: f64)  [1 &mut slice param]",
        "borrow_vec": (0.0, 0.0, 0.0, 0.3, 0.0),
        # B4=0.3: one &mut param (in-place mutation of output slice)
    },
    {
        "name": "fft",
        "complexity": "fft",
        "description": "fn fft(buf: &mut [Complex64])  [in-place + temp scratch]",
        "borrow_vec": (0.2, 0.1, 0.1, 0.3, 0.0),
        # B1=0.2: references across butterfly stages
        # B2=0.1: some explicit lifetimes for scratch buffer
        # B3=0.1: occasional clone of twiddle factors
        # B4=0.3: &mut buffer for in-place operation
    },
    {
        "name": "matvec",
        "complexity": "matvec",
        "description": "fn matvec(a: &[f64], x: &[f64], y: &mut [f64])  [2 in + 1 out]",
        "borrow_vec": (0.2, 0.0, 0.0, 0.4, 0.0),
        # B1=0.2: 2 input slice refs + 1 output across module boundary
        # B4=0.4: &mut output slice (exclusive write access)
    },
    {
        "name": "matmul",
        "complexity": "matmul",
        "description": "fn matmul(a: &[f64], b: &[f64], c: &mut [f64], n: usize)  [blocking+tiles]",
        "borrow_vec": (0.3, 0.2, 0.1, 0.5, 0.0),
        # B1=0.3: tile references across module boundaries (blocking algorithm)
        # B2=0.2: explicit lifetimes for tile sub-slices
        # B3=0.1: occasional clone for boundary tile padding
        # B4=0.5: &mut output matrix (complex access pattern)
    },
]

E_BORROW_WEIGHTS = (0.25, 0.20, 0.15, 0.20, 0.20)  # B1..B5 weights


def compute_e_borrow(borrow_vec) -> float:
    return sum(w * b for w, b in zip(E_BORROW_WEIGHTS, borrow_vec))


def compute_k_python(complexity_key: str) -> tuple[float, float, float]:
    """Get (log_ω₀_norm, log_Q_norm, ζ) for a Python function via static analysis."""
    alpha, Q = _COMPLEXITY_TABLE[complexity_key]
    pattern = MathPattern(
        complexity_class=complexity_key,
        complexity_exponent=alpha,
        arithmetic_intensity=Q,
        dominant_op=complexity_key,
        confidence=1.0,
    )
    return pattern_to_invariants(pattern)


def compute_k_rust(k_python: tuple, speedup: float) -> tuple[float, float, float]:
    """
    Estimate K_rust from K_python by applying measured speedup.

    Rust accelerates compute throughput (ω₀ increases) but preserves:
    - arithmetic intensity Q (same algorithm, same FLOP/mem ratio)
    - damping ratio ζ (same memory access pattern)

    ΔK = (Δlog_ω₀_norm, 0, 0) where Δlog_ω₀_norm = log₁₀(speedup)
    """
    log_w, log_q, zeta = k_python
    delta_log_w = math.log10(speedup)
    log_w_rust = float(np.clip(log_w + delta_log_w, -3.0, 3.0))
    return (log_w_rust, log_q, zeta)


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("Step 3: Cross-Domain Comparison — BorrowVector vs Duffing Separatrix")
    print("=" * 70)
    print(f"\nMeasured Rust speedup (from Step 1 benchmark): {RUST_SPEEDUP_MEASURED}×")
    print(f"Δlog_ω₀_norm (Python → Rust):                 +{DELTA_LOG_OMEGA0:.3f}")
    print()

    # ── Compute invariants for all function pairs ─────────────────────────────
    results = []
    for fp in FUNCTION_PAIRS:
        k_py = compute_k_python(fp["complexity"])
        k_rs = compute_k_rust(k_py, RUST_SPEEDUP_MEASURED)
        e_borrow = compute_e_borrow(fp["borrow_vec"])

        # E_python proxy: computational intensity = how "heavy" the Python function is
        # Use -log_ω₀_norm (slower algorithms have lower ω₀ → higher E_python)
        e_python = max(0.0, -k_py[0])   # clip to [0, ∞): element_wise ω₀=ref → E_py=0

        delta_k = (k_rs[0] - k_py[0], k_rs[1] - k_py[1], k_rs[2] - k_py[2])

        results.append({
            "name": fp["name"],
            "description": fp["description"],
            "k_python": k_py,
            "k_rust": k_rs,
            "delta_k": delta_k,
            "e_borrow": e_borrow,
            "e_python": e_python,
        })

    # ── Table: K_python, K_rust, ΔK ─────────────────────────────────────────
    print(f"{'Function':<14}  {'K_python':^26}  {'K_rust':^26}  {'ΔK':^14}")
    print(f"{'--------':<14}  {'(ω₀,Q,ζ)_norm':^26}  {'(ω₀,Q,ζ)_norm':^26}  {'Δω₀_norm':^14}")
    print("-" * 85)
    for r in results:
        kp = r["k_python"]
        kr = r["k_rust"]
        print(
            f"  {r['name']:<12}  ({kp[0]:+.2f},{kp[1]:+.2f},{kp[2]:.2f})      "
            f"  ({kr[0]:+.2f},{kr[1]:+.2f},{kr[2]:.2f})      "
            f"  +{r['delta_k'][0]:.3f}"
        )

    print()

    # ── Table: E_borrow vs E_python ────────────────────────────────────────
    print(f"{'Function':<14}  {'E_borrow':>10}  {'E_python':>10}  {'Rust implementation'}")
    print(f"{'--------':<14}  {'--------':>10}  {'--------':>10}  {'-' * 35}")
    for r in results:
        print(
            f"  {r['name']:<12}  {r['e_borrow']:>10.4f}  {r['e_python']:>10.4f}  "
            + next(fp["description"] for fp in FUNCTION_PAIRS if fp["name"] == r["name"])
        )

    print()

    # ── Spearman correlation ───────────────────────────────────────────────
    e_borrows = np.array([r["e_borrow"] for r in results])
    e_pythons = np.array([r["e_python"] for r in results])
    rho, pval = spearmanr(e_borrows, e_pythons)

    print(f"Spearman ρ(E_borrow, E_python) = {rho:.4f}  (p={pval:.4f}, n={len(results)})")
    print()
    if abs(rho) > 0.5:
        print("  → STRONG correlation: compute-intensive algorithms also need")
        print("    more complex Rust ownership patterns.")
    elif abs(rho) > 0.3:
        print("  → MODERATE correlation: complexity class predicts both E_python")
        print("    and E_borrow, but they measure partially distinct things.")
    else:
        print("  → WEAK correlation: E_borrow and E_python are largely independent.")
        print("    Compute intensity does not determine borrow complexity.")

    print()

    # ── Separatrix analogy ─────────────────────────────────────────────────
    print("-" * 70)
    print("Constraint Boundary Analogy")
    print("-" * 70)

    # Duffing separatrix parameters (from Step 1 knowledge):
    # α=1, β=-0.1 (softening) → E_sep = α²/(4|β|) = 1/(0.4) = 2.5 J
    # Near-separatrix override: E₀/E_sep > 0.85
    duffing_e_sep = 2.5      # example softening case
    duffing_near_sep_fraction = 0.85

    # BorrowVector constraint boundary (from Step 2 measurement):
    e_borrow_dsep = 0.43       # Overall E_borrow D_sep from fit_boundary.py
    e_borrow_compile_fail_threshold = 0.60   # per-dimension threshold for hard failures

    print(f"""
  Duffing oscillator (nonlinear dynamics):
    E_sep       = {duffing_e_sep:.2f} J           (separatrix energy, β<0 softening)
    near-sep    : E₀/E_sep > {duffing_near_sep_fraction}  → topology change (ω₀_eff → 0)
    Measurement : EDMD trust < 0.3 when E₀/E_sep > 0.85

  Rust borrow checker (program structure):
    D_sep       = {e_borrow_dsep:.2f}            (E_borrow boundary, from Step 2 logistic fit)
    compile fail: E_borrow > D_sep        → structural phase change (compile fails)
    Measurement : AUC = 0.9163, B1/B2/B4 D_sep ≈ 0.60

  Structural analogy:
    Duffing:  E_total / E_sep > 0.85  →  topology changes, borrow (capture) fails
    Rust:     E_borrow / D_sep > 1.0  →  constraint violated, compile fails
    Both:     energy > threshold       →  hard boundary / phase transition

  Key finding:
    All 6 numeric kernels have E_borrow ∈ [0.00, 0.23]  << D_sep = 0.43
    → Numeric algorithms live DEEP inside the Rust constraint manifold
    → No compile failures expected for standard numeric kernels
    → The D_sep boundary is approached only for architecturally complex code
      (shared mutable state across modules, complex lifetime dependencies)
""")

    # ── Validation gate ────────────────────────────────────────────────────
    n_pairs = len(results)
    gate_pairs = n_pairs >= 5
    gate_corr = abs(rho) > 0.5 or abs(rho) < 0.3

    print("=" * 70)
    print(f"Step 3 Validation Gate:")
    print(f"  {'[PASS]' if gate_pairs else '[FAIL]'} Function pairs measured: {n_pairs} (need ≥5)")
    print(f"  {'[PASS]' if gate_corr else '[---]'} Correlation: ρ={rho:.4f} "
          f"({'informative' if gate_corr else 'in gray zone 0.3-0.5'})")

    if gate_pairs and gate_corr:
        print()
        print("  [PASS] Step 3 complete — proceed to Step 4 (optional Rust parser)")
        print("         or Phase 3 (DNN training) after collecting BorrowVector data.")
    elif gate_pairs:
        print()
        print("  [NOTE] Correlation in gray zone (0.3–0.5). This is informative:")
        print("         E_borrow and E_python are partially independent measures.")
        print("  [PASS] Step 3 complete — proceed to Step 4 / Phase 3.")
    else:
        print()
        print("  [FAIL] Need more function pairs.")

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
