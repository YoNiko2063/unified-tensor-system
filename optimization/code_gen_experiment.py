"""
code_gen_experiment.py — Closed-loop invariant-guided code generation gate.

Experiment:
    Python kernel (dot product) → 3 Rust candidates at distinct BorrowVector
    levels → classifier predicts compile_success before compilation →
    compare prediction vs known ground truth → measure ΔE_total.

The three templates span the BorrowVector space:
    A  pure functional / owned     E_borrow ≈ 0.025  (safe zone)
    B  shared references           E_borrow ≈ 0.065  (safe zone)
    C  mutable aliasing (E0499)    E_borrow ≈ 0.495  (above D_sep=0.43)

BorrowVector components (B1..B5):
    B1 cross_module_borrows      weight 0.25
    B2 lifetime_annotations      weight 0.20
    B3 clone_density             weight 0.15  (protective)
    B4 mutable_references        weight 0.20
    B5 interior_mutability       weight 0.20  (protective)

Classifier: LogisticRegression trained on 250 samples from
    rust-constraint-manifold/metrics.jsonl (CV AUC=0.918, hold-out=0.951).

Compile ground truth: derived from template design using the same borrow
rules the Rust compiler enforces (B1>0.6 OR B2>0.6 OR B4>0.6 → failure).
If rustc is available, actual compilation is performed instead.

Usage:
    conda run -n tensor python optimization/code_gen_experiment.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

METRICS_JSONL = os.path.join(
    os.path.expanduser("~"),
    "projects/rust-constraint-manifold/metrics.jsonl",
)

# ── BorrowVector constants ────────────────────────────────────────────────────

WEIGHTS = np.array([0.25, 0.20, 0.15, 0.20, 0.20])   # B1..B5 weights
D_SEP   = 0.43                                          # compile failure boundary


def e_borrow(bv: Tuple[float, ...]) -> float:
    """Scalar borrow energy from 5-component BorrowVector."""
    return float(np.dot(WEIGHTS, bv))


def feature_vec(bv: Tuple[float, ...]) -> np.ndarray:
    """6D feature vector [B1..B5, E_borrow] for the classifier."""
    eb = e_borrow(bv)
    return np.array([*bv, eb], dtype=float)


# ── Template definitions ──────────────────────────────────────────────────────

@dataclass
class RustTemplate:
    name:               str
    bv:                 Tuple[float, ...]   # (B1, B2, B3, B4, B5)
    rust_code:          str
    expected_compile:   bool                # ground truth from borrow rules

    @property
    def eb(self) -> float:
        return e_borrow(self.bv)


# Python kernel being translated
PYTHON_KERNEL = """
def dot_product(a: list, b: list) -> float:
    \"\"\"Element-wise multiply-accumulate. O(N), no aliasing.\"\"\"
    return sum(x * y for x, y in zip(a, b))
"""

# E_python: from Step 3 function-pair table — element_wise class
E_PYTHON = 0.00   # reduction/element_wise: purely numeric, no borrow structure

TEMPLATES: List[RustTemplate] = [
    RustTemplate(
        name="A_pure_functional",
        bv=(0.10, 0.00, 0.00, 0.00, 0.00),
        rust_code="""
fn dot_product(a: Vec<f64>, b: Vec<f64>) -> f64 {
    // Owned values consumed. No references, no lifetimes.
    // B1=0.10 (minor cross-call boundary), B2=B3=B4=B5=0.
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
""",
        expected_compile=True,
    ),
    RustTemplate(
        name="B_shared_reference",
        bv=(0.20, 0.15, 0.00, 0.00, 0.00),
        rust_code="""
fn dot_product<'a>(a: &'a [f64], b: &'a [f64]) -> f64 {
    // Immutable borrows with explicit lifetime annotation.
    // B1=0.20 (borrows cross function boundary), B2=0.15 (lifetime 'a).
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
""",
        expected_compile=True,
    ),
    RustTemplate(
        name="C_mutable_aliasing",
        bv=(0.70, 0.70, 0.00, 0.90, 0.00),
        rust_code="""
fn dot_product_alias(a: &mut Vec<f64>, b: &mut Vec<f64>) -> f64 {
    // Simultaneous mutable aliases into the same vector → E0499.
    // B1=0.70 (cross-module mut ref), B2=0.70 (lifetime annotation),
    // B4=0.90 (mutable aliasing). All three exceed 0.6 threshold.
    let r1: &mut f64 = &mut a[0];           // first &mut borrow of a[0]
    let r2: &mut f64 = &mut a[0];           // E0499: second simultaneous &mut
    *r1 = b[0];
    *r2 = b[1];
    0.0
}
""",
        expected_compile=False,
    ),
]

# ── Classifier ────────────────────────────────────────────────────────────────

def load_classifier(metrics_path: str) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Load training data from metrics.jsonl and fit LogisticRegression.
    Returns (fitted_clf, fitted_scaler).
    """
    samples = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    X = np.array([
        [s["b1"], s["b2"], s["b3"], s["b4"], s["b5"], s["e_borrow"]]
        for s in samples
    ])
    y = np.array([int(s["compile_success"]) for s in samples])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    return clf, scaler


def predict(clf: LogisticRegression,
            scaler: StandardScaler,
            bv: Tuple[float, ...]) -> Tuple[bool, float]:
    """
    Predict compile_success for a BorrowVector.
    Returns (predicted_bool, probability).
    """
    fv = feature_vec(bv).reshape(1, -1)
    fv_scaled = scaler.transform(fv)
    prob = clf.predict_proba(fv_scaled)[0, 1]
    return (prob >= 0.5), float(prob)


# ── Compilation (if rustc available) ─────────────────────────────────────────

def try_compile(rust_code: str) -> Tuple[bool | None, str]:
    """
    Attempt to compile a Rust snippet with rustc.
    Returns (compile_success, stderr).
    Returns (None, '') if rustc is not available.
    """
    rustc = shutil.which("rustc")
    if rustc is None:
        return None, ""

    with tempfile.NamedTemporaryFile(suffix=".rs", mode="w", delete=False) as f:
        # Wrap snippet in a main function for standalone compilation
        f.write("fn main() {}\n")
        f.write(rust_code)
        tmppath = f.name

    try:
        result = subprocess.run(
            [rustc, "--edition", "2021", tmppath, "--out-dir", tempfile.gettempdir()],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0, result.stderr
    finally:
        os.unlink(tmppath)
        # Remove compiled binary if it exists
        binary = os.path.join(tempfile.gettempdir(),
                              os.path.basename(tmppath).replace(".rs", ""))
        if os.path.exists(binary):
            os.unlink(binary)


# ── ΔE_total ──────────────────────────────────────────────────────────────────

def delta_e_total(e_py: float, eb_rust: float) -> float:
    """
    ΔE_total = E_borrow_rust - E_python.
    Positive: added borrow structure; small positive = stable translation.
    Must remain below D_sep to stay in the safe zone.
    """
    return eb_rust - e_py


# ── Main experiment ───────────────────────────────────────────────────────────

@dataclass
class TemplateResult:
    template:        RustTemplate
    predicted_ok:    bool
    probability:     float
    actual_ok:       bool | None    # None if rustc unavailable
    actual_source:   str            # "rustc" | "design"
    delta_e:         float | None   # only if template passes gate


def run_experiment(metrics_path: str = METRICS_JSONL) -> List[TemplateResult]:
    clf, scaler = load_classifier(metrics_path)

    results = []
    for tmpl in TEMPLATES:
        pred_ok, prob   = predict(clf, scaler, tmpl.bv)
        actual_ok, _    = try_compile(tmpl.rust_code)
        actual_source   = "rustc" if actual_ok is not None else "design"
        if actual_ok is None:
            actual_ok = tmpl.expected_compile   # fall back to known ground truth

        de = delta_e_total(E_PYTHON, tmpl.eb) if actual_ok else None

        results.append(TemplateResult(
            template=tmpl,
            predicted_ok=pred_ok,
            probability=prob,
            actual_ok=actual_ok,
            actual_source=actual_source,
            delta_e=de,
        ))
    return results


def print_report(results: List[TemplateResult]) -> None:
    print()
    print("=" * 72)
    print("  Invariant-Guided Code Generation Gate Experiment")
    print("  Python kernel: dot_product (element_wise, E_python=0.00)")
    print(f"  Classifier: LogReg trained on 250 samples (D_sep={D_SEP})")
    print("=" * 72)
    print()
    print(f"  {'Template':<26} {'E_borrow':>8}  {'Pred':>5}  {'P(ok)':>6}  "
          f"{'Actual':>6}  {'Match':>5}  {'dE_total':>9}  {'Zone':>8}")
    print("  " + "-" * 70)

    n_correct = 0
    for r in results:
        match     = r.predicted_ok == r.actual_ok
        n_correct += int(match)
        de_str    = f"{r.delta_e:+.3f}" if r.delta_e is not None else "  n/a "
        zone      = "SAFE" if (r.delta_e is not None and r.delta_e < D_SEP) else \
                    ("FAIL" if r.delta_e is None else "WARN")
        src_tag   = f"[{r.actual_source}]"
        print(f"  {r.template.name:<26} {r.template.eb:>8.3f}  "
              f"{'OK' if r.predicted_ok else 'FAIL':>5}  {r.probability:>6.3f}  "
              f"{'OK' if r.actual_ok else 'FAIL':>6}{src_tag:>4}  "
              f"{'YES' if match else 'NO ':>5}  {de_str:>9}  {zone:>8}")

    print()
    accuracy = n_correct / len(results)
    passing  = [r for r in results if r.actual_ok and r.delta_e is not None]
    max_de   = max((r.delta_e for r in passing), default=0.0)

    print(f"  Classifier accuracy:  {n_correct}/{len(results)} ({100*accuracy:.0f}%)")
    print(f"  Passing templates:    {len(passing)}/{len(results)}")
    if passing:
        print(f"  Max ΔE_total (passing): {max_de:+.3f}  (gate: < {D_SEP})")
        print(f"  Safe zone confirmed:  {'YES' if max_de < D_SEP else 'NO'}")
    print()

    if accuracy == 1.0 and (not passing or max_de < D_SEP):
        print("  RESULT: Invariant-guided gate correctly classifies all 3 templates.")
        print("  Low-BorrowVector templates pass; high-BorrowVector template blocked.")
        print("  ΔE_total for passing templates bounded below D_sep — stable zone.")
    else:
        print("  RESULT: Partial — see mismatches above.")
    print("=" * 72)
    print()


def compute_summary(results: List[TemplateResult]) -> dict:
    n_correct = sum(r.predicted_ok == r.actual_ok for r in results)
    passing   = [r for r in results if r.actual_ok and r.delta_e is not None]
    return {
        "accuracy":       n_correct / len(results),
        "n_correct":      n_correct,
        "n_templates":    len(results),
        "n_passing":      len(passing),
        "max_delta_e":    max((r.delta_e for r in passing), default=None),
        "d_sep":          D_SEP,
        "safe_zone":      all(r.delta_e < D_SEP for r in passing),
    }


if __name__ == "__main__":
    results = run_experiment()
    print_report(results)
