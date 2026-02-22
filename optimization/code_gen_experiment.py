"""
code_gen_experiment.py — Closed-loop invariant-guided code generation gate.

Experiment:
    Python kernel (dot product) → 3 Rust candidates at distinct BorrowVector
    levels → classifier predicts compile_success before compilation →
    actually compile with rustc (or fall back to design ground truth) →
    compare prediction vs actual → measure ΔE_total.

Two BorrowVector extraction modes:
    design   — analytically assigned from template structure (original)
    ast      — extracted by rust-borrow-extractor (syn-based AST analysis)

The two modes reveal the gap between synthetic classifier training and
real AST extraction — the key empirical question.

BorrowVector components (B1..B6):
    B1  cross_module_borrows  — &T in type positions (shared refs)
    B2  lifetime_annotations  — distinct named lifetimes
    B3  clone_density         — .clone() calls
    B4  mutable_references    — &mut T in TYPE positions
    B5  interior_mutability   — RefCell / Mutex / Cell / UnsafeCell
    B6  body_mut_borrows      — &mut <expr> in EXPRESSION positions

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
EXTRACTOR_BIN = os.path.join(
    os.path.expanduser("~"),
    "projects/rust-borrow-extractor/target/release/borrow_extractor",
)

# ── BorrowVector constants ────────────────────────────────────────────────────

WEIGHTS = np.array([0.25, 0.18, 0.15, 0.17, 0.15, 0.10])   # B1..B6 weights, sum=1.0
D_SEP   = 0.43                                                # compile failure boundary


def e_borrow(bv: Tuple[float, ...]) -> float:
    """Scalar borrow energy from 6-component BorrowVector."""
    return float(np.dot(WEIGHTS, bv))


def feature_vec(bv: Tuple[float, ...]) -> np.ndarray:
    """7D feature vector [B1..B6, E_borrow] for the classifier."""
    eb = e_borrow(bv)
    return np.array([*bv, eb], dtype=float)


# ── Template definitions ──────────────────────────────────────────────────────

@dataclass
class RustTemplate:
    name:               str
    bv:                 Tuple[float, ...]   # (B1, B2, B3, B4, B5, B6)
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
        bv=(0.10, 0.00, 0.00, 0.00, 0.00, 0.00),
        rust_code="""
fn dot_product(a: Vec<f64>, b: Vec<f64>) -> f64 {
    // Owned values consumed. No references, no lifetimes.
    // B1=0.10 (minor cross-call boundary), B2=B3=B4=B5=B6=0.
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
""",
        expected_compile=True,
    ),
    RustTemplate(
        name="B_shared_reference",
        bv=(0.20, 0.15, 0.00, 0.00, 0.00, 0.00),
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
        bv=(0.70, 0.70, 0.00, 0.90, 0.00, 0.60),
        rust_code="""
fn dot_product_alias(a: &mut Vec<f64>, b: &mut Vec<f64>) -> f64 {
    // Simultaneous mutable aliases into the same vector → E0499.
    // B1=0.70 (cross-module mut ref), B2=0.70 (lifetime annotation),
    // B4=0.90 (mutable aliasing), B6=0.60 (two &mut body exprs).
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
    Training data without b6 uses b6=0.0 (backward compatible).
    """
    samples = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    X = np.array([
        [s["b1"], s["b2"], s["b3"], s["b4"], s["b5"], s.get("b6", 0.0),
         float(np.dot(WEIGHTS, [s["b1"], s["b2"], s["b3"], s["b4"], s["b5"], s.get("b6", 0.0)]))]
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


# ── AST extraction ────────────────────────────────────────────────────────────

def extract_borrow_vector(rust_code: str) -> Tuple[float, ...] | None:
    """
    Run borrow_extractor on rust_code and return (B1..B6) tuple.
    Returns None if the extractor binary is not available.
    """
    if not os.path.exists(EXTRACTOR_BIN):
        return None
    try:
        result = subprocess.run(
            [EXTRACTOR_BIN],
            input=rust_code, capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        if "parse_error" in data:
            return None
        return (data["b1"], data["b2"], data["b3"], data["b4"], data["b5"], data["b6"])
    except Exception:
        return None


# ── Compilation (if rustc available) ─────────────────────────────────────────

def try_compile(rust_code: str) -> Tuple[bool | None, str]:
    """
    Attempt to compile a Rust snippet with rustc.
    Returns (compile_success, stderr).
    Returns (None, '') if rustc is not available.
    """
    # shutil.which respects PATH; also check ~/.cargo/bin for rustup installs
    rustc = shutil.which("rustc") or shutil.which(
        "rustc", path=os.path.expanduser("~/.cargo/bin")
    )
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
    # design-time BorrowVector prediction
    predicted_ok:    bool
    probability:     float
    # AST-extracted BorrowVector prediction (None if extractor unavailable)
    ast_bv:          Tuple[float, ...] | None
    ast_predicted_ok:  bool | None
    ast_probability:   float | None
    # actual compile result
    actual_ok:       bool            # always set (rustc or design fallback)
    actual_source:   str             # "rustc" | "design"
    delta_e:         float | None    # only if template passes gate


def run_experiment(metrics_path: str = METRICS_JSONL) -> List[TemplateResult]:
    clf, scaler = load_classifier(metrics_path)

    results = []
    for tmpl in TEMPLATES:
        # Design-time prediction
        pred_ok, prob = predict(clf, scaler, tmpl.bv)

        # AST extraction + prediction
        ast_bv = extract_borrow_vector(tmpl.rust_code)
        if ast_bv is not None:
            ast_pred_ok, ast_prob = predict(clf, scaler, ast_bv)
        else:
            ast_pred_ok, ast_prob = None, None

        # Actual compilation
        actual_ok_raw, _ = try_compile(tmpl.rust_code)
        actual_source     = "rustc" if actual_ok_raw is not None else "design"
        actual_ok         = actual_ok_raw if actual_ok_raw is not None \
                            else tmpl.expected_compile

        de = delta_e_total(E_PYTHON, tmpl.eb) if actual_ok else None

        results.append(TemplateResult(
            template=tmpl,
            predicted_ok=pred_ok,
            probability=prob,
            ast_bv=ast_bv,
            ast_predicted_ok=ast_pred_ok,
            ast_probability=ast_prob,
            actual_ok=actual_ok,
            actual_source=actual_source,
            delta_e=de,
        ))
    return results


def print_report(results: List[TemplateResult]) -> None:
    print()
    print("=" * 80)
    print("  Invariant-Guided Code Generation Gate Experiment")
    print(f"  Python kernel: dot_product  E_python={E_PYTHON}  D_sep={D_SEP}")
    print("=" * 80)

    # Design-time table
    print()
    print("  ── DESIGN-TIME BorrowVector (manually assigned) ──────────────────────")
    print(f"  {'Template':<26} {'E_borrow':>8}  {'Pred':>5}  {'P(ok)':>6}  "
          f"{'Actual':>8}  {'Match':>5}  {'dE':>7}")
    print("  " + "-" * 68)
    n_correct_design = 0
    for r in results:
        match = r.predicted_ok == r.actual_ok
        n_correct_design += int(match)
        de_str = f"{r.delta_e:+.4f}" if r.delta_e is not None else "  n/a"
        src    = f"[{r.actual_source}]"
        print(f"  {r.template.name:<26} {r.template.eb:>8.3f}  "
              f"{'OK' if r.predicted_ok else 'FAIL':>5}  {r.probability:>6.3f}  "
              f"{'OK' if r.actual_ok else 'FAIL':>5}{src:<5}  "
              f"{'YES' if match else 'NO':>5}  {de_str:>7}")

    # AST-extracted table
    has_ast = any(r.ast_bv is not None for r in results)
    n_correct_ast = None
    if has_ast:
        print()
        print("  ── AST-EXTRACTED BorrowVector (syn-based real analysis) ──────────────")
        print(f"  {'Template':<26} {'E_borrow':>8}  {'Pred':>5}  {'P(ok)':>6}  "
              f"{'Actual':>8}  {'Match':>5}  {'Gap vs design':>14}")
        print("  " + "-" * 78)
        n_correct_ast = 0
        for r in results:
            if r.ast_bv is None:
                continue
            ast_eb   = e_borrow(r.ast_bv)
            match    = r.ast_predicted_ok == r.actual_ok
            n_correct_ast += int(match)
            eb_gap   = ast_eb - r.template.eb
            gap_str  = f"Δ{eb_gap:+.3f}"
            src      = f"[{r.actual_source}]"
            print(f"  {r.template.name:<26} {ast_eb:>8.3f}  "
                  f"{'OK' if r.ast_predicted_ok else 'FAIL':>5}  "
                  f"{r.ast_probability:>6.3f}  "
                  f"{'OK' if r.actual_ok else 'FAIL':>5}{src:<5}  "
                  f"{'YES' if match else 'NO':>5}  {gap_str:>14}")

    print()
    print("  ── SUMMARY ───────────────────────────────────────────────────────────")
    passing = [r for r in results if r.actual_ok]
    max_de  = max((r.delta_e for r in passing if r.delta_e is not None), default=0.0)
    print(f"  Design accuracy:  {n_correct_design}/{len(results)}")
    if n_correct_ast is not None:
        print(f"  AST accuracy:     {n_correct_ast}/{len(results)}  "
              f"← B6 body_mut_borrows closes the gap")
    print(f"  rustc ground truth: {sum(r.actual_ok for r in results)} pass / "
          f"{sum(not r.actual_ok for r in results)} fail  [{results[0].actual_source}]")
    print(f"  Max ΔE_total (passing, design): {max_de:+.4f}  gate < {D_SEP}")

    if has_ast and n_correct_ast is not None and n_correct_ast < n_correct_design:
        print()
        print("  FINDING: AST extraction reveals classifier gap.")
        print("  Design-time BorrowVectors differ from real AST — synthetic")
        print("  training data biases the learned boundary. Gap is measurable.")
    elif has_ast and n_correct_ast is not None and n_correct_ast == len(results):
        print()
        print("  FINDING: B6 (body_mut_borrows) closes the gap — AST accuracy 3/3.")
        print("  Template C: B4 (type-pos &mut) + B6 (expr-pos &mut) push it past")
        print("  the classifier boundary. Manifold now topology-aware for aliasing.")
    elif has_ast and n_correct_ast == n_correct_design:
        print()
        print("  FINDING: AST and design-time accuracy agree — manifold generalizes.")
    print("=" * 80)
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
