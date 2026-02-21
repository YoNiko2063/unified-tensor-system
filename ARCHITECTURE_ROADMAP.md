# Unified Architecture Roadmap
# Hardware → Software → DNN/SNN Optimization Chain

**Status:** Living document — updated as implementation proceeds
**Date:** 2026-02-21
**Directive:** ChatGPT isolation rules incorporated 2026-02-21

---

## Repo Isolation Structure (MANDATORY)

Three separate repos. Do not cross-pollinate.

```
Repo A: unified-tensor-system/       ← EXISTING — do not modify core invariant engine
Repo B: rust-physics-kernel/         ← Step 1 [IN PROGRESS] — RK4 acceleration
Repo C: rust-constraint-manifold/    ← Step 2 [TO DO] — BorrowVector experiment
```

Repo B feeds trajectory throughput back to Repo A via pyo3 bindings.
Repo C is purely measurement — no integration until Step 3 analysis is complete.

---

## What Already Exists (Not Aspirational)

The cross-domain invariant space is already operational across FOUR domains:

```
Physical systems:       RLC, spring-mass, Duffing → (log_ω₀_norm, log_Q_norm, ζ)
Optimization dynamics:  Gradient descent           → (log_ω₀_norm, log_Q_norm, ζ)
Algorithm complexity:   Python code (AST+timing)   → (log_ω₀_norm, log_Q_norm, ζ)
Nonlinear dynamics:     Koopman/EDMD manifold       → curvature profile, trust metric
```

Files doing this TODAY:
- `optimization/code_profiler.py`   — Python code → (ω₀, Q) via AST + dynamic timing
- `optimization/repo_learner.py`    — GitHub repo → function catalog → invariant memory
- `optimization/koopman_memory.py`  — cross-domain retrieval by invariant triple
- `optimization/duffing_evaluator.py` — separatrix detection, topology discrimination

The system already learns that an O(N³) matmul and an overdamped RLC circuit have
similar (ω₀, Q) signatures. This is the foundational cross-domain link.

---

## The Mathematical Framework

### Layer Stack

```
Layer 7: Hardware H
         └─ FLOPS/sec, cache miss rate, SIMD utilization, memory bandwidth
                ↑ (LLVM IR / profiler)

Layer 6: LLVM IR L
         └─ SSA form, register allocation, vectorization opportunities
                ↑ (rustc codegen)

Layer 5: Rust R  ← NEW RESEARCH REGION (Steps 1-2)
         └─ Borrow checker constraint manifold M_C
         └─ Type system as compile-time invariant enforcer
         └─ BorrowVector = (cross_module, lifetime, clone, mut_ref, interior_mut)
                ↑ (pyo3 / FFI)

Layer 4: Python P  ← currently instrumented
         └─ code_profiler.py: AST → complexity class → (ω₀, Q)
         └─ DynamicProfiler: timing → empirical (ω₀, Q)
         └─ ASTMathClassifier: matmul/fft/sort/reduction/stencil
                ↑ (AST / semantic analysis)

Layer 3: Behavioral Signature B
         └─ (input_types, output_type, side_effects, complexity, conserved_quantity)
                ↑ (intent parsing)

Layer 2: Koopman Invariant K = (log_ω₀_norm, log_Q_norm, ζ)
         └─ Universal representation across all layers
         └─ Separatrix detection: E₀/E_sep ratio (topology gate)
         └─ Trust metric: EDMD recon error (chaos/periodic classifier)

Layer 1: Intent I
         └─ "Optimize this", "Parse this", "Simulate this"
```

### Constraint Manifold (Rust Layer)

Rust's borrow checker defines a hard constraint manifold:

```
M_C = { p ∈ Programs | borrow_checker(p) = ✓ ∧ type_checker(p) = ✓ }
∂M_C = compile error surface (hard binary boundary, unlike Python test softness)
D_sep(p) = inf_{q ∈ ∂M_C} d(p, q)  — distance to constraint boundary
```

Structural analogy to Duffing separatrix:
```
Duffing:  E₀ / E_sep > 0.85         → topology change (near-separatrix override)
Rust:     E_borrow   > D_sep        → compile failure
Both:     state_energy > threshold  → structural phase change
```

BorrowVector (normalized by module count n):
```
B = (B1, B2, B3, B4, B5) where:
  B1 = cross_module_borrows / n     (shared references across modules)
  B2 = lifetime_annotations / n    (explicit lifetime constraints)
  B3 = clone_density / n           (copying vs borrowing)
  B4 = mutable_references / n      (exclusive write access)
  B5 = interior_mutability / n     (RefCell, Mutex — runtime borrow checking)

E_borrow = 0.25·B1 + 0.20·B2 + 0.15·B3 + 0.20·B4 + 0.20·B5
```

### Extended Energy Function

```
E_python  = f(log_ω₀_norm, log_Q_norm, ζ)              [from code_profiler.py]
E_topo    = g(D_sep, near_separatrix, curvature_spike)  [from duffing_evaluator.py]
E_borrow  = Σ wᵢ·Bᵢ                                     [NEW: Rust constraint layer]
E_hardware = h(cache_miss_rate, SIMD_util, memory_bw)   [future: LLVM perf]

E_total = E_python + λ₁·E_topo + λ₂·E_borrow + λ₃·E_hardware
          (λ₁=0.3, λ₂=0.5, λ₃=0.2 — to be calibrated after Step 2 data)
```

**Key insight:** E_borrow is to code structure what E_topo is to dynamical systems.
Both measure proximity to a constraint boundary (compile error vs. separatrix).
Both gate retrieval (wrong topology → wrong retrieval; wrong constraint level → compile fail).

### SNN Neuron Model for Code Generation

The DNN modeling SNN behavior maps to code generation as:

```
Membrane potential  V(t)    ↔  E_total(t)       accumulated structural energy
Firing threshold    V_th    ↔  D_sep            distance to compile/test boundary
Fire event                  ↔  "propose patch"
Refractory period           ↔  E_borrow         more constrained = slower recovery
Inhibitory feedback         ↔  compile_error_count  negative reward on failure
Reward signal               ↔  ΔE_total         energy reduction from patch
```

The DNN learns:
```
f: (B, K, BorrowVector) → {python_code | rust_code | reject}
```
such that E_total is minimized subject to behavioral equivalence B.

---

## Why Rust Is the Right Research Layer

Python gives soft boundaries (test failures, runtime errors, GC pressure).
Rust gives hard boundaries (the type system is a theorem prover; borrow checker is a linear logic oracle).

This is valuable because:

1. **Structured error signal**: Rust compiler errors are typed (E0505 = moved value,
   E0502 = borrow conflict, E0308 = type mismatch). Each error code is a specific
   constraint violation — richer training signal than "test failed."

2. **Hardware proximity**: Rust → LLVM IR → machine code with no runtime overhead.
   The mapping from code structure to hardware performance is nearly direct.
   Python has GIL, GC, and interpreter overhead obscuring this mapping.

3. **Behavioral equivalence provable**: If a Rust program type-checks, its memory
   safety is proven. Behavioral equivalence can be checked by the type system
   (same function signatures, no unexpected side effects). Python cannot prove this.

4. **WASM target**: Rust compiles to WebAssembly. Physics simulators, parsers, and
   EDMD computations written in Rust can run in the browser with near-native speed.

---

## Implementation Steps

### [IN PROGRESS] Step 1 — Rust Physics Kernel (Repo B)

**Repo:** `rust-physics-kernel/`
**Goal:** Accelerate RK4/Duffing trajectory generation. Validate Rust toolchain.
**Scope:** RK4 step + Duffing RHS + pyo3 bindings. Nothing else.
**Expected speedup:** 20-100× for trajectory generation.

```
rust-physics-kernel/
├── Cargo.toml              pyo3 extension-module
├── src/lib.rs              DuffingParams, rk4_step, generate_trajectory, pymodule
├── python/
│   ├── benchmark.py        Python RK4 baseline vs Rust — measure actual speedup
│   └── verify.py           Trajectory agreement check (max abs diff < 1e-10)
└── setup.sh                rustup install + maturin build
```

BorrowVector for this code (expected lowest complexity):
```
B1=0.1, B2=0, B3=0, B4=0, B5=0  →  E_borrow = 0.025  (pure functional, Copy types)
```

**Validation gate:** Do not proceed to Step 2 until:
- [ ] `cargo check` passes
- [ ] benchmark.py shows >10× speedup at n_steps=10000
- [ ] verify.py shows max trajectory divergence < 1e-10

---

### [TO DO] Step 2 — Rust Constraint Manifold Generator (Repo C)

**Repo:** `rust-constraint-manifold/`
**Goal:** Measure BorrowVector → compile_success mapping. Build D_sep estimate.
**Scope:** CLI program generator only. No physics. No integration.

```
rust-constraint-manifold/
├── Cargo.toml
├── src/
│   ├── main.rs             CLI: --b1 --b2 --b3 --b4 --b5 --n-samples
│   ├── generator.rs        Synthetic Rust program template engine
│   ├── runner.rs           cargo check subprocess + structured stderr parsing
│   └── metrics.rs          metrics.jsonl emitter
└── analysis/
    └── fit_boundary.py     Logistic regression: BorrowVector → P(compile_success)
```

Output per sample:
```json
{"b1": 0.2, "b2": 0.0, "b3": 0.1, "b4": 0.3, "b5": 0.0,
 "e_borrow": 0.11, "compile_success": true, "error_codes": [], "error_count": 0}
```

Target: 200-500 samples spanning E_borrow ∈ [0.0, 1.0].

**Validation gate:** Do not proceed to Step 3 until:
- [ ] 200+ samples collected
- [ ] fit_boundary.py logistic regression AUC > 0.75
- [ ] D_sep estimate exists for B1..B5 individually

---

### [TO DO] Step 3 — Cross-Domain Comparison

**Location:** `unified-tensor-system/optimization/` (analysis only — no new modules)
**Goal:** Compare E_borrow boundary to Duffing separatrix topology.

```
Week 1: Map same behavioral signatures to Python and Rust
        - Same function (e.g., matrix multiply) in Python and Rust
        - K(Python) = (log_ω₀_norm, log_Q_norm, ζ) from code_profiler.py
        - K(Rust)   = (log_ω₀_norm, log_Q_norm, ζ) estimated from complexity class
        - Measure ΔK = invariant shift from Python → Rust

Week 2: Compare curvature of BorrowVector boundary to Duffing separatrix curvature
        - Is D_sep(BorrowVector) a topological boundary in the same sense?
        - Does E_borrow correlate with E_python at same behavioral signature?
```

**Validation gate:** Do not proceed to Step 4 until:
- [ ] ΔK measured for at least 5 function pairs
- [ ] Correlation(E_borrow, E_python | same behavior) > 0.5 or < 0.3 (either is informative)

---

### [TO DO] Step 4 — Rust Parser for Ingestion (Optional, Independent)

**Repo:** `rust-html-parser/` (new, separate)
**Goal:** Replace regex-based arxiv_pdf_parser.py equation extraction.
**Scope:** html5ever or nom crate. WASM target optional.

Not blocked by Steps 2-3. Can run in parallel after Step 1 validates toolchain.

BorrowVector expected (medium complexity):
```
B1=0.3, B2=0.2, B3=0.1, B4=0.2, B5=0.1  →  E_borrow ≈ 0.20
```
Parser is a useful mid-complexity training point for constraint manifold.

---

### [TO DO] Phase 3 — DNN Training (Months 3-6)

**Prerequisite:** Step 2 data (200+ BorrowVector samples) + Step 3 cross-domain comparison.
**Goal:** DNN predicts Rust implementation quality from behavioral signature.

```
Training data: (B, K, BorrowVector) → (compile_success, E_total)
Architecture:  Feed-forward over invariant triple + BorrowVector
Objective:     Minimize predicted E_total subject to P(compile_success) > 0.8
Output:        Code template selection + constraint parameter injection
```

No DNN training until Step 2 data exists. The λ values in E_total are empirical.

---

### [TO DO] Phase 4 — Self-Optimization (Months 6-12)

**Prerequisite:** Phase 3 trained DNN.
**Goal:** System proposes its own Rust optimizations in isolated test branch.

```
- Identify Python bottlenecks via code_profiler.py (high E_python)
- DNN proposes Rust replacement candidate
- Generate Rust implementation via template + constraint satisfaction
- cargo check → accept/reject
- Run behavioral equivalence test
- If both pass: merge to optimization branch
- Log: ΔE_total, BorrowVector, K_before, K_after
```

---

## What Is Explicitly Forbidden (Now)

These are out of scope until their prerequisite phases complete:

- [ ] LLVM IR analysis → Phase 4+
- [ ] Multi-target hardware mapping → Phase 5+
- [ ] Full SNN architecture training → needs Phase 1-3 data first
- [ ] Generalized "AI efficiency function" → needs empirical λ calibration from Step 2
- [ ] Self-modification of dev-agent → needs Phase 1 measurement infrastructure
- [ ] Modifying dev-agent in any way
- [ ] Mixing repos (Rust code in unified-tensor-system)
- [ ] Cross-layer coupling before Step 3 analysis
- [ ] Patent expansion claims → measure first, claim second
- [ ] API calls of any kind → local-only constraint (permanent)

---

## Connection to Patent Claims

The constraint manifold experiment directly extends patentable claims:

**Existing Claim 2** (Energy-conditioned separatrix gate):
The Rust compile boundary IS a separatrix — E_borrow approaching D_sep triggers
the same retrieval gate logic. The patent claim extends naturally: "wherein the
constraint boundary is detected by a borrow energy metric computed from a
constraint vector of cross-module reference patterns, lifetime annotations,
and mutation density."

**New claimable area** (after Step 2 data): BorrowVector → compile_success predictor
as a topology-aware constraint manifold for code structure optimization.
This is novel — no existing Koopman/EDMD work applies to compiler constraint spaces.
Do not file expanded claims until D_sep measurement exists.

---

## Summary

The Rust experiment is not separate from the unified tensor system.
It IS the unified tensor system applied to a new domain: programs under
type-system and borrow-checker constraints.

The existing (ω₀, Q, ζ) invariant triple extends to programs.
The existing separatrix detection extends to compile boundaries.
The existing EDMD trust metric extends to code generation success rates.
The existing cross-domain retrieval works for code just as it works for circuits.

code_profiler.py already proved this for Python.
The Rust layer gives the hard constraint manifold that Python lacks.
Together they span the hardware→algorithm→physics dimension.
