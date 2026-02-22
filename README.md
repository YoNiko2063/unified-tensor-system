# Unified Tensor Systems

**Regime-Aware Spectral Acceleration for Stability-Critical Simulation**

*Patent Pending*

---

## Breakthrough

We have developed a regime-aware Koopman spectral method that accelerates nonlinear stability simulations by up to:

> **57,946×**
> on the IEEE 39-bus benchmark system
> with **<2.73% error** versus RK4 time-domain simulation.

This enables near-instant evaluation of critical clearing times (CCT) in power networks — turning a four-hour N-1 contingency screen into a sub-second computation.

---

## The Problem

Large-scale nonlinear systems — including power grids, inverter networks, and multiphysics PDE systems — require time-domain integration for stability evaluation.

These simulations are:

- Computationally expensive (156,000 ODE evaluations per CCT estimate)
- Difficult to scale (25,000 computations for a 500-bus N-1 screen)
- Poorly suited for real-time decision-making

At 0.1 ms per ODE evaluation in optimized Python, a full N-1 screen takes **over four hours**. Real-time contingency analysis requires orders-of-magnitude improvement.

---

## The Method

Our approach replaces brute-force time integration with a spectral regime classification pipeline:

1. Linearize local system dynamics via Modified Nodal Analysis
2. Construct a Koopman operator approximation (EDMD)
3. Identify locally commutative (LCA) stability regions
4. Detect transition boundaries between stability regimes
5. Compute critical stability thresholds analytically from spectral geometry

Instead of simulating trajectories step-by-step, the method identifies the structural stability manifold directly.

The architecture is domain-agnostic: any system expressible in the universal form **C·ẋ + G·x + h(x) = u(t)** can be analyzed.

---

## Validation

| Benchmark | Result |
|-----------|--------|
| System | IEEE 39-bus New England System (10 generators) |
| Speedup | **57,946×** over RK4 binary-search |
| CCT Error | **<2.73%** maximum deviation |
| Damping coverage | ζ = 0.00–0.20 (full realistic range) |
| Cross-validation | C(10,2) = 45 generator subsets, a = 1.51 ± 0.01 (range 2.3%) |
| Test suite | 2,239 automated validation tests passing |

The correction parameter a = 1.51 is stable across all cross-validation splits, indicating a structural property of the system rather than a fitting artifact.

---

## Quick Demonstration

```bash
git clone <repo-url>
cd unified-tensor-system
pip install -e .
python examples/ieee39_demo.py
```

**Output:**

```
IEEE 39-Bus CCT Benchmark
══════════════════════════════════════════════════════════════════════
  Gen    Bus    H[s]   CCT_EAC[s]  CCT_Ref[s]   Err%     Speedup
──────────────────────────────────────────────────────────────────────
  G1     39    500.0    0.548 s     0.534 s      2.6%    52,341×
  ...
  G10    30     42.0    0.231 s     0.228 s      1.3%    61,203×
──────────────────────────────────────────────────────────────────────
  Mean error: 1.84%   Max error: 2.73%   Mean speedup: 57,946×
```

---

## Applications

- **Power system stability** — real-time contingency screening, N-1 analysis
- **Renewable integration** — stability envelopes for inverter-dominated grids
- **Inverter and grid-forming control** — switching regime detection
- **Multiphysics simulation** — accelerated PDE stability screening
- **Stability-critical optimization** — fast objective evaluation

---

## Architecture

```
MNA linearization
      ↓
Koopman spectral estimator (EDMD)
      ↓
Regime classification (LCA / transition / nonabelian)
      ↓
Stability manifold mapper
      ↓
Analytic CCT / multi-objective optimizer
```

The same framework applies across circuits, power grids, and nonlinear oscillators — all expressible in the same universal dynamical form.

---

## Documentation

| Document | Description |
|----------|-------------|
| [TECHNICAL_BRIEF.md](TECHNICAL_BRIEF.md) | IEEE 39-bus method — full derivation, validation, cross-validation |
| [whitepaper/](whitepaper/) | Circuit optimizer — spectral geometry, Pareto results, Hurwitz enforcement |
| [patent/](patent/) | Provisional USPTO filing |

---

## Intellectual Property

A provisional patent has been filed covering:

- Regime classification via Koopman spectral geometry
- Detection of locally commutative (LCA) stability regions
- Computation of stability-critical transitions from spectral signatures

---

## Contact

For collaboration, licensing, or pilot deployments, contact us at:

**yoonikolas@gmail.com**
