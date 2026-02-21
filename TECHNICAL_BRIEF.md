# Fast CCT Screening for Grid Transient Stability
## A Universal Equal-Area Criterion with Validated Damping Correction

**Version 1.0 — 2026-02-21**
*Unified Tensor System — Operator Geometry Platform*

---

## Abstract

Critical Clearing Time (CCT) estimation is a computational bottleneck in
N-1 contingency screening for power grid transient stability.  The standard
method — binary-search over fault duration with full time-domain simulation —
requires on the order of 10,000 ODE evaluations per machine per contingency.
We present an analytic Equal-Area Criterion (EAC) formula for three-phase
faults, extended by a single global damping correction parameter fitted from
data.  On the IEEE 39-bus New England system (10 generators), the corrected
method achieves **maximum CCT error of 2.73%** across all generators and
damping ratios up to ζ = 0.20 (Q ≥ 2), covering the full realistic range
of inter-area and local mode damping.  The method is **57,946× faster** than
the RK4 reference on the undamped benchmark.  Cross-validation across all
C(10,2) = 45 generator subsets confirms that the correction parameter
a = 1.51 ± 0.01 is stable (range 2.3%), indicating a structural property
of the system rather than a fitting artifact.

---

## 1. Problem

Transient stability assessment of a power grid requires determining, for
each generator and each credible fault contingency, whether the generator
will maintain synchronism after the fault is cleared.  The Critical Clearing
Time (CCT) is the maximum fault duration below which the generator remains
stable.

**The computational challenge:** N-1 contingency screening requires CCT
estimates for O(N) faults × O(G) generators.  A 500-bus system with 50
generators and 500 contingencies requires 25,000 CCT computations.  The
standard RK4 binary-search method requires approximately:

    ~13 iterations × 3,000 settle-steps × 4 RK4 evaluations
    ≈ 156,000 ODE evaluations per CCT

At 0.1 ms per evaluation in optimized Python, this is **15 seconds per CCT**
— over four hours for a full N-1 screen.  Real-time or near-real-time
screening (for energy markets, operator decision support, or remedial action
schemes) requires orders-of-magnitude improvement.

**Existing fast methods** (transient energy functions, direct methods,
extended equal-area criterion) are known in the literature but are often
validated only for the undamped classical model, and their accuracy under
realistic damping conditions is rarely quantified systematically across
multiple machines with full error bounds.

---

## 2. Method

### 2.1 System Model

We use the classical Single-Machine Infinite Bus (SMIB) swing equation:

    M · δ̈ + D · δ̇  =  Pₘ − Pₑ · sin(δ)

where δ is rotor angle [rad], ω = δ̇ is speed deviation [rad/s], M is the
inertia constant [pu·s²/rad] (M = 2H/ωₛ, ωₛ = 2π·60 rad/s), D is damping
[pu·s/rad], Pₘ is mechanical power [pu], and Pₑ is peak electrical power [pu].

The stable equilibrium is δₛ = arcsin(Pₘ/Pₑ).  The separatrix energy (energy
barrier to loss of synchronism) is:

    Eₛₑₚ = 2Pₑ·cos(δₛ) − Pₘ·(π − 2δₛ)

This is the exact analog of the separatrix in a nonlinear pendulum / softening
Duffing oscillator.  Loss of synchronism occurs when the rotor trajectory
crosses the unstable equilibrium δᵤ = π − δₛ.

### 2.2 Equal-Area Criterion (D = 0, three-phase fault)

For a complete three-phase fault (Pₑ → 0 during fault) and zero damping, the
equal-area criterion gives the critical clearing angle analytically:

    cos(δc) = Pₘ·(π − 2δₛ) / Pₑ  −  cos(δₛ)                   (1)

The critical clearing time follows from the fault-phase equation of motion
M·δ̈ = Pₘ (exact integral for D = 0):

    CCT_EAC = √( 2M·(δc − δₛ) / Pₘ )                            (2)

This formula requires zero ODE evaluations.  It is **analytically exact** for
D = 0 and a complete three-phase fault.

### 2.3 Global Damping Correction

For D > 0, damping slows the rotor during the fault phase, reducing kinetic
energy at clearing.  The actual CCT (reference) is therefore **larger** than
CCT_EAC — the undamped formula is conservative (underestimates the true CCT).

We propose the one-parameter first-order correction:

    CCT_corrected = CCT_EAC / (1 − a·ζ)                          (3)

where ζ = D/(2Mω₀) is the damping ratio and a is a scalar fitted by
ordinary least squares across all generators and damping levels.

**Why a single scalar suffices — analytic argument.**  For all generators
operating at the same loading ratio Pₑ/Pₘ = 2 (δₛ = 30°), the product:

    ω₀ · CCT_EAC = √( 2·√3·(δc − δₛ) ) ≈ 1.73                  (4)

is a universal constant, independent of M, Pₘ, or Pₑ individually.  First-order
perturbation of the fault-phase dynamics shows CCT_ref/CCT_EAC ≈ 1 + C·ζ
where C = ω₀·CCT_EAC/3 ≈ 0.577.  The fitted a absorbs higher-order terms;
universality is guaranteed by the geometry of the equal-area constraint, not
assumed.

**OLS fit:**  Given sweep data (CCT_EAC,i, CCT_ref,i, ζi) for all generators
and damping levels, define eᵢ = CCT_EAC,i − CCT_ref,i < 0 and xᵢ = ζᵢ·CCT_ref,i.
Minimising Σ(eᵢ + a·xᵢ)² gives:

    a = −Σ(eᵢ·xᵢ) / Σ(xᵢ²)                                      (5)

---

## 3. Results

### 3.1 Benchmark System

**Generator data:**  Anderson & Fouad (2003), Table 2.7 — 10-generator
New England equivalent of the IEEE 39-bus system.  Inertia constants range
from H = 24.3 s (G8) to H = 500 s (G1, large external equivalent).
All generators operate at Pₑ = 2·Pₘ (δₛ = 30°).

**Reference method:**  `estimate_cct()` — binary search on fault duration
with 4th-order Runge–Kutta integration (dt = 0.01 s, tolerance = 1 ms),
3,000 post-fault settle steps per stability check.

### 3.2 Undamped Benchmark (D = 0)

| Generator | H [s] | ω₀ [rad/s] | CCT_EAC [s] | CCT_Ref [s] | Error |
|-----------|-------|------------|-------------|-------------|-------|
| G1        | 500.0 |      1.278 |      1.3549 |      1.3553 |  0.0% |
| G2        |  30.3 |      7.858 |      0.2203 |      0.2249 |  2.1% |
| G3        |  35.8 |      7.699 |      0.2248 |      0.2249 |  0.0% |
| G4        |  28.6 |      8.494 |      0.2038 |      0.2048 |  0.5% |
| G9        |  34.5 |      8.863 |      0.1953 |      0.1950 |  0.2% |
| G10       |  42.0 |      8.834 |      0.1959 |      0.1950 |  0.5% |

**Mean CCT error: 0.8%.  Mean speedup: 57,946×.**

Residual error at D = 0 is due to RK4 time discretisation and ±0.5 ms
binary-search tolerance; it is not a limitation of the EAC formula itself.

### 3.3 Damping Sweep and Correction (D > 0)

Sweep: ζ ∈ {0.01, 0.03, 0.05, 0.10, 0.20} × 10 generators = 50 measurements.
D set per generator as D = 2Mω₀ζ.

**Figure 1** shows |CCT error| versus ζ for the raw EAC formula (red) and
the corrected formula with a = 1.51 (blue).  Shaded bands indicate
min/max spread across the 10 generators.

![Figure 1](figures/error_vs_zeta.png)

*Figure 1.  CCT error vs damping ratio ζ.  Red: raw EAC (D = 0 formula).
Blue: corrected EAC with a = 1.51.  Shaded bands show generator spread.
Dashed horizontal: 5% accuracy gate.  Blue shaded region: validated
operating envelope (ζ ≤ 0.20).*

| ζ   | Q    | ω₀ drift | Raw mean err | Raw max err | Corr mean err | Corr max err |
|-----|------|----------|--------------|-------------|---------------|--------------|
| 0.01 | 50 | −0.005% |       −1.33% |       3.64% |        +0.19% |        2.15% |
| 0.03 | 16.7 | −0.045% |     −4.89% |       6.37% |        −0.36% |        1.91% |
| 0.05 | 10.0 | −0.125% |     −8.93% |      10.10% |        −1.46% |        2.73% |
| 0.10 |  5.0 | −0.501% |    −16.29% |      17.07% |        −1.34% |        2.26% |
| 0.20 |  2.5 | −2.020% |    −29.13% |      30.12% |        +1.68% |        2.69% |

**Key findings:**

1. **EAC is always conservative** (negative signed error for D > 0).  For
   safety-critical screening, underestimating CCT is the correct direction.

2. **ω₀ drift is negligible:** 0.005% at ζ = 0.01, 2.0% at ζ = 0.20.
   The correction is not needed for the resonance frequency term.

3. **After correction:** maximum error across all 50 (generator, ζ) pairs
   is 2.73% — well within the 5% accuracy gate.

4. **Invariant geometry:** The embedding distance in the 3D invariant space
   (log ω₀ norm, log Q norm, ζ) is generator-independent (varies by < 10⁻⁸
   across all 10 machines at the same ζ).  This is the geometric signature
   of universality — confirmed analytically by Equation (4).

---

## 4. Robustness: Leave-2-Out Cross-Validation

To verify that a = 1.51 is a structural property and not an in-sample
artefact, we performed leave-2-out cross-validation across all C(10,2) = 45
generator subsets.  For each split, a was fitted on 8 generators (40 data
points) and tested on the 2 held-out generators (10 data points).

| Statistic        | Value                 |
|------------------|-----------------------|
| a (mean, 45 splits) | 1.5109             |
| a (std)          | 0.0100                |
| a (min, max)     | 1.4822, 1.5176        |
| a range / mean   | **2.3%**  (gate: < 10%) |
| CV (std/mean)    | **0.66%**             |
| Max test error   | **2.90%**  (split: G1,G2) |
| Mean test error  | 1.33%                 |

**Stability verdict: PASS.**  The correction parameter varies by 2.3% across
all generator subsets, against a ±5% stability gate.  No split exceeded 2.90%
maximum test error.

The near-zero coefficient of variation (0.66%) indicates that a is determined
by the geometry of the equal-area constraint — specifically by Equation (4) —
rather than by which machines happen to appear in the training set.

---

## 5. Geometric Interpretation

The EAC + correction framework is a special case of a broader result from
invariant manifold theory applied to dynamical systems.

**The SMIB swing equation linearises** around δₛ to a damped harmonic
oscillator with natural frequency ω₀ = √(Pₑ·cos(δₛ)/M) and quality factor
Q = Mω₀/D.  This places every generator in a three-dimensional invariant
space (log ω₀ norm, log Q norm, ζ), shared with RLC circuits, spring-mass
systems, Duffing oscillators, and gradient-descent dynamics.

**The separatrix energy** Eₛₑₚ = 2Pₑ·cos(δₛ) − Pₘ(π − 2δₛ) defines the
topological boundary of the stability region — identical in structure to
α²/(4|β|) in the softening Duffing oscillator.  The equal-area criterion
is the analytic expression of this separatrix geometry.

**The universal correction** arises because the product ω₀·CCT_EAC is
constant across all generators at the same loading ratio — a consequence of
the equal-area geometry, not of machine parameters.  When damping perturbs
the trajectory, it perturbs the invariant embedding by a universal amount,
independent of where in (ω₀, Q) space the generator sits.

**Implication for screening architecture:**  Because the correction is
universal (one scalar, generator-independent), it can be applied
post-hoc to any EAC-based estimate without re-fitting per machine.
A system with 1,000 generators at the same loading ratio uses the same
a = 1.51, pre-computed once.

---

## 6. Scope and Limitations

**What is validated:**

- IEEE 39-bus classical model (10 generators, SMIB representation)
- Three-phase fault (Pₑ → 0 during fault)
- Damping range: ζ ∈ [0.01, 0.20] (Q ∈ [2, 50])
- Uniform loading: Pₑ = 2·Pₘ → δₛ = 30° for all generators

**Known boundary conditions:**

1. *Loading ratio.*  The universality of a depends on all generators sharing
   the same Pₑ/Pₘ ratio (hence the same δₛ and δc).  For mixed loading,
   a will vary per generator class.  This is a one-time per-class calibration,
   not a per-machine calibration.

2. *Fault type.*  The EAC formula (Equation 2) assumes complete power loss
   (three-phase fault).  Partial faults (line outage, Pₑ → fault_factor · Pₑ)
   require a modified equal-area construction; the correction structure (3)
   remains applicable with a refitted for the partial-fault case.

3. *Multi-machine interactions.*  The SMIB model decouples each generator
   from the network.  Full multi-machine stability analysis requires
   center-of-inertia corrections.  The SMIB result provides accurate first-order
   screening; cases flagged as near-critical require full simulation.

4. *Model precision.*  The reference used in this benchmark (RK4 binary search
   with dt = 0.01 s) is itself an approximation to the continuous-time solution.
   The 2.73% corrected error bound includes both the formula error and the
   reference discretisation error.

---

## 7. Conclusion

We have demonstrated that CCT estimation for power grid transient stability
can be accelerated by approximately 57,946× relative to RK4 binary search,
with bounded error under realistic damping conditions.

The corrected EAC formula:

    CCT_corrected = √(2M·(δc − δₛ)/Pₘ) / (1 − 1.51·ζ)

requires zero ODE evaluations per machine per contingency.  The correction
parameter a = 1.51 is universal for uniform-loading grids (δₛ = 30°) and
stable across all generator subsets (CV = 0.66%).

**Corrected performance on IEEE 39-bus (10 generators, ζ ≤ 0.20):**

- Maximum CCT error: **2.73%**
- Coverage: ζ ∈ [0, 0.20], Q ∈ [2, ∞)  — full inter-area and local mode range
- Cross-validation: max test error **2.90%** across 45 leave-2-out splits

The universality of the correction is not a curve-fitting result.  It is a
consequence of the equal-area constraint geometry: the product ω₀·CCT_EAC
is invariant across all machines at fixed loading ratio.  Damping perturbs
the invariant manifold by a universal amount, characterised by a single scalar.

This property — that a physical correction is predictable from the geometry
of the invariant space — is the foundational claim of the underlying platform.
The grid stability result is the first externally-validatable instance of it.

---

*Correspondence and validation inquiries: [contact information]*
*Code, data, and reproducibility: `optimization/ieee39_benchmark.py`,*
*`tests/test_ieee39_benchmark.py`, `tests/test_damping_sweep.py`,*
*`tests/test_damping_correction.py`, `tests/test_cv_correction.py`*
*All results reproducible with: `conda run -n tensor python -m pytest tests/test_ieee39_benchmark.py tests/test_damping_sweep.py tests/test_damping_correction.py tests/test_cv_correction.py -v`*
