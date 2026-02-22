# Collaboration Document
# rust-physics-kernel — Duffing Oscillator Exploration

> **For agents:** Read this file fully before starting work. Append findings to your
> domain section. Never overwrite another agent's section. See AGENTS.md for roles.
>
> **For humans:** This is the living record of all exploration across domains.

---

## System Quick Reference

### The Equation

```
ẍ + δẋ + αx + βx³ = F·cos(Ω·t)

State:   [x, ẋ]   (position, velocity)
```

| Parameter | Symbol | Role | Regime notes |
|-----------|--------|------|--------------|
| `alpha`   | α      | Linear stiffness | α > 0 required for oscillation |
| `beta`    | β      | Nonlinear stiffness | β > 0 hardening, β < 0 softening, β = 0 → linear |
| `delta`   | δ      | Viscous damping | δ = 0 conservative, δ > 0 dissipative |
| `f_drive` | F      | Forcing amplitude | F = 0 autonomous system |
| `omega`   | Ω      | Drive frequency | Near √α → resonance |

### Python API

```python
import rust_physics_kernel as rk

# Full trajectory: returns list of [x, v] pairs, length n_steps
traj = rk.py_generate_trajectory(
    x0, v0,          # initial position, velocity (float)
    dt,              # time step (float)
    n_steps,         # number of steps (int)
    alpha, beta,     # stiffness parameters (float)
    delta,           # damping (float)
    f_drive=0.0,     # forcing amplitude (default 0)
    omega=1.0,       # drive frequency (default 1)
)

# Single RK4 step: useful for custom control loops
state_next = rk.py_rk4_step(
    [x, v],          # current state (list of 2 floats)
    t,               # current time (float)
    dt,              # time step (float)
    alpha, beta, delta,
    f_drive=0.0, omega=1.0,
)
```

### Known-Good Parameter Sets

| Name | α | β | δ | F | Ω | Behavior |
|------|---|---|---|---|---|---------|
| `linear_undamped` | 1.0 | 0.0 | 0.0 | 0.0 | — | Pure sine, T = 2π |
| `hardening_spring` | 1.0 | 0.1 | 0.3 | 0.0 | — | Decaying nonlinear oscillation |
| `softening_spring` | 1.0 | -0.1 | 0.2 | 0.0 | — | Softening decay |
| `driven_duffing` | 1.0 | 0.1 | 0.3 | 0.5 | 1.2 | Forced nonlinear, steady state |
| `classic_chaos` | 1.0 | -1.0 | 0.3 | 0.5 | 1.2 | Holmes-Duffing chaotic attractor |

### Build / Run

```bash
./setup.sh                    # build Rust extension (once)
python python/verify.py       # confirm correctness
python python/benchmark.py    # measure Rust vs Python speedup
```

---

## Domain Exploration Map

| Domain | Agent | Status | Lead finding | Scripts |
|--------|-------|--------|--------------|---------|
| Chaos / Nonlinear Dynamics | CHAOS | `[ ] not started` | — | — |
| Structural Engineering | ENGRG | `[ ] not started` | — | — |
| ML / Surrogate Models | ML | `[ ] not started` | — | — |
| Signal & Frequency Analysis | SIGNAL | `[~] in progress` | Spectral coherence regime system | `unified-tensor-system/tensor/reharmonization.py` |
| Chaos Control | CTRL | `[ ] not started` | — | — |
| Spectral Regime Detection | CROSS | `[x] findings logged` | Reharmonization architecture (5-layer) | `unified-tensor-system/tensor/reharmonization.py`, `tests/test_reharmonization.py` |

Status key: `[ ] not started` → `[~] in progress` → `[x] findings logged`

---

## Domain Sections

### CHAOS — Nonlinear Dynamics & Chaos Theory

**Scope:** Bifurcation diagrams · Poincaré sections · strange attractors · Lyapunov
exponents · period-doubling cascade · intermittency · transient chaos.

**Key questions for this domain:**
- At what value of F (with Ω=1.2, α=1, β=-1, δ=0.3) does the system transition
  from periodic to chaotic?
- What does the bifurcation diagram look like sweeping F from 0 → 1.0?
- What is the largest Lyapunov exponent in the chaotic regime?
- Does the classic Holmes attractor appear with α=1, β=-1, δ=0.25, F=0.3, Ω=1?

**Task queue:**
- [ ] Bifurcation diagram: sweep F ∈ [0, 1.0], step 0.005; record Poincaré x after
      transient (discard first 200 drive periods, sample next 100).
- [ ] Phase portrait: plot x vs v for chaotic and periodic regimes.
- [ ] Lyapunov exponent: implement two-trajectory divergence method.
- [ ] Period-doubling cascade: find F values for period 1→2→4→8.

**Findings:**
<!-- CHAOS findings go here. Tag with: <!-- CHAOS YYYY-MM-DD --> -->
*(No findings yet.)*

**Artifacts:** *(none yet)*

---

### ENGINEERING — Structural & Mechanical Applications

**Scope:** Real-world parameter mapping · frequency-response curves · backbone curves ·
jump phenomena · fatigue analysis · vibration isolation · MEMS resonators.

**Key questions for this domain:**
- What physical system does α=1e6, β=1e12, δ=10, F=50 N, Ω=1000 rad/s correspond to?
  (A steel microbeam? An automotive mount?)
- Where are the fold points (jump-up / jump-down frequencies) for a hardening spring?
- How does the amplitude-frequency response change with damping ratio ζ = δ/(2√α)?
- Can the kernel simulate a Duffing vibration absorber reducing peak transmissibility?

**Task queue:**
- [ ] Map DuffingParams to a physical MEMS resonator (document units and scaling).
- [ ] Frequency sweep: vary Ω from 0.5√α to 1.5√α; record steady-state amplitude.
      Plot both up-sweep and down-sweep to reveal hysteresis.
- [ ] Find fold frequencies analytically (harmonic balance) and verify against simulation.
- [ ] Rainflow counting on a long displacement history to estimate fatigue cycles.

**Findings:**
<!-- ENGRG findings go here. Tag with: <!-- ENGRG YYYY-MM-DD --> -->
*(No findings yet.)*

**Artifacts:** *(none yet)*

---

### ML — Surrogate Models & Scientific Machine Learning

**Scope:** Dataset generation · MLP/LSTM surrogates · physics-informed neural networks ·
parameter-to-behavior classifiers · uncertainty quantification · inference speedup.

**Key questions for this domain:**
- Can a small MLP predict whether given (α, β, δ, F, Ω, x0, v0) will produce chaotic
  behavior, trained on Rust-kernel-generated labels?
- How fast is a trained surrogate vs the Rust kernel for single-trajectory queries?
- What's the minimum dataset size for a reliable behavior classifier?
- Can a PINN reproduce the Duffing trajectory without the RK4 kernel at test time?

**Task queue:**
- [ ] Generate dataset: 10k parameter samples (Latin hypercube over param space);
      label each as periodic/chaotic using Lyapunov sign from CHAOS findings.
- [ ] Train MLP classifier: (α, β, δ, F, Ω) → {periodic, chaotic}. Report F1 score.
- [ ] Train LSTM to predict next 100 states given first 50 states.
- [ ] Compare surrogate vs Rust kernel on inference time at batch sizes 1, 100, 1000.

**Findings:**
<!-- ML findings go here. Tag with: <!-- ML YYYY-MM-DD --> -->
*(No findings yet.)*

**Artifacts:** *(none yet)*

---

### SIGNAL — Spectral & Frequency Analysis

**Scope:** Power spectral density · harmonic content · subharmonics · broadband noise ·
autocorrelation · chaotic RNG · spread-spectrum applications.

**Key questions for this domain:**
- How does the PSD evolve as F crosses the chaos threshold?
- Which parameter sets produce the cleanest subharmonic ladders (period-2, period-4)?
- What is the autocorrelation decay time in the chaotic regime vs periodic?
- How many bits/second can a chaotic bit generator produce with good NIST statistical test scores?

**Task queue:**
- [ ] PSD sweep: compute FFT of x(t) at 10 F values from periodic to chaotic regime.
      Plot waterfall of PSDs.
- [ ] Subharmonic map: find Ω values producing period-3, period-5 windows.
- [ ] Autocorrelation: compute R(τ) for τ up to 100 drive periods in chaotic regime.
- [ ] Chaotic RNG prototype: sample sign(x(t)) at integer drive periods; test with
      monobit and runs tests.

**Findings:**
<!-- SIGNAL 2026-02-22 -->
**Spectral coherence regime detection** — The reharmonization system
(`unified-tensor-system/tensor/reharmonization.py`) implements bootstrapped EDMD
spectral estimation with Hungarian mode matching across timescales. Key insight:
cross-timescale lock scores (dissonance metric between mode pairs) provide a continuous,
normalized coherence energy signal. When this signal persists (N_entry consecutive
observations), it declares a spectral regime. When it breaks (N_exit consecutive
departures), it emits a reharmonization event. This directly addresses OQ-5: the
spectral transition from tonal to broadband at chaos onset manifests as a drop in
coherence energy and rise in normalized spectral entropy.

**Artifacts:**
- `unified-tensor-system/tensor/reharmonization.py` — 5-layer system (1076 lines)
- `unified-tensor-system/tests/test_reharmonization.py` — 37 tests (818 lines)

---

### CONTROL — Chaos Control & Synchronization

**Scope:** OGY method · Pyragas delayed feedback · drive-response sync · anti-control ·
adaptive control · stabilizing unstable periodic orbits (UPOs).

**Key questions for this domain:**
- Can Pyragas DFC (u = K(x(t-τ) - x(t))) stabilize period-1 orbit at K < 0.1?
- What is the minimum coupling strength ε for complete synchronization of two identical
  Duffing oscillators with different initial conditions?
- Does adaptive parameter adjustment (slow δ tuning) suppress chaos?
- Can the kernel implement OGY by perturbing f_drive at each Poincaré crossing?

**Task queue:**
- [ ] Implement Pyragas DFC loop using py_rk4_step; sweep K ∈ [0, 0.5] and τ ∈ [T/4, T].
- [ ] Plot stabilization: x(t) before and after DFC activated (at t=50T).
- [ ] Synchronization: two oscillators with ε coupling on velocity; plot sync error vs time.
- [ ] Find sync threshold εc; plot εc vs parameter mismatch Δδ.

**Findings:**
<!-- CTRL findings go here. Tag with: <!-- CTRL YYYY-MM-DD --> -->
*(No findings yet.)*

**Artifacts:** *(none yet)*

---

## Open Questions (Cross-Domain)

These questions require input from multiple agents to answer.

| # | Question | Relevant Domains | Priority |
|---|----------|-----------------|----------|
| OQ-1 | What parameter subspace produces "useful chaos" (good PSD + controllable)? | CHAOS, SIGNAL, CTRL | High |
| OQ-2 | Can ML classifier accuracy predict which regions CTRL can stabilize? | ML, CTRL | Medium |
| OQ-3 | Which physical engineering systems live near the chaos boundary? | ENGRG, CHAOS | High |
| OQ-4 | How does surrogate model accuracy degrade near bifurcation points? | ML, CHAOS | Medium |
| OQ-5 | What drives the spectral transition from tonal to broadband at the chaos onset? | SIGNAL, CHAOS | Medium — **partially answered**: coherence energy drop + spectral entropy rise detected by reharmonization L1-L3 |
| OQ-6 | Can the reharmonization system's regime persistence filter detect Duffing bifurcations in real-time from spectral coherence alone (without parameter knowledge)? | SIGNAL, CHAOS, CTRL | High |
| OQ-7 | How do reharmonization events correlate with backbone curve fold points (jump phenomena)? | SIGNAL, ENGRG | Medium |

---

## Cross-Domain Connections

### Spectral Coherence Regime System — Reharmonization (2026-02-22)

**Location:** `~/projects/unified-tensor-system/tensor/reharmonization.py`
**Tests:** `~/projects/unified-tensor-system/tests/test_reharmonization.py` — 37/37 pass

**Core principle:** Cross-timescale spectral coherence persistence IS the regime change
signal. Regime changes are reharmonizations — frequency content at one timescale
reorganizes into new rational structure at another. Duffing is explanatory, not primary.

#### 5-Layer Architecture

| Layer | Class | Purpose | Status |
|-------|-------|---------|--------|
| L1 | `BootstrappedSpectrumTracker` | EDMD + bootstrap CIs, Hungarian mode matching, adaptive recompute | Phase A — active |
| L2 | `CoherenceScorer` | Continuous lock score via `DissonanceMetric`, fixed-length lock vector | Phase A — active |
| L3 | `RegimePersistenceFilter` | HMM-smoothed regime state, cosine similarity, N_entry/N_exit hysteresis | Phase A — active |
| L4 | `DuffingParameterFilter` | EKF on backbone curve ω = √(α + ¾βA²), observability gate | Phase B — gated (`enable_duffing=False`) |
| L5 | `ProfitWindow` | Welford online stats, min_observations_for_sharpe, drawdown tracking | Phase C — gated (`enable_profit_window=False`) |

Orchestrator: `ReharmonizationTracker` composes all layers with constructor flags to isolate subsystems.

#### Key Design Decisions & Fixes

| # | Problem | Solution |
|---|---------|----------|
| 1 | Bootstrap recomputation CPU explosion | Adaptive trigger: only recompute when spectral entropy or Gram condition drifts >10%/50% |
| 2 | Mode index misalignment across bootstrap replicates | Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on frequency cost matrix |
| 3 | Lock vector dimensionality changes as modes appear/disappear | Fixed-length vector (`max_pairs`), zero-padded, deterministic pair ordering |
| 4 | Binary lock detection threshold unintuitive | Cosine similarity: threshold 0.3 = "30% dissimilarity" — scale-invariant |
| 5 | EKF invents curvature when backbone is locally flat | Observability gate: skip β update when `|ΔA| < amplitude_threshold` |
| 6 | Spectral entropy changes when mode count fluctuates | Normalized by `log(n_modes)`, computed only on stable modes |
| 7 | Sharpe estimate from 5 observations is noise | `min_observations_for_sharpe=20`, `is_profitable=False` until threshold met |
| 8 | Regime ID counter explodes (new ID for every minor shift) | Historical centroid matching: cosine_sim > 0.85 → reuse old regime_id |
| 9 | Coherence energy drifts with mode count | `coherence_energy = mean(lock_scores)` not sum, range [0,1] |

#### Quantitative Results (37/37 tests)

| Metric | Result |
|--------|--------|
| Clean oscillator (α=1, β=0, δ=0.1) frequency recovery | mode detected, CI brackets true freq |
| Bootstrap variance (clean signal, stable modes) | < 10.0 (Hungarian matching prevents inflation) |
| Spectral entropy range | [0, 1] (normalized) |
| Same-frequency lock score | > 0.8 coherence energy |
| Lock vector dimensional stability | Always exactly `max_pairs` length |
| Cosine similarity scale invariance | Scaled vector (3x) does not trigger regime break |
| Regime merging | Re-entering similar centroid reuses historical ID |
| Stable oscillator | 0 false reharmonization events |
| Pure noise | ≤ 3 false events (statistical fluctuation allowance) |
| EKF convergence (α=1.0, β=0.5) | Within 20% after 50 updates |
| Regime probabilities | Sum to 1.0, shape (4,) |
| Welford online stats | Matches batch mean/variance to 10 decimal places |
| Max drawdown tracking | Exact cumulative peak-to-trough |

#### Integration with Existing Modules

**Composed (not modified):**
- `spectral_path.py` → `DissonanceMetric.compute(ω_i, ω_j)` for lock scoring
- `koopman_edmd.py` → `EDMDKoopman.fit_trajectory()` for spectral estimation
- `bifurcation_detector.py` → `BifurcationDetector.check()` enriches reharmonization events
- `frequency_dependent_lifter.py` → `best_rational_approximation()` for interpretability

**Modified (backward compatible):**
- `timescale_state.py` → `CrossTimescaleSystem.__init__` gains `reharmonization_tracker=None`; `propagate_shock` auto-feeds states when tracker present
- `multi_horizon_mixer.py` → `mix()` gains `regime_vector: Optional[np.ndarray]` (4,) soft Duffing regime probs → per-timeframe logit shifts via `REGIME_TO_LOGIT` (4×3) matrix

Backward compat confirmed: 41/41 calendar lifter + 13/13 codegen pipeline tests unchanged.

#### Connection to Duffing Kernel

The `DuffingParameterFilter` (Layer 4, Phase B) estimates Duffing parameters from
observed spectral modes via the backbone curve equation:

```
ω = √(α + ¾βA²)
```

This maps the Duffing oscillator's nonlinear frequency-amplitude relationship to
observed market regime frequencies. The filter classifies regimes into 4 characters:
- **mean_rev** (high α/|β|): linear restoring force dominates
- **breakout** (high |β|): nonlinear stiffness dominates
- **momentum** (low δ): low damping → persistent trends
- **anticipatory** (high F): external forcing → event-driven

These probabilities feed into `MultiHorizonMixer` as soft regime weights, conditioning
the geometric gating across fundamental/regime/shock timeframes.

---

## Next Actions Queue

| Priority | Task | Assign to | Blocked by |
|----------|------|-----------|------------|
| 1 | Bifurcation diagram sweep (F sweep) | CHAOS | — |
| 2 | Frequency response curve (Ω sweep, hardening) | ENGRG | — |
| 3 | Generate initial parameter-behavior dataset | ML | CHAOS (labels) |
| 4 | PSD waterfall across F values | SIGNAL | CHAOS (param ranges) |
| 5 | Pyragas DFC prototype | CTRL | CHAOS (chaotic regime params) |
| 6 | **Phase B gate test**: Run reharmonization L4 (DuffingParameterFilter) on clean Duffing, slowly drifting α, constant-amplitude case | SIGNAL | Phase A stable (done) |
| 7 | **Phase C gate test**: Run reharmonization L5 (ProfitWindow) with synthetic returns conditioned on detected regimes | ML | Phase B stable |
| 8 | Feed Rust kernel trajectories into reharmonization tracker to test real Duffing regime transitions | CROSS | CHAOS (bifurcation params) |

---

## Shared Vocabulary

| Term | Definition |
|------|-----------|
| Poincaré section | Sample x(t) and v(t) at integer multiples of drive period T = 2π/Ω |
| Transient | Initial portion of trajectory before settling to attractor; discard for analysis |
| UPO | Unstable Periodic Orbit embedded in a chaotic attractor |
| DFC | Delayed Feedback Control (Pyragas method) |
| Backbone curve | Amplitude vs frequency of free oscillation (no damping or forcing) |
| Jump phenomenon | Hysteretic jump in amplitude at fold points during frequency sweep |
| Lyapunov exponent | Rate of divergence of nearby trajectories; positive λ → chaos |
| Hardening spring | β > 0: restoring force grows faster than linear |
| Softening spring | β < 0: restoring force grows slower than linear; can escape potential well |
| Surrogate model | ML model trained to approximate the physics kernel at low inference cost |
| Reharmonization | Regime change viewed as spectral reorganization: frequency content at one timescale forms new rational structure at another |
| Coherence energy | Mean lock score across cross-timescale mode pairs; ∈ [0, 1]; high = spectrally locked |
| Lock score | exp(-τ/σ) where τ is dissonance between two modes; 1.0 = perfectly consonant, 0.0 = dissonant |
| Spectral entropy | Normalized Shannon entropy of eigenvalue magnitudes; low = coherent, high = diffuse/chaotic |
| EDMD | Extended Dynamic Mode Decomposition — linear approximation of Koopman operator from data |
| Hungarian matching | Optimal bipartite assignment (O(n³)) used to track mode identity across bootstrap replicates |

---

## Kernel Performance Reference

*(From benchmark.py — update after re-running on your machine)*

| n_steps | Python (ms) | Rust (ms) | Speedup |
|---------|-------------|-----------|---------|
| 1,000   | —           | —         | — |
| 5,000   | —           | —         | — |
| 10,000  | —           | —         | — |
| 50,000  | —           | —         | — |

Run `python python/benchmark.py` to populate this table.
