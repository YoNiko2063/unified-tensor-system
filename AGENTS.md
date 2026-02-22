# AGENTS.md — Project Knowledge File
# Unified Tensor System

> Read this before starting any work session. It is the single authoritative
> source of project state for all agents working on this codebase.

---

## What This Project Is

A domain-agnostic stability assurance platform. Core idea: any physical or
computational system expressible as **C·ẋ + G·x + h(x) = u(t)** — power
grids, RLC circuits, spring-mass systems, Duffing oscillators, synthesized
Rust programs — can be analyzed by the same three-layer spectral pipeline.

The platform has been built, tested (2,294 tests passing), patented (5
provisional filings), and deployed as a FastAPI + React web interface.

---

## Permanent Constraints

- **NO external API calls for intelligence** — local models only
- **Active conda env**: `tensor`
- **Test command**: `python -m pytest tests/ platform/backend/tests/ -q`
- **sys.path pattern**: every test file needs `sys.path.insert(0, _ROOT)` — no root conftest
- **ecemath**: `sys.path.insert(0, 'ecemath/')` then `from src.backend... import ...`
- **ecemath invariant**: NEVER mutate `ir.params` without save/restore pattern
- **SymPy**: use `if expr is not None:` NOT `if expr:` (Relational can't bool-test)

---

## Three-Layer Architecture

```
Layer 1 — Regime Classification (tensor/*)
  Input: system parameters or trajectory data
  Process: linearize → EDMD → spectral invariant coords
  Trust gate: ε_rec < η_max AND γ_gap > γ_min
  Output: regime ∈ {LCA, transitional, chaotic}, trust T, coords (s₁,s₂,s₃)

Layer 2 — Structural Synthesis (codegen/*)
  Input: intent spec + Layer 1 outputs
  Process: BorrowVector B → energy E = w·B → compilation manifold
  Gate: E < D_sep_effective before attempting compile
  Output: compiled Rust program, hardware affinity label ∈ {GPU, CPU, general}

Layer 3 — Runtime Validation (tensor/*, optimization/*)
  Input: compiled candidate
  Process: execute → K from trajectories → R = min(R₁,R₂,R₃,R₄) → Monte Carlo β
  Certificate: CERT = C₁∧C₂∧C₃∧C₄ where C₄ = cross-layer regime consistency
  Output: unified stability certificate or feedback signal

Cross-layer signals (6 total):
  S_{1→2}:  spectral coords → Layer 2 (template pre-filtering)
  S_{1→2b}: regime label → Layer 2 AND Layer 3 (sets D_sep_eff, R_min_eff)
  S_{2→3}:  BorrowVector B + energy E → Layer 3 (traceability)
  S_{3→2}:  R < R_min_eff → Layer 2 (tighten D_sep, regenerate)
  S_{3→1}:  persistent failure → Layer 1 (reclassify with runtime data)
  S_{2→1}:  fires ONLY at CERT=TRUE (library update to Layer 1)
```

---

## Key Constants and Formulas

```
D_SEP = 0.43          BorrowVector energy boundary (compilation safety)
WEIGHTS = [0.25, 0.18, 0.15, 0.17, 0.15, 0.10]   (B₁..B₆ weights)
E = w·B               scalar energy metric
D_sep_effective = D_sep - delta_tighten·k_L3 - delta_cautious   (Eq. A)

Spectral invariant coords:
  s₁ = log(ω₀)   where ω₀ = |Im(λ_dominant)|   (log oscillation frequency)
  s₂ = log(Q)    where Q = ω₀/(2|Re(λ_dominant)|)  (log quality factor)
  s₃ = ζ         = |Re(λ_dominant)|/ω₀           (damping ratio)

Runtime robustness components:
  R₁ = -max_i(Re(λᵢ))                 stability margin
  R₂ = |λ₁|/|λ₂| - 1                  spectral separation
  R₃ = 1 - ε_rec/η_max                reconstruction fidelity
  R₄ = 1 - max_t(V(x(t)))/E_bound     Lyapunov energy headroom
  R  = min(R₁, R₂, R₃, R₄)

Regime classification (two-step):
  Step 1: if ε_rec >= η_max → Chaotic (no further analysis)
           if ε_rec < η_max AND γ_gap <= γ_min → escalate to simulation
           else → trusted, proceed to Step 2
  Step 2: max Re(λᵢ) < 0 → LCA;  max Re(λᵢ) >= 0 → Transitional

Hardware affinity from energy:
  E < 0.08  → SIMD/GPU   (pure functional, low mutable density)
  E < 0.30  → CPU cache  (shared immutable refs, low branching)
  E >= 0.30 → general    (complex ownership, defer to profiling)

Koopman trust gate: ε_rec < η_max=0.05 AND γ_gap > γ_min=1.2
EDMD fundamental frequency: sort by SMALLEST |Im(log λ)| (NOT largest)
Near-separatrix override: E₀/E_sep > 0.85 → ω₀_eff = ω_floor
```

---

## Module Map

```
tensor/
  integrated_hdv.py          HDV space (10k-dim), structural_encode(), overlap()
  calendar_regime.py         5-channel event encoder (earnings/fed/options/rebalance/holiday)
  frequency_dependent_lifter.py  Φ_{S→M}(t) = Σ_k A_k·φ_k(t), Arnold tongue resonance
  timescale_state.py         L/M/S state spaces + CrossTimescaleSystem lifting
  multi_horizon_mixer.py     (5,3) calendar modulation matrix + resonance vol penalty
  semantic_geometry.py       _TextEDMD observer state (use this, NOT EDMDKoopman)
  epistemic_geometry.py      Token trajectory → velocity/curvature/FFT validity score
  source_spectral_profile.py Per-source reliability model + trust ellipsoid
  semantic_flow_encoder.py   Articles/market windows/code intents → unified HDV
  intent_projector.py        HDV direction → IntentSpec → template selection
  domain_canonicalizer.py    DissonanceMetric τ, library matching (K=10, tau_threshold=0.5)
  eigenspace_mapper.py       system_factory(θ) → MapResult; scan_grid; scan_random
  parameter_space_walker.py  pure-numpy Adam MLP; predict_step(θ, eigvals, regime)
  scrapling_ingestion.py     Scrapling-first fetcher with requests fallback
  financial_ingestion.py     Source router by domain

codegen/
  pipeline.py                CodeGenPipeline: pre-gate → render → post-gate
  template_registry.py       24 Rust templates across 7 domain files
  intent_spec.py             IntentSpec, BorrowProfile, WEIGHTS, e_borrow()
  borrow_predictor.py        Wraps code_gen_experiment.py classifier

optimization/
  circuit_optimizer.py       CircuitSpec → analytic guess + Nelder-Mead → ParetoResult

platform/
  backend/main.py            FastAPI, 6 routers: /api/v1/{regime,calendar,codegen,hdv,physics,circuit}
  frontend/                  React/Tailwind, 6 tabs

ecemath/ (submodule)
  src/domains/trading.py     TradingSystem: C·ṗ+G·p+h(p)=u(t), AgentClass
  src/domains/circuits.py    RLC/RC/LC via MNA

tools/
  provisional_to_docx.py     Convert ip/*.txt → ~/Downloads/*.docx

ip/
  provisional_2_regime_classification.txt
  provisional_3_code_synthesis.txt
  provisional_4_runtime_validation.txt
  provisional_5_platform_integration.txt

tensor/data/
  function_library.json      575 entries, 53 DEQs, 282 raw_latex
  fed_dates.json             FOMC 2024-2026 + NYSE holidays
  hdv_state.json             HDV vectors
  universals.json            Cross-domain patterns
  ingestion_journal.json     92 papers journaled
  ingested/                  359 papers (RSS-fetched)
```

---

## Common Instantiation Patterns

```python
# Production HDV system
IntegratedHDVSystem(hdv_dim=10000, n_modes=150)

# Test HDV system
IntegratedHDVSystem(hdv_dim=1000, n_modes=10, embed_dim=64)

# EigenspaceMapper
EigenspaceMapper(system_factory, atlas, n_states=None, n_samples=30, sample_radius=0.01)
# system_factory(theta) → DynamicalSystem (has .rhs(t,x), .n_states()) OR plain f(x)->ẋ

# DomainCanonicalizer
DomainCanonicalizer(atlas, K=10, tau_threshold=0.5)
# recognize(eigvals) → Optional[CanonicalMatch] with fields: patch_id, domain, interval_ratio, confidence

# ParameterSpaceWalker
ParameterSpaceWalker(theta_keys, param_bounds, hidden=128)
# predict_step(theta, eigvals, regime) → Δθ

# ArxivPDFSourceParser — always rate_limit_seconds=0 in tests
ArxivPDFSourceParser(rate_limit_seconds=0)

# CodeGenPipeline
CodeGenPipeline.generate(IntentSpec) → compile-verified Rust with pyo3 bindings
```

---

## Multi-Horizon Financial Structure

```
Three timeframes: x(L) fundamentals / x(M) regime-technical / x(S) news-shock
Calendar Φ_{S→M}: earnings week 2.15× mid-quarter delta; Fed/earnings 2:1 Arnold tongue
Multi-horizon mixer: (5,3) calendar modulation — earnings boosts w_S, holiday boosts w_L
Source routing: arxiv/EDGAR → Fetcher; reuters/FT → StealthyFetcher; SA/MF → DynamicFetcher

compute_overlap_similarity() returns 0.0 if same domain for both vectors
softmax(x+c) = softmax(x) — mixer calendar modulation MUST be (5,3) matrix, not scalar
```

---

## Koopman / EDMD Rules

```
Driven systems (limit cycles): reconstruction-only trust gate (spectral gap = 0 is OK)
EAC correction parameter: a = 1.51 (fitted), max CCT error < 2.73% for ζ ≤ 0.20
Power domain: C·ẍ + G·ẋ + h(x) = u(t) via ecemath TradingSystem / circuits.py
LCA classification: Re(λ) < 0 for all eigenvalues (magnitude gap = 0 for conjugate pair)
```

---

## Platform Start

```bash
cd platform/backend && bash start.sh          # FastAPI on :8000
cd platform/frontend && npm run dev           # React on :5173
```

Defensive imports: all heavy model imports guarded with try/except, mock fallback always available.

---

## Intellectual Property Status

5 provisional patent applications in `ip/`. All filed 2026-02-21/22.
DOCX versions: `python3 tools/provisional_to_docx.py` → `~/Downloads/`

Provisional #1 (CCT/power systems, EAC + damping correction) filed separately.
Provisionals #2–5 cover the full three-layer platform.
Non-provisional target: single non-provisional claiming priority through all five.

---

## What Is NOT Done (from TODO.md)

- Patch classifier (PatchClassifier) with trust gates and hysteresis
- EDMD spectral tracking with continuous-spectrum detection
- Harmonic atlas / patch-graph navigation (Dijkstra/A*)
- Prediction-driven learning loop (entropy-guided concept selection)
- GPU/DEQ modeling (VeriGPU and tiny-gpu converters)
- ArXiv LaTeX equation parser (Task 6.4)
- GitHub capability extractor (Task 11.1)
- Cross-dimensional discovery (Task 9.5)
- G-code generator (Task 12.1)

See TODO.md for the full prioritized list with acceptance criteria.
