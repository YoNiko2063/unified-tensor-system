# Architecture

**Analysis Date:** 2026-02-22

## Pattern Overview

**Overall:** Unified multi-layer tensor framework with spectral-geometric operators across domains (circuits, finance, code generation, physics).

**Key Characteristics:**
- Four-level tensor stack: T ∈ R^(L x N x N x t) where L={0:market, 1:neural, 2:code, 3:hardware}
- Spectral operators (Koopman EDMD) for regime classification and nonlinear system acceleration
- Domain-generic eigenspace mapping with harmonic resonance detection
- Integrated HDV (hyperdimensional vector) system for cross-domain discovery
- Code generation via BorrowVector metric and Rust template matching
- Local-only inference (no external APIs) through FastAPI backend + React frontend

## Layers

**Layer 0 - Market/Signal Level:**
- Purpose: Financial system state tracking, regime detection, multi-horizon mixing
- Location: `tensor/calendar_regime.py`, `tensor/frequency_dependent_lifter.py`, `tensor/multi_horizon_mixer.py`
- Contains: Event encoders (5-channel: earnings/fed/options/rebalance/holiday), Arnold tongue resonance tracking, cross-timescale state lifting
- Depends on: ecemath MNA (Koopman trust gate), numpy/scipy
- Used by: Financial ingestion pipeline, trading system

**Layer 1 - Neural/SNN Level:**
- Purpose: Lifted semantic and epistemic state via forced nonlinear dynamical systems
- Location: `tensor/semantic_observer.py`, `tensor/epistemic_geometry.py`, `tensor/unified_network.py`
- Contains: Koopman-lifted observer (ẋ = A·x + g(x) + B·u), basis consolidation, semantic energy tracking
- Depends on: scipy Schur decomposition, torch (for neural embedding)
- Used by: Semantic geometry layer, HDV training

**Layer 2 - Code Level:**
- Purpose: Code structure vectorization, borrow pattern extraction, compilation prediction
- Location: `tensor/code_graph.py`, `codegen/borrow_predictor.py`, `codegen/structural_encoder.py`
- Contains: AST extraction (B1..B6 BorrowVector), intent specification, template matching
- Depends on: Rust AST parsing, sklearn logistic regression
- Used by: CodeGenPipeline for Rust synthesis

**Layer 3 - Hardware Topology (Future):**
- Purpose: Reserved for hardware-aware code optimization
- Location: Not yet implemented
- Contains: Placeholder for future GPU/SIMD targeting
- Depends on: Layer 2 code metrics

**Integration Layer - Unified HDV System:**
- Purpose: Cross-domain semantic bridging via 10k-dimensional hyperdimensional vectors
- Location: `tensor/integrated_hdv.py`, `tensor/function_basis.py`
- Contains: Sparse domain masks (which HDV dims per domain), learned neural embeddings, structural hash encoding
- Depends on: torch, numpy
- Used by: Cross-dimensional discovery, domain invariant matching

**Optimization/Evaluation Domains:**
- Purpose: Domain-specific eigenspace evaluation and parameter optimization
- Location: `optimization/` folder (RLC, spring-mass, power-grid, Duffing, GD)
- Contains: EigenvalueMapper, CircuitOptimizer, ParameterSpaceWalker, MonteCarloStabilityAnalyzer
- Depends on: ecemath (MNA, eigenvalue solvers)
- Used by: Benchmarking, regime acceleration

## Data Flow

**Code Generation Pipeline:**

1. User specifies `IntentSpec` (domain, operation, complexity_class, parameters)
2. `BorrowPredictor.from_intent()` → estimate BorrowVector (B1..B6) from semantic features
3. `CodeGenPipeline.pre_gate()` → check if E_borrow < D_SEP=0.43 (safety threshold)
4. `TemplateRegistry.best_match()` → select Rust template by domain + operation
5. `template.render(intent.parameters)` → generate Rust source code
6. `CodeGenPipeline.post_gate()` → AST extract actual BV, verify compilation
7. `FeedbackStore.record()` → log (BV, compile_result) for classifier training

**Regime Classification (Financial):**

1. Market time series feeds into `SemanticObserver` (forced ODE with Koopman lifting)
2. Calendar events trigger `frequency_dependent_lifter` (Φ_S→M with Arnold tongue flags)
3. Multi-horizon mixer combines L/M/S states with (5,3) calendar modulation matrix
4. Spectral eigenvalue analysis classifies regime: LCA (stable) vs nonabelian vs chaotic
5. Koopman trust gate enforces: EDMD reconstruction error < η_max AND spectral_gap > γ_min

**HDV Cross-Domain Discovery:**

1. Document/equation/code input → hash-encoded into sparse HDV dimension sets
2. Per-domain masks define which 10k HDV dimensions each domain occupies
3. Overlap dimensions (used by 2+ domains) capture universals
4. `compute_overlap_similarity()` measures cross-domain semantic closeness (0 if same domain)
5. Function library parser maps equations → basis → HDV indices

**Eigenspace Mapping:**

1. `EigenspaceMapper.map_point(theta, domain)` → evaluate system at parameter theta
2. Extract eigenvalues and compute harmonic signature (ratios, dissonance metric)
3. `DomainCanonicalizer.recognize()` → match eigenvalues to known domain/interval (e.g., "3:2")
4. `ParameterSpaceWalker` trains on transition data to predict Δθ toward harmonic targets
5. Jumper (energy-based) navigates across rational ridges; Walker performs local descent

## Key Abstractions

**UnifiedTensor (T ∈ R^(L x N x N x t)):**
- Purpose: Sparse multi-level matrix tensor storing MNA systems per domain level
- Examples: `tensor/core.py` (lines 46-100)
- Pattern: Stores (G, C) matrices indexed by [level][time_idx]; lazy computation of coarsening/lifting

**IntentSpec:**
- Purpose: Semantic specification for code generation
- Examples: `codegen/intent_spec.py`
- Pattern: Domain + operation + expected BorrowProfile → BorrowVector energy metric

**BorrowVector (B1..B6):**
- Purpose: 6-dimensional code ownership pattern metric for Rust compilation
- Examples: `codegen/borrow_predictor.py`
- Pattern: Extract from AST via binary at ~/.cargo/projects/rust-borrow-extractor, predict via logistic regression

**SemanticObserver (ẋ = A·x + g(x) + B·u):**
- Purpose: Forced nonlinear ODE for tracking lifted semantic state with Lyapunov energy cap
- Examples: `tensor/semantic_observer.py` (lines 1-100)
- Pattern: Real Schur decomposition for spectral truncation, tanh saturation for boundedness

**HarmonicSignature:**
- Purpose: Captures eigenvalue ratio structure for domain recognition
- Examples: `ecemath/src/core/sparse_solver.py`
- Pattern: Compute smallest |Im(log λ)| → ratio tuple (e.g., "3:2") → dissonance metric τ

**EigenspaceMapper/ParameterSpaceWalker:**
- Purpose: Navigate parameter space toward harmonic targets via spectral matching
- Examples: `tensor/eigenspace_mapper.py`, `tensor/eigen_walker.py`
- Pattern: Walker encodes ∂λ/∂θ gradients; Jumper fires on energy stagnation to escape local basins

## Entry Points

**Web Platform:**
- Location: `platform/backend/main.py`
- Triggers: Server startup (`python -m uvicorn platform.backend.main:app --reload`)
- Responsibilities: FastAPI server at :8000, 6 routers (/regime, /calendar, /codegen, /hdv, /physics, /circuit)

**Code Generation Experiment:**
- Location: `run_experiment.py`
- Triggers: Command-line execution
- Responsibilities: 2000-variant synthesis, adaptive constrained loop, spectral geometry analysis

**Benchmark Harnesses:**
- Location: `benchmarks/` folder
- Triggers: `python benchmarks/{walker_vs_random,transfer_eigen_walker_v2,rational_atlas}.py`
- Responsibilities: Domain-specific evaluation (RLC, MSD, TwoMassSpring, etc.)

**Autonomous Training Loop:**
- Location: `tensor/autonomous_training.py`, `run_autonomous.py`
- Triggers: Scheduled runs or manual invocation
- Responsibilities: Curriculum-driven learning across domains, semantic observer training

## Error Handling

**Strategy:** Multi-layer defensiveness with mock fallbacks and local alternatives.

**Patterns:**
- Try/except around heavy imports (torch, torch_geometric) with mock availability check
- AST parsing failures → fallback to structural_encode() hash-based BorrowVector
- Rust compilation failures → record to feedback store; adjust gate thresholds
- Eigenvalue solver ill-conditioning → condition number check; use regularized SVD
- Missing data → JSON persists; restore on restart from journal files

## Cross-Cutting Concerns

**Logging:** Pure console output (no external logging service). Debug traces in JSONL feedback files (`codegen/feedback.jsonl`).

**Validation:**
- Pre-gate checks expected BorrowProfile before template render
- Post-gate AST extracts actual BV and verifies compilation success
- Koopman trust gate checks EDMD reconstruction error and spectral gap

**Authentication:** None (local platform only). All inference uses local models.

**sys.path Management:**
- `conftest.py` re-pins project root at position 0 during pytest collection
- `platform/backend/main.py` ensures ecemath/src and project root insertion order
- Every heavy import module (tensor/core.py) inserts ecemath/src, then re-pins project root

---

*Architecture analysis: 2026-02-22*
