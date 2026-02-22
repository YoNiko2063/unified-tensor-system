# Codebase Structure

**Analysis Date:** 2026-02-22

## Directory Layout

```
unified-tensor-system/
├── tensor/                      # Core spectral-geometric framework (50+ modules)
│   ├── core.py                  # UnifiedTensor: 4-level matrix tensor T ∈ R^(L x N x N x t)
│   ├── semantic_observer.py     # Forced ODE: ẋ = A·x + g(x) + B·u (Lyapunov tracking)
│   ├── koopman_edmd.py          # EDMD eigenvalue decomposition, spectral analysis
│   ├── eigenspace_mapper.py     # Map parameters → eigenvalues → domain signatures
│   ├── eigen_walker.py          # ParameterSpaceWalker + JumperStrategy for harmonic search
│   ├── domain_registry.py       # Register 7 evaluation domains (RLC, MSD, power-grid, etc.)
│   ├── integrated_hdv.py        # 10k-dim HDV system: domain masks + learned embeddings
│   ├── function_basis.py        # Function library parser (575 entries, 53 DEQs)
│   ├── calendar_regime.py       # 5-channel event encoder + Arnold tongue resonance
│   ├── frequency_dependent_lifter.py  # Φ_S→M(t) cross-timescale state lifting
│   ├── multi_horizon_mixer.py   # (5,3) calendar modulation + volume penalties
│   ├── semantic_geometry.py     # _TextEDMD for observer state (NOT EDMDKoopman)
│   ├── epistemic_geometry.py    # Token trajectory → velocity/curvature/FFT validity
│   ├── autonomous_training.py   # Curriculum-driven learning orchestrator
│   ├── code_graph.py            # Code structure vectorization
│   ├── code_validator.py        # Compilation + execution validation
│   ├── data/                    # Persistent data files
│   │   ├── function_library.json        # 575 entries, 53 DEQs, 282 raw_latex
│   │   ├── hdv_state.json               # HDV vectors + domain masks
│   │   ├── fed_dates.json               # FOMC 2024-2026 + NYSE holidays
│   │   ├── universals.json              # Cross-domain universal patterns
│   │   ├── ingestion_journal.json       # 92 papers journaled
│   │   └── ingested/                    # 359+ RSS-fetched papers
│   └── agents/                  # Agent network for distributed learning
│
├── ecemath/                     # Physics kernel library (pure NumPy/SciPy)
│   └── src/
│       ├── core/                # Matrix algebra, MNA systems, sparse solvers
│       │   ├── matrix.py        # MNASystem: C·ẋ + G·x + h(x) = u(t)
│       │   ├── coarsening.py    # CoarseGrainingOperator for level compression
│       │   └── sparse_solver.py # compute_free_energy, HarmonicSignature, CorrectedLifter
│       ├── domains/             # Domain-specific implementations
│       │   ├── trading.py       # TradingSystem with crowd tanh nonlinearity
│       │   ├── circuits.py      # RLC/RC/LC via MNA, eigenvalue targeting
│       │   └── mechanics.py     # MassSpringDamper, TwoMassSpring (symmetric 2-DOF)
│       ├── circuits/            # Circuit analysis (RLC topology)
│       ├── analysis/            # Bifurcation, stability, spectral analysis
│       └── optimization/        # Parameter fitting and synthesis
│
├── codegen/                     # Rust code generation pipeline
│   ├── pipeline.py              # CodeGenPipeline: intent → render → compile
│   ├── intent_spec.py           # IntentSpec + BorrowProfile enum + e_borrow()
│   ├── borrow_predictor.py      # LogReg classifier on B1..B6 features
│   ├── template_registry.py     # TemplateRegistry: 24 Rust templates
│   ├── templates/               # 7 domain folders + 24 template files
│   │   ├── numeric_kernel.py    # PURE_FUNCTIONAL kernels
│   │   ├── market_model.py      # SHARED_REFERENCE trading models
│   │   ├── api_handler.py       # ASYNC_IO HTTP handlers
│   │   ├── text_parser.py       # LaTeX/markup parsing
│   │   ├── html_navigator.py    # DOM traversal
│   │   ├── physics_sim.py       # Numerical ODE solvers
│   │   └── trading_sim.py       # Market agent simulation
│   ├── structural_encoder.py    # AST-free hash-based BV encoder
│   ├── rust_graph_extractor.py  # Binary calls to rust-borrow-extractor v0.3
│   ├── feedback_store.py        # JSONL: (BV, compile_result) pairs
│   ├── experiment_runner.py     # 2000-variant synthesis experiment
│   ├── analysis.py              # SpectralGeometryAnalysis + feature importance
│   └── feedback.jsonl           # Training data: 4000+ records
│
├── optimization/                # Domain evaluators + optimization
│   ├── circuit_optimizer.py     # CircuitOptimizer: EigenvalueMapper + Pareto search
│   ├── code_gen_experiment.py   # BorrowVector classifier training (LogReg, 16D features)
│   ├── code_profiler.py         # Extract B1..B6 from code samples
│   ├── duffing_evaluator.py     # Duffing oscillator (ẍ + 2ζω₀·ẋ + ω₀²x + α·x³ = u)
│   ├── spring_mass_system.py    # MSD/TMS parameter optimization
│   ├── power_grid_evaluator.py  # IEEE 39-bus power system
│   ├── harmonic_navigator.py    # Spectral resonance tracking (rational ridges)
│   ├── hdv_optimizer.py         # HDV overlap optimization
│   └── phase3_predictor.py      # Transfer learning predictor
│
├── platform/                    # Web platform (FastAPI + React)
│   ├── backend/
│   │   ├── main.py              # FastAPI app, 6 routers at /api/v1/*
│   │   ├── routers/
│   │   │   ├── regime.py        # LCA/nonabelian/chaotic classification
│   │   │   ├── calendar.py      # 5-channel event encoding
│   │   │   ├── codegen.py       # IntentSpec → compiled Rust
│   │   │   ├── hdv.py           # HDV overlap + visualization
│   │   │   ├── physics.py       # Domain evaluator endpoints
│   │   │   └── circuit.py       # CircuitOptimizer Pareto frontend
│   │   ├── tests/
│   │   │   └── test_*.py        # Backend integration tests
│   │   └── start.sh             # Startup script
│   └── frontend/                # React 18 + Vite + Tailwind
│       ├── src/
│       │   ├── components/      # 6 main tabs
│       │   │   ├── RegimeDashboard.jsx         # LCA/nonabelian/chaotic live viz
│       │   │   ├── CalendarOverlay.jsx         # 5-channel event amplitude
│       │   │   ├── CodeGenPanel.jsx            # Intent → BV → Rust
│       │   │   ├── HDVExplorer.jsx             # PCA projection + domain coloring
│       │   │   ├── PhysicsSimulator.jsx        # Live ODE integration
│       │   │   └── CircuitOptimizer.jsx        # Pareto frontier + frequency response
│       │   ├── api/
│       │   │   └── client.js    # HTTP wrapper for /api/v1/* endpoints
│       │   ├── App.jsx          # Tab router
│       │   └── main.jsx         # React 18 entry
│       ├── package.json
│       ├── vite.config.js
│       └── tailwind.config.js
│
├── tests/                       # Comprehensive test suite (90+ files)
│   ├── test_codegen_pipeline.py         # Full code gen flow (35K lines)
│   ├── test_calendar_lifter.py          # Multi-horizon mixing (59K lines)
│   ├── test_eigenspace_mapper.py        # Domain mapping + recognition (17K lines)
│   ├── test_circuit_optimizer.py        # Pareto search validation (20K lines)
│   ├── test_calendar_regime.py          # Event encoding (29K lines)
│   ├── test_duffing_koopman.py          # Duffing acceleration (34K lines)
│   ├── test_bifurcation_regimes.py      # Stability detection (29K lines)
│   └── [87 more test files] — 2261 passed, 4 skipped (as of 2026-02-22)
│
├── benchmarks/                  # Performance evaluation + transfer learning
│   ├── walker_vs_random.py              # Random vs Walker vs Hybrid (100 trials, JSON+plots)
│   ├── instability_lead_time.py         # Spectral vs time-domain detection
│   ├── transfer_eigen_walker_v2.py      # Energy-gradient training (5k synthetic)
│   ├── transfer_rlc_to_msd.py           # Cross-domain parameter transfer
│   ├── multimode_eigen_walker.py        # r=2 two-mode benchmark (TwoMassSpring)
│   ├── competing_surfaces.py            # Z_{2:1} → Z_{3:2} navigation
│   ├── rational_atlas.py                # Farey graph topology
│   ├── coupled_trimode_atlas.py         # r=3 three-mode extensions
│   ├── results/                         # Benchmark outputs (JSON + PNG)
│   └── __init__.py
│
├── stability_engine/            # Containerized stability analysis
│   ├── core/                    # Core computation modules
│   ├── api/                     # REST API wrapper
│   ├── validation/              # Output validators
│   └── Dockerfile
│
├── data/                        # Global data directory (sparse)
├── examples/                    # Demo scripts
│   └── ieee39_demo.py           # Power grid optimization example
│
├── run_experiment.py            # Main code generation experiment entry (produces 2000+ variants)
├── run_autonomous.py            # Autonomous training orchestrator
├── run_explorer.py              # Interactive configuration explorer
├── run_system.py                # Full system bootstrap
├── conftest.py                  # pytest: pin project root at sys.path[0]
├── pyproject.toml               # Project config (setuptools, pytest, dependencies)
└── README.md
```

## Directory Purposes

**tensor/:**
- Purpose: Core mathematical framework for spectral analysis, Koopman lifting, HDV encoding
- Contains: 50+ Python modules for dynamics, semantic geometry, code analysis
- Key files: `core.py` (UnifiedTensor), `semantic_observer.py` (forced ODE), `eigenspace_mapper.py` (domain mapping)

**ecemath/src/:**
- Purpose: Physics kernel library (MNA, Koopman, eigensolvers) — pure NumPy/SciPy, no torch
- Contains: Domain implementations (trading, circuits, mechanics), optimization, analysis
- Key files: `core/matrix.py` (MNASystem), `core/sparse_solver.py` (harmonic signatures), `domains/*.py` (trading/circuits/mechanics)

**codegen/:**
- Purpose: Intent-driven Rust code generation with BorrowVector safety gating
- Contains: Pipeline orchestrator, 24 templates, BorrowVector predictor, feedback loop
- Key files: `pipeline.py` (orchestrator), `intent_spec.py` (semantic spec), `templates/` (7 domains)

**optimization/:**
- Purpose: Domain-specific evaluators for eigenspace navigation and parameter optimization
- Contains: Circuit/RLC/spring-mass/power-grid/Duffing evaluators, spectral searchers
- Key files: `circuit_optimizer.py`, `spring_mass_system.py`, `power_grid_evaluator.py`

**platform/backend/:**
- Purpose: FastAPI server exposing tensor/codegen/optimization to web UI
- Contains: 6 routers (/regime, /calendar, /codegen, /hdv, /physics, /circuit)
- Key files: `main.py` (app), `routers/` (endpoints)

**platform/frontend/:**
- Purpose: React UI for spectral visualization, code generation, circuit optimization
- Contains: 6 component tabs, API client wrapper
- Key files: `components/*.jsx`, `api/client.js`

**tests/:**
- Purpose: Comprehensive validation (2261 passing tests)
- Contains: Per-module test files + integration tests
- Key files: `test_codegen_pipeline.py`, `test_eigenspace_mapper.py`, `test_circuit_optimizer.py`

**benchmarks/:**
- Purpose: Performance evaluation and transfer learning experiments
- Contains: Walker vs random, spectral lead time, cross-domain transfer, multi-mode search
- Key files: `walker_vs_random.py`, `transfer_eigen_walker_v2.py`, `rational_atlas.py`

## Key File Locations

**Entry Points:**
- `run_experiment.py`: Code generation experiment (2000 variants, produces experiment.jsonl + plots)
- `run_autonomous.py`: Autonomous training loop with curriculum learning
- `platform/backend/main.py`: FastAPI server at :8000 with 6 routers
- `examples/ieee39_demo.py`: Power grid optimization example

**Configuration:**
- `pyproject.toml`: Dependencies, test paths, package discovery
- `conftest.py`: pytest fixture (repins project root at sys.path[0])
- `platform/backend/start.sh`: Server startup script
- `platform/frontend/vite.config.js`, `tailwind.config.js`: Frontend build config

**Core Logic:**
- `tensor/core.py`: UnifiedTensor 4-level matrix storage and coarsening
- `tensor/semantic_observer.py`: Forced ODE with Lyapunov energy tracking
- `ecemath/src/core/matrix.py`: MNA system definition (C·ẋ + G·x + h(x) = u)
- `codegen/pipeline.py`: Intent → compile pipeline with pre/post gating
- `codegen/borrow_predictor.py`: LogReg classifier for BorrowVector energy
- `optimization/circuit_optimizer.py`: Pareto eigenvalue search

**Testing:**
- `tests/test_codegen_pipeline.py`: Full code generation flow
- `tests/test_eigenspace_mapper.py`: Domain mapping + harmonic recognition
- `tests/test_circuit_optimizer.py`: Optimizer Pareto frontier
- `tests/test_calendar_lifter.py`: Multi-horizon mixing validation
- `conftest.py`: Root pytest configuration

**Data:**
- `tensor/data/function_library.json`: 575 equations + LaTeX
- `tensor/data/hdv_state.json`: HDV vectors + domain masks
- `codegen/feedback.jsonl`: (BV, compile_result) training pairs (4K+ records)

## Naming Conventions

**Files:**
- `{module}_observer.py`: Forced ODE tracking system (semantic_observer.py, epistemic_geometry.py)
- `{domain}_evaluator.py`: Domain-specific eigenspace evaluation (duffing_evaluator.py, spring_mass_system.py)
- `test_{module}.py`: Test file for module (test_codegen_pipeline.py)
- `{name}_mapper.py`: Parameter/state mapping system (eigenspace_mapper.py)

**Directories:**
- `tensor/`: Core framework
- `ecemath/src/`: Physics library (pure NumPy/SciPy)
- `codegen/`: Code generation
- `optimization/`: Domain evaluators
- `platform/{backend,frontend}/`: Web platform
- `tests/`: Test suite
- `benchmarks/`: Experiment harnesses

**Classes:**
- `*Observer`: Forced ODE system (SemanticObserver, TensorObserver)
- `*Evaluator`: Domain evaluator (CircuitOptimizer, DuffingEvaluator)
- `*Mapper`: Parameter/eigenspace mapping (EigenspaceMapper, DomainCanonicalizer)
- `*Pipeline`: Orchestration (CodeGenPipeline, CircuitOptimizer)
- `*Registry`: Lookup/registration (TemplateRegistry, DomainRegistry)
- `*Predictor`: ML classifier (BorrowPredictor, ParameterSpaceWalker)

**Functions:**
- `truncate_spectrum()`: Spectral filtering (tensor/semantic_observer.py)
- `compute_overlap_similarity()`: Cross-domain HDV similarity (tensor/integrated_hdv.py)
- `compute_free_energy()`: Harmonic energy functional (ecemath/src/core/sparse_solver.py)
- `rotate_operator()`: Basis-change matrix similarity (ecemath/src/core/matrix.py)

## Where to Add New Code

**New Feature (e.g., novel optimization domain):**
- Primary code: `optimization/{domain_name}_evaluator.py`
- Domain registration: `tensor/domain_registry.py` (register_domain call)
- Tests: `tests/test_{domain_name}_evaluator.py`
- Example: `examples/{domain_name}_demo.py` (optional)
- API endpoint: `platform/backend/routers/physics.py` (add POST handler)

**New Component/Module:**
- Implementation: `tensor/{component}.py` (if framework) or `codegen/{component}.py` (if code gen)
- Exports: `tensor/__init__.py` or `codegen/__init__.py` (add to _IMPORTS list)
- Tests: `tests/test_{component}.py`
- Integration: Hook via parent orchestrator (e.g., autonomous_training.py)

**Utilities:**
- Shared helpers: `tensor/` (general framework utilities)
- ecemath helpers: `ecemath/src/core/` (physics-specific)
- codegen helpers: `codegen/` (code gen specific)
- Avoid top-level `utils.py` — organize by domain

**Frontend Component:**
- Implementation: `platform/frontend/src/components/{ComponentName}.jsx`
- API integration: Update `platform/frontend/src/api/client.js`
- Styling: Use Tailwind classes (no CSS files)
- Route: Add tab to `App.jsx`

**Test File Structure:**
- Imports: `sys.path.insert(0, _ROOT)` at top (where `_ROOT = os.path.dirname(...) upward to project`)
- Use pytest fixtures from `conftest.py` for sys.path management
- Organize by feature: one test file per module or integration point

**ecemath Domain:**
- Implement system class at `ecemath/src/domains/{domain}.py`
- Must have `.rhs(t, x)` method or be callable `f(x) -> ẋ`
- Register at `ecemath/src/domains/__init__.py`
- Use save/restore pattern for `ir.params` (immutable parameter snapshot)

## Special Directories

**tensor/data/:**
- Purpose: Persistent data files (JSON, database)
- Generated: Yes (function_library.json populated by parsers)
- Committed: Yes (initial seed data committed)
- Contents: 575+ equations, HDV state, ingestion journal, visualization data

**codegen/feedback.jsonl:**
- Purpose: Training data for BorrowVector classifier
- Generated: Yes (CodeGenPipeline appends feedback records)
- Committed: Yes (initial seed records in repo)
- Format: One JSON object per line (BV, compile_result, template, etc.)

**tests/__pycache__/ and platform/backend/__pycache__/:**
- Purpose: Python bytecode cache
- Generated: Yes (by Python during import)
- Committed: No (.gitignored)

**benchmarks/results/:**
- Purpose: Benchmark outputs (JSON + PNG plots)
- Generated: Yes (by benchmark harnesses)
- Committed: Selectively (JSON committed, PNG may be too large)

**platform/frontend/node_modules/:**
- Purpose: npm dependencies
- Generated: Yes (by npm install)
- Committed: No (.gitignored)

---

*Structure analysis: 2026-02-22*
