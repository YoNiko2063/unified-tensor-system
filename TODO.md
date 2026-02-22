# Unified Tensor System — Master TODO
<!-- Consolidates: deep-research-report.md, GPU_HARDWARE_LEARNING.md,
     PREDICTION_DRIVEN_LEARNING.md, BOOTSTRAP_DIRECTIVE.md,
     FICUTS_v3.0_UNIFIED.md, ARCHITECTURE.md
     Those files may be deleted once this list is confirmed complete. -->

---

## A. Regime-Aware Operator Framework
*Goal: Runtime that detects which mathematical regime the system is in (flat/LCA, curved/Koopman, chaotic) and chooses the right tool automatically. Grounded in circuit MNA and citable operator-theory methods.*

### A1. Patch Detection Subsystem
- [ ] Implement `PatchClassifier` that takes a state `x ∈ R^n` and an evolution function `f` (or Jacobian `J = Df(x)`) and outputs:
  - `patch_type ∈ {lca, nonabelian, chaotic}`
  - `operator_rank r`
  - `commutator_score δ = ||[J, J^T]||_F`
  - `curvature_ratio ρ = ||∇J|| / λ_max`
  - `spectrum Λ` (dominant eigenvalues, complex)
  - `koopman_trust τ_K ∈ [0, 1]`
- [ ] Integrate Jacobian estimation: exact autodiff where autograd is available, finite-difference JVP otherwise
- [ ] Add non-smooth mode: when operator has discontinuities (switching circuits, comparators), classify as `discontinuous` and route to differential-inclusion handling instead of smooth Jacobian
- [ ] Validate patch classifier on MNA circuit: RLC (linear) should classify as `lca`; RLC + diode nonlinearity at high amplitude should classify as `nonabelian` or `chaotic`

### A2. Runtime Mode Machine with Trust Gates
- [ ] Implement three-mode controller that the system switches between at runtime:
  - **Mode LCA** (Laplace/Pontryagin): active when `ρ < ε₁` AND `δ < δ₁` AND spectrum well-conditioned. Compute local linear operator A, Laplace transfer objects, pole/zero summaries. Eigenmodes interpreted as "frequencies" only here.
  - **Mode Transition** (monitoring corridor): active when `ρ` is rising toward threshold or eigenvalues drift quickly. Use bifurcation criteria: eigenvalue real-part crossing zero = stability boundary. Do not commit to Koopman yet.
  - **Mode Koopman** (EDMD spectral tracking): enter ONLY when ALL of: `ρ > ε₂`, EDMD reconstruction error `< η_max`, spectral gap `> γ_min`, eigenfunction drift over short window `< s_max`.
- [ ] Parameterize thresholds `ε₁, ε₂, δ₁, η_max, γ_min, s_max` — do not hard-code; make tunable and eventually meta-learned by Optuna
- [ ] Implement trust-gate checks as standalone functions that return (bool, diagnostic_dict) — never a bare boolean
- [ ] Add hysteresis to mode transitions (prevent rapid oscillation near thresholds)

### A3. Koopman / EDMD Spectral Tracking
- [ ] Implement EDMD pipeline: given trajectory data `{x_k, x_{k+1}}` and observable dictionary `g(x)`, compute approximate Koopman eigenvalues and eigenfunctions
- [ ] Observable dictionary options: monomials up to degree d, radial basis functions, delay-coordinate embeddings — make configurable
- [ ] Compute trust metrics after each EDMD fit:
  - Reconstruction error: `||x_reconstructed - x_actual||`
  - Spectral gap: distance between dominant and next eigenvalue
  - Eigenfunction stability: perturbation of eigenfunctions under small data perturbation
- [ ] Detect continuous-spectrum regimes (gap collapse) and flag as "Koopman unreliable"
- [ ] Wire trust metrics back into mode machine (A2) as inputs to `koopman_trust τ_K`

### A4. Harmonic Atlas and Patch-Graph Navigation
- [ ] For each LCA patch P, extract dominant frequencies: `Ω(P) = {|Im(λ_i)|}` and decay magnitudes `Σ(P) = {|Re(λ_i)|}`
- [ ] Implement interval operator `I(P, Q) = {log(ω_j(Q) / ω_i(P))}` as a set-valued mapping between patches
- [ ] Implement Minkowski-sum composition of intervals: `I(P, R) ⊆ I(P, Q) ⊕ I(Q, R)` with tolerance pruning
- [ ] Build patch graph with edge cost: `w(P,Q) = α·(curvature integral) + β·(interval misalignment) + γ·(Koopman risk)`
  - curvature integral = `∫ ρ(x) dt` along path between patches
  - interval misalignment = `min_{u ∈ I(P,Q)} ||u - u*||²` where `u*` is target interval
  - Koopman risk = trust penalty from A3
- [ ] Implement Dijkstra/A* shortest path over patch graph for navigation
- [ ] "Exploration" flag: allow larger interval misalignment only when a mission flag says "controlled modulation"

### A5. SPDE Energy Functional and Stability Guarantee
- [ ] Implement discrete-time update for operator field A:
  `A_{k+1} = A_k - η·(∇_A E_total(A_k)) + √η·σ·ζ_k`
  where `ζ_k ~ N(0, I)` and `E_total = ||J - Π_A J||² + λ·||F_A||²`
  with `F_A = dA + A ∧ A` (curvature penalty)
- [ ] Implement projection consistency loss: `||J - Π_A J||²` where `Π_A` is the projection onto the operator basis induced by A
- [ ] Implement curvature penalty `||F_A||²` using commutator norm as discrete proxy before introducing full gauge formalism
- [ ] Implement stability scaffold: verify `E_total` decreases in expectation (Lyapunov descent), log energy trajectory
- [ ] Add step-size schedule: start with stability criterion; switch to adaptive when descent is confirmed
- [ ] For eventual "research submission grade": add smoothness + boundedness assumptions, prove (or verify empirically) monotonic decrease, cite LaSalle-style argument in comments

### A6. Minimal Proof-of-Concept (Validation)
- [ ] Build MNA model for pure RLC circuit (linear): verify patch classifier outputs `lca`, eigenvalues stable, Laplace mode valid across parameter sweep
- [ ] Add nonlinear element (diode-like: `i = I_s·(exp(V/V_T) - 1)`): sweep amplitude; verify that at small signal, patch stays `lca`; at large signal, transitions to `nonabelian`
- [ ] At high amplitude / chaotic-like operating point: run EDMD; verify leading modes reappear in observable space when trust conditions satisfied
- [ ] Construct patch graph from operating-point sweep; show shortest path corresponds to low-curvature traversal
- [ ] Write a single self-contained script `ecemath/examples/poc_patch_circuit.py` that runs this end-to-end and prints a pass/fail summary

---

## B. FICUTS 5-Dimension Learning System
*Goal: Single unified network (150 modes) learns simultaneously from math papers, code repos, code execution, architecture search, and physical hardware. Cross-dimensional overlaps = universal patterns.*

### B1. Dimension 1 — Mathematical: ArXiv Ingestion → Function Basis Library
- [ ] **Task 6.4**: Implement `ArxivPDFSourceParser` in `tensor/arxiv_pdf_parser.py`
  - Handle both `/abs/` and `/pdf/` URLs by downloading LaTeX source from `https://arxiv.org/e-print/{paper_id}`
  - Extract `.tex` files from tar.gz, parse equation environments: `equation`, `align`, `align*`, `\[...\]`, `$$...$$`
  - Return `{paper_id, equations: [str], num_equations: int}`
  - Test: parse `https://arxiv.org/abs/2602.13213`, assert `num_equations > 0`
- [ ] **Task 8.4**: Implement `populate_library_from_arxiv()` in `tensor/function_basis.py`
  - Iterate all `tensor/data/ingested/*.json` files where URL contains `arxiv.org`
  - Parse each with `ArxivPDFSourceParser`
  - Call `library._add_equation(paper_id, latex, domain)` for each extracted equation
  - Save library; assert `len(library.library) >= 50` after running on the 359 ingested papers
  - Infer domain from arXiv category code in URL (cs.AI, cs.LG, physics, etc.)

### B2. Dimension 2 — Behavioral: GitHub + DeepWiki → Dev-Agent Templates
- [ ] **Task 11.1**: Implement `GitHubCapabilityExtractor` in `tensor/github_ingestion.py`
  - Clone repo to temp dir → analyze → extract capability map → delete clone (save ~99.99% disk)
  - Capability map fields: `repo_url`, `intent` (from README), `patterns` (key classes/functions via AST), `parameters` (type signatures), `dependencies` (requirements.txt / pyproject.toml)
  - Use `tempfile.TemporaryDirectory()` as context manager so repo auto-deletes
  - Test: `extract_capability_map('https://github.com/wmjordan/PDFPatcher')` → intent contains "PDF"
- [ ] **Task 11.2**: Implement `DeepWikiIntegration` in `tensor/deepwiki_integration.py`
  - Method `search_patterns(intent, top_k=20)` → list of capability maps from DeepWiki API
  - Method `get_behavioral_template(repo_url)` → pre-built capability map
  - Fall back to direct GitHub API if DeepWiki API unavailable
  - Test: search "PDF manipulation", assert len > 0
- [ ] **Task 11.3**: Implement `DevAgentWithTemplates` in `tensor/dev_agent_bridge.py` (or extend existing)
  - `handle_intent(user_intent, parameters)` → first searches capability library, then DeepWiki, then generates from scratch
  - `fill_template(template, parameters)` → use unified network to instantiate behavioral template
  - Target: 10x reduction in lines of generated code vs. scratch generation for known intents
  - Test: "Merge PDFs" → returns working code using PDFPatcher template pattern

### B3. Dimension 3 — Execution: Already Implemented
- [ ] Review `tensor/execution_validator.py`: confirm Lyapunov energy update (E decreases on success, increases on failure) is wired and tested
- [ ] Add logging of Hebbian reinforcement: which patterns strengthened, which suppressed, per run

### B4. Dimension 4 — Optimization: Optuna Meta-Architecture Search
- [ ] Implement / verify `tensor/meta_optimizer.py` with `objective(trial)` that:
  - Suggests `hdv_dim, embed_dim, num_heads, num_layers, learning_rate`
  - Builds and trains `UnifiedTensorNetwork`
  - Returns `universals / total_patterns + phi_bonus`
  - phi_bonus = +0.2 if any attention eigenvalue ratio is within 0.05 of φ = 1.618...
- [ ] Run Optuna study with TPE sampler; store results in `tensor/data/optuna_study.db`
- [ ] After study, persist optimal config to `tensor/data/best_config.json`

### B5. Dimension 5 — Physical: Parameters → G-code
- [ ] **Task 12.1**: Implement `ParameterToGCodeGenerator` in `tensor/gcode_generator.py`
  - Load slicing behavioral template from capability library (Cura or PrusaSlicer pattern)
  - `parameters_to_gcode(params: Dict) → str` where params includes `thickness`, `infill`, `material`, `geometry`
  - Use unified network to fill template with parameters
  - Validate output: G-code must contain valid header, at least one `G1` move, and `M104 S0` footer
- [ ] Wire physical prediction feedback: measure actual print properties, compute error vs. predicted, call `network.reinforce_physical_model(params_hdv)` on success or `learn_residual(...)` on miss

### B6. Cross-Dimensional Discovery
- [ ] **Task 9.5**: Implement `CrossDimensionalDiscovery` in `tensor/cross_dimensional.py`
  - `record_pattern(dimension, pattern_hdv, metadata)` — stores per-dimension
  - `find_universals(similarity_threshold=0.95)` — cosine similarity search across all dimension pairs
  - Returns list of `{dimensions, similarity, patterns, type: 'cross_dimensional_universal'}`
  - Test: create math HDV for `exp(-t/τ)` and nearly-identical code HDV for a rate limiter; assert universal is found
- [ ] Integrate into training loop: after each batch, call `find_universals()`; log discoveries; promote to foundational when confirmed in ≥ 3 dimensions
- [ ] Target: ≥ 5 cross-dimensional universals discovered from first 1000 papers + 100 repos

### B7. Continuous Learning Execution
- [ ] Set up multi-terminal execution instructions (document in `README` or `run_ficuts.sh`):
  - Terminal 1: `WebIngestionLoop` on arXiv RSS feeds (cs.AI, cs.LG, physics), 1-hour interval
  - Terminal 2: `populate_library_from_arxiv()` (one-shot, re-run as papers accumulate)
  - Terminal 3: `GitHubCapabilityExtractor` batch over curated repo list
  - Terminal 4: `UnifiedTensorNetwork` training loop (all dimensions feeding HDV space)
  - Terminal 5: `meta_optimizer.py` Optuna study
- [ ] Target medium milestone: 1000+ papers, 5000+ equations, 100+ GitHub capability maps, ≥ 5 universals, dev-agent using templates

---

## C. Prediction-Driven Learning Loop
*Goal: System learns by predicting → testing → updating, grounded in mutual information, entropy, MDL, and Lyapunov energy. Not just "encode text to HDV" — actively verify understanding.*

### C1. Structured Concept Extraction
- [ ] Implement `StructuredTextExtractor` in `tensor/concept_extractor.py`
  - `extract_concepts(text: str) → ConceptGraph`
  - Chunk text into sections (chapters, headings)
  - Extract concept names per section
  - Build concept dependency graph: edge `(c1, c2)` with weight = `I(c1; c2)` where `I = H(c1) + H(c2) - H(c1, c2)` estimated from co-occurrence within paragraphs
  - Threshold: add edge only when MI > configurable threshold

### C2. Entropy-Guided Concept Selection
- [ ] Implement `PredictiveConceptLearner` in `tensor/concept_learner.py`
  - `predict_next_concept() → Concept`: selects from learnable candidates (prerequisites met)
  - Selection criterion: maximize `ΔI = H_before - H_after` where entropy is over unknown concepts weighted by out-degree (how many others depend on them)
  - Log information gain at each step for diagnostics

### C3. Problem Generation and Verification
- [ ] Implement `ProblemGenerator` in `tensor/problem_generator.py`
  - `generate_problem(concept: Concept) → Problem` based on concept type: operation / theorem / algorithm
  - Problem difficulty = |prerequisites|: easy (0) = direct application; medium (1-2) = chained; hard (3+) = synthesis
  - `verify_solution(problem, solution) → (is_correct, confidence)`:
    - If ground truth exists: exact comparison
    - Otherwise: MDL-based confidence = `1 / (1 + description_length(solution, problem))`
    - Threshold at 0.7 for is_correct

### C4. HDV-Based Problem Solving
- [ ] Implement `KnowledgeBasedProblemSolver` in `tensor/knowledge_solver.py`
  - `solve(problem: Problem) → (solution, confidence)`
  - Encode problem → HDV query; retrieve similar patterns from HDV space (threshold 0.7)
  - If patterns have code: use dev-agent to combine templates with problem parameters
  - If patterns are mathematical: apply math patterns directly
  - Confidence = geometric mean of retrieved pattern similarities

### C5. Verification, Feedback, and Gap Detection
- [ ] Implement `PredictionVerifier` in `tensor/prediction_verifier.py`
  - `verify_and_update(concept, problem, solution, is_correct)`:
    - Correct: strengthen connection in HDV (`strengthen_connection(concept_hdv, problem_hdv, weight=0.1)`); log Lyapunov energy decrease
    - Wrong: call `_identify_knowledge_gap(concept, problem, solution)`
  - `_identify_knowledge_gap` → `Gap(type, concept)`:
    - `missing_prerequisite`: prerequisite not in learned set
    - `no_relevant_patterns`: HDV similarity search returned nothing
    - `synthesis_error`: patterns found but synthesis failed
  - On gap: focused re-learning targeting the gap type

### C6. Continuous Learning Loop
- [ ] Implement `ContinuousLearningLoop` in `tensor/learning_loop.py`
  - `run(source: str)`: fetch text → extract concepts → while not all learned: predict → learn → generate problem → solve → verify → update → repeat
  - Log per-concept status: concept name, problems attempted, pass rate, Lyapunov delta
  - Halt criterion: all concepts learned with ≥ 0.8 confidence, or max iterations exceeded
  - Wire into FICUTS Dimension 1 so textbook / paper learning uses this loop, not raw ingestion

---

## D. GPU Hardware Modeling via Differential Equations
*Goal: Convert GPU verification specs and implementation patterns into DEQs, then co-simulate with physics (heat, power) to verify and optimize hardware designs. "All hardware optimization becomes differential equation solving."*

### D1. VeriGPU → DEQ Converter
- [ ] Implement `VeriGPUToDEQConverter` in `tensor/verigpu_to_deq.py`
  - `convert(verification_file: str) → DifferentialEquation`
  - Parse Verilog verification specs; extract LTL properties (□, ◊, U operators)
  - LTL → DEQ conversion rules:
    - `□P` → equilibrium: `∂P/∂t = 0`
    - `◊P` → growth: `∂P/∂t = λ·(1 - P)`
    - `P U Q` → `∂P/∂t = Q - P`
  - State vars: `coherence`, `race_free`, `deadlock_free`
  - Parameters: `cache_policy`, `memory_ordering`, `lock_protocol`
  - Constraints: coherence > 0.99, race_free == 1.0, deadlock_free == 1.0
- [ ] Clone VeriGPU: `git clone https://github.com/hughperkins/VeriGPU`
- [ ] Test: convert `coherence.v` example; assert DEQ has coherence state var and appropriate constraint
- [ ] Encode resulting DEQ to HDV (physical dimension)

### D2. tiny-gpu → DEQ Converter
- [ ] Implement `TinyGPUToDEQConverter` in `tensor/tiny_gpu_to_deq.py`
  - `convert(gpu_source: str) → DifferentialEquation`
  - Parse Verilog source; identify state machines (fetch/decode/execute/writeback)
  - State machine → continuous DEQ: `∂stage/∂t = next_stage(stage) - stage`
  - Identify performance bottlenecks: memory bandwidth limit → throughput constraint
  - Constraints: throughput ≤ bandwidth / data_size, latency ≤ ALU latency, power ≤ max_power
  - `extract_optimization_patterns(deq) → List[Dict]`: detect pipeline, cache hierarchy, warp scheduling patterns
- [ ] Clone tiny-gpu: `git clone https://github.com/adam-maj/tiny-gpu`
- [ ] Test: convert `shader_core.v`; assert pipeline and cache_hierarchy patterns found
- [ ] Encode optimization patterns to HDV

### D3. GPU Physics Simulator
- [ ] Implement `GPUPhysicsSimulator` in `tensor/gpu_physics_simulator.py`
  - `simulate(gpu_deq, parameters: Dict) → Dict` with keys: `temperature`, `power`, `performance`, `violations`, `energy_efficiency`
  - Coupled DEQ system:
    - GPU logic: `∂s/∂t = f(s, θ)`
    - Heat: `∂T/∂t = α·∇²T - P/c`
    - Power: `P = activity(s)·C·V²·f`
  - Solve with `scipy.integrate.solve_ivp`, method RK45
  - Constraint checking: T > 85°C → thermal_violation; P > 250W → power_violation; IPC < target → performance_violation
  - Test: simulate tiny-gpu DEQ at 1.5 GHz, 4096 cores, V=1.2V; assert result keys present

### D4. Unified DEQ Solver Integration
- [ ] Add VeriGPU and tiny-gpu converters to `UnifiedDEQSolver` (or create it if not yet existing):
  ```python
  solver.converters['verigpu'] = VeriGPUToDEQConverter()
  solver.converters['tiny_gpu'] = TinyGPUToDEQConverter()
  ```
- [ ] Implement `solver.optimize(gpu_deq, constraints)` → finds optimal parameters satisfying constraints via gradient descent on coupled DEQ
- [ ] Cross-learning: after both VeriGPU DEQ and tiny-gpu DEQ are encoded to HDV, run `CrossDimensionalDiscovery` — expect to find "coherence verification ≈ cache implementation" as a universal
- [ ] Full example: learn from VeriGPU → learn from tiny-gpu → find universals → design new GPU DEQ → optimize → simulate → verify

---

## E. Bootstrap: External Resource Integration
*Goal: System autonomously integrates 4 external resources to expand capabilities. Try first without human help; report blockers only.*

### E1. Scrapling — JS-Capable Web Scraping
- [ ] Implement `BootstrapManager.attempt_scrapling_integration()` in `tensor/bootstrap_manager.py`
  - Install Scrapling: `pip install scrapling`
  - Replace `requests.get` in `tensor/deepwiki_navigator.py` (line ~45) with Scrapling fetcher
  - Scrapling handles JavaScript-rendered pages, Cloudflare bypass, auto-retry with exponential backoff
  - Test: navigate `https://deepwiki.com/D4Vinci/Scrapling`; assert non-empty page data extracted
  - Fallback if blocked: fetch `https://raw.githubusercontent.com/D4Vinci/Scrapling/master/README.md` and parse code blocks for API usage

### E2. Open3D — Geometry Operations → Physical HDV
- [ ] Implement `BootstrapManager.attempt_open3d_integration()`
  - Use GitHub API to list Python example files under `isl-org/Open3D` that contain "geometry"
  - Fetch top 10 example files from `https://raw.githubusercontent.com/isl-org/Open3D/master/{path}`
  - Extract geometry transformation patterns: rotate, translate, scale, deformation → transformation matrix + parameters
  - Map each transformation → equivalent G-code commands (e.g., `rotation(90°, Z)` → G-code arc)
  - Encode each `(geometry_operation → G-code)` pair to HDV physical dimension
  - Target: ≥ 20 geometry operations extracted and encoded
  - Test: after extraction, HDV physical dimension has ≥ 20 new patterns

### E3. PrusaSlicer — Slicing Algorithms → Behavioral HDV
- [ ] Implement `BootstrapManager.attempt_prusaslicer_integration()`
  - Navigate DeepWiki page for `prusa3d/PrusaSlicer`
  - Find files with "gcode" or "slic" in path (e.g., `src/libslic3r/GCode.cpp`)
  - Extract ≥ 5 parameter → G-code mappings:
    - `layer_height` → Z increment: `G1 Z{layer * layer_height}`
    - infill pattern generation (gyroid, honeycomb, rectilinear)
    - travel move optimization
    - retraction parameters → `G1 E-{retract_length}` + speed
  - Encode each mapping as a behavioral pattern in HDV behavioral dimension
  - Cross-dimensional link: math (geometry params) + code (PrusaSlicer algorithm) + physical (G-code) → confirm universal discovered via C9.5
  - Test: behavioral dimension gains ≥ 5 new PrusaSlicer patterns after integration

### E4. Book of Secret Knowledge — Meta-Resource for Capability Gap Filling
- [ ] Implement `BootstrapManager.attempt_secret_knowledge_integration()`
  - Fetch `https://raw.githubusercontent.com/trimstray/the-book-of-secret-knowledge/master/README.md`
  - Parse all `[Title](URL)` markdown links (regex `r'\[([^\]]+)\]\(([^\)]+)\)'`)
  - Categorize links by topic: networking, security, programming, data, system design, crypto
  - For each category, check if it fills a current capability gap (compare against known dimensions)
  - Extract patterns from top 3 repos per relevant category
  - Target: ≥ 50 new patterns added to behavioral dimension
  - Test: behavioral dimension count increases by ≥ 50 after integration

### E5. BootstrapManager Orchestrator
- [ ] Wire all 4 integration attempts into `BootstrapManager.run_bootstrap()`:
  - Run all 4 attempts; collect successes and blockers
  - Print structured report: successes (✓), blockers (✗ with reason)
  - Only request human help on blocked resources
- [ ] Add `--bootstrap` flag to `run_autonomous.py`:
  ```python
  from tensor.bootstrap_manager import BootstrapManager
  bootstrap = BootstrapManager(hdv_system)
  bootstrap.run_bootstrap()
  ```
- [ ] Run bootstrap; confirm all 4 resources integrated or file specific blockers for human review

---

## F. Web Ingestion Pipeline Hardening
*Goal: Make ingestion research-grade: trustworthy, auditable, compliant, and deduplicated.*

- [ ] **URL canonicalization**: normalize URLs before storing (strip tracking params like `utm_*`, `ref=`, fragment identifiers); hash canonical URL as dedup key
- [ ] **Near-duplicate content deduplication**: hash extracted text blocks (SimHash or MinHash); reject if similarity > 0.9 to an existing stored block
- [ ] **Dynamic source credibility scoring**: assign each source a `credibility_prior ∈ [0, 1]`; update using cross-source corroboration (same claim in ≥ 3 independent sources → boost) and retraction signals (flagged contradictions → penalize); store per-domain in `tensor/data/source_trust.json`
- [ ] **Citation-preserving ingestion**: for each ingested document, store: `{url_hash, raw_html_snapshot_hash, extracted_text_blocks, extraction_provenance (CSS selectors used), timestamp}`; enable audit trail back to source
- [ ] **Compliance**: respect `robots.txt` before crawling any URL; enforce rate limiting (≥ 1s between requests to same domain); skip paywalled content (detect 402/login redirect); flag and skip content with restrictive licenses; lean on arXiv official feeds for scientific content

---

## G. Local Hardware Optimization
*Goal: System auto-detects hardware and chooses optimal compute strategy — matching the "patch geometry" principle (LCA/linear ops → analog-like SIMD; nonlinear/switching → digital-like branches).*

- [ ] **CPU feature detection**: at startup, detect SIMD width (SSE2/AVX2/AVX512), cache hierarchy (L1/L2/L3 sizes and latencies), thread count; store in L3 MNA (already in `tensor/hardware_profiler.py` — verify these fields are present)
- [ ] **Adaptive batching + precision**: based on detected SIMD width and available RAM, choose: batch size for EDMD, float32 vs float64 for operator computations, number of Koopman modes to track simultaneously
- [ ] **GPU strategy** (if GPU detected): choose quantization scheme (int8/fp16) for unified network inference; keep HDVS atlas in pinned memory; use CUDA streams for overlapping EDMD and network forward pass
- [ ] **Atlas memory layout**: store patch graph and HDV atlas in contiguous row-major arrays ordered by patch ID; benchmark cache miss rate before and after; target < 5% L3 miss rate during navigation
- [ ] **Sparse operator bases**: patch Jacobians are sparse (especially in circuit MNA); verify sparse matrix format (CSR/CSC) is used throughout `ecemath/` and that sparse solve is used when density < 20%

---

## H. Cross-Cutting Infrastructure

- [ ] **Unified DifferentialEquation dataclass**: create `tensor/deq.py` with `DifferentialEquation(equations, state_vars, parameters, constraints, domain)` used consistently across GPU (D), physical synthesis (B5), and prediction-verifier (C5)
- [ ] **HDV dimension registry**: create `tensor/hdv_registry.py` that tracks pattern counts per dimension (math, behavioral, execution, physical) and can be queried by any module — eliminates scattered print statements and enables monitoring
- [ ] **Lyapunov energy logging**: centralize energy tracking in `tensor/energy_tracker.py`; all modules call `tracker.log(E_new, context)` rather than implementing their own; add assertion that energy is non-increasing in expectation over a sliding window of 100 steps
- [ ] **Test coverage target**: bring test count from 271 (FICUTS claim) / 71 (ARCHITECTURE claim) to a consistent number with CI; ensure all new modules (A1–G) have at least one test each; add a `tests/test_new_modules.py` as the landing point for new tests
- [ ] **Single `run_ficuts.sh` script**: orchestrate all five FICUTS terminals (web ingestion, library population, GitHub ingestion, network training, Optuna) with proper conda env activation and logging to `logs/ficuts_{date}.log`

---

## Priority Order for First Implementation Sprint

1. **A6** — PoC circuit demo (validates the whole theoretical framework with something runnable)
2. **B1 (Task 6.4)** — ArXiv LaTeX parser (unblocks function library and training)
3. **B1 (Task 8.4)** — Populate function library from 359 ingested papers
4. **B6 (Task 9.5)** — Cross-dimensional discovery (enables universal detection)
5. **B2 (Task 11.1)** — GitHub capability extractor (unblocks behavioral dimension)
6. **A1** — Patch classifier (foundation for A2–A5)
7. **C1–C6** — Prediction-driven learning loop (transforms ingestion into active learning)
8. **D1–D4** — GPU/DEQ modeling (adds hardware domain)
9. **E1–E5** — Bootstrap resource integration (expands capability surface)
10. **A2–A5, F, G** — Runtime mode machine, ingestion hardening, HW optimization
