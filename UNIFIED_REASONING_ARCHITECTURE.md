# Unified Reasoning Architecture
# FICUTS — Fractals In Compositions of Unified Tensor Systems

**Status:** Reference document for autonomous activation
**Purpose:** Hand this to a fresh Claude instance — it has everything needed to resume the project
**Branch:** `claude/consolidate-repo-math-structure-lOXEH`
**Tests passing:** 271+ (run `PYTHONPATH=. python -m pytest tests/ -q`)

---

## The One Sentence

> Everything — math papers, circuits, markets, code, hardware, 3D models — reduces to the
> same differential equation, encoded in a shared HDV space, where universal patterns
> emerge as cross-dimensional overlaps, and the system gets smarter every time it runs.

---

## The Core Equation (Governs Everything)

```
C(Θ)·ẋ + G(Θ)·x + h(x;Θ) = u(t)
```

This is the **Modified Nodal Analysis (MNA)** equation. It is not just for circuits.
It is the universal form for any network where:

| Symbol | Meaning | Circuit | Market | Code | Hardware |
|--------|---------|---------|--------|------|----------|
| `x` | state vector | node voltages | prices | module coupling | thermal state |
| `G(Θ)` | conductance matrix | resistors | correlations | import edges | bandwidth |
| `C(Θ)` | capacitance matrix | capacitors | momentum | inheritance | thermal mass |
| `h(x;Θ)` | nonlinear terms | diodes/MOSFETs | spread/slippage | recursion | throttling |
| `u(t)` | external drive | voltage sources | market events | user requests | workload |

Every subsystem of this project builds and updates one or more MNA matrices.
The tensor `T ∈ ℝ^(L × N × N × t)` holds all four levels simultaneously.

---

## The Four Levels of the Tensor

| Level | Domain | What the nodes are | What the edges are |
|-------|--------|-------------------|-------------------|
| **L0** | Finance / Market | Tickers (AAPL, BTC…) | Price correlations, sentiment |
| **L1** | Neural / SNN | Spiking neurons | Synaptic weights |
| **L2** | Codebase | Python modules/files | Import, call, inheritance |
| **L3** | Hardware | CPU cores, memory, GPU | Thermal coupling, bandwidth |

All four levels are **active simultaneously** in the running system.
Cross-level interactions are the emergent intelligence.

---

## The Four Reasoning Domains — How They Connect

### 1. Math Reasoning (Papers → Function Basis)

**What it does:** Ingests arXiv LaTeX source, extracts equations with SymPy, classifies them
into a universal function basis library that every other domain can use.

**The pipeline:**
```
arXiv /e-print/{id}  →  tar.gz with .tex  →  LaTeX equations  →  SymPy parse
→  classify (exponential_decay | power_law | oscillation | linear | …)
→  encode to HDV vector  →  store in math dimension of HDV space
```

**Mathematical grounding:**
- Each equation class is a solution family of some DEQ
- `exponential_decay`: `∂x/∂t = -x/τ` (same as RC circuit, same as rate limiter)
- `oscillation`: `∂²x/∂t² + ω²x = 0` (same as LC circuit, same as pendulum)
- `power_law`: `x(t) ∝ t^α` (scale-invariant → appears at criticality)

**Key insight — universality test:** If the same function class appears in ≥2 other
domains with cosine similarity > 0.95 in HDV space, it is promoted to **universal**.
That universal then informs all other domains.

**Files:**
- `tensor/arxiv_pdf_parser.py` — LaTeX source download + equation extraction
- `tensor/function_basis.py` — universal function basis library
- `tensor/deq_system.py` — `PaperToDEQConverter`, `UnifiedDEQSolver`

**How to feed it:**
```bash
python run_autonomous.py --populate --max-papers 50
python run_autonomous.py --deq test_gradient_descent.txt --deq-type paper
```

---

### 2. Finance Reasoning (Markets → Trading Signals)

**What it does:** Converts real-time market price/sentiment data into an L0 MNA system.
Detects market regimes (calm / volatile / crisis) via eigenvalue gap analysis.
Enhances FinBERT sentiment scores with tensor-derived signals.

**The pipeline:**
```
WebSocket tick (price, volume)  →  MarketGraph  →  L0 MNA update
FinBERT sentiment + tensor_signal + regime_weight  →  enhanced trading signal
```

**Enhanced score formula:**
```
signal = α·finbert_score + β·free_energy(L0_node) + γ·regime_weight
```

Where:
- `free_energy(node) = E - τS + γH` — high tension = strong directional signal
- `regime_weight` — calm:1.0, volatile:0.5, crisis:0.1 (from eigenvalue gap)
- `harmonic_signature` — consonance score → confidence multiplier

**Regime detection (eigenvalue gap):**
```python
gap = λ_1 - λ_2   # from G matrix eigenspectrum
# Large gap → stable, small/narrowing gap → regime transition imminent
```

**What the MNA looks like for markets:**
```
G[i,j] = correlation(ticker_i, ticker_j)   # off-diagonal conductance
G[i,i] = sentiment_weight(ticker_i)         # self-conductance
C[i,i] = price_momentum(ticker_i)           # inertia
u[i]   = news_event_strength(ticker_i)      # external drive
```

**Files:**
- `tensor/market_graph.py` — `MarketGraph`, `MarketNode`, `sentiment_injection()`
- `tensor/trading_bridge.py` — FinBERT enhancement + tensor signal fusion
- `tensor/scraper_bridge.py` — HTML articles → ticker mentions → L0 update
- `tensor/realtime_feed.py` — WebSocket feed (Yahoo, Coinbase, mock)
- `tradingCode/` — trading bot pipeline
- `tradingBot/` — deployment

**How to run standalone:**
```python
from tensor.realtime_feed import RealtimeFeed
from tensor.trading_bridge import TradingBridge
feed = RealtimeFeed(tensor)
feed.connect_mock()   # or connect_yahoo() / connect_coinbase()
bridge = TradingBridge(tensor)
signal = bridge.enhanced_signal('AAPL')
```

---

### 3. Circuit Reasoning (ECEMath → Solver + Fisher Guidance)

**What it does:** Pure NumPy circuit math library. Solves DC/AC circuits via MNA stamping,
computes Fisher Information Matrix to identify highest-information improvement directions,
detects dynamical regimes, runs stochastic simulations.

**This is the mathematical core of the entire system** — every other domain borrows
from these primitives.

**The ECEMath stack (`ecemath/src/core/`):**

| Module | What it does | Key classes |
|--------|-------------|-------------|
| `matrix.py` | MNA system + builder | `MNASystem`, `ExtendedMNABuilder` |
| `components.py` | Stamp R/C/L/V/MOSFET into G,C | `Resistor`, `Capacitor`, `VoltageSource`, `MOSFET` |
| `graph.py` | Circuit topology | `CircuitGraph`, `Node`, `Edge` |
| `dynamics.py` | `C·dv/dt = -G·v - h(v) + u` | `CircuitDynamics` |
| `solver.py` | Equilibrium + transient | `CircuitSolver.find_equilibrium()`, `.simulate()` |
| `fisher.py` | FIM = J^T·Σ⁻¹·J | `FisherInformation.compute()` |
| `regime.py` | Markov regime switching | `RegimeSwitchingSystem` |
| `stochastic.py` | SDE solvers | Euler-Maruyama, Milstein |
| `sparse_solver.py` | Harmonic signature | free energy, consonance scoring |
| `coarsening.py` | φ operator | `CoarseGrainingOperator` |

**How HomeworkSolver works (DC circuit example):**
```python
from ecemath.examples.homework_solver import HomeworkSolver
solver = HomeworkSolver()
result = solver.solve_dc({
    'nodes': [0, 1, 2],
    'components': [
        {'type': 'V', 'value': 15, 'node_p': 1, 'node_n': 0},
        {'type': 'R', 'value': 2200, 'node_p': 1, 'node_n': 0},
        {'type': 'R', 'value': 3300, 'node_p': 1, 'node_n': 2},
    ]
})
# result.voltages, result.currents, result.power_balance
```

**Fisher-guided improvement (critical bridge to L2 code graph):**
```
G_matrix (code coupling) → normalize → FIM = (J^T·J) → eigendecompose
→ top-k eigenvectors = priority directions (which modules to improve first)
```
*Note: G must be normalized before FIM — raw G with 1e-6 diagonals blows up to 1e+12 eigenvalues.*

**Circuit as iso-functional manifold (the deeper insight):**
- All circuits that compute the same boolean function form a manifold
- Optimization = walk along the manifold surface (gradient descent in tangent space)
- Changing R/C values without changing function = staying on the manifold
- The system can navigate this manifold to minimize power while preserving behavior

**Files:**
- `ecemath/src/core/` — the complete math library
- `ecemath/examples/homework_solver.py` — DC solver interface
- `tensor/math_connections.py` — 7 bridges connecting ECEMath → tensor loop

**The 7 Math Connections:**

| # | Bridge | What it does |
|---|--------|-------------|
| 1 | Fisher → GSD | FIM eigenvalues → priority indices → which modules to improve |
| 2 | Regime → monitoring | eigenvalue gap → `should_pause` GSD during transitions |
| 3 | Stochastic → explorer | Monte Carlo noise → robustness score |
| 4 | Neural error → GSD | predicted vs actual L1 → weight GSD tasks |
| 5 | SNN firing → L1 | free energy → activation mask for neural update |
| 6 | Pytest → jump events | test pass rate → L2 regime discontinuity |
| 7 | Feed health → monitoring | staleness/status → L0 data quality warnings |

---

### 4. Physics / Hardware Reasoning (DEQs → Optimization → G-code)

**What it does:** Converts hardware specs, GPU verification docs, and 3D model geometry
into differential equations representing physical constraints. Optimizes within those
constraints. Generates G-code for physical fabrication.

**Every physical phenomenon becomes a DEQ:**
```
GPU heat dissipation:  ∂T/∂t = α·∇²T - β·(T - T_ambient)
GPU coherence:         ∂coherence/∂t = λ·(1 - coherence)
Pipeline stage:        ∂stage/∂t = (n - stage)/n
3D print cooling:      ∂T_layer/∂t = -T_layer/τ_cool
Memory bandwidth:      ∂load/∂t = (demand - load)/τ_memory
```

**Hardware Profiler → L3 MNA:**
```python
# CPU cores → nodes, bandwidth → conductance, thermal → capacitance
profile = HardwareProfiler().profile()
hw_mna = profiler.to_mna(profile)
tensor.update_level(3, hw_mna, t)
```

**GPU learning sources (encode these as DEQs, not separate projects):**
- VeriGPU: formal verification docs → temporal logic properties → DEQs
- tiny-gpu: pipeline state machines → state evolution DEQs

**Physics simulation verification:**
```python
# Constraints the system checks before accepting a GPU design
constraints = {
    'temperature': '< 85',   # °C
    'power': '< 300',        # W
    'coherence': '> 0.99',   # cache coherence
    'performance': '> 15'    # IPC
}
```

**3D printing pipeline:**
```
Optimization parameters  →  GeometricHDVPopulator  →  HDV encoding
→  UnifiedDEQSolver (solve for optimal G-code parameters)
→  G-code output  →  print  →  measure actual vs predicted
→  feed error back to network (Lyapunov energy update)
```

**Files:**
- `tensor/hardware_profiler.py` — L3 MNA builder
- `tensor/deq_system.py` — `UnifiedDEQSolver`, `GPUPhysicsSimulator`, `CircuitToDEQConverter`
- `tensor/compiler_stack.py` — φ/φ⁻¹ (compilation = coarse-graining)
- `GPU_HARDWARE_LEARNING.md` — VeriGPU + tiny-gpu integration plan
- `UNIFIED_DEQ_ARCHITECTURE.md` — complete DEQ conversion patterns

---

## The Unified HDV Space (Where Everything Meets)

All four domains project into a single **High-Dimensional Vector (HDV) space**:

```
dim = 10,000   (configurable)
universal overlap = first 33% (dims 0–3332) shared by ALL domains
domain-specific = remaining 67% (sparse random subsets per domain)
```

**How overlaps work:**
```python
hdv_math = encode_equation("∂x/∂t = -x/τ")      # math dimension active
hdv_code = encode_pattern("exponential_backoff")  # code dimension active
hdv_circuit = encode_circuit("RC_filter_1kHz")    # circuit dimension active

# All three activate the same universal dimensions (0–3332)
# cosine_similarity(hdv_math, hdv_code) ≈ 0.95  → UNIVERSAL FOUND
```

**Universal discovery = what makes the system get smarter:**
```
Same abstract structure (e.g., first-order exponential decay) found in:
  math dimension:    "decay equation" (arXiv paper)
  circuit dimension: "RC low-pass filter"
  code dimension:    "rate limiter with backoff"
  finance dimension: "mean-reversion signal"

→ Promote to UNIVERSAL → all domains now use the optimized version
→ Isometric constraint ensures distances preserved across projections
```

**Isometric regularization (ICLR 2025 foundation):**
```
||z₁ - z₂|| ≈ ||f(z₁) - f(z₂)||

Latent space distances must equal data manifold distances.
Minimal intrinsic curvature → robust representations even for small/noisy datasets.
```

---

## The FICUTS Learning Loop (5 Dimensions, Always Running)

```
┌─────────────────────────────────────────────────────────────────┐
│           UNIFIED TENSOR NETWORK (150 modes, 10k HDV)           │
│                                                                  │
│  Dim 1: Math       arXiv papers  → equations  → function basis  │
│  Dim 2: Behavioral GitHub/DeepWiki → patterns → dev-agent tmpl  │
│  Dim 3: Execution  Run code      → validate   → reinforce/supp  │
│  Dim 4: Optimize   Optuna trials → architecture → φ emergence   │
│  Dim 5: Physical   Parameters   → G-code      → measure → feed  │
│                                                                  │
│  All dimensions project into shared HDV space (33% overlap)     │
│  Cross-dimensional overlaps = universal pattern discovery        │
│  φ = 1.618... emerges in coupling ratios — not hardcoded        │
└─────────────────────────────────────────────────────────────────┘
```

**The 6-stage prediction loop (runs continuously):**
1. Extract concepts from current knowledge (mutual information ranking)
2. Predict next concept to learn (entropy minimization — explore sparse HDV dims)
3. Generate test problems to verify understanding (MDL)
4. Solve using current HDV patterns (unified DEQ solver)
5. Verify solution (substitution + Lyapunov + physics constraints)
6. Update HDV weights (Lyapunov energy decreases → learning is provably stable)

**Mathematical invariants of the loop:**
- `E(t+1) < E(t)` — Lyapunov energy strictly decreasing during learning
- FIM always positive semi-definite
- Eigenvalue ratios preserved under φ (coarse-graining)
- KCL satisfied at every node in every MNA system

---

## The GSD Loop (Autonomous Code Improvement)

```
Fisher priorities (L2 FIM top-k)
    ↓
create_improvement_project()   — define scope
    ↓
plan_phase(i)                  — atomic task plans from Fisher directions
    ↓
execute_phase(i)               — dev-agent writes/modifies code
    ↓
verify_phase(i)                — CodeValidator: re-parse → MNA → consonance delta
    ↓
if consonance improved AND tests pass → accept
else → rollback
    ↓
loop
```

**Consonance = structural health metric:**
```
eigenvalue ratios of G → musical intervals → consonance score (0–1)
octave (2:1) = perfect, fifth (3:2) = good, dissonance = tension
consonance > 0.75 → stable, healthy codebase
```

**Regime detection pauses GSD:**
```
eigenvalue gap narrows → transition_probability rises → should_pause = True
GSD waits → gap stabilizes → GSD resumes
```

---

## Key Mathematical Invariants (Never Break These)

1. **Eigenvalue ratios preserved under φ** — coarsening preserves computational semantics
2. **FIM always PSD** — normalize G before computing FIM (divide by mean diagonal)
3. **Free energy minimum = equilibrium** — system finds stable states naturally
4. **Consonance = structural health** — eigenvalue ratios near musical intervals = good code
5. **KCL at every node** — current conservation in all MNA systems
6. **Lyapunov decreasing** — `E(t+1) < E(t)` proves learning convergence
7. **Isometric constraint** — latent distances ≈ data distances (prevents overfitting)

---

## Current Repo State

### What exists and works
```
tensor/
├── core.py                    ✅ UnifiedTensor T ∈ ℝ^(L×N×N×t)
├── code_graph.py              ✅ AST → L2 MNA (import/call/inheritance edges)
├── market_graph.py            ✅ tickers → L0 MNA
├── neural_bridge.py           ✅ SNN → L1 MNA (free energy firing)
├── hardware_profiler.py       ✅ CPU/GPU/thermal → L3 MNA
├── compiler_stack.py          ✅ φ/φ⁻¹ compilation as coarse-graining
├── math_connections.py        ✅ 7 ECEMath→tensor bridges
├── gsd_bridge.py              ✅ GSD autonomous improvement cycle
├── dev_agent_bridge.py        ✅ dev-agent ↔ tensor interface
├── trading_bridge.py          ✅ FinBERT + tensor signal fusion
├── scraper_bridge.py          ✅ HTML → sentiment → L0
├── realtime_feed.py           ✅ WebSocket market data → L0
├── explorer.py                ✅ NAND/bandpass/SNN config search
├── observer.py                ✅ tensor snapshots + markdown reporting
├── integrated_hdv.py          ✅ HDV space (bug: find_overlaps returns 0, should be 33%)
├── deq_system.py              ✅ DEQ solver (bug: solve() returns dict, needs DEQ object)
├── arxiv_pdf_parser.py        ✅ LaTeX source download + equation extraction
├── function_basis.py          ✅ universal function basis library
├── cross_dimensional_discovery.py  ✅ universal pattern detection
├── curriculum_trainer.py      ✅ progressive learning (freeCodeCamp, books, Open3D)
├── deepwiki_navigator.py      ✅ DeepWiki + GitHub API integration
├── bootstrap_manager.py       ✅ autonomous resource integration
├── meta_optimizer.py          ✅ Optuna hyperparameter search
├── prediction_driven_learning.py  ✅ 6-stage prediction loop
└── geometric_population.py    🔲 TODO: structure-based HDV from raw LaTeX

ecemath/src/core/              ✅ Complete circuit math library (all 10 modules)
ecemath/examples/              ✅ HomeworkSolver (DC circuit solving)
dev-agent/                     ✅ 136-module autonomous coding agent
run_autonomous.py              ✅ Full autonomous learning CLI
run_system.py                  ✅ Full system orchestrator (4 threads)
```

### Known bugs to fix (in priority order)
1. `tensor/integrated_hdv.py` — `find_overlaps()` returns `set()` instead of `set(range(hdv_dim // 3))`
2. `tensor/deq_system.py` — `UnifiedDEQSolver.solve()` returns `dict` instead of `DifferentialEquation`
3. `tensor/deq_system.py` — `GPUPhysicsSimulator` accesses `.variables` but DEQ uses `.state_vars`
4. `tensor/math_connections.py` — FIM: G must be normalized before FIM (divide by `G.diagonal().mean()`)

### What to build next (in priority order)
1. Fix the 4 bugs above — they block cross-dimensional discovery
2. `tensor/geometric_population.py` — HDV from raw LaTeX structure (no semantic understanding needed)
3. `tensor/autonomous_training.py` — `ParallelPaperIngester` (ThreadPoolExecutor, learn during ingest)
4. Live dashboard HTTP server for Optuna visualization during optimization
5. GPU hardware DEQ integration (VeriGPU formal verification → temporal logic → DEQs)

---

## Activation (Press Start)

```bash
# Activate environment
conda activate tensor

# Full autonomous loop (this is the "leave it running" command)
python run_autonomous.py --populate --curriculum --discover --predict --optimize --trials 30

# What happens:
# → Downloads arXiv papers → extracts equations → encodes to HDV (math dim)
# → Trains on freeCodeCamp challenges → GitHub patterns → Open3D geometry
# → Runs prediction loop: predict → test → verify → update Lyapunov energy
# → Optuna tunes network architecture (watches for φ = 1.618 emergence in coupling ratios)
# → Discovers universals where math, code, and circuit HDVs overlap
# → GSD loop autonomously improves dev-agent code via Fisher priorities
# → FICUTS.md gets updated with discoveries as the system learns

# Status check
python run_autonomous.py --status

# Run tests
PYTHONPATH=. python -m pytest tests/ -q

# System orchestrator (all 4 tensor levels + trading + neural)
python run_system.py
```

**What the system does without you after activation:**
1. Ingests papers from arXiv, extracts equations, builds function basis
2. Scans GitHub/DeepWiki for behavioral patterns, encodes capability maps
3. Runs generated code, validates it, reinforces or suppresses HDV patterns
4. Searches for capability gaps in HDV space, finds repos to fill them
5. Detects cross-dimensional universals, promotes them to shared foundation
6. Improves its own codebase (dev-agent) using Fisher-guided GSD loop
7. Updates FICUTS docs with what it learned — journal of its own evolution

---

## The Yin-Yang of the System

The playful insight behind all of this:

- **Yin (exploration):** Stochastic perturbation, entropy maximization, sparse HDV overlap discovery
- **Yang (convergence):** Lyapunov stability, Fisher guidance, eigenvalue gap monitoring

Neither side dominates. The system explores when it has low Lyapunov energy and converges
when it has high entropy — exactly like thermodynamics. φ = 1.618 emerges at the
fixed point of the renormalization group — it is not put in, it is discovered.

The human is in the loop not because the system needs commands, but because this is a
**shared workspace**. The FICUTS document is the game board. Every session — human, Claude
Code, running system — adds its move. The math is the beauty. The system is singing along.

---

## File Transfer Protocol

To hand this session to a new Claude instance:
1. Pass this file (`UNIFIED_REASONING_ARCHITECTURE.md`) as initial context
2. Also pass `FICUTS_v3.0_UNIFIED.md` for the full task list with completion status
3. Run `python run_autonomous.py --status` and paste the output

To update this file from a running system, the `FICUTSUpdater` class in
`tensor/ficuts_updater.py` can call `append_hypothesis()` to record what was learned.

---

*Last updated by: Claude Code (branch: claude/consolidate-repo-math-structure-lOXEH)*
*Universals discovered so far: 0 (run the system to find them)*
*Next action: Fix the 4 bugs listed above, then `python run_autonomous.py --populate --curriculum`*
