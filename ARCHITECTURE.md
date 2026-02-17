# Unified Tensor System — Architecture

## Core Equation

Everything derives from MNA (Modified Nodal Analysis):

```
C(Θ)·v̇ + G(Θ)·v + h(v;Θ) = u(t)
```

Where `C` = capacitance (state storage), `G` = conductance (energy flow), `h` = nonlinear terms, `u` = external drive. This equation governs circuits, but here it governs **software systems, markets, neural networks, and hardware** — all represented as the same mathematical object.

## The Tensor: T ∈ R^(L × N × N × t)

A 4-level time-varying tensor where each level is an MNA system:

| Level | Domain | Nodes | Edges |
|-------|--------|-------|-------|
| **L0** | Market data | Tickers (AAPL, MSFT...) | Correlations, sentiment |
| **L1** | Neural network | SNN neurons | Synaptic weights |
| **L2** | Codebase | Python modules/files | Import, call, inheritance |
| **L3** | Hardware | CPU cores, memory, GPU | Thermal/bandwidth coupling |

Each level has its own MNA matrices (G, C), state vector, and dynamics. Cross-level interactions create a unified system.

**Key operations:**
- `update_level(i, mna, t)` — inject new MNA at level i
- `coarsen_to(from, to)` — φ: compress from→to (compiler = coarsening)
- `lift_from(from, to)` — φ⁻¹: expand from→to (hardware optimization = lifting)
- `eigenvalue_gap(level)` — regime indicator (gap narrows → instability)
- `free_energy_map(level)` — F = E - τS + γH per node (firing rule)
- `harmonic_signature(level)` — eigenvalue ratios → musical intervals → consonance score

## Subsystems

### ECEMath (`ecemath/src/core/`)

Pure NumPy circuit math library. No ML dependencies.

| Module | Purpose |
|--------|---------|
| `matrix.py` | MNASystem dataclass, ExtendedMNABuilder (R/C/L/V stamps) |
| `components.py` | Resistor, Capacitor, Inductor, VoltageSource, MOSFET |
| `graph.py` | CircuitGraph with Node/Edge |
| `dynamics.py` | CircuitDynamics: C·dv/dt = -G·v - h(v) + u |
| `solver.py` | CircuitSolver: find_equilibrium(), simulate() |
| `fisher.py` | FisherInformation: I(Θ) = J^T·Σ⁻¹·J, certainty ellipsoids |
| `regime.py` | RegimeSwitchingSystem: Markov chain over dynamics regimes |
| `stochastic.py` | Euler-Maruyama, Milstein SDE solvers |
| `sparse_solver.py` | Harmonic signature, free energy firing, consonance scoring |
| `coarsening.py` | CoarseGrainingOperator: φ preserves eigenvalue ratios |

### Dev-Agent Bridge (`tensor/dev_agent_bridge.py`)

Connects the tensor to the dev-agent (136-module autonomous coding system at `dev-agent/`).

**How it works:**
1. `CodeGraph.from_directory()` parses dev-agent source via AST
2. Import/call/inheritance relationships → MNA edges → L2 tensor
3. `free_energy_map()` identifies high-tension modules (need improvement)
4. `harmonic_signature()` scores structural consonance
5. `proposal_weights()` ranks which modules to improve next
6. Hotspot boost (complexity × centrality) ensures high-impact modules get priority

**The GSD loop uses this to decide WHAT to improve.**

### GSD Bridge (`tensor/gsd_bridge.py`)

**G**et **S**hit **D**one — the autonomous improvement cycle:

```
create_improvement_project()  →  defines scope
    plan_phase(i)             →  creates atomic task plans
    execute_phase(i)          →  dev-agent executes tasks
    verify_phase(i)           →  tensor validator checks results
```

- Plans are derived from Fisher-guided priorities (FIM eigenvalues)
- Execution delegates to dev-agent
- Verification uses `CodeValidator` (re-parse → MNA → consonance delta)
- If consonance improved: accept. If degraded: rollback.

### Trading Bot Bridge (`tensor/trading_bridge.py`)

Enhances FinBERT sentiment scores with tensor intelligence:

```
enhanced_score = α·finbert + β·tensor_signal + γ·regime_weight
```

Where:
- `tensor_signal` = free energy at ticker's L0 node (high tension = strong signal)
- `regime_weight` = market regime (calm/volatile/crisis) from eigenvalue gap
- `harmonic_signature` = structural market consonance → confidence multiplier

### Scraper Bridge (`tensor/scraper_bridge.py`)

HTML article → ticker mentions (regex) → sentiment (lexicon) → `MarketGraph.sentiment_injection()` → L0 tensor update. Feeds the trading bridge with real-time article data.

### Compiler Stack (`tensor/compiler_stack.py`)

**φ (coarse-graining) IS the compiler. φ⁻¹ (lifting) IS hardware optimization.**

```
L_python (AST)  →φ→  L_bytecode (dis)  →φ→  L_asm (x86)  →φ→  L_hardware (gates)
L_hardware      →φ⁻¹→  L_asm           →φ⁻¹→  optimal high-level
```

- `python_to_mna()` — AST nodes as circuit, complexity as resistance
- `bytecode_to_mna()` — opcodes as nodes, control flow as edges
- `asm_to_mna()` — x86 instructions, latency as resistance
- `phi_between(high, low)` — coarsening operator between levels
- Each φ preserves eigenvalue ratios (computational semantics preserved)
- `cross_language_report()` — consonance preservation across all levels

### Neural Bridge (`tensor/neural_bridge.py`)

SNN (Spiking Neural Network) layer at L1:

- `NeuralBridge(tensor, n_neurons)` — creates SNN MNA
- `forward()` — one SNN timestep
- `update_tensor()` — inject SNN state into L1
- `run_continuous()` — background thread, continuous SNN simulation
- Free energy firing rule: F(node) = E - τS + γH; fire when F < θ

### Realtime Feed (`tensor/realtime_feed.py`)

WebSocket-based market data → L0:

- `connect_yahoo()`, `connect_coinbase()`, `connect_mock()`
- `on_tick()` — price update → MarketGraph → L0 MNA
- Background thread, reconnect on failure
- `status()` → used by `check_feed_health()` for monitoring

### Hardware Profiler (`tensor/hardware_profiler.py`)

L3 = hardware layer:

- `profile()` → CPU cores, memory, GPU, thermals
- `to_mna()` → cores as nodes, bandwidth as conductance, thermal as capacitance
- Refreshed periodically in `run_system.py`

## Math Connections (`tensor/math_connections.py`)

Seven bridges from ECEMath to the tensor improvement loop:

| # | Connection | Input → Output | Purpose |
|---|-----------|----------------|---------|
| 1 | Fisher → GSD | FIM eigenvalues → priority indices | Which modules to improve first |
| 2 | Regime → monitoring | Eigenvalue gap → should_pause | Pause GSD during regime transitions |
| 3 | Stochastic → explorer | Monte Carlo noise → robustness score | Validate configuration stability |
| 4 | Neural error → GSD | Predicted vs actual L1 → error weights | Weight GSD tasks by prediction error |
| 5 | SNN firing → L1 | Free energy → activation mask | Selective neural layer update |
| 6 | Pytest → jump events | Test pass rate → discontinuity | Test results as L2 regime jumps |
| 7 | Feed health → monitoring | Staleness/status → warnings | L0 data quality monitoring |

All available as standalone functions or via `MathConnections(tensor)` class.

## The Autonomous Loop (`run_system.py`)

```
┌─────────────────────────────────────────────────────┐
│                   SystemRunner                       │
│                                                      │
│  Thread 1: RealtimeFeed ──→ MarketGraph ──→ L0      │
│  Thread 2: NeuralBridge ──→ SNN step ──→ L1         │
│  Thread 3: HardwareProfiler (every 5min) ──→ L3     │
│                                                      │
│  Main loop:                                          │
│    1. GSD: plan_phase → execute_phase → verify_phase │
│    2. Explorer: NAND target optimization             │
│    3. Observer: snapshot_markdown every N seconds     │
│    4. MathConnections: regime check, feed health     │
│                                                      │
│  Fisher priorities guide GSD task selection           │
│  Regime detection pauses GSD during transitions      │
│  Test results create jump events in L2               │
│  Feed health monitors L0 data quality                │
└─────────────────────────────────────────────────────┘
```

## How Dev-Agent Uses This

The dev-agent is a 136-module autonomous coding system. The tensor system gives it:

1. **What to work on**: Fisher-guided priorities from L2 code graph
2. **When to pause**: Regime detection (eigenvalue gap narrowing = instability)
3. **How to validate**: Consonance delta before/after changes
4. **Market context**: L0 regime affects trading bot parameters
5. **Hardware awareness**: L3 profile informs resource allocation
6. **Self-improvement**: GSD loop autonomously plans, executes, validates code changes

The dev-agent executes GSD tasks. The tensor validates results. If consonance improves and tests pass, changes are accepted. The cycle repeats.

## Key Mathematical Invariants

1. **Eigenvalue ratios preserved under φ** — coarsening doesn't destroy computational semantics
2. **FIM PSD** — Fisher Information Matrix is always positive semi-definite
3. **Free energy minimum = equilibrium** — system naturally finds stable states
4. **Consonance = structural health** — eigenvalue ratios near musical intervals = well-structured code
5. **KCL at every node** — current conservation holds in all MNA systems (verified by tests)

## File Map

```
unified-tensor-system/
├── tensor/                    # Core tensor framework
│   ├── core.py                # UnifiedTensor T ∈ R^(L×N×N×t)
│   ├── code_graph.py          # Codebase → L2 MNA (AST parsing)
│   ├── market_graph.py        # Market → L0 MNA
│   ├── neural_bridge.py       # SNN → L1 MNA
│   ├── hardware_profiler.py   # Hardware → L3 MNA
│   ├── compiler_stack.py      # φ/φ⁻¹ cross-level compilation
│   ├── math_connections.py    # 7 ECEMath→tensor bridges + MathConnections class
│   ├── gsd_bridge.py          # GSD autonomous improvement cycle
│   ├── dev_agent_bridge.py    # Dev-agent ↔ tensor interface
│   ├── trading_bridge.py      # Trading bot enhancement
│   ├── scraper_bridge.py      # HTML → sentiment → L0
│   ├── realtime_feed.py       # WebSocket market data → L0
│   ├── explorer.py            # Configuration space search (NAND, bandpass, SNN)
│   ├── bootstrap.py           # Bootstrap orchestrator
│   ├── code_validator.py      # Consonance-based code validation
│   ├── observer.py            # Tensor snapshots + markdown reporting
│   └── skill_writer.py        # Skill library management
├── ecemath/                   # Circuit-theoretic math (pure NumPy)
│   ├── src/core/              # MNA, dynamics, Fisher, regime, stochastic, coarsening
│   └── examples/              # HomeworkSolver (DC circuit solving)
├── dev-agent/                 # 136-module autonomous coding agent
├── tradingCode/               # Trading bot pipeline
├── tradingBot/                # Trading bot deployment
├── run_system.py              # Full system orchestrator
├── run_explorer.py            # Standalone explorer runner
└── tests/                     # 71 tests across all subsystems
```

## Test Coverage (71 tests)

- `test_tensor_all.py` — UnifiedTensor, CodeGraph, MarketGraph, HardwareProfiler
- `test_bridges.py` — DevAgentBridge, TradingBridge, Observer, full system integration
- `test_full_stack.py` — Bootstrap, CompilerStack, CodeValidator, Explorer, SkillWriter
- `test_math_connections.py` — All 7 math connections + full loop integration
- `test_homework.py` — DC circuit solver (KCL, power balance, manifold)
- `test_integration.py` — Cross-subsystem integration
- `test_explorer.py` — NAND, bandpass, SNN, code structure targets
- `test_tracks_abc.py` — Track-based testing
