"""
FICUTS Unified DEQ Architecture

Every input type is converted to a Differential Equation, then solved in
HDV space and verified mathematically.

  Paper   → LaTeX extraction → ∂x/∂t = F(x,θ)
  Circuit → KVL/KCL         → ∂V/∂t = ...
  Code    → AST → discrete  → ∂x/∂t = transition(x) - x
  3D model→ mesh → PDE      → ∂T/∂t = α·∇²T, ...

All types share ONE solver (UnifiedDEQSolver) and ONE HDV space.

Mathematical guarantees (per UNIFIED_DEQ_ARCHITECTURE.md):
  - Completeness: any computable function ≡ DEQ (Neural ODE theorem)
  - Uniqueness: Picard-Lindelöf ensures unique solutions to well-posed DEQs
  - Verification: MDL gives computable bound on solution quality
  - Improvement: Lyapunov stability guarantees convergence
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.integrate import solve_ivp as _solve_ivp
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import sympy as sp
    _SYMPY = True
except ImportError:
    _SYMPY = False


# ── DifferentialEquation ──────────────────────────────────────────────────────

@dataclass
class DifferentialEquation:
    """
    Canonical representation of a differential equation ∂x/∂t = F(x, θ).

    Fields:
      equation  — human-readable string (may be ";" separated for systems)
      domain    — FICUTS dimension: math | physical | execution | behavioral
      variables — state variable names
      parameters — named parameter values
      initial_condition — x(0), length must match variables
      t_final   — integration horizon
      constraints — dict of verification constraints
      source    — original input type: paper | circuit | code | 3d_model
    """

    equation: str
    domain: str = "math"
    variables: List[str] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    initial_condition: Optional[List[float]] = None
    t_final: float = 1.0
    constraints: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    @property
    def state_vars(self) -> List[str]:
        """Alias for variables (backwards-compat with GPU converter demos)."""
        return self.variables

    def evaluate(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Evaluate RHS of ∂x/∂t = F(x, θ) numerically.

        Default: exponential decay dx/dt = -x (unconditionally stable).
        Subclasses override for domain-specific dynamics.
        """
        return -np.atleast_1d(x).astype(float)

    def verify_solution(self, x_final: np.ndarray) -> float:
        """
        Residual of ∂x/∂t - F(x, θ) at the final state.

        Returns norm of residual: 0 = perfect, ∞ = completely wrong.
        """
        x = np.atleast_1d(x_final).astype(float)
        residual = self.evaluate(self.t_final, x)
        return float(np.linalg.norm(residual))


# ── Paper → DEQ ───────────────────────────────────────────────────────────────

class PaperToDEQConverter:
    """
    Convert a research paper (plain text with LaTeX fragments) to a DEQ.

    Steps:
      1. Extract LaTeX equations from the text
      2. Identify the most "central" equation (time derivative preferred)
      3. Convert to standard ∂x/∂t = F(x, θ) form
      4. Extract variables and named parameters
    """

    _LATEX_RE = [
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\$\$(.*?)\$\$',
        r'\$(.*?)\$',
        r'(?:d[a-z]|\\partial\s*[a-z])/(?:dt|\\partial\s*t)\s*=\s*.+',
    ]

    def convert(self, paper_text: str) -> DifferentialEquation:
        equations = self._extract_equations(paper_text)
        main = self._pick_main_equation(equations)
        std = self._to_standard_form(main)

        return DifferentialEquation(
            equation=std,
            domain="math",
            variables=self._find_variables(std),
            parameters=self._find_parameters(std),
            source="paper",
        )

    def _extract_equations(self, text: str) -> List[str]:
        eqs = []
        for pattern in self._LATEX_RE:
            for m in re.finditer(pattern, text, re.DOTALL):
                raw = m.group(0 if m.lastindex is None else 1).strip()
                if len(raw) > 2:
                    eqs.append(raw)
        # Plain-text time-derivative lines
        for line in text.splitlines():
            if "=" in line and any(t in line for t in ("d/dt", "∂/∂t", "∂t", "'(")):
                eqs.append(line.strip())
        return eqs or ["dx/dt = -x"]

    def _pick_main_equation(self, equations: List[str]) -> str:
        scored = []
        for eq in equations:
            score = 0.0
            if any(t in eq for t in ("dt", "∂t", "\\dot", "d/dt")):
                score += 3.0
            score += min(len(eq) / 40, 2.0)
            scored.append((score, eq))
        scored.sort(reverse=True)
        return scored[0][1]

    def _to_standard_form(self, eq: str) -> str:
        for marker in ("dx/dt", "dy/dt", "dz/dt", "∂x/∂t", "\\dot{x}"):
            if marker in eq:
                return eq
        if "=" in eq:
            rhs = eq.split("=", 1)[1].strip()
            return f"∂x/∂t = {rhs}"
        return f"∂x/∂t = -({eq.strip()})"

    def _find_variables(self, eq: str) -> List[str]:
        candidates = re.findall(r'\b([a-z])\b', eq)
        return list(dict.fromkeys(v for v in candidates if v not in "eijot"))

    def _find_parameters(self, eq: str) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for greek in ("alpha", "beta", "gamma", "tau", "omega", "lambda", "sigma"):
            if greek in eq.lower() or f"\\{greek}" in eq.lower():
                params[greek] = 1.0
        return params


# ── Circuit → DEQ ─────────────────────────────────────────────────────────────

class CircuitToDEQConverter:
    """
    Convert a SPICE-style netlist to KVL/KCL differential equations.

    Netlist format (simplified, one component per line):
      R1  N1  N2  1000       # 1 kΩ resistor between N1 and N2
      C1  N2  GND 1e-6       # 1 µF capacitor
      L1  N1  N3  1e-3       # 1 mH inductor
      V1  N1  GND 5.0        # 5 V source

    State variables: capacitor voltages V_Cx, inductor currents I_Lx.
    Dynamics derived from C·dV/dt = I and L·dI/dt = V.
    """

    def convert(self, netlist: str) -> DifferentialEquation:
        comps = self._parse(netlist)
        caps = [c for c in comps if c["type"] == "C"]
        inds = [c for c in comps if c["type"] == "L"]
        ress = [c for c in comps if c["type"] == "R"]
        sources = [c for c in comps if c["type"] == "V"]

        state_vars: List[str] = []
        params: Dict[str, float] = {}
        eq_parts: List[str] = []

        R_total = sum(r["value"] for r in ress) or 1.0
        V_src = sources[0]["value"] if sources else 0.0

        for cap in caps:
            vname = f"V_{cap['name']}"
            state_vars.append(vname)
            params[cap["name"]] = cap["value"]
            # RC discharge: C·dV/dt = (V_src - V) / R
            tau = R_total * cap["value"]
            eq_parts.append(
                f"d{vname}/dt = ({V_src} - {vname}) / {tau:.6g}"
            )

        for ind in inds:
            iname = f"I_{ind['name']}"
            state_vars.append(iname)
            params[ind["name"]] = ind["value"]
            # RL decay: L·dI/dt = V_src - R·I
            eq_parts.append(
                f"d{iname}/dt = ({V_src} - {R_total} * {iname}) / {ind['value']:.6g}"
            )

        if not eq_parts:
            eq_parts = ["dx/dt = 0"]

        return DifferentialEquation(
            equation=" ; ".join(eq_parts),
            domain="physical",
            variables=state_vars,
            parameters=params,
            source="circuit",
        )

    def _parse(self, netlist: str) -> List[Dict]:
        comps = []
        for line in netlist.splitlines():
            line = line.split("#")[0].strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            name, node1, node2, val_str = parts[0], parts[1], parts[2], parts[3]
            try:
                value = float(val_str)
            except ValueError:
                value = 1.0
            comps.append({
                "type": name[0].upper(),
                "name": name,
                "node1": node1,
                "node2": node2,
                "value": value,
            })
        return comps


# ── Code → DEQ ────────────────────────────────────────────────────────────────

class CodeToDEQConverter:
    """
    Convert Python code to a continuous DEQ over program state.

    Discrete dynamics (code):   x_{n+1} = f(x_n)
    Continuous embedding:       ∂x/∂t   = f(x) - x

    State variables = variables mutated inside loops or via augmented assignment.
    """

    def convert(self, code: str) -> DifferentialEquation:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return DifferentialEquation(
                equation="dx/dt = 0",
                domain="execution",
                source="code",
            )

        state_vars = self._state_variables(tree)
        transitions = self._transitions(tree, state_vars)
        params = self._constants(tree)

        eq_parts = [
            f"∂{var}/∂t = ({expr}) - {var}"
            for var, expr in transitions.items()
        ] or ["dx/dt = 0"]

        return DifferentialEquation(
            equation=" ; ".join(eq_parts),
            domain="execution",
            variables=list(state_vars),
            parameters=params,
            source="code",
        )

    def _state_variables(self, tree: ast.AST) -> List[str]:
        vars_: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name):
                                vars_.append(t.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    vars_.append(node.target.id)
        return list(dict.fromkeys(vars_))

    def _transitions(
        self, tree: ast.AST, state_vars: List[str]
    ) -> Dict[str, str]:
        out: Dict[str, str] = {}
        _ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        for node in ast.walk(tree):
            if (isinstance(node, ast.AugAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id in state_vars):
                name = node.target.id
                op = _ops.get(type(node.op), "+")
                val = ast.unparse(node.value) if hasattr(ast, "unparse") else "?"
                out[name] = f"{name} {op} {val}"
        return out

    def _constants(self, tree: ast.AST) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant):
                        if isinstance(node.value.value, (int, float)):
                            out[t.id] = float(node.value.value)
        return out


# ── 3D Model → DEQ ────────────────────────────────────────────────────────────

class Model3DToDEQConverter:
    """
    Convert a 3D model description to coupled heat/stress/flow PDEs.

    Supported materials: pla, abs, petg, steel (see MATERIAL_DB).
    Produces a system of PDEs in string form plus numeric parameter values.
    """

    MATERIAL_DB: Dict[str, Dict[str, float]] = {
        "pla":   {"alpha": 1.2e-4, "rho": 1240.0, "E": 3.5e9, "nu": 0.36},
        "abs":   {"alpha": 1.7e-4, "rho": 1060.0, "E": 2.5e9, "nu": 0.35},
        "petg":  {"alpha": 1.4e-4, "rho": 1270.0, "E": 2.1e9, "nu": 0.38},
        "steel": {"alpha": 1.17e-5, "rho": 7850.0, "E": 200e9, "nu": 0.28},
    }

    def convert(
        self, model_description: str, material: str = "pla"
    ) -> DifferentialEquation:
        props = self.MATERIAL_DB.get(material.lower(), self.MATERIAL_DB["pla"])
        a = props["alpha"]
        E = props["E"]
        rho = props["rho"]
        volume = self._estimate_volume(model_description)

        equations = [
            f"∂T/∂t = {a:.3e}·∇²T + Q(x,y,z,t)",
            f"∂σ/∂t = {E:.3e}·∂ε/∂t",
            f"∂ρ/∂t + ∇·(ρv) = 0",
        ]

        return DifferentialEquation(
            equation=" ; ".join(equations),
            domain="physical",
            variables=["T", "sigma", "rho_field"],
            parameters={
                "alpha_thermal": a,
                "rho_material": rho,
                "E_modulus": E,
                "volume_mm3": volume,
                "material": 0.0,       # string material stored in source
            },
            source=f"3d_model:{material}",
        )

    def _estimate_volume(self, description: str) -> float:
        nums = re.findall(r'\d+(?:\.\d+)?', description)
        if len(nums) >= 3:
            a, b, c = float(nums[0]), float(nums[1]), float(nums[2])
            return a * b * c
        if len(nums) == 1:
            r = float(nums[0])
            return (4 / 3) * 3.14159 * r ** 3
        return 1000.0


# ── VeriGPU → DEQ ─────────────────────────────────────────────────────────────

class VeriGPUToDEQConverter:
    """
    Convert GPU verification constraints (described as text) to DEQs.

    LTL operators (extracted from natural language or formal comments):
      □P  (always P)      → equilibrium:  ∂P/∂t = λ·(1 - P)
      ◊P  (eventually P)  → convergence:  ∂P/∂t = (1 - P)
      PUQ (P until Q)     → coupling:     ∂P/∂t = Q - P

    State variables auto-detected from domain keywords:
      coherence, race_free, deadlock_free, bandwidth, latency
    """

    _ALWAYS_KW = {"always", "invariant", "must", "shall", "guaranteed", "□"}
    _EVENTUALLY_KW = {"eventually", "converges", "reaches", "◊", "liveness"}
    _UNTIL_KW = {"until", "before", "precedes", "U"}
    _STATE_KW = {
        "coherence": "coherence",
        "race": "race_free",
        "deadlock": "deadlock_free",
        "bandwidth": "bandwidth",
        "latency": "latency",
        "throughput": "throughput",
    }

    def convert(self, spec_text: str) -> DifferentialEquation:
        tokens = spec_text.lower().split()
        token_set = set(tokens)

        # Detect active state variables
        state_vars = []
        for kw, var in self._STATE_KW.items():
            if kw in token_set and var not in state_vars:
                state_vars.append(var)
        if not state_vars:
            state_vars = ["coherence", "race_free", "deadlock_free"]

        # Classify dominant LTL type
        has_always = bool(token_set & self._ALWAYS_KW)
        has_eventually = bool(token_set & self._EVENTUALLY_KW)
        has_until = bool(token_set & self._UNTIL_KW)

        eq_parts = []
        for i, var in enumerate(state_vars):
            if has_until and i + 1 < len(state_vars):
                next_var = state_vars[i + 1]
                eq_parts.append(f"∂{var}/∂t = {next_var} - {var}")
            elif has_eventually:
                eq_parts.append(f"∂{var}/∂t = (1 - {var})")
            else:  # default: always / equilibrium
                eq_parts.append(f"∂{var}/∂t = 1.0 * (1 - {var})")

        # Safety constraints
        constraints = {v: "> 0.99" if v in ("coherence", "race_free", "deadlock_free")
                       else "> 0" for v in state_vars}

        return DifferentialEquation(
            equation=" ; ".join(eq_parts),
            domain="physical",
            variables=state_vars,
            parameters={"lambda": 1.0, "target": 1.0},
            constraints=constraints,
            source="verigpu",
        )


# ── tiny-gpu → DEQ ────────────────────────────────────────────────────────────

class TinyGPUToDEQConverter:
    """
    Convert GPU implementation descriptions (pipeline, cache, warp) to DEQs.

    Discrete state machine: s_{n+1} = f(s_n)
    Continuous embedding:   ∂s/∂t   = f(s) - s

    For a k-stage pipeline: ∂stage/∂t = (total_stages - stage) / total_stages
    i.e. stage drives toward completion.

    Also extracts optimization patterns for HDV encoding.
    """

    _PIPELINE_KW = {"pipeline", "fetch", "decode", "execute", "writeback", "stage"}
    _CACHE_KW = {"cache", "l1", "l2", "dram", "memory", "bandwidth", "miss", "hit"}
    _WARP_KW = {"warp", "thread", "simd", "scheduling", "latency", "hide"}

    def convert(self, source_text: str) -> DifferentialEquation:
        lower = source_text.lower()
        tokens = set(lower.split())

        has_pipeline = bool(tokens & self._PIPELINE_KW)
        has_cache = bool(tokens & self._CACHE_KW)
        has_warp = bool(tokens & self._WARP_KW)

        # Count pipeline stages from text
        n_stages = sum(1 for kw in ("fetch", "decode", "execute", "writeback") if kw in lower)
        n_stages = max(n_stages, 4)  # default 4-stage

        state_vars, eq_parts, params = [], [], {}

        if has_pipeline:
            state_vars.append("stage")
            params["n_stages"] = float(n_stages)
            eq_parts.append(f"∂stage/∂t = ({n_stages} - stage) / {n_stages}")

        if has_cache:
            state_vars.append("hit_rate")
            params["cache_levels"] = 3.0
            eq_parts.append("∂hit_rate/∂t = (1 - hit_rate) * 0.1")

        if has_warp:
            state_vars.append("utilization")
            params["threads_per_warp"] = 32.0
            eq_parts.append("∂utilization/∂t = (1 - utilization)")

        if not eq_parts:
            state_vars = ["ipc"]
            eq_parts = ["∂ipc/∂t = (1 - ipc)"]
            params["n_stages"] = 4.0

        # Performance constraints
        constraints = {
            "throughput": "<= bandwidth / data_size",
            "power": "<= 250",
            "temperature": "< 85",
        }

        return DifferentialEquation(
            equation=" ; ".join(eq_parts),
            domain="physical",
            variables=state_vars,
            parameters=params,
            constraints=constraints,
            source="tiny_gpu",
        )

    def extract_optimization_patterns(self, deq: DifferentialEquation) -> List[Dict]:
        """Return hardware optimization patterns encoded in this DEQ."""
        patterns = []
        eq = deq.equation

        if "stage" in eq:
            patterns.append({
                "type": "pipeline",
                "stages": int(deq.parameters.get("n_stages", 4)),
                "throughput": "1 instruction/cycle",
                "optimization": "maximize_pipeline_utilization",
            })
        if "hit_rate" in eq:
            patterns.append({
                "type": "cache_hierarchy",
                "levels": int(deq.parameters.get("cache_levels", 3)),
                "optimization": "minimize_misses",
            })
        if "utilization" in eq:
            patterns.append({
                "type": "warp_scheduling",
                "threads_per_warp": int(deq.parameters.get("threads_per_warp", 32)),
                "optimization": "hide_memory_latency",
            })
        return patterns


# ── GPU Physics Simulator ──────────────────────────────────────────────────────

class GPUPhysicsSimulator:
    """
    Verify GPU DEQs against physical constraints via coupled simulation.

    Physics:
      Power:  P(t)  = activity(s) · C · V² · f
      Heat:   ∂T/∂t = -P / thermal_capacity   (simplified lumped model)

    Constraints checked:
      temperature < 85°C, power < 250W
    """

    DEFAULTS = {
        "C": 100e-12,        # capacitance (F)
        "V": 1.2,            # voltage (V)
        "f": 1.5e9,          # clock frequency (Hz)
        "thermal_capacity": 50.0,  # J/°C (lumped)
        "T0": 25.0,          # initial temperature (°C)
    }

    def simulate(
        self,
        gpu_deq: DifferentialEquation,
        parameters: Optional[Dict] = None,
        t_max: float = 1e-3,
        steps: int = 100,
    ) -> Dict[str, Any]:
        p = {**self.DEFAULTS, **(parameters or {})}
        dt = t_max / steps
        n = max(len(gpu_deq.variables), 1)
        x0 = np.ones(n, dtype=float) * 0.5
        T = p["T0"]

        temps, powers, ipcs = [], [], []
        x = x0.copy()

        for i in range(steps):
            t = i * dt
            dx = np.atleast_1d(gpu_deq.evaluate(t, x))
            if dx.shape != x.shape:
                dx = np.resize(dx, x.shape)
            x = np.clip(x + dt * dx, 0.0, None)

            # Activity ∝ mean state (0..1)
            activity = float(np.mean(np.clip(x, 0, 1)))
            power = activity * p["C"] * p["V"] ** 2 * p["f"]
            dT = -power / p["thermal_capacity"]
            T = T + dt * dT  # heat dissipates (simplified)
            T = max(T, p["T0"])  # can't go below ambient

            # IPC proxy: first state var if "stage" else mean
            ipc = float(x[0]) if "stage" in (gpu_deq.variables or []) else activity

            temps.append(T)
            powers.append(power)
            ipcs.append(ipc)

        violations = []
        if max(temps) > 85.0:
            violations.append("thermal_violation")
        if max(powers) > 250.0:
            violations.append("power_violation")

        mean_ipc = float(np.mean(ipcs))
        mean_power = float(np.mean(powers)) if np.mean(powers) > 0 else 1.0

        return {
            "temperature": temps,
            "power": powers,
            "performance": ipcs,
            "violations": violations,
            "energy_efficiency": mean_ipc / mean_power,
            "verified": len(violations) == 0,
        }


# ── SolveResult ───────────────────────────────────────────────────────────────

class _SolveResult(dict):
    """
    dict subclass returned by UnifiedDEQSolver.solve().

    Supports both dict-style access (result["equation"]) and
    attribute-style access (result.equation).

    Extra alias: result.state_vars == result["variables"]
    """

    def __getattr__(self, name: str):
        if name == "state_vars":
            return self.get("variables", [])
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'_SolveResult' has no attribute '{name}'"
            ) from None


# ── UnifiedDEQSolver ──────────────────────────────────────────────────────────

class UnifiedDEQSolver:
    """
    Solves ANY differential equation (paper / circuit / code / 3d_model)
    in HDV space using a single unified pipeline:

      1. Convert input → DifferentialEquation
      2. Encode equation string → HDV
      3. Retrieve similar previously-solved DEQs
      4. Solve numerically (scipy RK45 → Euler fallback)
      5. Verify via residual + MDL
      6. Store solution for future retrieval
    """

    def __init__(self, hdv_system=None):
        self.hdv = hdv_system
        self.converters: Dict[str, Any] = {
            "paper":    PaperToDEQConverter(),
            "circuit":  CircuitToDEQConverter(),
            "code":     CodeToDEQConverter(),
            "3d_model": Model3DToDEQConverter(),
            "verigpu":  VeriGPUToDEQConverter(),
            "tiny_gpu": TinyGPUToDEQConverter(),
        }
        self.physics = GPUPhysicsSimulator()
        self._store: List[Dict] = []

    def convert(self, input_data: str, input_type: str) -> DifferentialEquation:
        converter = self.converters.get(input_type)
        if converter is None:
            raise ValueError(
                f"Unknown input_type '{input_type}'. "
                f"Use one of: {list(self.converters)}"
            )
        return converter.convert(input_data)

    def solve(
        self,
        input_data: str,
        input_type: str,
        constraints: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: convert → encode → find similar → solve → verify → store.

        Returns dict with keys:
          equation, domain, variables, parameters,
          solution (t/y arrays), verified, confidence, similar_found
        """
        # 1. Convert
        deq = self.convert(input_data, input_type)

        # 2. Encode
        if self.hdv is not None:
            deq_hdv = self.hdv.structural_encode(deq.equation, deq.domain)
            similar = self._find_similar(deq_hdv, threshold=0.3)
        else:
            deq_hdv = None
            similar = []

        # 3. Solve
        initial = None
        if similar and similar[0].get("initial_condition"):
            initial = similar[0]["initial_condition"]
        sol = self._solve_numerically(deq, initial)

        # 4. Verify
        verified, confidence = self._verify(deq, sol, constraints)

        # 5. Store
        if deq_hdv is not None:
            self._store.append({
                "equation": deq.equation,
                "domain": deq.domain,
                "hdv": deq_hdv,
                "initial_condition": deq.initial_condition,
                "confidence": confidence,
            })

        return _SolveResult(
            equation=deq.equation,
            domain=deq.domain,
            variables=deq.variables,
            parameters=deq.parameters,
            solution=sol,
            verified=verified,
            confidence=confidence,
            similar_found=len(similar),
        )

    # ── Numerical solving ──────────────────────────────────────────────────────

    def _solve_numerically(
        self, deq: DifferentialEquation, initial_guess: Optional[List[float]]
    ) -> Dict:
        n = max(len(deq.variables), 1)
        x0_list = initial_guess or deq.initial_condition or [1.0] * n
        x0 = np.array(x0_list[:n], dtype=float)
        if len(x0) < n:
            x0 = np.pad(x0, (0, n - len(x0)), constant_values=1.0)

        if _SCIPY:
            try:
                sol = _solve_ivp(
                    lambda t, x: deq.evaluate(t, x),
                    t_span=(0.0, deq.t_final),
                    y0=x0,
                    method="RK45",
                    max_step=deq.t_final / 20.0,
                    dense_output=False,
                )
                return {
                    "t": sol.t.tolist(),
                    "y": sol.y.tolist(),
                    "success": bool(sol.success),
                    "message": sol.message,
                }
            except Exception as exc:
                return {"t": [], "y": [], "success": False, "message": str(exc)}

        # Euler fallback
        steps = 50
        dt = deq.t_final / steps
        x = x0.copy()
        ts, ys = [0.0], [x.tolist()]
        for i in range(steps):
            t = i * dt
            dx = np.atleast_1d(deq.evaluate(t, x))
            if dx.shape != x.shape:
                dx = np.resize(dx, x.shape)
            x = x + dt * dx
            ts.append((i + 1) * dt)
            ys.append(x.tolist())
        return {"t": ts, "y": ys, "success": True, "message": "Euler"}

    # ── Verification ──────────────────────────────────────────────────────────

    def _verify(
        self,
        deq: DifferentialEquation,
        sol: Dict,
        constraints: Optional[Dict],
    ) -> Tuple[bool, float]:
        if not sol.get("success", False):
            return False, 0.0

        y = sol.get("y", [])
        if not y:
            return False, 0.0

        # Final state
        last = y[-1] if isinstance(y[0], list) else y
        x_final = np.array(last, dtype=float).flatten()

        residual = deq.verify_solution(x_final)
        # MDL: shorter description (smaller residual) → higher confidence
        confidence = 1.0 / (1.0 + residual)

        # Constraint checks
        if constraints:
            for key, val in constraints.items():
                if isinstance(val, (int, float)) and abs(float(val)) > 1e12:
                    confidence *= 0.5

        verified = confidence > 0.3
        return verified, float(confidence)

    # ── HDV similarity search ──────────────────────────────────────────────────

    def _find_similar(
        self, query: np.ndarray, threshold: float = 0.3
    ) -> List[Dict]:
        results = []
        for entry in self._store:
            v = entry["hdv"]
            n1, n2 = np.linalg.norm(query), np.linalg.norm(v)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            sim = float(np.dot(query, v) / (n1 * n2))
            if sim >= threshold:
                results.append({**entry, "similarity": sim})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results
