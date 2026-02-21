"""
Code → Math → Hardware invariant extractor.

Two-level analysis of Python functions:

  Level 1 (static): AST pattern matching → complexity class → theoretical (ω₀, Q)
    - Works on any source code without execution
    - Less precise but safe and fast
    - domain = "hardware_static"

  Level 2 (dynamic): time execution at N=[64..4096] → empirical (ω₀, Q)
    - Requires controlled execution (trusted code only)
    - Precise: actual throughput and memory intensity
    - domain = "hardware_dynamic"

Both produce KoopmanInvariantDescriptor and store in KoopmanExperienceMemory
using the same (log_omega0_norm, log_Q_norm, damping_ratio) space as RLC/spring-mass.

Mapping to the shared invariant space:
  log_omega0_norm = (log(ω₀) - _HW_OMEGA0_REF) / _LOG_OMEGA0_SCALE
    where ω₀ = FLOPS/sec at N=1024 (dynamic) or N^α / τ_ref (static)
    reference: _HW_OMEGA0_REF = log(1e6) — 1 MFLOPS baseline
  log_Q_norm = log(arithmetic_intensity) / _LOG_OMEGA0_SCALE
    where arithmetic_intensity = FLOP / memory_byte_accesses
  damping_ratio = memory_fraction = 1 / (1 + arithmetic_intensity)

Cross-domain transfer: a new function with similar (ω₀, Q) to a stored
"hardware" experience retrieves that experience → suggests the same
optimization strategy (blocking, loop order, vectorisation hint).
"""

from __future__ import annotations

import ast
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from optimization.koopman_signature import (
    KoopmanInvariantDescriptor,
    compute_invariants,
    _LOG_OMEGA0_SCALE,
)
from optimization.koopman_memory import (
    KoopmanExperienceMemory,
    OptimizationExperience,
)
from tensor.koopman_edmd import EDMDKoopman

# ── Hardware reference constants ──────────────────────────────────────────────

# Reference: 1 MFLOP/sec (reasonable baseline for pure-Python functions).
# Normalised: ω₀ = 1e6 → log_omega0_norm = 0.0.
# GBlas routines (~1e9 FLOPS/sec) → log_omega0_norm ≈ +1.0 (one decade above ref).
_HW_OMEGA0_REF: float = math.log(1e6)          # log(1 MFLOPS/sec)

# Known complexity class → (complexity_exponent α, base_arithmetic_intensity Q)
# α: log(time) ≈ α × log(N)  — e.g. O(N²) → α=2
# Q: FLOP / memory_access ratio (dimensionless)
_COMPLEXITY_TABLE: Dict[str, Tuple[float, float]] = {
    "element_wise":   (1.0, 0.5),   # O(N),      memory-bound:  Q≈0.5
    "sort":           (1.32, 0.3),  # O(N log N), comparison-bound
    "fft":            (1.32, 1.5),  # O(N log N), moderate arithmetic
    "matvec":         (2.0, 2.0),   # O(N²),      moderate compute
    "matmul":         (3.0, 8.0),   # O(N³),      compute-bound: Q≈8+
    "conv_direct":    (2.5, 4.0),   # O(N²K),     depends on kernel
    "stencil":        (2.0, 1.0),   # O(N²),      moderate memory
    "reduction":      (1.0, 1.0),   # O(N),        minimal reuse
    "unknown":        (1.5, 1.0),   # conservative fallback
}

# Default profiling sizes (powers of 2, avoids edge effects)
_DEFAULT_SIZES: List[int] = [64, 128, 256, 512, 1024, 2048, 4096]
_MIN_PROFILE_POINTS: int = 5   # minimum sizes for reliable Koopman fit


# ── Static analysis ────────────────────────────────────────────────────────────


@dataclass
class MathPattern:
    """Classification of a function's mathematical structure."""
    complexity_class: str          # key into _COMPLEXITY_TABLE
    complexity_exponent: float     # α from log-linear fit
    arithmetic_intensity: float    # Q (dimensionless)
    dominant_op: str               # human-readable description
    confidence: float              # 0–1
    notes: str = ""


class ASTMathClassifier:
    """
    Static AST analysis → MathPattern.

    Detects:
      - numpy calls (np.dot, np.matmul, np.fft.*, np.sort, np.convolve…)
      - nested for/while loops over arrays → stencil or matmul-like
      - comprehensions with arithmetic → reduction or element-wise
      - recursive calls → divide-and-conquer (sort / FFT structure)

    Returns MathPattern with theoretical complexity.
    """

    # numpy attribute calls → pattern
    _NUMPY_PATTERNS: Dict[str, str] = {
        "dot":           "matvec",
        "matmul":        "matmul",
        "inner":         "matmul",
        "outer":         "matmul",
        "tensordot":     "matmul",
        "einsum":        "matmul",
        "fft":           "fft",
        "ifft":          "fft",
        "rfft":          "fft",
        "irfft":         "fft",
        "fftn":          "fft",
        "sort":          "sort",
        "argsort":       "sort",
        "convolve":      "conv_direct",
        "correlate":     "conv_direct",
        "sum":           "reduction",
        "cumsum":        "reduction",
        "prod":          "reduction",
        "mean":          "reduction",
        "std":           "reduction",
        "var":           "reduction",
        "max":           "reduction",
        "min":           "reduction",
        "add":           "element_wise",
        "multiply":      "element_wise",
        "subtract":      "element_wise",
        "divide":        "element_wise",
        "exp":           "element_wise",
        "log":           "element_wise",
        "sqrt":          "element_wise",
        "abs":           "element_wise",
    }

    # scipy/sklearn top-level call patterns
    _FUNC_NAME_PATTERNS: Dict[str, str] = {
        "fft":       "fft",
        "ifft":      "fft",
        "sort":      "sort",
        "sorted":    "sort",
        "matmul":    "matmul",
        "dot":       "matvec",
        "convolve":  "conv_direct",
    }

    def classify(self, source: str) -> MathPattern:
        """
        Parse source code and return MathPattern.
        Falls back to "unknown" on syntax error or ambiguity.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._unknown("syntax error")

        votes: Dict[str, int] = {}   # complexity_class → vote count
        dominant_ops: List[str] = []

        for node in ast.walk(tree):
            # numpy.X() or np.X() calls
            if isinstance(node, ast.Call):
                name = self._extract_call_name(node)
                if name in self._NUMPY_PATTERNS:
                    key = self._NUMPY_PATTERNS[name]
                    votes[key] = votes.get(key, 0) + 3   # high weight
                    dominant_ops.append(name)
                elif name in self._FUNC_NAME_PATTERNS:
                    key = self._FUNC_NAME_PATTERNS[name]
                    votes[key] = votes.get(key, 0) + 2
                    dominant_ops.append(name)
                # @ operator turns into ast.BinOp with ast.MatMult
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                votes["matmul"] = votes.get("matmul", 0) + 3
                dominant_ops.append("@")

        # Nested loops → stencil or matmul candidate
        loop_depth = self._max_loop_depth(tree)
        if loop_depth >= 3:
            votes["matmul"] = votes.get("matmul", 0) + 2
        elif loop_depth == 2:
            votes["stencil"] = votes.get("stencil", 0) + 1
        elif loop_depth == 1:
            votes["element_wise"] = votes.get("element_wise", 0) + 1

        # Recursive calls → divide-and-conquer
        if self._has_recursion(tree):
            votes["sort"] = votes.get("sort", 0) + 1   # safe default for D&C

        if not votes:
            return self._unknown("no patterns detected")

        best_class = max(votes, key=votes.__getitem__)
        total_votes = sum(votes.values())
        confidence = min(1.0, votes[best_class] / max(total_votes, 1))

        α, Q = _COMPLEXITY_TABLE[best_class]
        dom_op = dominant_ops[0] if dominant_ops else best_class
        return MathPattern(
            complexity_class=best_class,
            complexity_exponent=α,
            arithmetic_intensity=Q,
            dominant_op=dom_op,
            confidence=confidence,
            notes=f"votes={votes}  loop_depth={loop_depth}",
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_call_name(node: ast.Call) -> str:
        """Extract the last attribute/name from a call node."""
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return ""

    @staticmethod
    def _max_loop_depth(tree: ast.AST) -> int:
        """Return the maximum nesting depth of for/while loops."""
        max_depth = [0]

        def walk(node, depth):
            if isinstance(node, (ast.For, ast.While)):
                depth += 1
                max_depth[0] = max(max_depth[0], depth)
            for child in ast.iter_child_nodes(node):
                walk(child, depth)

        walk(tree, 0)
        return max_depth[0]

    @staticmethod
    def _has_recursion(tree: ast.AST) -> bool:
        """Detect if any function calls itself (direct recursion)."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_name = node.name
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        name = ""
                        if isinstance(child.func, ast.Name):
                            name = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            name = child.func.attr
                        if name == fn_name:
                            return True
        return False

    @staticmethod
    def _unknown(reason: str) -> MathPattern:
        α, Q = _COMPLEXITY_TABLE["unknown"]
        return MathPattern(
            complexity_class="unknown",
            complexity_exponent=α,
            arithmetic_intensity=Q,
            dominant_op="unknown",
            confidence=0.0,
            notes=reason,
        )


def pattern_to_invariants(
    pattern: MathPattern,
    n_ref: int = 1024,
    tau_ref_sec: float = 1e-6,   # assumed 1 µs per base operation
) -> Tuple[float, float, float]:
    """
    Convert a MathPattern to (log_omega0_norm, log_Q_norm, damping_ratio).

    Theoretical estimate (no execution required):
      ω₀ = n_ref^α / (τ_ref × n_ref^α) = 1/τ_ref   — throughput at N=n_ref
      (The N^α terms cancel, so ω₀ is dominated by τ_ref, the per-operation latency.)

    Actually more useful: ω₀ = n_ref / time_est where time_est = n_ref^α × τ_ref
    = 1 / (n_ref^(α-1) × τ_ref)  — effective throughput in ops/sec at N=n_ref.

    This makes O(N) algorithms (α=1) have the highest ω₀ (fastest per element),
    and O(N³) algorithms (α=3) have the lowest ω₀ (most work per element).
    """
    α = pattern.complexity_exponent
    Q = max(pattern.arithmetic_intensity, 1e-12)

    # ω₀ = throughput at N_ref: elements processed per second
    omega0 = 1.0 / (max(n_ref, 1) ** max(α - 1.0, 0.0) * max(tau_ref_sec, 1e-30))
    log_omega0_norm = float(np.clip(
        (math.log(max(omega0, 1e-30)) - _HW_OMEGA0_REF) / _LOG_OMEGA0_SCALE,
        -3.0, 3.0,
    ))
    log_Q_norm = float(np.clip(
        math.log(Q) / _LOG_OMEGA0_SCALE,
        -3.0, 3.0,
    ))
    zeta = 1.0 / (1.0 + Q)   # memory fraction: Q→∞ means compute-bound (ζ→0)
    return log_omega0_norm, log_Q_norm, float(np.clip(zeta, 0.0, 1.0))


# ── Dynamic profiling ─────────────────────────────────────────────────────────


@dataclass
class TimingRow:
    N: int
    wall_sec: float
    flops_estimate: float
    mem_bytes_estimate: float


class DynamicProfiler:
    """
    Time a callable at multiple input sizes → empirical (ω₀, Q, ζ).

    Args:
        n_warmup: number of warm-up calls before timing (to fill CPU caches)
        n_repeat: number of timed repetitions (median is used)
    """

    def __init__(self, n_warmup: int = 2, n_repeat: int = 5) -> None:
        self.n_warmup = n_warmup
        self.n_repeat = n_repeat

    def profile(
        self,
        fn: Callable,
        input_gen: Callable[[int], Any],
        sizes: Optional[List[int]] = None,
    ) -> List[TimingRow]:
        """
        Run fn(input_gen(N)) for each size N.

        Args:
            fn:         function to profile
            input_gen:  N → input (e.g. lambda N: np.random.randn(N))
            sizes:      list of N values (default: powers of 2 from 64 to 4096)

        Returns:
            List of TimingRow sorted by N.
        """
        sizes = sizes or _DEFAULT_SIZES
        rows = []
        for N in sizes:
            inp = input_gen(N)
            # Warm up
            for _ in range(self.n_warmup):
                try:
                    fn(inp)
                except Exception:
                    break
            # Time
            times = []
            for _ in range(self.n_repeat):
                t0 = time.perf_counter()
                try:
                    fn(inp)
                except Exception:
                    times.append(float("inf"))
                    break
                times.append(time.perf_counter() - t0)
            wall = float(np.median(times))
            # Estimate flops and memory (heuristic: N elements, each touched once)
            flops_est = float(N)          # lower bound; overridden by caller if known
            mem_est   = float(N * 8)      # 8 bytes per float64
            rows.append(TimingRow(N=N, wall_sec=wall,
                                  flops_estimate=flops_est,
                                  mem_bytes_estimate=mem_est))
        return rows

    def fit_scaling(self, rows: List[TimingRow]) -> Tuple[float, float]:
        """
        Fit log-linear model: log(time) = α × log(N) + β.
        Returns (α, β) where α is the complexity exponent.
        """
        valid = [(r.N, r.wall_sec) for r in rows
                 if r.wall_sec > 0 and math.isfinite(r.wall_sec)]
        if len(valid) < 2:
            return 1.0, 0.0
        log_N = np.array([math.log(n) for n, _ in valid])
        log_t = np.array([math.log(t) for _, t in valid])
        # Least-squares fit
        A = np.stack([log_N, np.ones_like(log_N)], axis=1)
        result = np.linalg.lstsq(A, log_t, rcond=None)
        α, β = float(result[0][0]), float(result[0][1])
        return α, β

    def empirical_invariants(
        self,
        rows: List[TimingRow],
        n_ref: int = 1024,
        flops_fn: Optional[Callable[[int], float]] = None,
        mem_fn: Optional[Callable[[int], float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute (log_omega0_norm, log_Q_norm, damping_ratio) from timing rows.

        ω₀ = flops(n_ref) / time(n_ref)   [FLOPS/sec]
        Q  = flops(n_ref) / mem(n_ref)     [FLOP/byte] = arithmetic intensity

        Args:
            flops_fn:  N → total_flops (if None: uses N)
            mem_fn:    N → total_bytes_accessed (if None: uses N×8)
        """
        # Interpolate time at n_ref
        t_at_ref = self._interpolate_time(rows, n_ref)
        if t_at_ref <= 0:
            t_at_ref = 1e-6

        flops_at_ref = flops_fn(n_ref) if flops_fn else float(n_ref)
        mem_at_ref   = mem_fn(n_ref)   if mem_fn   else float(n_ref * 8)

        omega0 = flops_at_ref / t_at_ref       # FLOPS/sec
        Q      = flops_at_ref / max(mem_at_ref, 1.0)   # FLOP/byte

        log_omega0_norm = float(np.clip(
            (math.log(max(omega0, 1e-30)) - _HW_OMEGA0_REF) / _LOG_OMEGA0_SCALE,
            -3.0, 3.0,
        ))
        log_Q_norm = float(np.clip(
            math.log(max(Q, 1e-30)) / _LOG_OMEGA0_SCALE,
            -3.0, 3.0,
        ))
        zeta = 1.0 / (1.0 + Q)
        return log_omega0_norm, log_Q_norm, float(np.clip(zeta, 0.0, 1.0))

    @staticmethod
    def _interpolate_time(rows: List[TimingRow], n_target: int) -> float:
        """Linear interpolation in log-log space."""
        valid = sorted([(r.N, r.wall_sec) for r in rows
                        if r.wall_sec > 0 and math.isfinite(r.wall_sec)])
        if not valid:
            return 1e-6
        # Find bracketing rows
        for i in range(len(valid) - 1):
            n0, t0 = valid[i]
            n1, t1 = valid[i + 1]
            if n0 <= n_target <= n1:
                frac = math.log(n_target / n0) / math.log(n1 / n0)
                return math.exp(math.log(t0) + frac * (math.log(t1) - math.log(t0)))
        # Extrapolate from last two points
        if len(valid) >= 2:
            n0, t0 = valid[-2]
            n1, t1 = valid[-1]
            frac = math.log(n_target / n0) / math.log(max(n1 / n0, 1.0001))
            return math.exp(math.log(t0) + frac * (math.log(t1) - math.log(t0)))
        return valid[0][1]

    def build_koopman_trace(self, rows: List[TimingRow]) -> Optional[np.ndarray]:
        """
        Build a state-space trace for EDMD Koopman from timing rows.

        State vector at each N: [log(N)/10, log(time)/10, log(N/time)/10]
        Normalised by /10 to keep values in [-1, +1] range.

        Returns (T, 3) array or None if too few points.
        """
        valid = [(r.N, r.wall_sec) for r in rows
                 if r.wall_sec > 0 and math.isfinite(r.wall_sec)]
        if len(valid) < 3:
            return None
        states = []
        for N, t in valid:
            states.append([
                math.log(max(N, 1)) / 10.0,
                math.log(max(t, 1e-30)) / 10.0,
                math.log(max(N / max(t, 1e-30), 1e-30)) / 10.0,
            ])
        return np.array(states)


# ── Top-level profiler ────────────────────────────────────────────────────────


class CodeHardwareProfiler:
    """
    Profile a Python function and store the result in KoopmanExperienceMemory.

    Combines static AST analysis (always) with optional dynamic timing
    (when the function is trusted / safe to execute).

    Usage:
        profiler = CodeHardwareProfiler()

        # Static only:
        inv = profiler.analyse_static(source_code, fn_name="my_fn")

        # Dynamic (trusted code):
        inv = profiler.analyse_dynamic(fn, lambda N: np.random.randn(N), fn_name="my_fn")

        # Store in memory:
        profiler.store(inv, memory, fn_name="my_fn", hints={"block_size": 32})
    """

    def __init__(self, n_warmup: int = 2, n_repeat: int = 5) -> None:
        self._classifier = ASTMathClassifier()
        self._dyn = DynamicProfiler(n_warmup=n_warmup, n_repeat=n_repeat)
        self._edmd = EDMDKoopman(observable_degree=1)

    # ── Static analysis ────────────────────────────────────────────────────────

    def analyse_static(
        self,
        source: str,
        fn_name: str = "unknown",
    ) -> KoopmanInvariantDescriptor:
        """
        AST analysis only → theoretical KoopmanInvariantDescriptor.
        No execution required.

        The descriptor has empty eigenvalues (no execution trace) but
        carries the (log_omega0_norm, log_Q_norm, damping_ratio) from
        complexity analysis.
        """
        pattern = self._classifier.classify(source)
        log_w, log_q, zeta = pattern_to_invariants(pattern)
        return compute_invariants(
            np.array([]), np.zeros((0, 0)), [],
            log_omega0_norm=log_w,
            log_Q_norm=log_q,
            damping_ratio=zeta,
        )

    # ── Dynamic profiling ──────────────────────────────────────────────────────

    def analyse_dynamic(
        self,
        fn: Callable,
        input_gen: Callable[[int], Any],
        fn_name: str = "unknown",
        sizes: Optional[List[int]] = None,
        flops_fn: Optional[Callable[[int], float]] = None,
        mem_fn: Optional[Callable[[int], float]] = None,
    ) -> KoopmanInvariantDescriptor:
        """
        Profile fn and return KoopmanInvariantDescriptor with empirical (ω₀, Q).

        Fits EDMDKoopman to the scaling trace (log_N, log_time, throughput)
        to get eigenvalues that encode scaling behaviour.

        Args:
            fn:         callable to profile
            input_gen:  N → input argument for fn
            fn_name:    name for labelling
            sizes:      list of N values (default powers of 2, 64–4096)
            flops_fn:   N → total FLOPS performed (for arithmetic intensity)
            mem_fn:     N → total bytes accessed (for arithmetic intensity)
        """
        rows = self._dyn.profile(fn, input_gen, sizes)
        log_w, log_q, zeta = self._dyn.empirical_invariants(
            rows, flops_fn=flops_fn, mem_fn=mem_fn
        )

        # Fit Koopman to the scaling trace for eigenvalues (fine-verify stage)
        kt = self._dyn.build_koopman_trace(rows)
        eigenvalues = np.array([])
        eigenvectors = np.zeros((0, 0))
        if kt is not None and len(kt) >= 3:
            try:
                pairs = [(kt[i], kt[i + 1]) for i in range(len(kt) - 1)]
                self._edmd.fit(pairs)
                kr = self._edmd.eigendecomposition()
                eigenvalues = kr.eigenvalues
                eigenvectors = kr.eigenvectors
            except Exception:
                pass

        return compute_invariants(
            eigenvalues, eigenvectors, [],
            log_omega0_norm=log_w,
            log_Q_norm=log_q,
            damping_ratio=zeta,
        )

    # ── Memory storage ─────────────────────────────────────────────────────────

    def store(
        self,
        invariant: KoopmanInvariantDescriptor,
        memory: KoopmanExperienceMemory,
        fn_name: str,
        complexity_class: str = "unknown",
        hints: Optional[Dict] = None,
        domain: str = "hardware_static",
    ) -> None:
        """
        Store a profiled function's invariant in KoopmanExperienceMemory.

        best_params encodes optimization hints for the code:
          {"fn_name": ..., "complexity": ..., "block_size": ...,
           "vectorise": ..., "use_fft": ..., "notes": ...}

        The stored experience can be retrieved for new functions with
        similar (ω₀, Q) to suggest the same optimization approach.
        """
        from tensor.koopman_edmd import KoopmanResult
        # Build a minimal placeholder signature for confirm_match()
        n_eig = max(len(invariant.top_k_real), 1)
        eigs = np.array([0.9 + 0j] * n_eig)
        vecs = np.eye(n_eig, dtype=complex)
        sig = KoopmanResult(
            eigenvalues=eigs,
            eigenvectors=vecs,
            K_matrix=np.diag(np.real(eigs)),
            spectral_gap=0.0,
            is_stable=True,
        )

        bp = {"fn_name": fn_name, "complexity": complexity_class}
        if hints:
            bp.update(hints)

        exp = OptimizationExperience(
            bottleneck_operator=complexity_class,
            replacement_applied="code_analysis",
            runtime_improvement=0.5,
            n_observations=1,
            hardware_target="cpu",
            best_params=bp,
            domain=domain,
        )
        memory.add(invariant, sig, exp)


# ── Convenience: profile + store in one call ──────────────────────────────────


def profile_and_store(
    source: str,
    memory: KoopmanExperienceMemory,
    fn_name: str = "unknown",
    fn: Optional[Callable] = None,
    input_gen: Optional[Callable[[int], Any]] = None,
    sizes: Optional[List[int]] = None,
    flops_fn: Optional[Callable[[int], float]] = None,
    mem_fn: Optional[Callable[[int], float]] = None,
    hints: Optional[Dict] = None,
) -> KoopmanInvariantDescriptor:
    """
    Analyse source code (statically, or dynamically if fn+input_gen provided)
    and store the result in memory.

    Returns the KoopmanInvariantDescriptor for inspection.

    If fn and input_gen are provided, performs dynamic profiling (preferred).
    Otherwise falls back to static AST analysis.
    """
    profiler = CodeHardwareProfiler()
    classifier = ASTMathClassifier()
    pattern = classifier.classify(source)

    if fn is not None and input_gen is not None:
        inv = profiler.analyse_dynamic(fn, input_gen, fn_name=fn_name,
                                       sizes=sizes, flops_fn=flops_fn, mem_fn=mem_fn)
        domain = "hardware_dynamic"
    else:
        inv = profiler.analyse_static(source, fn_name=fn_name)
        domain = "hardware_static"

    profiler.store(inv, memory, fn_name=fn_name,
                   complexity_class=pattern.complexity_class,
                   hints=hints, domain=domain)
    return inv
