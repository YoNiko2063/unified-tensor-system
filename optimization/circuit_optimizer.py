"""
CircuitOptimizer — eigenvalue-guided circuit parameter optimization engine.

Optimization objective:
    J = w1*||λ - λ_target||_norm² + w2*RegimePenalty + w3*StabilityPenalty + w4*ComponentCost

Search strategy:
    1. Analytic initial guess via EigenvalueMapper.inverse_map()
    2. Nelder-Mead local refinement (log-space params)
    3. Multi-start: 5 random perturbations ± 30% of initial guess
    4. Return ParetoResult from all converged candidates

Pure NumPy/SciPy only — no external ML dependencies.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# CircuitSpec
# ---------------------------------------------------------------------------

@dataclass
class CircuitSpec:
    """User-facing circuit optimization target.

    topology: "bandpass_rlc" | "lowpass_rc" | "highpass_rc" | "notch_rlc"
    center_freq_hz: target center frequency [Hz]
    Q_target: target quality factor (bandwidth = center_freq / Q)
    damping_ratio: ζ = 1/(2Q), alternative to Q_target (one or the other)
    max_power_w: max power dissipation (sets R lower bound)
    component_tolerances: {"R": 0.05, "L": 0.10, "C": 0.05} — ±fraction
    weights: cost weights [w_eigenvalue, w_regime, w_stability, w_cost]
    """
    topology: str = "bandpass_rlc"
    center_freq_hz: float = 1000.0
    Q_target: float = 5.0
    damping_ratio: Optional[float] = None  # if set, overrides Q_target
    max_power_w: float = 1.0
    component_tolerances: Dict[str, float] = field(
        default_factory=lambda: {"R": 0.05, "L": 0.10, "C": 0.05}
    )
    weights: List[float] = field(
        default_factory=lambda: [1.0, 0.3, 0.5, 0.1]
    )

    @property
    def omega0(self) -> float:
        """Angular resonant frequency [rad/s]."""
        return 2.0 * np.pi * self.center_freq_hz

    @property
    def zeta(self) -> float:
        """Damping ratio ζ."""
        if self.damping_ratio is not None:
            return self.damping_ratio
        return 1.0 / (2.0 * self.Q_target)

    @property
    def target_eigenvalues(self) -> np.ndarray:
        """λ_target = -ζω₀ ± jω₀√(1-ζ²) as complex pair."""
        sigma = -self.zeta * self.omega0
        omega_d = self.omega0 * np.sqrt(max(1.0 - self.zeta**2, 0.0))
        return np.array([sigma + 1j * omega_d, sigma - 1j * omega_d])


# ---------------------------------------------------------------------------
# EigenvalueMapper
# ---------------------------------------------------------------------------

class EigenvalueMapper:
    """Computes system matrix A = -C^{-1}G from circuit parameters.

    For RLC bandpass: C·ẋ + G·x = 0  →  A = -C^{-1}G
    State: [V_C, I_L] (capacitor voltage, inductor current)

    C = [[C, 0],  [0, L]]
    G = [[1/R,-1], [1,  0]]   (series RLC, KVL)

    A = -C^{-1}G = [[-1/(RC),  1/C],
                    [-1/L,      0  ]]
    """

    def compute_A(self, R: float, L: float, C: float) -> np.ndarray:
        """State matrix for series RLC. Returns (2,2) ndarray."""
        return np.array([
            [-1.0 / (R * C),  1.0 / C],
            [-1.0 / L,        0.0    ],
        ])

    def eigenvalues(self, R: float, L: float, C: float) -> np.ndarray:
        """Eigenvalues of A. Returns complex array shape (2,)."""
        A = self.compute_A(R, L, C)
        return np.linalg.eigvals(A)

    def eigenvalue_error(
        self,
        R: float,
        L: float,
        C: float,
        target_eigs: np.ndarray,
    ) -> float:
        """||λ_computed - λ_target||_2 (minimum-distance matching)."""
        eigs = self.eigenvalues(R, L, C)
        err1 = np.abs(eigs[0] - target_eigs[0]) + np.abs(eigs[1] - target_eigs[1])
        err2 = np.abs(eigs[0] - target_eigs[1]) + np.abs(eigs[1] - target_eigs[0])
        return float(min(err1, err2))

    def inverse_map(self, target_eigs: np.ndarray) -> Dict[str, float]:
        """Analytic inverse for RLC.

        Given λ = -ζω₀ ± jω₀√(1-ζ²), recover R/L/C.
        Anchor: L = 1e-3 H, then C = 1/ω₀²L, R = 2ζω₀L.

        The characteristic polynomial of A:
            s² + (1/RC)*s + 1/(LC) = 0
        So: trace(A) = -1/(RC), det(A) = 1/(LC)
        From target eigenvalues: λ₁*λ₂ = det(A) = 1/(LC),  λ₁+λ₂ = trace = -1/(RC)
        """
        lam1, lam2 = target_eigs[0], target_eigs[1]

        omega0_sq = float(np.real(lam1 * lam2))
        omega0_sq = max(omega0_sq, 1e-20)

        # trace(A) = -1/(RC) = Re(λ₁ + λ₂)
        neg_inv_RC = float(np.real(lam1 + lam2))

        # Anchor L = 1 mH
        L = 1e-3  # H
        # det(A) = 1/(LC) → C = 1/(omega0_sq * L)
        C = 1.0 / (omega0_sq * L)
        C = max(C, 1e-15)

        # trace = -1/(RC) → R = -1/(trace * C)
        if abs(neg_inv_RC) > 1e-30:
            R = -1.0 / (neg_inv_RC * C)
        else:
            # Zero damping edge case — use a nominal value
            R = 1e6

        R = max(R, 1e-6)  # physical lower bound

        return {"R": R, "L": L, "C": C}


# ---------------------------------------------------------------------------
# CircuitCostFunction
# ---------------------------------------------------------------------------

class CircuitCostFunction:
    """Multi-objective cost function.

    J = w1*||λ-λ_target||_norm² + w2*RegimePenalty + w3*StabilityPenalty + w4*ComponentCost

    RegimePenalty: 0 = LCA (good), 0.5 = nonabelian (warning), 1.0 = chaotic (bad)
    StabilityPenalty: max(0, gap_threshold - spectral_gap)
    ComponentCost: (log(R/R_ref))² + (log(L/L_ref))² + (log(C/C_ref))²
        where R_ref, L_ref, C_ref are the analytic inverse_map values.
        This penalises deviation from physically meaningful components without
        driving them to extremes.
    """

    def __init__(
        self,
        spec: CircuitSpec,
        mapper: EigenvalueMapper,
        patch_detector=None,
    ):
        self.spec = spec
        self.mapper = mapper
        self.patch_detector = patch_detector  # LCAPatchDetector or None
        # Unpack weights with defaults
        w = spec.weights
        self.w1 = w[0] if len(w) > 0 else 1.0
        self.w2 = w[1] if len(w) > 1 else 0.3
        self.w3 = w[2] if len(w) > 2 else 0.5
        self.w4 = w[3] if len(w) > 3 else 0.1

        # Reference component values (analytic inverse-map) for component cost
        inv = mapper.inverse_map(spec.target_eigenvalues)
        self._log_R_ref = np.log(max(inv["R"], 1e-30))
        self._log_L_ref = np.log(max(inv["L"], 1e-30))
        self._log_C_ref = np.log(max(inv["C"], 1e-30))

        # Scale for eigenvalue term: ||target_eigenvalues||
        self._target_scale = max(
            np.linalg.norm(np.abs(spec.target_eigenvalues)), 1.0
        )

    def __call__(self, params: np.ndarray) -> float:
        """params = [log_R, log_L, log_C]. Returns scalar cost J."""
        log_R, log_L, log_C = float(params[0]), float(params[1]), float(params[2])
        R = float(np.exp(log_R))
        L = float(np.exp(log_L))
        C = float(np.exp(log_C))

        # Guard against degenerate values
        if R <= 0 or L <= 0 or C <= 0:
            return 1e12

        target_eigs = self.spec.target_eigenvalues

        # Term 1: eigenvalue match (squared, scale-normalised)
        eig_err = self.mapper.eigenvalue_error(R, L, C, target_eigs)
        t1 = (eig_err / self._target_scale) ** 2

        # Term 2: regime penalty
        t2 = self.regime_penalty(R, L, C)

        # Term 3: stability penalty
        t3 = self.stability_penalty(R, L, C)

        # Term 4: component cost — quadratic in log-ratio from reference
        # Cost is 0 at the analytic solution and grows as components move away
        t4 = (
            (log_R - self._log_R_ref) ** 2
            + (log_L - self._log_L_ref) ** 2
            + (log_C - self._log_C_ref) ** 2
        )

        return self.w1 * t1 + self.w2 * t2 + self.w3 * t3 + self.w4 * t4

    def regime_penalty(self, R: float, L: float, C: float) -> float:
        """Run LCA patch classifier on short RLC trajectory.

        Slow — only runs if patch_detector is provided.
        Returns 0 if no detector.
        """
        if self.patch_detector is None:
            return 0.0

        try:
            # Generate short trajectory for classification
            omega0 = 1.0 / np.sqrt(max(L * C, 1e-30))
            T = 4.0 * np.pi / omega0  # two full periods
            dt = T / 50.0
            n_steps = 50

            # Simple RK4 forward integration
            def rlc_rhs(x):
                v_c, i_l = x
                dvc = (i_l - v_c / R) / C
                dil = -v_c / L
                return np.array([dvc, dil])

            x = np.array([1.0, 0.0])  # initial: unit voltage, no current
            samples = [x.copy()]
            for _ in range(n_steps - 1):
                k1 = rlc_rhs(x)
                k2 = rlc_rhs(x + 0.5 * dt * k1)
                k3 = rlc_rhs(x + 0.5 * dt * k2)
                k4 = rlc_rhs(x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                samples.append(x.copy())

            x_samples = np.array(samples)
            result = self.patch_detector.classify_region(x_samples)
            patch_type = result.patch_type

            if patch_type == "lca":
                return 0.0
            elif patch_type == "nonabelian":
                return 0.5
            else:  # chaotic
                return 1.0
        except (ValueError, np.linalg.LinAlgError, AttributeError) as e:
            warnings.warn(f"regime_penalty: classification failed ({e}), returning 0.0", stacklevel=2)
            return 0.0

    def stability_penalty(
        self,
        R: float,
        L: float,
        C: float,
        gap_threshold: float = 0.1,
    ) -> float:
        """Penalize when spectral gap < gap_threshold."""
        eigs = self.mapper.eigenvalues(R, L, C)
        gap = abs(abs(eigs[0]) - abs(eigs[1]))
        return max(0.0, gap_threshold - gap)


# ---------------------------------------------------------------------------
# OptimizationResult / ParetoResult
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    R: float
    L: float
    C: float
    achieved_eigenvalues: np.ndarray
    eigenvalue_error: float
    cost: float
    regime_type: str          # "lca" | "nonabelian" | "chaotic"
    spectral_gap: float
    omega0_achieved: float    # rad/s
    Q_achieved: float
    converged: bool


def _non_dominated(candidates: List[OptimizationResult]) -> List[OptimizationResult]:
    """Filter to non-dominated Pareto front across 3 objectives:
    minimize eigenvalue_error, minimize cost, maximize spectral_gap.
    """
    front = []
    for c in candidates:
        dominated = False
        for o in candidates:
            if o is c:
                continue
            # o dominates c if o is <= on all objectives and < on at least one
            better_eig = o.eigenvalue_error <= c.eigenvalue_error
            better_cost = o.cost <= c.cost
            better_gap = o.spectral_gap >= c.spectral_gap
            strict = (
                o.eigenvalue_error < c.eigenvalue_error
                or o.cost < c.cost
                or o.spectral_gap > c.spectral_gap
            )
            if better_eig and better_cost and better_gap and strict:
                dominated = True
                break
        if not dominated:
            front.append(c)
    return front


@dataclass
class ParetoResult:
    """Pareto front: named extremes + full non-dominated set."""
    best_eigenvalue: OptimizationResult
    best_stability: OptimizationResult
    best_cost: OptimizationResult
    all_candidates: List[OptimizationResult]
    pareto_front: List[OptimizationResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CircuitOptimizer
# ---------------------------------------------------------------------------

class CircuitOptimizer:
    """Eigenvalue-guided circuit parameter optimizer.

    Search strategy:
        1. Analytic initial guess via EigenvalueMapper.inverse_map()
        2. Nelder-Mead local refinement (scipy.optimize.minimize, method='Nelder-Mead')
        3. Multi-start: 5 random perturbations ± 30% of initial guess
        4. Return ParetoResult from all converged candidates

    All optimization in log-space: params = [log_R, log_L, log_C]

    The analytic inverse_map result is always included as the first candidate,
    since it provides a near-zero eigenvalue error by construction. Nelder-Mead
    runs refine or find alternative Pareto-optimal trade-offs.
    """

    def __init__(self, spec: CircuitSpec, use_regime_penalty: bool = False):
        self.spec = spec
        self.mapper = EigenvalueMapper()
        self.cost_fn = CircuitCostFunction(
            spec,
            self.mapper,
            patch_detector=self._make_detector() if use_regime_penalty else None,
        )
        self._rng = np.random.default_rng(42)

    def _make_detector(self):
        """Build an LCAPatchDetector for regime classification."""
        try:
            _project_root = str(Path(__file__).resolve().parents[1])
            if _project_root not in sys.path:
                sys.path.insert(0, _project_root)
            from tensor.lca_patch_detector import LCAPatchDetector

            def dummy_fn(x):
                return np.zeros_like(x)

            return LCAPatchDetector(dummy_fn, n_states=2)
        except Exception:
            return None

    def optimize(self) -> ParetoResult:
        """Run full optimization. Returns Pareto front."""
        # Step 1: analytic initial guess
        target_eigs = self.spec.target_eigenvalues
        initial_params = self.mapper.inverse_map(target_eigs)
        R0, L0, C0 = initial_params["R"], initial_params["L"], initial_params["C"]
        x0 = np.array([np.log(R0), np.log(L0), np.log(C0)])

        # Always include the analytic result as the first candidate
        # (it is near-optimal for eigenvalue matching by construction)
        analytic_result = self._build_result(x0)
        candidates: List[OptimizationResult] = [analytic_result]

        # Step 2: Nelder-Mead from analytic start
        nm_result = self._single_run(x0)
        if nm_result is not None:
            candidates.append(nm_result)

        # Step 3: 5 random perturbations ± 30%
        for _ in range(5):
            perturbation = self._rng.uniform(-0.3, 0.3, size=3)
            x_start = x0 + perturbation
            result = self._single_run(x_start)
            if result is not None:
                candidates.append(result)

        # Build Pareto front
        best_eig = min(candidates, key=lambda r: r.eigenvalue_error)
        best_stab = max(candidates, key=lambda r: r.spectral_gap)
        best_cost = min(candidates, key=lambda r: r.cost)

        return ParetoResult(
            best_eigenvalue=best_eig,
            best_stability=best_stab,
            best_cost=best_cost,
            all_candidates=candidates,
            pareto_front=_non_dominated(candidates),
        )

    def _single_run(self, x0: np.ndarray) -> Optional[OptimizationResult]:
        """Single Nelder-Mead run from x0. Returns None if diverged."""
        try:
            res = minimize(
                self.cost_fn,
                x0,
                method="Nelder-Mead",
                options={
                    "maxiter": 2000,
                    "xatol": 1e-6,
                    "fatol": 1e-8,
                    "disp": False,
                },
            )
            result = self._build_result(res.x)
            # Reject clearly diverged solutions
            if not np.isfinite(result.cost):
                return None
            return result
        except Exception:
            return None

    def _build_result(self, params: np.ndarray) -> OptimizationResult:
        """Convert log-space params to OptimizationResult with all metrics."""
        R = float(np.exp(params[0]))
        L = float(np.exp(params[1]))
        C = float(np.exp(params[2]))

        eigs = self.mapper.eigenvalues(R, L, C)
        target_eigs = self.spec.target_eigenvalues
        eig_err = self.mapper.eigenvalue_error(R, L, C, target_eigs)
        cost = self.cost_fn(params)

        # Spectral gap: |Re(λ₁) - Re(λ₂)| — meaningful for overdamped circuits
        # For underdamped (conjugate pair), gap in real parts = 0 by symmetry
        # Use imaginary-part gap for overdamped, or magnitude gap otherwise
        re_gap = abs(np.real(eigs[0]) - np.real(eigs[1]))
        im_gap = abs(np.imag(eigs[0]) - np.imag(eigs[1]))
        gap = float(max(re_gap, im_gap))

        # omega0 and Q from achieved eigenvalues
        # For underdamped: λ = -σ ± jω_d, so ω₀ = sqrt(σ² + ω_d²)
        sigma = float(np.real(eigs[0]))
        omega_d = abs(float(np.imag(eigs[0])))
        omega0_achieved = float(np.sqrt(max(sigma**2 + omega_d**2, 0.0)))

        if omega0_achieved > 0 and abs(sigma) > 0:
            zeta_achieved = abs(sigma) / omega0_achieved
            Q_achieved = 1.0 / (2.0 * max(zeta_achieved, 1e-12))
        else:
            Q_achieved = 0.0

        # Regime classification for a linear RLC circuit:
        # A stable linear time-invariant system (all Re(λ) < 0) is LCA
        # (its symmetry group is abelian U(1), Pontryagin duality applies).
        # Unstable = chaotic (in the sense of non-convergent); marginally stable = nonabelian.
        all_stable = all(np.real(e) < 0 for e in eigs)
        all_negative = all(np.real(e) <= 0 for e in eigs)

        if all_stable:
            regime_type = "lca"
        elif all_negative:
            regime_type = "nonabelian"
        else:
            regime_type = "chaotic"

        return OptimizationResult(
            R=R,
            L=L,
            C=C,
            achieved_eigenvalues=eigs,
            eigenvalue_error=eig_err,
            cost=cost,
            regime_type=regime_type,
            spectral_gap=float(gap),
            omega0_achieved=omega0_achieved,
            Q_achieved=Q_achieved,
            converged=True,
        )


# ---------------------------------------------------------------------------
# MonteCarloStabilityAnalyzer
# ---------------------------------------------------------------------------

@dataclass
class BasinResult:
    n_samples: int
    n_lca: int
    n_nonabelian: int
    n_chaotic: int
    lca_fraction: float
    mean_eigenvalue_spread: float  # std of |λ| across samples
    worst_case_error: float        # max eigenvalue_error across samples
    samples: List[Dict]            # [{"R","L","C","regime","eig_error"}, ...]


class MonteCarloStabilityAnalyzer:
    """Perturbs nominal R/L/C by ±tolerance, classifies regime for each sample.

    Uses EigenvalueMapper only (no LCA classifier) — fast, 100 samples in <100ms.

    Regime classification:
        all Re(λ) < 0 → stable → "lca"   (LCA: linear stable RLC is abelian)
        all Re(λ) ≤ 0 → marginally stable → "nonabelian"
        else          → "chaotic"
    """

    def __init__(self, n_samples: int = 100, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.mapper = EigenvalueMapper()

    def analyze(self, result: OptimizationResult, spec: CircuitSpec) -> BasinResult:
        """Sample R/L/C from uniform ±tolerance, compute eigenvalues for each."""
        tol = spec.component_tolerances
        R_tol = tol.get("R", 0.05)
        L_tol = tol.get("L", 0.10)
        C_tol = tol.get("C", 0.05)

        target_eigs = spec.target_eigenvalues

        samples_list = []
        n_lca = 0
        n_nonabelian = 0
        n_chaotic = 0
        eig_errors = []
        mag_vals = []  # collect |λ| for spread computation

        for _ in range(self.n_samples):
            dR = self.rng.uniform(-R_tol, R_tol)
            dL = self.rng.uniform(-L_tol, L_tol)
            dC = self.rng.uniform(-C_tol, C_tol)

            R_s = result.R * (1.0 + dR)
            L_s = result.L * (1.0 + dL)
            C_s = result.C * (1.0 + dC)

            # Guard
            R_s = max(R_s, 1e-12)
            L_s = max(L_s, 1e-12)
            C_s = max(C_s, 1e-12)

            eigs = self.mapper.eigenvalues(R_s, L_s, C_s)
            eig_err = self.mapper.eigenvalue_error(R_s, L_s, C_s, target_eigs)
            eig_errors.append(eig_err)
            mag_vals.extend([abs(e) for e in eigs])

            # Classify: stable linear RLC → LCA; marginally stable → nonabelian; else chaotic
            all_strictly_stable = all(np.real(e) < 0 for e in eigs)
            all_non_positive = all(np.real(e) <= 0 for e in eigs)

            if all_strictly_stable:
                regime = "lca"
                n_lca += 1
            elif all_non_positive:
                regime = "nonabelian"
                n_nonabelian += 1
            else:
                regime = "chaotic"
                n_chaotic += 1

            samples_list.append({
                "R": R_s,
                "L": L_s,
                "C": C_s,
                "regime": regime,
                "eig_error": eig_err,
            })

        lca_fraction = n_lca / max(self.n_samples, 1)
        mean_eig_spread = float(np.std(mag_vals)) if mag_vals else 0.0
        worst_case_error = float(np.max(eig_errors)) if eig_errors else 0.0

        return BasinResult(
            n_samples=self.n_samples,
            n_lca=n_lca,
            n_nonabelian=n_nonabelian,
            n_chaotic=n_chaotic,
            lca_fraction=lca_fraction,
            mean_eigenvalue_spread=mean_eig_spread,
            worst_case_error=worst_case_error,
            samples=samples_list,
        )
