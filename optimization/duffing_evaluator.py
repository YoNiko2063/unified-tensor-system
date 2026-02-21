"""
Duffing Oscillator Domain — nonlinear test of the dynamical invariant framework.

Equation of motion:
    ẍ + δẋ + αx + βx³ = 0

where:
    α  — linear stiffness     (ω₀_linear = √α  [rad/s])
    β  — cubic stiffness      (nonlinearity parameter; β=0 → linear)
    δ  — viscous damping      (Q_linear = √α / δ  at β=0)

Physical correspondence
-----------------------
At β=0 this is the linear damped harmonic oscillator, exactly equivalent to the
spring-mass and RLC domains.  β>0 introduces amplitude-dependent stiffness:

    ω₀_eff(A) ≈ ω₀_linear · √(1 + 3β·A²/(4α))    (hardening spring: β>0)

So the effective resonant frequency INCREASES with amplitude — unlike all three
linear domains where it is fixed.

Koopman strategy
----------------
We fit EDMD with degree-3 polynomial observables on the simulated trajectory.
This lifts [x, ẋ] into a 9-dimensional observable space that can represent the
cubic term exactly. The dominant eigenvalue pair of the Koopman matrix then
encodes the CURRENT dynamical regime (ω₀_eff, Q_eff).

4D invariant
------------
The existing 3D invariant (log_ω₀_norm, log_Q_norm, ζ) is extended with energy:

    log_E = log₁₀(max_t (x² + ẋ²/ω₀²))     [dimensionless, ref = 1.0]

Energy is stored inside OptimizationExperience.best_params as {"log_E": ...},
preserving full backward compatibility with all existing memory infrastructure.

Energy-conditional retrieval
-----------------------------
    HarmonicNavigator.cross_domain_abelian_map(memory, omega0, log_E) retrieves
    candidates that match BOTH the 3D invariant (via existing to_query_vector())
    AND are within a log_E tolerance (secondary filter).

Low-energy Duffing (β·A²/α << 1) is indistinguishable from linear spring-mass in
the 3D space — memory entries transfer seamlessly.  High-energy Duffing (β·A²/α
>> 1) lives in a distinct log_E band and retrieves only nonlinear entries.

Normalisation constants (Duffing-specific)
------------------------------------------
ω₀ is in [rad/s], same as physical domains (RLC, spring-mass).
We use the same _LOG_OMEGA0_REF (2π×1kHz) so entries from all physical domains
share one coordinate system.  GD uses its own reference (1 rad/step).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from optimization.koopman_signature import (
    KoopmanInvariantDescriptor,
    compute_invariants,
    _LOG_OMEGA0_REF,
    _LOG_OMEGA0_SCALE,
)
from optimization.koopman_memory import (
    KoopmanExperienceMemory,
    OptimizationExperience,
    _MemoryEntry,
)
from tensor.koopman_edmd import EDMDKoopman, KoopmanResult


# ── Module constants ───────────────────────────────────────────────────────────

_DUFFING_LOG_E_REF: float = 0.0          # reference energy = 1.0 (x₀²-normalised)
_DUFFING_OBS_DEGREE: int = 3             # degree-3 polynomial observables
_MIN_TRAJ_STEPS: int = 50               # minimum steps for EDMD fit
_STABILITY_TOL: float = 1.05            # |λ| ≤ this → stable Koopman mode
_OMEGA_FLOOR_FRACTION: float = 0.005    # ω floor = 0.5% of ω₀_linear (prevents log→-∞)
_NEAR_SEP_ENERGY_RATIO: float = 0.85    # E₀/E_s > this → near_separatrix flag


# ── Parameters ────────────────────────────────────────────────────────────────


@dataclass
class DuffingParams:
    """
    Normalised Duffing oscillator parameters (mass = 1).

    Equation: ẍ + δẋ + αx + βx³ = 0

    Args:
        alpha:  linear stiffness   [> 0]   ω₀_linear = √α  [rad/s]
        beta:   cubic stiffness    [≥ 0]   β=0 → purely linear
        delta:  damping coefficient [≥ 0]  Q_linear = √α/δ at β=0
    """

    alpha: float   # linear stiffness
    beta: float    # cubic nonlinearity
    delta: float   # damping

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        # beta < 0 → softening spring: saddle at A_s = √(α/|β|), orbit escapes for E > E_s
        # beta = 0 → linear harmonic oscillator
        # beta > 0 → hardening spring (no separatrix, always bounded)
        if self.delta < 0:
            raise ValueError(f"delta must be ≥ 0, got {self.delta}")

    @property
    def omega0_linear(self) -> float:
        """ω₀ = √α  [rad/s]  — exact at β=0, approximate for β≠0."""
        return float(math.sqrt(self.alpha))

    @property
    def Q_linear(self) -> float:
        """Q = √α/δ — exact at β=0; changes with amplitude for β≠0."""
        return float(self.omega0_linear / max(self.delta, 1e-30))

    @property
    def is_softening(self) -> bool:
        """True for β < 0 (softening spring with separatrix)."""
        return self.beta < 0

    @property
    def separatrix_energy(self) -> float:
        """
        Separatrix energy E_s = α²/(4|β|) for softening spring (β<0).

        This is the energy of the homoclinic orbit connecting the saddle points
        at ±A_s = ±√(α/|β|).  Trajectories with E < E_s are trapped; E ≥ E_s escape.

        Topology is energy-defined: the separatrix IS an energy surface.

        Returns float('inf') for β≥0 (no separatrix).
        """
        if self.beta >= 0:
            return float("inf")
        return float(self.alpha**2 / (4.0 * abs(self.beta)))

    @property
    def separatrix_amplitude(self) -> float:
        """A_s = √(α/|β|) — saddle-point position for β<0. Returns inf for β≥0."""
        if self.beta >= 0:
            return float("inf")
        return float(math.sqrt(self.alpha / abs(self.beta)))

    def nonlinearity_strength(self, amplitude: float) -> float:
        """β·A²/α — dimensionless nonlinearity parameter.  << 1 → linear regime."""
        return float(self.beta * amplitude**2 / max(self.alpha, 1e-30))

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "delta": self.delta}


# ── Simulation ─────────────────────────────────────────────────────────────────


class DuffingSimulator:
    """
    4th-order Runge–Kutta integrator for ẍ + δẋ + αx + βx³ = 0.

    State vector: [x, ẋ]
    """

    def __init__(self, params: DuffingParams, dt: float = 0.05) -> None:
        self.params = params
        self.dt = dt

    def rhs(self, state: np.ndarray) -> np.ndarray:
        """ẋ = [v, -(δv + αx + βx³)]"""
        x, v = float(state[0]), float(state[1])
        p = self.params
        ax = -(p.delta * v + p.alpha * x + p.beta * x**3)
        return np.array([v, ax])

    def potential_energy(self, x: float) -> float:
        """V(x) = αx²/2 + βx⁴/4  (Duffing potential)."""
        p = self.params
        return 0.5 * p.alpha * x**2 + 0.25 * p.beta * x**4

    def total_energy(self, x: float, v: float) -> float:
        """E = ½v² + V(x) = ½v² + αx²/2 + βx⁴/4."""
        return 0.5 * v**2 + self.potential_energy(x)

    def system_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return a Callable compatible with LCAPatchDetector(system_fn=...)."""
        return self.rhs

    def run(self, x0: float, v0: float, n_steps: int) -> np.ndarray:
        """
        Integrate for n_steps steps with RK4.

        For softening springs (β<0): applies an energy-based separatrix guard.
        The guard fires when E_total > 0.9 × E_s, preventing numerical escape.
        Topology is energy-defined: the separatrix IS an energy surface, so the
        guard is energy-based — NOT position-based — to maintain invariant consistency.

        Returns trajectory of shape (k+1, 2), k ≤ n_steps.
        """
        E_sep = self.params.separatrix_energy  # inf for β≥0
        E_guard = 0.9 * E_sep

        # Guard: if initial energy already exceeds separatrix, return trivial trajectory
        E_init = self.total_energy(x0, v0)
        if E_init >= E_sep:
            return np.array([[x0, v0]])

        state = np.array([x0, v0], dtype=float)
        traj = [state.copy()]
        dt = self.dt

        for _ in range(n_steps):
            k1 = self.rhs(state)
            k2 = self.rhs(state + 0.5 * dt * k1)
            k3 = self.rhs(state + 0.5 * dt * k2)
            k4 = self.rhs(state + dt * k3)
            new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Energy guard: stop before numerical drift across separatrix (β<0 only)
            if E_sep < float("inf"):
                E_new = self.total_energy(float(new_state[0]), float(new_state[1]))
                if E_new >= E_guard:
                    break

            state = new_state
            traj.append(state.copy())

        return np.array(traj)


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class DuffingResult:
    """Full evaluation result for one (x₀, v₀) initial condition."""

    params: DuffingParams
    x0: float
    v0: float
    omega0_linear: float      # √α — exact linear resonance
    omega0_eff: float         # extracted from dominant Koopman eigenpair [rad/s]
    Q_linear: float           # √α/δ — linear limit quality factor
    Q_eff: float              # extracted from Koopman decay rate
    log_E: float              # log₁₀(normalised peak energy)
    eigenvalues: np.ndarray   # all Koopman eigenvalues
    koopman_result: KoopmanResult
    is_linear_regime: bool    # |ω₀_eff − ω₀_linear| / ω₀_linear < 0.05
    omega0_shift: float       # (ω₀_eff − ω₀_linear) / ω₀_linear  [fractional]
    nonlinearity: float       # β·x₀²/α  — dimensionless nonlinearity at x₀
    # Bifurcation extension (β<0 softening support)
    near_separatrix: bool = False       # E₀/E_sep > 0.85 — approaching homoclinic orbit
    max_eigenvalue_modulus: float = 0.0 # max|λ| from EDMD; → 1 near bifurcation

    def __str__(self) -> str:
        lin = "linear" if self.is_linear_regime else "nonlinear"
        return (
            f"Duffing α={self.params.alpha:.2g} β={self.params.beta:.2g} "
            f"δ={self.params.delta:.2g}  "
            f"x₀={self.x0:.3g}  ω₀_eff={self.omega0_eff:.4f}  "
            f"Q_eff={self.Q_eff:.3f}  log_E={self.log_E:.3f}  [{lin}]"
        )


# ── Evaluator ─────────────────────────────────────────────────────────────────


class DuffingEvaluator:
    """
    Evaluate a Duffing oscillator at given initial conditions.

    Uses EDMD with degree-3 polynomial observables to extract effective
    (ω₀_eff, Q_eff) from the Koopman dominant eigenpair.  These map into
    the same invariant space as RLC / spring-mass / GD domains.

    Args:
        params:    Duffing parameters (α, β, δ)
        dt:        integration timestep [s]  (default 0.05)
        n_steps:   simulation steps per evaluation (default 800)
    """

    def __init__(
        self,
        params: DuffingParams,
        dt: float = 0.05,
        n_steps: int = 800,
    ) -> None:
        self.params = params
        self.dt = dt
        self.n_steps = n_steps
        self._sim = DuffingSimulator(params, dt)

    # ── Domain-invariant quantities ────────────────────────────────────────────

    def evaluate(self, x0: float, v0: float = 0.0) -> DuffingResult:
        """
        Simulate from (x₀, v₀), fit EDMD, extract 4D invariant.

        Returns DuffingResult with (ω₀_eff, Q_eff, log_E, is_linear_regime, ...).
        """
        traj = self._sim.run(x0, v0, self.n_steps)

        # Energy metric: max(x² + v²/ω₀²)  — dimensionless
        omega0_lin = self.params.omega0_linear
        E_seq = traj[:, 0] ** 2 + traj[:, 1] ** 2 / max(omega0_lin ** 2, 1e-30)
        E = float(np.max(E_seq))
        log_E = float(math.log10(max(E, 1e-30)))

        # Near-separatrix detection: energy-based (topology is energy-defined).
        # Use initial mechanical energy E₀ = ½v₀² + V(x₀) vs separatrix energy E_s.
        E_sep = self.params.separatrix_energy
        E0 = self._sim.total_energy(x0, v0)
        near_sep = (E_sep < float("inf")) and (E0 / max(E_sep, 1e-30) > _NEAR_SEP_ENERGY_RATIO)

        # EDMD Koopman fit
        koop_result, omega0_eff, Q_eff = self._fit_koopman(traj)
        max_eig_mod = float(np.max(np.abs(koop_result.eigenvalues)))

        # Near-separatrix override: EDMD cannot reliably detect ω₀_eff → 0 when
        # the period greatly exceeds the simulation window (less than one full cycle).
        # Use the ω floor directly — the physics dictates ω₀_eff → 0 near E_s.
        # This is an energy-based override: near_sep is determined by E₀/E_sep ratio.
        if near_sep:
            omega_min = max(_OMEGA_FLOOR_FRACTION * omega0_lin, 1e-6)
            omega0_eff = omega_min   # force to floor: period → ∞ as E₀ → E_s

        is_linear = (
            abs(omega0_eff - omega0_lin) / max(omega0_lin, 1e-30) < 0.05
        )
        shift = (omega0_eff - omega0_lin) / max(omega0_lin, 1e-30)

        return DuffingResult(
            params=self.params,
            x0=x0,
            v0=v0,
            omega0_linear=omega0_lin,
            omega0_eff=omega0_eff,
            Q_linear=self.params.Q_linear,
            Q_eff=Q_eff,
            log_E=log_E,
            eigenvalues=koop_result.eigenvalues,
            koopman_result=koop_result,
            is_linear_regime=is_linear,
            omega0_shift=shift,
            nonlinearity=self.params.nonlinearity_strength(abs(x0)),
            near_separatrix=near_sep,
            max_eigenvalue_modulus=max_eig_mod,
        )

    def dynamical_quantities(
        self, x0: float, v0: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """Return (ω₀_eff, Q_eff, ζ_eff, log_E) — 4D invariant."""
        r = self.evaluate(x0, v0)
        zeta = 1.0 / (2.0 * max(r.Q_eff, 1e-12))
        return r.omega0_eff, r.Q_eff, float(zeta), r.log_E

    def system_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Callable for LCAPatchDetector(system_fn=evaluator.system_fn(), n_states=2)."""
        return self._sim.system_fn()

    # ── Koopman fit ───────────────────────────────────────────────────────────

    # ── Trajectory utilities ──────────────────────────────────────────────────

    def _trim_trajectory(
        self, traj: np.ndarray, min_amp_fraction: float = 0.03
    ) -> np.ndarray:
        """
        Discard tail of trajectory where amplitude has decayed below threshold.

        For a damped oscillator with high Q, the signal decays slowly enough
        that EDMD gets many useful cycles.  For low Q, the signal decays fast —
        keeping near-zero pairs corrupts the Koopman fit.

        Keeps at least _MIN_TRAJ_STEPS rows.
        """
        initial_amp = float(np.linalg.norm(traj[0]))
        if initial_amp < 1e-12:
            return traj[:_MIN_TRAJ_STEPS]

        threshold = min_amp_fraction * initial_amp
        norms = np.linalg.norm(traj, axis=1)
        above = np.where(norms > threshold)[0]

        if len(above) < _MIN_TRAJ_STEPS:
            return traj[:_MIN_TRAJ_STEPS]

        keep = int(above[-1]) + 1
        return traj[:keep]

    def _fit_koopman(
        self, traj: np.ndarray
    ) -> Tuple[KoopmanResult, float, float]:
        """
        Fit degree-3 polynomial EDMD and extract dominant eigenpair.

        Returns (KoopmanResult, omega0_eff, Q_eff).
        Falls back to analytic linear estimates on failure.
        """
        omega0_lin = self.params.omega0_linear
        Q_lin = self.params.Q_linear

        if len(traj) < _MIN_TRAJ_STEPS + 1:
            return self._fallback_koopman(omega0_lin, Q_lin), omega0_lin, Q_lin

        # Trim low-amplitude tail so EDMD sees the informative oscillatory portion
        traj_trimmed = self._trim_trajectory(traj)
        if len(traj_trimmed) < _MIN_TRAJ_STEPS + 1:
            traj_trimmed = traj[:_MIN_TRAJ_STEPS + 1]

        try:
            edmd = EDMDKoopman(observable_degree=_DUFFING_OBS_DEGREE)
            edmd.fit_trajectory(traj_trimmed)
            koop = edmd.eigendecomposition()

            omega0_eff, Q_eff = self._dominant_pair_invariants(
                koop.eigenvalues, omega0_lin, Q_lin
            )
            return koop, omega0_eff, Q_eff

        except Exception:
            return self._fallback_koopman(omega0_lin, Q_lin), omega0_lin, Q_lin

    def _dominant_pair_invariants(
        self,
        eigvals: np.ndarray,
        omega0_fallback: float,
        Q_fallback: float,
    ) -> Tuple[float, float]:
        """
        Extract (ω₀_eff, Q_eff) from the dominant oscillatory Koopman eigenvalue.

        Selection: filter stable complex eigenvalues, pick the one with the
        largest |Im(log λ)| — that's the fundamental oscillation frequency.
        """
        dt = self.dt

        # Keep stable eigenvalues (|λ| ≤ 1 + tolerance) that are complex
        stable_complex = eigvals[
            (np.abs(eigvals) <= _STABILITY_TOL) & (np.abs(np.imag(eigvals)) > 1e-10)
        ]

        if len(stable_complex) == 0:
            return omega0_fallback, Q_fallback

        # Log eigenvalues → continuous-time rates
        log_eigvals = np.log(stable_complex + 1e-30)

        # Sort by |Im| ASCENDING — pick the smallest non-zero imaginary frequency.
        # The fundamental is the lowest-frequency mode; harmonics (2ω, 3ω, ...) appear
        # at higher |Im| values.  Picking the smallest avoids mistaking a harmonic
        # for the fundamental.
        freqs = np.abs(np.imag(log_eigvals))
        order = np.argsort(freqs)
        # Skip any near-zero modes (pure decay, not oscillatory)
        for idx in order:
            if freqs[idx] > 1e-4:
                dominant = log_eigvals[idx]
                break
        else:
            return omega0_fallback, Q_fallback

        freq = float(abs(np.imag(dominant))) / dt        # ω₀_eff [rad/s]
        decay = float(abs(np.real(dominant))) / dt       # γ [1/s]

        if freq < 1e-12:
            return omega0_fallback, Q_fallback

        # ω floor: near the separatrix (β<0), ω₀_eff → 0 and log(ω) → -∞.
        # Floor at 0.5% of ω₀_linear to keep curvature_profile finite and log_omega0_norm
        # clipped. This is geometric regularisation, not physical approximation.
        omega_min = max(_OMEGA_FLOOR_FRACTION * omega0_fallback, 1e-6)
        freq = max(freq, omega_min)

        Q_eff = freq / max(2.0 * decay, 1e-12)
        return float(freq), float(Q_eff)

    @staticmethod
    def _fallback_koopman(omega0: float, Q: float) -> KoopmanResult:
        """Minimal KoopmanResult using analytic linear estimates."""
        lam = 0.5 + 0.1j
        return KoopmanResult(
            eigenvalues=np.array([lam, lam.conjugate()]),
            eigenvectors=np.eye(2, dtype=complex),
            K_matrix=np.eye(2) * abs(lam),
            spectral_gap=0.0,
            is_stable=True,
            reconstruction_error=float("inf"),
            koopman_trust=0.0,
        )

    # ── Invariant descriptor (memory-compatible) ───────────────────────────────

    def invariant_descriptor(
        self, x0: float, v0: float = 0.0
    ) -> Tuple[KoopmanInvariantDescriptor, KoopmanResult, float]:
        """
        Return (invariant, koopman_result, log_E) for memory storage.

        The invariant uses the PHYSICAL log_omega0_norm reference (same as RLC
        and spring-mass), so Duffing at low amplitude will retrieve entries from
        all physical domains.
        """
        result = self.evaluate(x0, v0)

        log_omega0_norm = float(np.clip(
            (math.log(max(result.omega0_eff, 1e-30)) - _LOG_OMEGA0_REF)
            / _LOG_OMEGA0_SCALE,
            -3.0, 3.0,
        ))
        log_Q_norm = float(np.clip(
            math.log(max(result.Q_eff, 1e-30)) / _LOG_OMEGA0_SCALE,
            -3.0, 3.0,
        ))
        zeta = float(np.clip(1.0 / (2.0 * max(result.Q_eff, 1e-12)), 0.0, 10.0))

        invariant = compute_invariants(
            result.koopman_result.eigenvalues,
            result.koopman_result.eigenvectors,
            ["duffing_oscillation"],
            k=min(5, len(result.koopman_result.eigenvalues)),
            log_omega0_norm=log_omega0_norm,
            log_Q_norm=log_Q_norm,
            damping_ratio=zeta,
        )

        return invariant, result.koopman_result, result.log_E

    def store_in_memory(
        self,
        memory: KoopmanExperienceMemory,
        x0: float,
        v0: float = 0.0,
        label: str = "duffing",
    ) -> None:
        """Evaluate and store result as a memory entry (4D: invariant + log_E)."""
        invariant, koop, log_E = self.invariant_descriptor(x0, v0)
        experience = OptimizationExperience(
            bottleneck_operator="duffing_oscillation",
            replacement_applied="analytic_trajectory",
            runtime_improvement=max(0.0, 1.0 - abs(self.params.beta) / 2.0),
            n_observations=1,
            hardware_target="cpu",
            best_params={
                **self.params.as_dict(),
                "x0": x0,
                "v0": v0,
                "log_E": log_E,                 # 4th invariant dimension
            },
            domain=f"duffing_{label}",
        )
        memory.add(invariant, koop, experience)
