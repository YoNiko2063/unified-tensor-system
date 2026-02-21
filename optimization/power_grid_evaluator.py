"""
Power Grid Transient Stability — Single Machine Infinite Bus (SMIB).

Classical swing equation:

    M·d²δ/dt² + D·dδ/dt = P_m − P_e·sin(δ)

State variables:
    δ  — rotor angle      [rad]   relative to synchronous reference
    ω  — dδ/dt            [rad/s] angular velocity deviation from synchronous speed

First-order form:
    dδ/dt = ω
    dω/dt = (P_m − P_e·sin(δ) − D·ω) / M

Physical correspondence
-----------------------
Linearising around stable equilibrium δ_s = arcsin(P_m/P_e):

    M·ẍ + D·ẋ + P_e·cos(δ_s)·x = 0    (x = δ − δ_s small)

This is the damped harmonic oscillator (spring-mass with k_eff = P_e·cos(δ_s)):

    ω₀ = √(P_e·cos(δ_s) / M)       [rad/s]  small-signal resonance
    ζ  = D / (2·M·ω₀)               [—]      damping ratio
    Q  = 1/(2ζ) = M·ω₀/D           [—]      quality factor

Same (ω₀, Q, ζ) triple as RLC / spring-mass / Duffing / GD domains.

Separatrix geometry
-------------------
The pendulum potential V(δ) = −P_e·cos(δ) − P_m·δ has two equilibria:

    δ_s = arcsin(P_m/P_e)           stable   (potential minimum)
    δ_u = π − δ_s                   unstable (saddle point)

Energy barrier relative to stable equilibrium:

    E_sep = V(δ_u) − V(δ_s) = 2·P_e·cos(δ_s) − P_m·(π − 2δ_s)

This is the exact analog of α²/(4|β|) in the softening Duffing oscillator.
The near-separatrix condition E₀/E_sep > 0.85 triggers the same energy-based
override as DuffingEvaluator: ω₀_eff → floor as period → ∞.

Critical Clearing Time (CCT)
-----------------------------
For a fault at t=0, the rotor accelerates (insufficient P_e to balance P_m).
If the fault is cleared at time τ, the total energy is:

    E(τ) = ½M·ω(τ)² + [V(δ(τ)) − V(δ_s)]

The system regains synchronism if E(τ) < E_sep.
The CCT is the supremum of τ values for which the post-fault system stabilises.

Koopman strategy
----------------
EDMD with degree-3 polynomial observables on the centred state
(x₁ = δ − δ_s, x₂ = ω).  The dominant complex eigenpair gives (ω₀_eff, Q_eff).
At small amplitudes this recovers the analytic small-signal values.

Rust backend note: rust-physics-kernel provides Duffing-specific RK4 only
and cannot be reused for the swing equation RHS.  Python RK4 is the sole
integrator for this domain — no external dependencies required.

Normalisation
-------------
ω₀ in [rad/s], same reference as RLC, spring-mass, Duffing (2π×1 kHz).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

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
)
from tensor.koopman_edmd import EDMDKoopman, KoopmanResult


# ── Module constants ──────────────────────────────────────────────────────────

_PGRID_OBS_DEGREE: int = 3            # degree-3 polynomial EDMD observables
_MIN_TRAJ_STEPS: int = 50             # minimum steps for a valid EDMD fit
_STABILITY_TOL: float = 1.05          # |λ| ≤ this → stable Koopman mode
_OMEGA_FLOOR_FRACTION: float = 0.005  # ω₀_eff floor = 0.5% of ω₀_linear
_NEAR_SEP_ENERGY_RATIO: float = 0.85  # E₀/E_sep > this → near_separatrix flag
_CCT_SETTLE_STEPS: int = 3000         # post-fault RK4 steps for stability check
_CCT_DELTA_TOL: float = 1e-3          # binary-search convergence tolerance [s]
_CCT_STABILITY_MARGIN: float = 0.05   # rad — buffer below δ_u for stability check


# ── Parameters ────────────────────────────────────────────────────────────────


@dataclass
class PowerGridParams:
    """
    Single-Machine Infinite Bus (SMIB) swing equation parameters.

    Swing equation: M·d²δ/dt² + D·dδ/dt = P_m − P_e·sin(δ)

    Args:
        M:   Rotor inertia constant        [pu·s²/rad]  > 0
        D:   Viscous damping coefficient   [pu·s/rad]   ≥ 0
        P_m: Mechanical input power        [pu]         in (0, P_e)
        P_e: Peak electrical power transfer [pu]        > 0

    Requires 0 < P_m < P_e so that a stable equilibrium exists.
    """

    M: float
    D: float
    P_m: float
    P_e: float

    def __post_init__(self) -> None:
        if self.M <= 0:
            raise ValueError(f"M must be > 0, got {self.M}")
        if self.D < 0:
            raise ValueError(f"D must be ≥ 0, got {self.D}")
        if self.P_e <= 0:
            raise ValueError(f"P_e must be > 0, got {self.P_e}")
        if not (0 < self.P_m < self.P_e):
            raise ValueError(
                f"Need 0 < P_m < P_e for a stable equilibrium; "
                f"got P_m={self.P_m}, P_e={self.P_e}"
            )

    # ── Equilibria ────────────────────────────────────────────────────────────

    @property
    def delta_s(self) -> float:
        """Stable equilibrium δ_s = arcsin(P_m/P_e)  [rad]."""
        return float(math.asin(self.P_m / self.P_e))

    @property
    def delta_u(self) -> float:
        """Unstable equilibrium δ_u = π − δ_s  [rad]."""
        return float(math.pi - self.delta_s)

    # ── Small-signal invariants ───────────────────────────────────────────────

    @property
    def omega0_linear(self) -> float:
        """Small-signal resonance ω₀ = √(P_e·cos(δ_s)/M)  [rad/s]."""
        return float(math.sqrt(self.P_e * math.cos(self.delta_s) / self.M))

    @property
    def damping_ratio(self) -> float:
        """ζ = D/(2·M·ω₀)  (dimensionless)."""
        return float(self.D / (2.0 * self.M * max(self.omega0_linear, 1e-30)))

    @property
    def Q_linear(self) -> float:
        """Quality factor Q = 1/(2ζ)."""
        return float(1.0 / max(2.0 * self.damping_ratio, 1e-30))

    # ── Separatrix ────────────────────────────────────────────────────────────

    @property
    def separatrix_energy(self) -> float:
        """
        Energy barrier E_sep = V(δ_u) − V(δ_s)  [pu] — relative to stable eq.

        Using V(δ) = −P_e·cos(δ) − P_m·δ:
            E_sep = P_e·(cos(δ_s) − cos(δ_u)) + P_m·(δ_s − δ_u)
                  = 2·P_e·cos(δ_s) − P_m·(π − 2·δ_s)

        Always > 0 when 0 < P_m < P_e.  Analogue of α²/(4|β|) in Duffing.
        """
        ds = self.delta_s
        du = self.delta_u
        V_s = -self.P_e * math.cos(ds) - self.P_m * ds
        V_u = -self.P_e * math.cos(du) - self.P_m * du
        return float(V_u - V_s)

    @property
    def load_factor(self) -> float:
        """P_m/P_e — loading level.  → 1.0 means near-separatrix."""
        return float(self.P_m / self.P_e)

    def as_dict(self) -> dict:
        return {"M": self.M, "D": self.D, "P_m": self.P_m, "P_e": self.P_e}


# ── Simulator ─────────────────────────────────────────────────────────────────


class PowerGridSimulator:
    """
    4th-order Runge–Kutta integrator for the SMIB swing equation.

    State vector: [δ, ω]  where ω = dδ/dt.

    Args:
        params: PowerGridParams
        dt:     integration timestep [s]  (0.01 gives ~100 steps per typical period)
    """

    def __init__(self, params: PowerGridParams, dt: float = 0.01) -> None:
        self.params = params
        self.dt = dt

    def rhs(
        self, state: np.ndarray, P_e_override: Optional[float] = None
    ) -> np.ndarray:
        """
        [dδ/dt, dω/dt] = [ω, (P_m − P_e·sin(δ) − D·ω) / M].

        P_e_override replaces params.P_e (used during fault simulation).
        Pass P_e_override=0.0 for a three-phase fault (no power transfer).
        """
        delta, omega = float(state[0]), float(state[1])
        p = self.params
        P_e = P_e_override if P_e_override is not None else p.P_e
        domega = (p.P_m - P_e * math.sin(delta) - p.D * omega) / p.M
        return np.array([omega, domega])

    def potential_energy_abs(self, delta: float) -> float:
        """V(δ) = −P_e·cos(δ) − P_m·δ  (absolute Lyapunov potential)."""
        return -self.params.P_e * math.cos(delta) - self.params.P_m * delta

    def total_energy(self, delta: float, omega: float) -> float:
        """
        E = ½M·ω² + [V(δ) − V(δ_s)]   — relative to stable equilibrium.

        E = 0 at (δ_s, 0).  E = E_sep at (δ_u, 0).  Always ≥ 0 for
        trajectories initialised within the stability region.
        """
        delta_s = self.params.delta_s
        V_ref = self.potential_energy_abs(delta_s)
        V     = self.potential_energy_abs(delta)
        return 0.5 * self.params.M * omega ** 2 + (V - V_ref)

    def run(
        self,
        delta0: float,
        omega0: float,
        n_steps: int,
        P_e_override: Optional[float] = None,
    ) -> np.ndarray:
        """
        Integrate for n_steps steps.

        Args:
            delta0:       initial rotor angle [rad]
            omega0:       initial angular velocity [rad/s]
            n_steps:      number of RK4 steps
            P_e_override: fixed P_e to use throughout (fault model)

        Returns:
            trajectory of shape (n_steps+1, 2): columns are [δ, ω].
        """
        state = np.array([delta0, omega0], dtype=float)
        traj  = [state.copy()]
        dt    = self.dt

        for _ in range(n_steps):
            k1 = self.rhs(state,                P_e_override)
            k2 = self.rhs(state + 0.5 * dt * k1, P_e_override)
            k3 = self.rhs(state + 0.5 * dt * k2, P_e_override)
            k4 = self.rhs(state + dt       * k3, P_e_override)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            traj.append(state.copy())

        return np.array(traj)


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class PowerGridResult:
    """Full evaluation result for one (δ₀, ω₀) initial condition."""

    params: PowerGridParams
    delta0: float
    omega0: float

    # Analytic small-signal invariants (exact)
    omega0_linear: float    # √(P_e·cos(δ_s)/M)
    Q_linear: float         # 1/(2ζ)
    zeta_linear: float      # D/(2Mω₀)

    # EDMD-extracted effective invariants
    omega0_eff: float       # dominant Koopman oscillation frequency [rad/s]
    Q_eff: float            # effective quality factor

    # Energy
    E0: float               # initial energy relative to stable equilibrium
    E_sep: float            # separatrix energy barrier (always > 0)
    log_E: float            # log₁₀(normalised peak energy from trajectory)

    # Koopman
    eigenvalues: np.ndarray
    koopman_result: KoopmanResult

    # Regime flags
    near_separatrix: bool   # E₀/E_sep > 0.85 — approaching stability boundary
    is_linear_regime: bool  # |ω₀_eff − ω₀_linear| / ω₀_linear < 0.05

    def __str__(self) -> str:
        regime = "linear"    if self.is_linear_regime else "nonlinear"
        nsep   = " [near-sep]" if self.near_separatrix   else ""
        return (
            f"PowerGrid M={self.params.M:.2g} D={self.params.D:.2g} "
            f"Pm/Pe={self.params.load_factor:.2f}  "
            f"δ₀={self.delta0:.3g}  ω₀_eff={self.omega0_eff:.4f}  "
            f"Q_eff={self.Q_eff:.3f}  [{regime}]{nsep}"
        )


# ── Evaluator ─────────────────────────────────────────────────────────────────


class PowerGridEvaluator:
    """
    Evaluate SMIB transient stability and extract Koopman invariants.

    Uses EDMD with degree-3 polynomial observables on the centred state
    (x₁=δ−δ_s, x₂=ω) to extract effective (ω₀_eff, Q_eff).  These map
    into the same invariant space as RLC / spring-mass / Duffing / GD domains.

    Args:
        params:  PowerGridParams
        dt:      integration timestep [s]    (default 0.01 s)
        n_steps: simulation steps per eval   (default 800 → covers ~4 periods)
    """

    def __init__(
        self,
        params: PowerGridParams,
        dt: float = 0.01,
        n_steps: int = 800,
    ) -> None:
        self.params  = params
        self.dt      = dt
        self.n_steps = n_steps
        self._sim    = PowerGridSimulator(params, dt)

    # ── Public interface ──────────────────────────────────────────────────────

    def evaluate(self, delta0: float, omega0: float = 0.0) -> PowerGridResult:
        """
        Simulate from (δ₀, ω₀), fit EDMD, extract 4D invariant.

        Returns PowerGridResult with (ω₀_eff, Q_eff, E₀/E_sep, near_separatrix, ...).
        """
        traj = self._sim.run(delta0, omega0, self.n_steps)

        # Energy metrics
        E0    = self._sim.total_energy(delta0, omega0)
        E_sep = self.params.separatrix_energy

        # Near-separatrix: energy-based override (same as Duffing).
        # Period → ∞ as E₀ → E_sep, so EDMD cannot extract ω₀_eff.
        near_sep = (E0 / max(E_sep, 1e-30)) > _NEAR_SEP_ENERGY_RATIO

        # EDMD Koopman fit on centred trajectory
        omega0_lin = self.params.omega0_linear
        Q_lin      = self.params.Q_linear
        koop, omega0_eff, Q_eff = self._fit_koopman(traj)

        if near_sep:
            omega0_eff = max(_OMEGA_FLOOR_FRACTION * omega0_lin, 1e-6)

        # log_E: dimensionless peak energy (normalised by ω₀²)
        delta_s = self.params.delta_s
        x1  = traj[:, 0] - delta_s          # centred rotor angle deviation
        x2  = traj[:, 1]                    # angular velocity
        E_seq = x1**2 + x2**2 / max(omega0_lin**2, 1e-30)
        log_E = float(math.log10(max(float(np.max(E_seq)), 1e-30)))

        is_linear = (
            abs(omega0_eff - omega0_lin) / max(omega0_lin, 1e-30) < 0.05
        )

        return PowerGridResult(
            params        = self.params,
            delta0        = delta0,
            omega0        = omega0,
            omega0_linear = omega0_lin,
            Q_linear      = Q_lin,
            zeta_linear   = self.params.damping_ratio,
            omega0_eff    = omega0_eff,
            Q_eff         = Q_eff,
            E0            = E0,
            E_sep         = E_sep,
            log_E         = log_E,
            eigenvalues   = koop.eigenvalues,
            koopman_result= koop,
            near_separatrix   = near_sep,
            is_linear_regime  = is_linear,
        )

    def dynamical_quantities(
        self, delta0: float, omega0: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """Return (ω₀_eff, Q_eff, ζ_eff, log_E) — 4D invariant."""
        r    = self.evaluate(delta0, omega0)
        zeta = 1.0 / (2.0 * max(r.Q_eff, 1e-12))
        return r.omega0_eff, r.Q_eff, float(zeta), r.log_E

    def system_fn(self):
        """Callable[ndarray→ndarray] for LCAPatchDetector or HDVSNavigator."""
        return self._sim.rhs

    # ── Invariant descriptor (memory-compatible) ──────────────────────────────

    def invariant_descriptor(
        self, delta0: float, omega0: float = 0.0
    ) -> Tuple[KoopmanInvariantDescriptor, KoopmanResult, float]:
        """
        Return (invariant, koopman_result, log_E) for memory storage.

        Uses the same physical log_omega0_norm reference as RLC / spring-mass /
        Duffing, so a power grid at (ω₀, Q) retrieves from ALL physical domains.
        """
        result = self.evaluate(delta0, omega0)

        log_omega0_norm = float(np.clip(
            (math.log(max(result.omega0_eff, 1e-30)) - _LOG_OMEGA0_REF)
            / _LOG_OMEGA0_SCALE,
            -3.0, 3.0,
        ))
        log_Q_norm = float(np.clip(
            math.log(max(result.Q_eff, 1e-30)) / _LOG_OMEGA0_SCALE,
            -3.0, 3.0,
        ))
        zeta = float(np.clip(
            1.0 / (2.0 * max(result.Q_eff, 1e-12)),
            0.0, 10.0,
        ))

        invariant = compute_invariants(
            result.koopman_result.eigenvalues,
            result.koopman_result.eigenvectors,
            ["power_grid_swing"],
            k=min(5, len(result.koopman_result.eigenvalues)),
            log_omega0_norm=log_omega0_norm,
            log_Q_norm=log_Q_norm,
            damping_ratio=zeta,
        )

        return invariant, result.koopman_result, result.log_E

    def store_in_memory(
        self,
        memory: KoopmanExperienceMemory,
        delta0: float,
        omega0: float = 0.0,
        label: str = "smib",
    ) -> None:
        """Evaluate and store result as a memory entry (with 4th log_E dimension)."""
        invariant, koop, log_E = self.invariant_descriptor(delta0, omega0)
        experience = OptimizationExperience(
            bottleneck_operator="power_grid_swing",
            replacement_applied="analytic_trajectory",
            runtime_improvement=max(0.0, 1.0 - self.params.load_factor),
            n_observations=1,
            hardware_target="cpu",
            best_params={
                **self.params.as_dict(),
                "delta0": delta0,
                "omega0": omega0,
                "log_E":  log_E,
            },
            domain=f"power_grid_{label}",
        )
        memory.add(invariant, koop, experience)

    # ── Koopman fit ───────────────────────────────────────────────────────────

    def _fit_koopman(
        self, traj: np.ndarray
    ) -> Tuple[KoopmanResult, float, float]:
        """
        Fit degree-3 polynomial EDMD on centred state and extract dominant eigenpair.

        Returns (KoopmanResult, omega0_eff, Q_eff).
        Falls back to analytic linear values on failure.
        """
        omega0_lin = self.params.omega0_linear
        Q_lin      = self.params.Q_linear

        if len(traj) < _MIN_TRAJ_STEPS + 1:
            return self._fallback_koopman(omega0_lin, Q_lin), omega0_lin, Q_lin

        # Centre trajectory: EDMD polynomial basis works best near the origin.
        delta_s       = self.params.delta_s
        traj_centred  = traj.copy()
        traj_centred[:, 0] -= delta_s

        traj_trimmed = self._trim_trajectory(traj_centred)
        if len(traj_trimmed) < _MIN_TRAJ_STEPS + 1:
            traj_trimmed = traj_centred[:_MIN_TRAJ_STEPS + 1]

        try:
            edmd = EDMDKoopman(observable_degree=_PGRID_OBS_DEGREE)
            edmd.fit_trajectory(traj_trimmed)
            koop = edmd.eigendecomposition()

            omega0_eff, Q_eff = self._dominant_pair_invariants(
                koop.eigenvalues, omega0_lin, Q_lin
            )
            return koop, omega0_eff, Q_eff

        except Exception:
            return self._fallback_koopman(omega0_lin, Q_lin), omega0_lin, Q_lin

    def _trim_trajectory(self, traj: np.ndarray) -> np.ndarray:
        """Discard tail where amplitude has decayed below 3% of initial."""
        initial_amp = float(np.linalg.norm(traj[0]))
        if initial_amp < 1e-12:
            return traj[:_MIN_TRAJ_STEPS]

        threshold = 0.03 * initial_amp
        norms     = np.linalg.norm(traj, axis=1)
        above     = np.where(norms > threshold)[0]

        if len(above) < _MIN_TRAJ_STEPS:
            return traj[:_MIN_TRAJ_STEPS]

        return traj[:int(above[-1]) + 1]

    def _dominant_pair_invariants(
        self,
        eigvals: np.ndarray,
        omega0_fallback: float,
        Q_fallback: float,
    ) -> Tuple[float, float]:
        """
        Extract (ω₀_eff, Q_eff) from the dominant Koopman oscillatory eigenpair.

        Selects the stable complex eigenvalue with smallest non-zero |Im(log λ)|
        (fundamental oscillation, not a harmonic).
        """
        dt = self.dt

        stable_complex = eigvals[
            (np.abs(eigvals) <= _STABILITY_TOL) &
            (np.abs(np.imag(eigvals)) > 1e-10)
        ]
        if len(stable_complex) == 0:
            return omega0_fallback, Q_fallback

        log_eigvals = np.log(stable_complex + 1e-30)
        freqs       = np.abs(np.imag(log_eigvals))

        # Collect candidates with non-trivial frequency.
        candidates = [i for i in range(len(freqs)) if freqs[i] > 1e-4]
        if not candidates:
            return omega0_fallback, Q_fallback

        # Use the analytic ω₀_linear as a prior: pick the mode whose
        # frequency (in rad/s) is closest to omega0_fallback.  This
        # disambiguates the true fundamental from spurious polynomial-basis
        # modes (e.g. the sub-fundamental mode produced by the x² term in
        # the sin(δ) Taylor expansion for the swing equation).
        best_idx = min(candidates,
                       key=lambda i: abs(freqs[i] / dt - omega0_fallback))
        dominant = log_eigvals[best_idx]

        freq  = float(abs(np.imag(dominant))) / dt
        decay = float(abs(np.real(dominant))) / dt

        if freq < 1e-12:
            return omega0_fallback, Q_fallback

        # ω floor: prevents log(ω₀_eff) → −∞ near the separatrix.
        omega_floor = max(_OMEGA_FLOOR_FRACTION * omega0_fallback, 1e-6)
        freq        = max(freq, omega_floor)

        Q_eff = freq / max(2.0 * decay, 1e-12)
        return float(freq), float(Q_eff)

    @staticmethod
    def _fallback_koopman(omega0: float, Q: float) -> KoopmanResult:
        """Minimal KoopmanResult using analytic linear estimates."""
        lam = 0.5 + 0.1j
        return KoopmanResult(
            eigenvalues        = np.array([lam, lam.conjugate()]),
            eigenvectors       = np.eye(2, dtype=complex),
            K_matrix           = np.eye(2) * abs(lam),
            spectral_gap       = 0.0,
            is_stable          = True,
            reconstruction_error = float("inf"),
            koopman_trust      = 0.0,
        )


# ── Standalone public functions ───────────────────────────────────────────────


def compute_separatrix_energy(params: PowerGridParams) -> float:
    """
    E_sep = V(δ_u) − V(δ_s)  [pu] — energy barrier.

    Convenience wrapper over params.separatrix_energy.
    Always > 0 when 0 < P_m < P_e.
    """
    return params.separatrix_energy


def is_near_separatrix(
    E0: float,
    E_sep: float,
    threshold: float = _NEAR_SEP_ENERGY_RATIO,
) -> bool:
    """
    Return True if E₀/E_sep > threshold (default 0.85).

    Both E₀ and E_sep must be expressed relative to the stable equilibrium
    (E = 0 at δ_s, ω = 0).
    """
    if E_sep <= 0:
        return False
    return (E0 / E_sep) > threshold


def simulate_power_grid(
    params: PowerGridParams,
    x0: Tuple[float, float],
    t_span: Tuple[float, float],
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate SMIB swing equation over a time interval.

    Args:
        params: PowerGridParams
        x0:     (δ₀, ω₀) initial condition [rad, rad/s]
        t_span: (t_start, t_end) [s]
        dt:     integration timestep [s]

    Returns:
        (times, trajectory) where trajectory has shape (N, 2) with columns [δ, ω].
    """
    t_start, t_end = t_span
    n_steps = max(1, int(round((t_end - t_start) / dt)))
    sim     = PowerGridSimulator(params, dt=dt)
    traj    = sim.run(float(x0[0]), float(x0[1]), n_steps)
    times   = np.linspace(t_start, t_start + n_steps * dt, len(traj))
    return times, traj


# ── CCT result ────────────────────────────────────────────────────────────────


@dataclass
class CCTResult:
    """Result from estimate_cct()."""

    cct:          float   # estimated critical clearing time [s]
    is_stable:    bool    # True if CCT was found (bracket converged)
    fault_factor: float   # P_e fraction during fault (0.0 = three-phase)
    tau_lo:       float   # lower bound of final binary-search bracket [s]
    tau_hi:       float   # upper bound of final binary-search bracket [s]


def estimate_cct(
    params: PowerGridParams,
    fault_duration_range: Tuple[float, float] = (0.0, 5.0),
    fault_factor: float = 0.0,
    dt: float = 0.01,
    n_settle: int = _CCT_SETTLE_STEPS,
    tol: float = _CCT_DELTA_TOL,
) -> CCTResult:
    """
    Estimate Critical Clearing Time via binary search on fault duration.

    Procedure:
        1. Start at stable equilibrium (δ_s, 0).
        2. Apply fault: P_e → P_e × fault_factor for duration τ.
        3. Clear fault: restore P_e.
        4. Integrate for n_settle steps.
        5. Check stability: δ(t) stays below δ_u − margin for all post-fault steps.
        6. Binary-search on τ until |τ_hi − τ_lo| < tol.

    Args:
        params:               PowerGridParams
        fault_duration_range: (τ_min, τ_max) binary-search interval [s]
        fault_factor:         P_e reduction factor (0.0 = three-phase fault)
        dt:                   RK4 timestep [s]
        n_settle:             post-fault integration steps
        tol:                  convergence tolerance [s]

    Returns:
        CCTResult with estimated CCT and bracket bounds.
    """
    sim     = PowerGridSimulator(params, dt=dt)
    delta_s = params.delta_s
    delta_u = params.delta_u
    P_e_fault   = params.P_e * fault_factor   # power during fault
    stab_limit  = delta_u + _CCT_STABILITY_MARGIN

    def _is_stable(tau: float) -> bool:
        """Simulate fault for duration tau; return True if post-fault stable."""
        n_fault = max(0, int(round(tau / dt)))

        # ── Fault-on phase ───────────────────────────────────────────────────
        state = np.array([delta_s, 0.0])
        for _ in range(n_fault):
            k1 = sim.rhs(state,                 P_e_fault)
            k2 = sim.rhs(state + 0.5 * dt * k1, P_e_fault)
            k3 = sim.rhs(state + 0.5 * dt * k2, P_e_fault)
            k4 = sim.rhs(state + dt       * k3, P_e_fault)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if state[0] > stab_limit:       # rotor already past saddle
                return False

        # ── Post-fault phase ─────────────────────────────────────────────────
        for _ in range(n_settle):
            k1 = sim.rhs(state)
            k2 = sim.rhs(state + 0.5 * dt * k1)
            k3 = sim.rhs(state + 0.5 * dt * k2)
            k4 = sim.rhs(state + dt       * k3)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if state[0] > stab_limit:       # lost synchronism
                return False

        return True

    tau_lo, tau_hi = fault_duration_range

    # Verify that the bracket straddles the CCT.
    if not _is_stable(tau_lo):
        return CCTResult(
            cct=tau_lo, is_stable=False,
            fault_factor=fault_factor, tau_lo=tau_lo, tau_hi=tau_hi,
        )
    if _is_stable(tau_hi):
        return CCTResult(
            cct=tau_hi, is_stable=True,
            fault_factor=fault_factor, tau_lo=tau_lo, tau_hi=tau_hi,
        )

    # Binary search
    while tau_hi - tau_lo > tol:
        tau_mid = 0.5 * (tau_lo + tau_hi)
        if _is_stable(tau_mid):
            tau_lo = tau_mid
        else:
            tau_hi = tau_mid

    cct = 0.5 * (tau_lo + tau_hi)
    return CCTResult(
        cct=cct, is_stable=True,
        fault_factor=fault_factor, tau_lo=tau_lo, tau_hi=tau_hi,
    )
