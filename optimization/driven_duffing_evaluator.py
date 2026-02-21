"""
Driven Duffing Oscillator — Route to Chaos analysis.

Equation of motion:
    ẍ + δẋ + αx + βx³ = F·cos(Ωt)

The unforced Duffing (F=0) with damping always decays to the origin.  With
periodic forcing F>0, the system can lock into a limit cycle (period-1, period-2,
period-4, …) or, at large enough F, exhibit a strange attractor (chaos).

This is the Feigenbaum period-doubling cascade — a classical route to chaos.

Architecture honesty test
--------------------------
The key architectural question: when chaos is detected (no coherent Koopman
structure), does the system return `is_abelian = False` (honest) or
`is_abelian = True` (hallucinating structure)?

A well-designed architecture should:
  Period-1 limit cycle:  trust ≈ 1.0,  is_abelian = True
  Period-2 limit cycle:  trust > 0.3,  is_abelian = True  (still periodic)
  Chaotic attractor:     trust < 0.3,  is_abelian = False  (honest uncertainty)

EDMD on the driven system
--------------------------
We run the simulator to steady state (discard transient), then apply EDMD to the
limit cycle or chaotic trajectory.  For a periodic orbit:
  - Eigenvalues cluster near the unit circle
  - Reconstruction error is low (Koopman model faithfully predicts next state)

For a chaotic orbit:
  - No K satisfies K·ψ(xₖ) ≈ ψ(xₖ₊₁) globally → reconstruction error is high
  - Eigenvalues spread inside the unit disk

Trust metric for driven systems
---------------------------------
The composite trust score in EDMDKoopman uses a spectral gap gate, which is
unreliable for driven systems: on a periodic limit cycle all Koopman eigenvalues
lie exactly on the unit circle, giving gap ≈ 0 even for perfect periodicity.

We therefore use a reconstruction-only trust:
    trust = max(0, 1 - reconstruction_error / η_max)

This cleanly separates periodic (low recon error → trust ≈ 1) from chaotic
(high recon error → trust ≈ 0).

Period detection via Poincaré section
---------------------------------------
We sample the state (x, v) at stroboscopic intervals T_drive = 2π/Ω:
  Period-1: all Poincaré points cluster at a single location.
  Period-2: points alternate between two clusters.
  Period-4: four clusters.
  Chaos: points fill the attractor continuously (many clusters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from tensor.koopman_edmd import EDMDKoopman, KoopmanResult
from optimization.duffing_evaluator import (
    _DUFFING_OBS_DEGREE,
    _MIN_TRAJ_STEPS,
    _STABILITY_TOL,
)


# ── Module constants ────────────────────────────────────────────────────────────

_ETA_MAX: float = 1.0            # reconstruction error threshold (periodic ≈ 0, chaos >> 1)
_CHAOS_TRUST_THRESHOLD: float = 0.3    # trust < this → chaotic regime
_ABELIAN_TRUST_THRESHOLD: float = 0.3  # trust ≥ this → abelian (coherent structure)
_POINCARE_GAP_THRESHOLD: float = 0.12  # relative gap > this → separate Poincaré cluster
_MAX_PERIOD: int = 32            # cap period number at 32 (beyond this → chaos)


# ── Parameters ─────────────────────────────────────────────────────────────────


@dataclass
class DrivenDuffingParams:
    """
    Driven Duffing oscillator parameters.

    Equation: ẍ + δẋ + αx + βx³ = F·cos(Ωt)

    Args:
        alpha:  linear stiffness  [> 0]   ω₀ = √α  [rad/s]
        beta:   cubic stiffness   [any]   β>0 hardening, β<0 softening, β=0 linear
        delta:  damping           [≥ 0]
        F:      forcing amplitude [≥ 0]   F=0 → unforced (decays to fixed point)
        Omega:  driving frequency [> 0]   [rad/s]
    """

    alpha: float
    beta: float
    delta: float
    F: float
    Omega: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.delta < 0:
            raise ValueError(f"delta must be ≥ 0, got {self.delta}")
        if self.F < 0:
            raise ValueError(f"F must be ≥ 0, got {self.F}")
        if self.Omega <= 0:
            raise ValueError(f"Omega must be > 0, got {self.Omega}")

    @property
    def omega0_linear(self) -> float:
        """ω₀ = √α  [rad/s] — linear resonant frequency (exact at β=0)."""
        return float(math.sqrt(self.alpha))

    @property
    def Q_linear(self) -> float:
        """Q = ω₀/δ — linear quality factor."""
        return float(self.omega0_linear / max(self.delta, 1e-30))

    @property
    def forcing_period(self) -> float:
        """T_drive = 2π/Ω — fundamental period of the external forcing."""
        return float(2.0 * math.pi / self.Omega)

    @property
    def frequency_ratio(self) -> float:
        """Ω/ω₀ — ratio of driving to natural frequency."""
        return float(self.Omega / max(self.omega0_linear, 1e-30))

    @property
    def nonlinearity_strength(self) -> float:
        """β/α — normalised nonlinearity parameter (domain-invariant shape)."""
        return float(self.beta / max(self.alpha, 1e-30))

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "delta": self.delta,
            "F": self.F,
            "Omega": self.Omega,
        }


# ── Simulator ──────────────────────────────────────────────────────────────────


class DrivenDuffingSimulator:
    """
    4th-order Runge–Kutta integrator for ẍ + δẋ + αx + βx³ = F·cos(Ωt).

    Unlike the autonomous DuffingSimulator, the right-hand side has explicit
    time dependence through the forcing term.

    State vector: [x, ẋ]
    """

    def __init__(self, params: DrivenDuffingParams, dt: float = 0.05) -> None:
        self.params = params
        self.dt = dt

    def rhs(self, state: np.ndarray, t: float) -> np.ndarray:
        """ẋ = [v, -(δv + αx + βx³ - F·cos(Ωt))]"""
        x, v = float(state[0]), float(state[1])
        p = self.params
        ax = -(p.delta * v + p.alpha * x + p.beta * x**3 - p.F * math.cos(p.Omega * t))
        return np.array([v, ax])

    def run(
        self, x0: float, v0: float, n_steps: int, t_start: float = 0.0
    ) -> np.ndarray:
        """
        Integrate for n_steps with RK4, starting at (x0, v0) at time t_start.

        Returns trajectory of shape (n_steps+1, 2).
        """
        state = np.array([x0, v0], dtype=float)
        traj = [state.copy()]
        t = t_start
        dt = self.dt

        for _ in range(n_steps):
            k1 = self.rhs(state, t)
            k2 = self.rhs(state + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self.rhs(state + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self.rhs(state + dt * k3, t + dt)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += dt
            traj.append(state.copy())

        return np.array(traj)  # (n_steps+1, 2)

    def run_steady_state(
        self,
        x0: float,
        v0: float,
        n_total: int,
        transient_fraction: float = 0.5,
    ) -> np.ndarray:
        """
        Integrate n_total steps, discard the first transient_fraction as transient.

        Returns the steady-state portion: shape ((1−transient_fraction)·n_total + 1, 2).
        """
        traj = self.run(x0, v0, n_total)
        n_transient = int(n_total * transient_fraction)
        return traj[n_transient:]


# ── Result ─────────────────────────────────────────────────────────────────────


@dataclass
class DrivenDuffingResult:
    """
    Full evaluation result for driven Duffing at steady state.

    Fields:
        params:               DrivenDuffingParams used for this evaluation
        koopman_result:       raw EDMD KoopmanResult (eigenvalues, K_matrix, ...)
        koopman_trust:        reconstruction-based trust ∈ [0,1]
                              = max(0, 1 - reconstruction_error / η_max)
                              Periodic orbit → trust ≈ 1.  Chaos → trust ≈ 0.
        is_periodic:          True if poincare_clusters ≤ 8 (up to period-8 orbit)
        period_number:        1, 2, 4, 8, 16, 32 (from Poincaré cluster count)
        is_chaotic:           True if koopman_trust < _CHAOS_TRUST_THRESHOLD
        is_abelian:           True if koopman_trust ≥ _ABELIAN_TRUST_THRESHOLD
                              Abelian = U(1) structure (coherent periodic orbit).
                              Non-abelian = broadband / chaotic spectrum.
        dominant_frequency:   lowest non-trivial EDMD oscillatory frequency [rad/s]
                              ≈ Ω for period-1, ≈ Ω/2 for period-2, etc.
        poincare_clusters:    raw cluster count from Poincaré section
        reconstruction_error: EDMD mean squared prediction error on steady state
    """

    params: DrivenDuffingParams
    koopman_result: KoopmanResult
    koopman_trust: float
    is_periodic: bool
    period_number: int
    is_chaotic: bool
    is_abelian: bool
    dominant_frequency: float
    poincare_clusters: int
    reconstruction_error: float

    def __str__(self) -> str:
        status = "chaotic" if self.is_chaotic else f"period-{self.period_number}"
        return (
            f"DrivenDuffing α={self.params.alpha:.2g} β={self.params.beta:.2g} "
            f"F={self.params.F:.3g} Ω={self.params.Omega:.3g} | "
            f"{status} | trust={self.koopman_trust:.3f} recon_err={self.reconstruction_error:.3g} | "
            f"abelian={self.is_abelian}"
        )


# ── Evaluator ──────────────────────────────────────────────────────────────────


class DrivenDuffingEvaluator:
    """
    Evaluate driven Duffing at steady state and extract chaos indicators.

    Workflow:
      1. Run simulator to steady state (discard transient).
      2. Fit EDMD with degree-3 polynomial observables.
      3. Compute reconstruction-based trust score.
      4. Extract Poincaré section → count clusters → estimate period.
      5. Extract dominant oscillatory frequency.
      6. Compute is_chaotic and is_abelian from trust.

    Args:
        params:             DrivenDuffingParams
        dt:                 integration timestep [s]  (default 0.05)
        n_total:            total integration steps per evaluation (default 2000)
        transient_fraction: fraction of n_total discarded as transient (default 0.5)
    """

    def __init__(
        self,
        params: DrivenDuffingParams,
        dt: float = 0.05,
        n_total: int = 2000,
        transient_fraction: float = 0.5,
    ) -> None:
        self.params = params
        self.dt = dt
        self.n_total = n_total
        self.transient_fraction = transient_fraction
        self._sim = DrivenDuffingSimulator(params, dt)

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, x0: float = 0.1, v0: float = 0.0) -> DrivenDuffingResult:
        """
        Simulate from (x₀, v₀), run to steady state, extract chaos indicators.

        Returns DrivenDuffingResult with trust, period number, and abelian flag.
        """
        # 1. Run to steady state
        steady = self._sim.run_steady_state(
            x0, v0, self.n_total, self.transient_fraction
        )

        # 2. EDMD on steady-state trajectory
        koop_result = self._fit_koopman(steady)
        recon_err = float(koop_result.reconstruction_error)

        # 3. Reconstruction-based trust (gap score is unreliable for driven systems:
        #    periodic limit cycles have all Koopman eigenvalues on the unit circle,
        #    so the spectral gap ≈ 0 regardless of periodicity quality).
        trust = _reconstruction_trust(recon_err)

        # 4. Dominant oscillatory frequency from EDMD spectrum
        dom_freq = self._dominant_frequency(koop_result.eigenvalues)

        # 5. Poincaré section → cluster count → period number.
        #    Pass orbit_range to normalise the cluster gap threshold: without this,
        #    phase-drift between stroboscopic samples (≈ 1-3% of orbit range) would
        #    be falsely split into N clusters (one per sample point).
        poincare_x, orbit_range = self._poincare_x_and_range(steady)
        n_clusters = _count_clusters(poincare_x, orbit_range=orbit_range)
        period_num = _period_from_clusters(n_clusters)

        # 6. Chaos / abelian classification (trust is the primary gate)
        is_periodic = n_clusters <= 8
        is_chaotic = trust < _CHAOS_TRUST_THRESHOLD
        is_abelian = trust >= _ABELIAN_TRUST_THRESHOLD

        return DrivenDuffingResult(
            params=self.params,
            koopman_result=koop_result,
            koopman_trust=trust,
            is_periodic=is_periodic,
            period_number=period_num,
            is_chaotic=is_chaotic,
            is_abelian=is_abelian,
            dominant_frequency=dom_freq,
            poincare_clusters=n_clusters,
            reconstruction_error=recon_err,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fit_koopman(self, steady: np.ndarray) -> KoopmanResult:
        """Fit degree-3 polynomial EDMD on steady-state trajectory."""
        if len(steady) < _MIN_TRAJ_STEPS + 1:
            return _fallback_koopman()
        try:
            edmd = EDMDKoopman(observable_degree=_DUFFING_OBS_DEGREE)
            edmd.fit_trajectory(steady)
            return edmd.eigendecomposition()
        except Exception:
            return _fallback_koopman()

    def _poincare_x_and_range(self, steady: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Extract Poincaré x-values and orbit range from steady-state trajectory.

        orbit_range = max(x_steady) - min(x_steady) — the full amplitude of the
        attractor in x.  Used to normalise the cluster detection: for period-1,
        all Poincaré points cluster within < 10% of orbit_range (only phase drift
        from stroboscopic sampling error); for period-2 they split into two groups
        separated by >> 10% of orbit_range.

        Returns (poincare_x, orbit_range).
        """
        T_drive = self.params.forcing_period
        steps_per_period = max(1, round(T_drive / self.dt))
        n = len(steady)
        indices = np.arange(0, n, steps_per_period)
        orbit_range = float(np.max(steady[:, 0]) - np.min(steady[:, 0]))
        if len(indices) == 0:
            return steady[:, 0], orbit_range
        return steady[indices, 0], orbit_range

    def _poincare_x(self, steady: np.ndarray) -> np.ndarray:
        """Return Poincaré x-values (without orbit range).  Used in tests."""
        x, _ = self._poincare_x_and_range(steady)
        return x

    def _dominant_frequency(self, eigenvalues: np.ndarray) -> float:
        """
        Extract the smallest non-trivial oscillatory frequency from EDMD eigenvalues.

        For period-1 driven at Ω: smallest frequency ≈ Ω.
        For period-2: smallest frequency ≈ Ω/2 (subharmonic).
        Falls back to Ω if no valid eigenvalue found.
        """
        dt = self.dt
        Omega = self.params.Omega

        # Keep stable complex eigenvalues
        stable_complex = eigenvalues[
            (np.abs(eigenvalues) <= _STABILITY_TOL)
            & (np.abs(np.imag(eigenvalues)) > 1e-10)
        ]
        if len(stable_complex) == 0:
            return Omega

        log_eigs = np.log(stable_complex + 1e-30j)
        freqs = np.abs(np.imag(log_eigs)) / dt

        # Pick the smallest non-trivial frequency (fundamental, not harmonics)
        valid = freqs[freqs > 1e-4]
        if len(valid) == 0:
            return Omega

        return float(np.min(valid))


# ── Module-level helpers ────────────────────────────────────────────────────────


def _reconstruction_trust(recon_err: float, eta_max: float = _ETA_MAX) -> float:
    """
    Reconstruction-based trust score ∈ [0, 1].

    trust = max(0, 1 - reconstruction_error / η_max)

    Suitable for driven (non-autonomous) systems where the spectral gap metric
    is unreliable (all eigenvalues on the unit circle for any limit cycle).

    Periodic orbit: recon_err ≈ 0  → trust ≈ 1.0
    Chaotic orbit:  recon_err >> 1 → trust ≈ 0.0
    """
    return float(max(0.0, 1.0 - float(recon_err) / max(float(eta_max), 1e-12)))


def _count_clusters(
    x_poincare: np.ndarray,
    orbit_range: float = 0.0,
    gap_threshold: float = _POINCARE_GAP_THRESHOLD,
) -> int:
    """
    Count distinct clusters in Poincaré x-values using orbit-normalised gap detection.

    The key insight: for a period-1 orbit, Poincaré points at successive stroboscopic
    times drift by ≈ 1–3% of orbit_range (because T_drive/dt is rarely an integer).
    Normalising gaps by span-of-section-points (not orbit_range) would split these
    nearly-identical points into N separate clusters (17% relative gap for 7 points).

    Instead we normalise by orbit_range (full attractor x-amplitude):
      - Period-1: span ≈ 1-3% of orbit_range → below 10% → 1 cluster ✓
      - Period-2: two groups separated by ≫ 10% of orbit_range → 2 clusters ✓
      - Chaos:    spread ≈ 50-100% of orbit_range, many internal gaps > 12% → >2 ✓

    Args:
        x_poincare:   x-values at Poincaré section (stroboscopic samples)
        orbit_range:  max(x_steady) - min(x_steady), the full attractor amplitude.
                      If 0, falls back to span of section points.
        gap_threshold: relative gap (as fraction of section span) to split clusters.

    Returns integer cluster count ≥ 1.
    """
    n = len(x_poincare)
    if n <= 1:
        return 1

    x_sorted = np.sort(x_poincare)
    span = float(x_sorted[-1] - x_sorted[0])

    # Period-1 fast path: if all Poincaré points are clustered within 10% of the
    # orbit amplitude, treat as single cluster (phase drift is small).
    ref = orbit_range if orbit_range > 1e-6 else max(span, 1e-8)
    if span / ref < 0.10:
        return 1

    if span < 1e-8:
        return 1  # all numerically identical

    # General: count internal gaps that exceed gap_threshold of section span
    gaps = np.diff(x_sorted) / span
    n_clusters = 1 + int(np.sum(gaps > gap_threshold))
    return min(n_clusters, n)  # can't have more clusters than points


def _period_from_clusters(n_clusters: int) -> int:
    """
    Round cluster count to nearest power-of-2 period number.

    Period-doubling cascade: 1 → 2 → 4 → 8 → 16 → 32 → chaos.
    """
    if n_clusters <= 1:
        return 1
    if n_clusters <= 2:
        return 2
    if n_clusters <= 4:
        return 4
    if n_clusters <= 8:
        return 8
    if n_clusters <= 16:
        return 16
    return _MAX_PERIOD


def _fallback_koopman() -> KoopmanResult:
    """Minimal KoopmanResult for degenerate cases (too-short trajectory, EDMD failure)."""
    lam = 0.5 + 0.1j
    return KoopmanResult(
        eigenvalues=np.array([lam, lam.conjugate()]),
        eigenvectors=np.eye(2, dtype=complex),
        K_matrix=np.eye(2) * abs(lam),
        spectral_gap=0.0,
        is_stable=False,
        reconstruction_error=float("inf"),
        koopman_trust=0.0,
    )
