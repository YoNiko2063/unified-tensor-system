"""Frequency-Dependent Lifting Operator with Arnold Tongue Resonance Detection.

Replaces static Phi with Fourier-composed calendar-aware operator:

  Phi(t) = A_0 + sum_k A_k * phi_k(theta_k(t), a_k(t))

where phi_k are von Mises basis functions locked to calendar event cycles,
A_k are learned coefficient matrices, and theta_k(t) are phase angles
from CalendarRegimeEncoder.

Von Mises basis: phi_k(theta, a, kappa) = a * exp(kappa * (cos(theta) - 1))
  - theta=0 (event now) -> phi_k = a (maximum)
  - theta=pi (maximally far) -> phi_k ~ 0

Arnold tongue resonance: detects when multiple event cycles lock into
rational frequency ratios (e.g., Fed/earnings 2:1), amplifying volatility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from tensor.calendar_regime import (
    CHANNEL_NAMES,
    CYCLE_PERIODS,
    N_CHANNELS,
    CalendarPhase,
)
from tensor.timescale_state import LiftingOperator

# Von Mises concentration per channel
# Higher kappa = narrower peak around the event
KAPPA = np.array([4.0, 8.0, 3.0, 4.0, 2.0])


def von_mises_basis(theta: float, amplitude: float, kappa: float) -> float:
    """Von Mises basis function.

    phi_k(theta, a, kappa) = a * exp(kappa * (cos(theta) - 1))

    Returns value in [0, amplitude].
    """
    return float(amplitude * np.exp(kappa * (np.cos(theta) - 1.0)))


def von_mises_vector(phase: CalendarPhase) -> np.ndarray:
    """Evaluate all N_CHANNELS basis functions from a CalendarPhase.

    Returns (N_CHANNELS,) vector of basis weights.
    """
    weights = np.zeros(N_CHANNELS)
    for k in range(N_CHANNELS):
        weights[k] = von_mises_basis(phase.theta[k], phase.amplitudes[k], KAPPA[k])
    return weights


@dataclass
class LiftResult:
    """Result of a calendar-aware lift operation."""
    delta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    basis_weights: np.ndarray = field(default_factory=lambda: np.zeros(N_CHANNELS))
    resonance: Optional[ResonanceReport] = None
    dominant_channel: str = ""


@dataclass
class ResonanceReport:
    """Arnold tongue resonance detection result."""
    frequency_ratios: List[Tuple[str, str, float]] = field(default_factory=list)
    nearest_rationals: List[Tuple[str, str, int, int]] = field(default_factory=list)
    tongue_widths: List[Tuple[str, str, float]] = field(default_factory=list)
    is_resonant: bool = False
    resonant_pairs: List[Tuple[str, str]] = field(default_factory=list)


def best_rational_approximation(x: float, max_denom: int = 8) -> Tuple[int, int]:
    """Best rational approximation p/q to x via continued fractions.

    Returns (p, q) with q <= max_denom and p/q closest to x.
    """
    if x < 0:
        p, q = best_rational_approximation(-x, max_denom)
        return (-p, q)

    best_p, best_q = round(x), 1
    best_err = abs(x - best_p)

    # Continued fraction convergents
    # h_{-1}=1, h_0=a_0; k_{-1}=0, k_0=1
    h_prev, h_curr = 1, int(x)
    k_prev, k_curr = 0, 1
    remainder = x - int(x)

    for _ in range(20):  # max iterations
        if abs(remainder) < 1e-12:
            break
        remainder = 1.0 / remainder
        a = int(remainder)
        remainder -= a

        h_new = a * h_curr + h_prev
        k_new = a * k_curr + k_prev

        if k_new > max_denom:
            break

        h_prev, h_curr = h_curr, h_new
        k_prev, k_curr = k_curr, k_new

        err = abs(x - h_curr / k_curr)
        if err < best_err:
            best_p, best_q = h_curr, k_curr
            best_err = err

    return (best_p, best_q)


def detect_resonance(phase: CalendarPhase, amplitude_threshold: float = 0.05) -> ResonanceReport:
    """Arnold tongue resonance detection across active calendar cycles.

    For each pair of active cycles, compute:
      1. Frequency ratio omega_i / omega_j
      2. Best rational approximation p/q
      3. Arnold tongue width: width ~ epsilon^q * 2/(p+q)
         where epsilon = a_i * a_j (coupling strength)
      4. Flag resonant if distance from exact rational < tongue width

    Key structural resonance: 63/31.5 = 2.0 exactly (Fed/earnings 2:1 lock).
    """
    # Find active channels
    active = [
        i for i in range(N_CHANNELS)
        if phase.amplitudes[i] > amplitude_threshold
    ]

    frequency_ratios = []
    nearest_rationals = []
    tongue_widths = []
    resonant_pairs = []

    for idx_a in range(len(active)):
        for idx_b in range(idx_a + 1, len(active)):
            i, j = active[idx_a], active[idx_b]
            name_i = CHANNEL_NAMES[i]
            name_j = CHANNEL_NAMES[j]

            # Frequency ratio (using cycle periods: freq = 1/period)
            freq_i = 1.0 / CYCLE_PERIODS[i]
            freq_j = 1.0 / CYCLE_PERIODS[j]
            ratio = CYCLE_PERIODS[i] / CYCLE_PERIODS[j]  # period ratio

            frequency_ratios.append((name_i, name_j, ratio))

            # Best rational approximation
            p, q = best_rational_approximation(ratio, max_denom=8)

            nearest_rationals.append((name_i, name_j, p, q))

            # Arnold tongue width
            epsilon = phase.amplitudes[i] * phase.amplitudes[j]
            if q > 0:
                width = float(epsilon ** q * 2.0 / (abs(p) + q))
            else:
                width = 0.0
            tongue_widths.append((name_i, name_j, width))

            # Distance from exact rational
            if q > 0:
                exact_rational = p / q
                distance = abs(ratio - exact_rational)
                if distance < width:
                    resonant_pairs.append((name_i, name_j))

    is_resonant = len(resonant_pairs) > 0

    return ResonanceReport(
        frequency_ratios=frequency_ratios,
        nearest_rationals=nearest_rationals,
        tongue_widths=tongue_widths,
        is_resonant=is_resonant,
        resonant_pairs=resonant_pairs,
    )


class FrequencyDependentLifter:
    """Calendar-aware lifting operator using Fourier-composed von Mises basis.

    Phi(t) = A_0 + sum_k A_k * phi_k(theta_k(t), a_k(t))

    Extends the LiftingOperator API with calendar awareness.
    Backward compatible: lift() uses A_0 only, lift_at() uses full composition.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        max_rank: int = 10,
        spectral_radius_bound: float = 0.95,
    ) -> None:
        self._source_dim = source_dim
        self._target_dim = target_dim
        self._max_rank = max_rank
        self._sr_bound = spectral_radius_bound

        # Baseline operator A_0
        self._baseline = LiftingOperator(
            source_dim=source_dim,
            target_dim=target_dim,
            max_rank=max_rank,
            spectral_radius_bound=spectral_radius_bound,
        )

        # Per-cycle coefficient matrices A_k (low-rank: U_k @ V_k^T)
        self._cycle_U: List[Optional[np.ndarray]] = [None] * N_CHANNELS
        self._cycle_V: List[Optional[np.ndarray]] = [None] * N_CHANNELS
        self._cycle_fitted: List[bool] = [False] * N_CHANNELS

    def fit(
        self,
        source_states: np.ndarray,
        target_deltas: np.ndarray,
        regularization: float = 1.0,
    ) -> None:
        """Fit baseline A_0 (backward compatible with LiftingOperator)."""
        self._baseline.fit(source_states, target_deltas, regularization)

    def fit_calendar(
        self,
        source_states: np.ndarray,
        target_deltas: np.ndarray,
        calendar_phases: List[CalendarPhase],
        regularization: float = 1.0,
        min_active_samples: int = 5,
    ) -> None:
        """Two-stage regression for calendar-aware lifting.

        Stage 1: Fit A_0 (baseline, time-averaged).
        Stage 2: For each cycle k, weight-regress A_k from residuals
                 where phi_k is active.
        """
        n = source_states.shape[0]
        if n < 3:
            return

        # Stage 1: Fit baseline
        self._baseline.fit(source_states, target_deltas, regularization)
        if not self._baseline.is_fitted:
            return

        # Compute residuals
        residuals = np.zeros_like(target_deltas)
        for i in range(n):
            residuals[i] = target_deltas[i] - self._baseline.lift(source_states[i])

        # Stage 2: Per-cycle regression
        for k in range(N_CHANNELS):
            # Compute basis weights for this cycle
            phi_weights = np.array([
                von_mises_basis(cp.theta[k], cp.amplitudes[k], KAPPA[k])
                for cp in calendar_phases
            ])

            # Active samples: where phi_k is non-negligible
            active_mask = phi_weights > 0.01
            n_active = int(np.sum(active_mask))
            if n_active < min_active_samples:
                continue

            # Weighted regression: A_k from phi_k-weighted residuals
            # r_i = A_k @ x_i * phi_k_i  =>  r_i / phi_k_i = A_k @ x_i
            # Use phi_k as sample weights
            active_sources = source_states[active_mask]
            active_residuals = residuals[active_mask]
            active_phi = phi_weights[active_mask]

            # Weight sources and residuals by sqrt(phi_k) for WLS
            sqrt_phi = np.sqrt(active_phi)[:, np.newaxis]
            weighted_sources = active_sources * sqrt_phi
            weighted_residuals = active_residuals * sqrt_phi

            # Ridge regression
            d = self._source_dim
            XtX = weighted_sources.T @ weighted_sources + regularization * np.eye(d)
            XtY = weighted_sources.T @ weighted_residuals

            try:
                A_k = np.linalg.solve(XtX, XtY).T  # (target_dim, source_dim)
            except np.linalg.LinAlgError:
                continue

            # SVD truncation (same as LiftingOperator)
            U, S, Vt = np.linalg.svd(A_k, full_matrices=False)
            rank = min(self._max_rank, len(S), int(np.sum(S > 1e-10 * S[0])))
            rank = max(rank, 1)

            S_truncated = S[:rank].copy()
            # Scale down if needed (but not by sr_bound directly -- we verify sum later)
            scale = self._sr_bound / max(S_truncated[0], 1e-15)
            if scale < 1.0:
                S_truncated *= scale

            self._cycle_U[k] = U[:, :rank] * S_truncated[np.newaxis, :]
            self._cycle_V[k] = Vt[:rank, :].T
            self._cycle_fitted[k] = True

        # Verify worst-case spectral radius of A_0 + sum(A_k)
        self._enforce_worst_case_bound()

    def _enforce_worst_case_bound(self) -> None:
        """Ensure rho(A_0 + sum_k A_k) < spectral_radius_bound."""
        if not self._baseline.is_fitted:
            return

        worst_case = self._get_baseline_matrix().copy()
        for k in range(N_CHANNELS):
            if self._cycle_fitted[k]:
                worst_case += self._cycle_U[k] @ self._cycle_V[k].T

        # Check spectral radius
        if min(worst_case.shape) == 0:
            return
        sv = np.linalg.svd(worst_case, compute_uv=False)
        sr = sv[0] if len(sv) > 0 else 0.0

        if sr > self._sr_bound:
            # Scale all cycle matrices uniformly
            ratio = self._sr_bound / sr * 0.99  # slight safety margin
            for k in range(N_CHANNELS):
                if self._cycle_fitted[k]:
                    self._cycle_U[k] = self._cycle_U[k] * ratio

    def _get_baseline_matrix(self) -> np.ndarray:
        """Reconstruct full baseline matrix A_0."""
        if not self._baseline.is_fitted:
            return np.zeros((self._target_dim, self._source_dim))
        return self._baseline._U @ self._baseline._V.T

    def lift(self, source_state: np.ndarray) -> np.ndarray:
        """Backward compatible lift using A_0 only."""
        return self._baseline.lift(source_state)

    def lift_at(
        self,
        source_state: np.ndarray,
        phase: CalendarPhase,
        compute_resonance: bool = False,
    ) -> LiftResult:
        """Calendar-aware lift.

        delta = A_0 @ x + sum_k A_k * phi_k(theta_k, a_k) @ x
        """
        # Baseline
        delta = self._baseline.lift(source_state).copy()

        # Von Mises basis weights
        basis_weights = von_mises_vector(phase)

        # Calendar modulation
        for k in range(N_CHANNELS):
            if self._cycle_fitted[k] and basis_weights[k] > 1e-10:
                A_k_x = self._cycle_U[k] @ (self._cycle_V[k].T @ source_state)
                delta += basis_weights[k] * A_k_x

        # Dominant channel
        dominant_idx = int(np.argmax(basis_weights))
        dominant_channel = CHANNEL_NAMES[dominant_idx]

        # Optional resonance detection
        resonance = None
        if compute_resonance:
            resonance = detect_resonance(phase)

        return LiftResult(
            delta=delta,
            basis_weights=basis_weights,
            resonance=resonance,
            dominant_channel=dominant_channel,
        )

    @property
    def is_fitted(self) -> bool:
        return self._baseline.is_fitted

    @property
    def spectral_radius(self) -> float:
        return self._baseline.spectral_radius

    @property
    def rank(self) -> int:
        return self._baseline.rank

    def worst_case_spectral_radius(self) -> float:
        """Spectral radius of A_0 + sum_k A_k (all cycles active at max)."""
        if not self._baseline.is_fitted:
            return 0.0
        worst = self._get_baseline_matrix().copy()
        for k in range(N_CHANNELS):
            if self._cycle_fitted[k]:
                worst += self._cycle_U[k] @ self._cycle_V[k].T
        sv = np.linalg.svd(worst, compute_uv=False)
        return float(sv[0]) if len(sv) > 0 else 0.0


    # ── Convenience wrappers matching CalendarRegimeEncoder-based spec ─────────

    def lift_shock_to_regime(
        self,
        x_S: np.ndarray,
        dt: "date",
        encoder: "Optional[CalendarRegimeEncoder]" = None,
    ) -> np.ndarray:
        """Lift shock state to regime perturbation.

        If encoder is provided, uses calendar-aware lift_at().
        Otherwise falls back to baseline lift().

        Args:
            x_S: (source_dim,) shock feature vector.
            dt: Calendar date for calendar-aware lifting.
            encoder: Optional CalendarRegimeEncoder instance.

        Returns:
            (target_dim,) regime perturbation vector.
        """
        if encoder is not None:
            phase = encoder.encode(dt)
            return self.lift_at(x_S, phase).delta
        return self.lift(x_S)

    def lift_regime_to_fundamental(
        self,
        x_M: np.ndarray,
        dt: "date",
        encoder: "Optional[CalendarRegimeEncoder]" = None,
    ) -> np.ndarray:
        """Lift regime state to fundamental perturbation.

        If encoder is provided, uses calendar-aware lift_at().
        Otherwise falls back to baseline lift().

        Args:
            x_M: (source_dim,) regime feature vector.
            dt: Calendar date for calendar-aware lifting.
            encoder: Optional CalendarRegimeEncoder instance.

        Returns:
            (target_dim,) fundamental perturbation vector.
        """
        if encoder is not None:
            phase = encoder.encode(dt)
            return self.lift_at(x_M, phase).delta
        return self.lift(x_M)

    def fit_with_dates(
        self,
        source_states: np.ndarray,
        target_deltas: np.ndarray,
        dates: "List[date]",
        encoder: "Optional[CalendarRegimeEncoder]" = None,
        regularization: float = 1.0,
    ) -> None:
        """Fit with date list and optional CalendarRegimeEncoder.

        If encoder is provided, wraps dates into CalendarPhase objects and
        calls fit_calendar(). Otherwise falls back to fit() baseline.

        Args:
            source_states: (N, source_dim) source timescale states.
            target_deltas: (N, target_dim) target timescale changes.
            dates: List of N calendar dates (one per observation).
            encoder: Optional CalendarRegimeEncoder instance.
            regularization: Ridge regularization strength.
        """
        if encoder is not None and len(dates) == len(source_states):
            calendar_phases = [encoder.encode(d) for d in dates]
            self.fit_calendar(
                source_states,
                target_deltas,
                calendar_phases,
                regularization=regularization,
            )
        else:
            self.fit(source_states, target_deltas, regularization)
