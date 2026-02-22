"""Spectral Coherence Regime System — Reharmonization with Statistical Persistence.

5-layer architecture:
  L1: BootstrappedSpectrumTracker — EDMD + bootstrap CIs + adaptive recomputation
  L2: CoherenceScorer — continuous lock score, fixed-length lock vector
  L3: RegimePersistenceFilter — HMM-smoothed lock state with hysteresis
  L4: DuffingParameterFilter — EKF Duffing params (Phase B, gated)
  L5: ProfitWindow — vol targeting + drawdown (Phase C, gated)

Orchestrator: ReharmonizationTracker composes all layers.

Core principle: cross-timescale spectral coherence persistence is the signal.
Regime changes are reharmonizations — frequency content at one timescale
reorganizes into new rational structure at another.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from tensor.bifurcation_detector import BifurcationDetector, BifurcationStatus
from tensor.koopman_edmd import EDMDKoopman
from tensor.spectral_path import DissonanceMetric
from tensor.frequency_dependent_lifter import best_rational_approximation


# ── Layer 1: Spectral Estimation ─────────────────────────────────────────────

@dataclass
class FrequencyEstimate:
    """A single spectral mode with bootstrap confidence intervals."""
    frequency: float          # median across bootstrap replicates
    damping: float            # median damping rate
    confidence_low: float     # 5th percentile of frequency
    confidence_high: float    # 95th percentile
    variance: float           # Var[omega_k] across replicates
    stability_score: float    # fraction of replicates where mode exists
    amplitude: float          # RMS amplitude of this mode in state space


@dataclass
class TimescaleSpectrum:
    """Bootstrapped spectrum for one timescale."""
    modes: List[FrequencyEstimate]   # sorted by amplitude (dominant first)
    koopman_trust: float             # from EDMD trust gate
    gram_condition: float            # condition number of Gram matrix
    spectral_entropy: float          # normalized: -sum p_k log p_k / log(k), in [0, 1]
    timestamp: float


class BootstrappedSpectrumTracker:
    """Layer 1: Bootstrapped EDMD with spectral stability scoring.

    Per-timescale sliding window buffers. Adaptive recomputation to avoid
    CPU explosion: full bootstrap only when spectral state changes significantly.
    Hungarian mode matching prevents artificial variance inflation.
    """

    def __init__(
        self,
        window_size: int = 60,
        n_bootstrap: int = 20,
        observable_degree: int = 2,
        dt: float = 1.0,
        n_modes: int = 5,
        min_stability: float = 0.6,
        recompute_interval: int = 5,
    ) -> None:
        self.window_size = window_size
        self.n_bootstrap = n_bootstrap
        self.observable_degree = observable_degree
        self.dt = dt
        self.n_modes = n_modes
        self.min_stability = min_stability
        self.recompute_interval = recompute_interval

        # Per-scale buffers and cached spectra
        self._buffers: Dict[str, deque] = {}
        self._cached_spectra: Dict[str, Optional[TimescaleSpectrum]] = {}
        self._ticks_since_recompute: Dict[str, int] = {}
        self._last_entropy: Dict[str, float] = {}
        self._last_gram_cond: Dict[str, float] = {}

    def push_state(self, scale: str, state: np.ndarray) -> None:
        """Add a state observation to the specified timescale buffer."""
        if scale not in self._buffers:
            self._buffers[scale] = deque(maxlen=self.window_size)
            self._cached_spectra[scale] = None
            self._ticks_since_recompute[scale] = 0
            self._last_entropy[scale] = -1.0
            self._last_gram_cond[scale] = -1.0
        self._buffers[scale].append(np.asarray(state, dtype=np.float64))
        self._ticks_since_recompute[scale] += 1

    def _needs_recompute(self, scale: str) -> bool:
        """Check if full bootstrap recomputation is needed."""
        ticks = self._ticks_since_recompute.get(scale, 0)
        if ticks >= self.recompute_interval:
            return True
        cached = self._cached_spectra.get(scale)
        if cached is None:
            return True
        # Quick rolling EDMD to check entropy/condition drift
        buf = self._buffers.get(scale)
        if buf is None or len(buf) < 3:
            return False
        try:
            traj = np.array(list(buf))
            edmd = EDMDKoopman(observable_degree=self.observable_degree)
            edmd.fit_trajectory(traj)
            result = edmd.eigendecomposition()
            entropy = self._compute_entropy(result.eigenvalues)
            gram_cond = edmd._gram_cond
        except Exception:
            return True

        last_ent = self._last_entropy.get(scale, -1.0)
        last_gc = self._last_gram_cond.get(scale, -1.0)
        if last_ent >= 0 and abs(entropy - last_ent) > 0.1 * max(last_ent, 1e-10):
            return True
        if last_gc > 0 and abs(gram_cond - last_gc) > 0.5 * max(last_gc, 1e-10):
            return True
        return False

    def _compute_entropy(self, eigenvalues: np.ndarray) -> float:
        """Normalized spectral entropy from eigenvalue magnitudes."""
        mags = np.abs(eigenvalues)
        mags = mags[mags > 1e-15]
        if len(mags) == 0:
            return 0.0
        p = mags / mags.sum()
        raw = -float(np.sum(p * np.log(p + 1e-30)))
        k = max(len(p), 2)
        return raw / math.log(k)

    def compute_spectrum(self, scale: str) -> Optional[TimescaleSpectrum]:
        """Compute bootstrapped spectrum for the given timescale.

        Returns cached spectrum if no recompute is triggered.
        """
        buf = self._buffers.get(scale)
        if buf is None or len(buf) < 3:
            return None

        if not self._needs_recompute(scale):
            return self._cached_spectra.get(scale)

        traj = np.array(list(buf))
        n_samples = len(traj)

        # Base EDMD fit
        base_edmd = EDMDKoopman(observable_degree=self.observable_degree)
        base_edmd.fit_trajectory(traj)
        base_result = base_edmd.eigendecomposition()
        base_eigvals = base_result.eigenvalues

        # Extract base mode frequencies (imaginary parts give oscillation freq)
        base_freqs = np.abs(np.imag(base_eigvals))
        base_dampings = -np.real(base_eigvals)
        base_amplitudes = np.abs(base_eigvals)

        # Sort by amplitude (magnitude)
        sort_idx = np.argsort(base_amplitudes)[::-1]
        base_freqs = base_freqs[sort_idx]
        base_dampings = base_dampings[sort_idx]
        base_amplitudes = base_amplitudes[sort_idx]
        n_track = min(self.n_modes, len(base_freqs))

        if n_track == 0:
            return None

        # Bootstrap: resample and re-fit
        boot_freqs = [[] for _ in range(n_track)]  # boot_freqs[mode_k] = list of matched freqs
        boot_dampings = [[] for _ in range(n_track)]
        rng = np.random.default_rng()

        for _ in range(self.n_bootstrap):
            idx = rng.choice(n_samples - 1, size=n_samples - 1, replace=True)
            # Build trajectory pairs from resampled indices
            pairs = [(traj[i], traj[i + 1]) for i in idx]
            try:
                boot_edmd = EDMDKoopman(observable_degree=self.observable_degree)
                boot_edmd.fit(pairs)
                boot_result = boot_edmd.eigendecomposition()
            except Exception:
                continue

            boot_eig = boot_result.eigenvalues
            boot_f = np.abs(np.imag(boot_eig))
            boot_d = -np.real(boot_eig)
            boot_a = np.abs(boot_eig)
            b_sort = np.argsort(boot_a)[::-1]
            boot_f = boot_f[b_sort]
            boot_d = boot_d[b_sort]

            n_b = min(n_track, len(boot_f))
            if n_b == 0:
                continue

            # Hungarian assignment for mode matching
            cost = np.full((n_track, n_b), 1e10)
            for k in range(n_track):
                for j in range(n_b):
                    cost[k, j] = abs(base_freqs[k] - boot_f[j])

            row_ind, col_ind = linear_sum_assignment(cost)
            for k, j in zip(row_ind, col_ind):
                # Reject if cost too high (mode doesn't exist in this replicate)
                threshold = 0.2 * max(base_freqs[k], 0.1)
                if cost[k, j] <= threshold:
                    boot_freqs[k].append(boot_f[j])
                    boot_dampings[k].append(boot_d[j])

        # Build FrequencyEstimate per mode
        modes: List[FrequencyEstimate] = []
        for k in range(n_track):
            n_matched = len(boot_freqs[k])
            stability = n_matched / max(self.n_bootstrap, 1)
            if n_matched >= 2:
                arr = np.array(boot_freqs[k])
                med_freq = float(np.median(arr))
                med_damp = float(np.median(boot_dampings[k]))
                ci_low = float(np.percentile(arr, 5))
                ci_high = float(np.percentile(arr, 95))
                var = float(np.var(arr))
            elif n_matched == 1:
                med_freq = boot_freqs[k][0]
                med_damp = boot_dampings[k][0]
                ci_low = med_freq
                ci_high = med_freq
                var = 0.0
            else:
                med_freq = float(base_freqs[k])
                med_damp = float(base_dampings[k])
                ci_low = med_freq
                ci_high = med_freq
                var = float('inf')

            modes.append(FrequencyEstimate(
                frequency=med_freq,
                damping=med_damp,
                confidence_low=ci_low,
                confidence_high=ci_high,
                variance=var,
                stability_score=stability,
                amplitude=float(base_amplitudes[k]),
            ))

        # Sort modes by amplitude (dominant first)
        modes.sort(key=lambda m: m.amplitude, reverse=True)

        # Normalized spectral entropy from stable modes
        stable_modes = [m for m in modes if m.stability_score >= self.min_stability]
        if len(stable_modes) > 0:
            amps = np.array([m.amplitude for m in stable_modes])
            amps = amps[amps > 1e-15]
            if len(amps) > 0:
                p = amps / amps.sum()
                raw_ent = -float(np.sum(p * np.log(p + 1e-30)))
                k_norm = max(self.n_modes, 2)
                spectral_entropy = raw_ent / math.log(k_norm)
            else:
                spectral_entropy = 0.0
        else:
            spectral_entropy = 1.0  # no stable modes → maximum entropy

        spectrum = TimescaleSpectrum(
            modes=modes,
            koopman_trust=base_result.koopman_trust,
            gram_condition=base_edmd._gram_cond,
            spectral_entropy=min(1.0, max(0.0, spectral_entropy)),
            timestamp=0.0,
        )

        # Update cache
        self._cached_spectra[scale] = spectrum
        self._ticks_since_recompute[scale] = 0
        self._last_entropy[scale] = spectral_entropy
        self._last_gram_cond[scale] = base_edmd._gram_cond

        return spectrum


# ── Layer 2: Lock Coherence Score ─────────────────────────────────────────────

@dataclass
class LockScore:
    """Coherence score between a pair of modes from two timescales."""
    freq_a_idx: int
    freq_b_idx: int
    tau: float               # raw dissonance from DissonanceMetric
    lock_score: float        # exp(-tau/sigma), in [0, 1]
    nearest_rational: Tuple[int, int]
    rational_distance: float


@dataclass
class CoherenceState:
    """Cross-scale coherence snapshot."""
    lock_scores: List[LockScore]
    coherence_energy: float            # mean(lock_scores), NOT sum
    dominant_lock: Optional[LockScore]
    lock_vector: np.ndarray            # FIXED-LENGTH (max_pairs,), zero-padded
    spectral_entropy_drop: float       # normalized entropy(a) - normalized entropy(b)
    timestamp: float


class CoherenceScorer:
    """Layer 2: Continuous lock score per mode pair.

    Fixed-length lock vector with deterministic pair ordering and zero-padding.
    Coherence energy = mean(lock_scores), not sum, for normalization.
    """

    def __init__(
        self,
        sigma: float = 0.05,
        K: int = 10,
        max_rational_denom: int = 8,
        max_pairs: int = 15,
        min_stability: float = 0.6,
        variance_threshold: float = 0.05,
    ) -> None:
        self.sigma = sigma
        self.max_rational_denom = max_rational_denom
        self.max_pairs = max_pairs
        self.min_stability = min_stability
        self.variance_threshold = variance_threshold
        self._dissonance = DissonanceMetric(K=K)

    def score(
        self,
        spectrum_a: TimescaleSpectrum,
        spectrum_b: TimescaleSpectrum,
        timestamp: float = 0.0,
    ) -> CoherenceState:
        """Score coherence between two timescale spectra."""
        # Filter modes by stability and variance
        modes_a = [
            (i, m) for i, m in enumerate(spectrum_a.modes)
            if m.stability_score >= self.min_stability
            and m.variance < self.variance_threshold
        ]
        modes_b = [
            (i, m) for i, m in enumerate(spectrum_b.modes)
            if m.stability_score >= self.min_stability
            and m.variance < self.variance_threshold
        ]

        # Deterministic pair ordering: lexicographic by (idx_a, idx_b)
        lock_scores: List[LockScore] = []
        lock_vector = np.zeros(self.max_pairs)
        pair_idx = 0

        for ia, ma in modes_a:
            for ib, mb in modes_b:
                if pair_idx >= self.max_pairs:
                    break
                tau = self._dissonance.compute(ma.frequency, mb.frequency)
                ls = math.exp(-tau / max(self.sigma, 1e-15))
                p, q = best_rational_approximation(
                    ma.frequency / max(mb.frequency, 1e-15),
                    max_denom=self.max_rational_denom,
                )
                rat_dist = abs(ma.frequency / max(mb.frequency, 1e-15) - p / max(q, 1))

                score = LockScore(
                    freq_a_idx=ia,
                    freq_b_idx=ib,
                    tau=tau,
                    lock_score=ls,
                    nearest_rational=(p, q),
                    rational_distance=rat_dist,
                )
                lock_scores.append(score)
                lock_vector[pair_idx] = ls
                pair_idx += 1
            if pair_idx >= self.max_pairs:
                break

        # Coherence energy = mean of non-zero lock scores
        active_scores = [s.lock_score for s in lock_scores]
        if active_scores:
            coherence_energy = float(np.mean(active_scores))
        else:
            coherence_energy = 0.0

        # Dominant lock
        dominant = max(lock_scores, key=lambda s: s.lock_score) if lock_scores else None

        # Spectral entropy drop
        entropy_drop = spectrum_a.spectral_entropy - spectrum_b.spectral_entropy

        return CoherenceState(
            lock_scores=lock_scores,
            coherence_energy=coherence_energy,
            dominant_lock=dominant,
            lock_vector=lock_vector,
            spectral_entropy_drop=entropy_drop,
            timestamp=timestamp,
        )


# ── Layer 3: Regime Persistence ───────────────────────────────────────────────

@dataclass
class PersistentRegime:
    """A detected persistent regime.

    Named to avoid collision with timescale_state.RegimeState.
    """
    regime_id: int
    centroid: np.ndarray            # mean normalized lock_vector
    coherence_energy_mean: float
    coherence_energy_std: float
    entry_time: float
    exit_time: Optional[float]
    n_observations: int
    confidence: float               # persistence_count / N_entry, clamped [0, 1]
    duffing_probabilities: Optional[np.ndarray] = None  # (4,) from Layer 4


@dataclass
class ReharmonizationEvent:
    """Emitted when spectral regime changes."""
    timestamp: float
    old_regime: PersistentRegime
    new_regime: PersistentRegime
    transition_magnitude: float     # 1 - cosine(old_centroid, new_centroid)
    spectral_entropy_change: float
    bifurcation_status: Optional[BifurcationStatus]


class RegimePersistenceFilter:
    """Layer 3: Hidden state tracking with hysteresis.

    Cosine similarity for threshold interpretability.
    Regime merging to prevent ID explosion.
    Welford online statistics for coherence energy.
    """

    def __init__(
        self,
        N_entry: int = 5,
        N_exit: int = 3,
        similarity_threshold: float = 0.3,
        ema_alpha: float = 0.1,
        merge_threshold: float = 0.85,
        max_pairs: int = 15,
    ) -> None:
        self.N_entry = N_entry
        self.N_exit = N_exit
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha
        self.merge_threshold = merge_threshold
        self.max_pairs = max_pairs

        self._regime_counter = 0
        self._current_regime: Optional[PersistentRegime] = None
        self._break_count = 0
        self._candidate_buffer: List[np.ndarray] = []
        self._candidate_energies: List[float] = []
        self._historical_centroids: Dict[int, np.ndarray] = {}
        self._events: List[ReharmonizationEvent] = []

        # Welford accumulators for current regime
        self._welford_n = 0
        self._welford_mean = 0.0
        self._welford_M2 = 0.0

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize vector, return zero vector if norm is zero."""
        norm = np.linalg.norm(v)
        if norm < 1e-15:
            return np.zeros_like(v)
        return v / norm

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na = self._normalize(a)
        nb = self._normalize(b)
        return float(np.dot(na, nb))

    def _welford_update(self, x: float) -> None:
        """Online mean and variance via Welford's algorithm."""
        self._welford_n += 1
        delta = x - self._welford_mean
        self._welford_mean += delta / self._welford_n
        delta2 = x - self._welford_mean
        self._welford_M2 += delta * delta2

    def _welford_std(self) -> float:
        """Current std from Welford accumulator."""
        if self._welford_n < 2:
            return 0.0
        return math.sqrt(self._welford_M2 / (self._welford_n - 1))

    def _find_matching_historical(self, centroid: np.ndarray) -> Optional[int]:
        """Find a historical regime with similar centroid."""
        for rid, hist_centroid in self._historical_centroids.items():
            sim = self._cosine_similarity(centroid, hist_centroid)
            if sim > self.merge_threshold:
                return rid
        return None

    def _promote_candidate(self, timestamp: float) -> PersistentRegime:
        """Promote candidate buffer to a new (or merged) regime."""
        candidate_centroid = self._normalize(np.mean(self._candidate_buffer, axis=0))
        candidate_energy_mean = float(np.mean(self._candidate_energies))

        # Check for merge with historical regime
        merged_id = self._find_matching_historical(candidate_centroid)
        if merged_id is not None:
            regime_id = merged_id
        else:
            self._regime_counter += 1
            regime_id = self._regime_counter

        regime = PersistentRegime(
            regime_id=regime_id,
            centroid=candidate_centroid,
            coherence_energy_mean=candidate_energy_mean,
            coherence_energy_std=0.0,
            entry_time=timestamp,
            exit_time=None,
            n_observations=len(self._candidate_buffer),
            confidence=min(1.0, len(self._candidate_buffer) / max(self.N_entry, 1)),
        )

        # Store historical centroid
        self._historical_centroids[regime_id] = candidate_centroid.copy()

        # Reset Welford for new regime
        self._welford_n = 0
        self._welford_mean = 0.0
        self._welford_M2 = 0.0
        for e in self._candidate_energies:
            self._welford_update(e)

        return regime

    @property
    def current_regime(self) -> Optional[PersistentRegime]:
        return self._current_regime

    @property
    def events(self) -> List[ReharmonizationEvent]:
        return list(self._events)

    def update(
        self,
        coherence_state: CoherenceState,
        bifurcation_status: Optional[BifurcationStatus] = None,
    ) -> Optional[ReharmonizationEvent]:
        """Update persistence filter with new coherence observation.

        Returns ReharmonizationEvent if regime transition detected.
        """
        lv = coherence_state.lock_vector.copy()
        norm_lv = self._normalize(lv)

        # Bootstrap: no current regime yet
        if self._current_regime is None:
            self._candidate_buffer.append(norm_lv)
            self._candidate_energies.append(coherence_state.coherence_energy)
            if len(self._candidate_buffer) >= self.N_entry:
                self._current_regime = self._promote_candidate(
                    coherence_state.timestamp
                )
                self._candidate_buffer.clear()
                self._candidate_energies.clear()
                self._break_count = 0
            return None

        # Compute distance to current regime centroid
        sim = self._cosine_similarity(norm_lv, self._current_regime.centroid)
        distance = 1.0 - sim

        if distance < self.similarity_threshold:
            # Within current regime
            self._break_count = 0
            self._current_regime.n_observations += 1
            self._current_regime.confidence = min(
                1.0,
                self._current_regime.n_observations / max(self.N_entry, 1),
            )

            # EMA centroid update + re-normalize
            alpha = self.ema_alpha
            self._current_regime.centroid = self._normalize(
                (1 - alpha) * self._current_regime.centroid + alpha * norm_lv
            )

            # Update historical centroid
            rid = self._current_regime.regime_id
            self._historical_centroids[rid] = self._current_regime.centroid.copy()

            # Welford update for coherence energy
            self._welford_update(coherence_state.coherence_energy)
            self._current_regime.coherence_energy_mean = self._welford_mean
            self._current_regime.coherence_energy_std = self._welford_std()

            # Clear any candidate state
            self._candidate_buffer.clear()
            self._candidate_energies.clear()
            return None

        # Outside current regime
        self._break_count += 1
        self._candidate_buffer.append(norm_lv)
        self._candidate_energies.append(coherence_state.coherence_energy)

        if self._break_count < self.N_exit:
            return None

        # Regime break confirmed
        old_regime = self._current_regime
        old_regime.exit_time = coherence_state.timestamp

        # Promote candidate if enough observations
        if len(self._candidate_buffer) >= self.N_entry:
            new_regime = self._promote_candidate(coherence_state.timestamp)
        else:
            # Not enough for full promotion, but regime clearly broke
            new_regime = self._promote_candidate(coherence_state.timestamp)

        self._current_regime = new_regime
        self._candidate_buffer.clear()
        self._candidate_energies.clear()
        self._break_count = 0

        transition_mag = 1.0 - self._cosine_similarity(
            old_regime.centroid, new_regime.centroid
        )

        event = ReharmonizationEvent(
            timestamp=coherence_state.timestamp,
            old_regime=old_regime,
            new_regime=new_regime,
            transition_magnitude=transition_mag,
            spectral_entropy_change=coherence_state.spectral_entropy_drop,
            bifurcation_status=bifurcation_status,
        )
        self._events.append(event)
        return event


# ── Layer 4: Strategy Conditioning (Phase B, gated) ──────────────────────────

@dataclass
class DuffingEstimate:
    """EKF-smoothed Duffing parameter estimates."""
    alpha: float
    alpha_var: float
    beta: float
    beta_var: float
    delta: float
    delta_var: float
    f_drive: float
    omega_drive: float
    regime_probabilities: np.ndarray  # (4,)
    dominant_character: str
    character_confidence: float


class DuffingParameterFilter:
    """Layer 4: EKF-smoothed parameter estimation with observability gating.

    State: theta = [alpha, beta, delta]
    Measurement: h(theta, A) = sqrt(alpha + 3/4 * beta * A^2)  (backbone curve)

    Only active when enable_duffing=True in ReharmonizationTracker.
    """

    CHARACTER_NAMES = ["mean_rev", "breakout", "momentum", "anticipatory"]

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        amplitude_threshold: float = 0.01,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.amplitude_threshold = amplitude_threshold

        # EKF state: [alpha, beta, delta]
        self._theta = np.array([1.0, 0.1, 0.1])
        self._P = np.eye(3) * 1.0
        self._Q = np.eye(3) * process_noise
        self._R = np.array([[measurement_noise]])
        self._last_amplitude: Optional[float] = None
        self._f_drive = 0.0
        self._omega_drive = 1.0

    def predict(self) -> None:
        """Random walk prediction step."""
        self._P = self._P + self._Q

    def update(self, frequency_estimate: FrequencyEstimate, amplitude: float) -> None:
        """EKF update with observability gate on beta."""
        self.predict()

        omega_obs = frequency_estimate.frequency
        alpha, beta, delta = self._theta

        # Predicted frequency from backbone: omega_pred = sqrt(alpha + 3/4 * beta * A^2)
        backbone = alpha + 0.75 * beta * amplitude ** 2
        if backbone <= 0:
            backbone = 1e-10
        omega_pred = math.sqrt(backbone)

        # Innovation
        y = omega_obs - omega_pred

        # Jacobian H = dh/d[alpha, beta, delta]
        # h = sqrt(alpha + 3/4 * beta * A^2)
        # dh/dalpha = 1 / (2 * omega_pred)
        # dh/dbeta = 3*A^2 / (8 * omega_pred)
        # dh/ddelta = 0
        H = np.array([[
            1.0 / (2.0 * omega_pred),
            3.0 * amplitude ** 2 / (8.0 * omega_pred),
            0.0,
        ]])

        # Observability gate: mask beta if amplitude variation too small
        mask_beta = False
        if self._last_amplitude is not None:
            if abs(amplitude - self._last_amplitude) < self.amplitude_threshold:
                mask_beta = True
        self._last_amplitude = amplitude

        # Kalman gain
        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T / max(float(S[0, 0]), 1e-15)

        if mask_beta:
            K[1] = 0.0  # Zero out beta gain

        # State update
        self._theta = self._theta + (K @ np.array([[y]])).flatten()
        I_KH = np.eye(3) - K @ H
        self._P = I_KH @ self._P

    def classify(self) -> np.ndarray:
        """Soft classification into 4 Duffing characters via softmax."""
        alpha, beta, delta = self._theta
        # Features: [alpha/max(|beta|, eps), |beta|, delta, f_drive]
        features = np.array([
            alpha / max(abs(beta), 1e-6),  # mean_rev indicator
            abs(beta),                      # breakout indicator
            max(0.0, 1.0 - delta),         # momentum (low damping)
            self._f_drive,                  # anticipatory (external forcing)
        ])
        # Temperature-scaled softmax
        logits = features * 2.0
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()

    def get_estimate(self) -> DuffingEstimate:
        """Current parameter estimate with variances."""
        probs = self.classify()
        dominant_idx = int(np.argmax(probs))
        return DuffingEstimate(
            alpha=float(self._theta[0]),
            alpha_var=float(self._P[0, 0]),
            beta=float(self._theta[1]),
            beta_var=float(self._P[1, 1]),
            delta=float(self._theta[2]),
            delta_var=float(self._P[2, 2]),
            f_drive=self._f_drive,
            omega_drive=self._omega_drive,
            regime_probabilities=probs,
            dominant_character=self.CHARACTER_NAMES[dominant_idx],
            character_confidence=float(probs[dominant_idx]),
        )


# ── Layer 5: Risk Overlay (Phase C, gated) ───────────────────────────────────

@dataclass
class ProfitWindow:
    """Per-regime profit tracking with online statistics."""
    regime: PersistentRegime
    duffing_estimate: Optional[DuffingEstimate]
    start_time: float
    end_time: Optional[float] = None
    n_returns: int = 0
    cumulative_return: float = 0.0
    return_variance: float = 0.0
    _M2: float = 0.0
    _mean: float = 0.0
    max_drawdown: float = 0.0
    sharpe_estimate: float = 0.0
    is_profitable: bool = False
    position_scale: float = 1.0
    _peak_cumulative: float = 0.0
    min_observations_for_sharpe: int = 20

    def update_returns(self, r: float) -> None:
        """Update online statistics with a new return observation."""
        self.n_returns += 1
        self.cumulative_return += r

        # Welford online variance
        delta = r - self._mean
        self._mean += delta / self.n_returns
        delta2 = r - self._mean
        self._M2 += delta * delta2

        if self.n_returns >= 2:
            self.return_variance = self._M2 / (self.n_returns - 1)

        # Max drawdown tracking
        if self.cumulative_return > self._peak_cumulative:
            self._peak_cumulative = self.cumulative_return
        drawdown = self._peak_cumulative - self.cumulative_return
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Sharpe and profitability only after sufficient observations
        if self.n_returns >= self.min_observations_for_sharpe:
            std = math.sqrt(max(self.return_variance, 1e-15))
            self.sharpe_estimate = self._mean / std
            self.is_profitable = self.sharpe_estimate > 0.0
        else:
            self.sharpe_estimate = 0.0
            self.is_profitable = False
            self.position_scale = 1.0


# ── Orchestrator: ReharmonizationTracker ─────────────────────────────────────

class ReharmonizationTracker:
    """Orchestrates all 5 layers of the spectral coherence regime system.

    Constructor flags isolate subsystems:
      enable_duffing=False — Layer 4 EKF off by default (Phase B)
      enable_profit_window=False — Layer 5 profit tracking off (Phase C)
    """

    def __init__(
        self,
        # Layer 1
        window_size: int = 60,
        n_bootstrap: int = 20,
        observable_degree: int = 2,
        dt_S: float = 1.0,
        dt_M: float = 1.0,
        dt_L: float = 1.0,
        n_modes: int = 5,
        min_stability: float = 0.6,
        recompute_interval: int = 5,
        # Layer 2
        sigma: float = 0.05,
        K: int = 10,
        max_pairs: int = 15,
        # Layer 3
        N_entry: int = 5,
        N_exit: int = 3,
        similarity_threshold: float = 0.3,
        merge_threshold: float = 0.85,
        # Layer 4 (Phase B)
        enable_duffing: bool = False,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        amplitude_threshold: float = 0.01,
        # Layer 5 (Phase C)
        enable_profit_window: bool = False,
        min_observations_for_sharpe: int = 20,
    ) -> None:
        self.enable_duffing = enable_duffing
        self.enable_profit_window = enable_profit_window
        self._min_obs_sharpe = min_observations_for_sharpe
        self._n_modes = n_modes
        self._min_stability = min_stability

        # Layer 1: per-scale spectrum trackers
        self._spectrum_tracker = BootstrappedSpectrumTracker(
            window_size=window_size,
            n_bootstrap=n_bootstrap,
            observable_degree=observable_degree,
            dt=dt_S,
            n_modes=n_modes,
            min_stability=min_stability,
            recompute_interval=recompute_interval,
        )

        # Layer 2: coherence scorer
        self._coherence_scorer = CoherenceScorer(
            sigma=sigma,
            K=K,
            max_pairs=max_pairs,
            min_stability=min_stability,
        )

        # Layer 3: regime persistence
        self._persistence = RegimePersistenceFilter(
            N_entry=N_entry,
            N_exit=N_exit,
            similarity_threshold=similarity_threshold,
            merge_threshold=merge_threshold,
            max_pairs=max_pairs,
        )

        # Layer 4: Duffing EKF (gated)
        self._duffing_filter: Optional[DuffingParameterFilter] = None
        if enable_duffing:
            self._duffing_filter = DuffingParameterFilter(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                amplitude_threshold=amplitude_threshold,
            )

        # Layer 5: profit window (gated)
        self._current_profit_window: Optional[ProfitWindow] = None

        # Bifurcation detectors per scale pair
        self._bifurcation_detectors: Dict[str, BifurcationDetector] = {}

    def update(
        self,
        shock_state: Optional[np.ndarray] = None,
        regime_state: Optional[np.ndarray] = None,
        fundamental_state: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        calendar_phase=None,
        observed_return: Optional[float] = None,
    ) -> Optional[ReharmonizationEvent]:
        """Push new states and run all active layers.

        Returns ReharmonizationEvent if a regime transition is detected.
        """
        # L1: push states into per-scale buffers
        if shock_state is not None:
            self._spectrum_tracker.push_state("S", shock_state)
        if regime_state is not None:
            self._spectrum_tracker.push_state("M", regime_state)
        if fundamental_state is not None:
            self._spectrum_tracker.push_state("L", fundamental_state)

        # L1: compute spectra
        spec_S = self._spectrum_tracker.compute_spectrum("S")
        spec_M = self._spectrum_tracker.compute_spectrum("M")

        # L2: score coherence between S and M (primary pair)
        coherence: Optional[CoherenceState] = None
        if spec_S is not None and spec_M is not None:
            coherence = self._coherence_scorer.score(spec_S, spec_M, timestamp)

        # Also try M→L if available
        spec_L = self._spectrum_tracker.compute_spectrum("L")
        coherence_ML: Optional[CoherenceState] = None
        if spec_M is not None and spec_L is not None:
            coherence_ML = self._coherence_scorer.score(spec_M, spec_L, timestamp)

        # Use whichever coherence is available (prefer S→M)
        primary_coherence = coherence or coherence_ML
        if primary_coherence is None:
            # Not enough data yet
            if self.enable_profit_window and observed_return is not None:
                self._update_profit_window(observed_return)
            return None

        # Bifurcation detection (optional enrichment)
        bif_status = None
        if spec_S is not None:
            eigvals = np.array([
                complex(-m.damping, m.frequency)
                for m in spec_S.modes
                if m.stability_score >= self._min_stability
            ])
            if len(eigvals) > 0:
                if "S_M" not in self._bifurcation_detectors:
                    self._bifurcation_detectors["S_M"] = BifurcationDetector()
                det = self._bifurcation_detectors["S_M"]
                # Reset if eigenvalue count changed (prevents shape mismatch)
                if det._prev_eigvals is not None and len(det._prev_eigvals) != len(eigvals):
                    det._prev_eigvals = None
                    det._prev_min_real = None
                bif_status = det.check(eigvals)

        # L3: regime persistence
        event = self._persistence.update(primary_coherence, bif_status)

        # L4: Duffing EKF (gated)
        if self.enable_duffing and self._duffing_filter is not None:
            regime = self._persistence.current_regime
            if regime is not None and spec_S is not None:
                stable_modes = [
                    m for m in spec_S.modes
                    if m.stability_score >= self._min_stability
                ]
                if stable_modes:
                    dom = stable_modes[0]
                    self._duffing_filter.update(dom, dom.amplitude)
                    est = self._duffing_filter.get_estimate()
                    regime.duffing_probabilities = est.regime_probabilities

        # L5: profit window (gated)
        if self.enable_profit_window and observed_return is not None:
            self._update_profit_window(observed_return)
            if event is not None:
                self._start_new_profit_window(event.new_regime, timestamp)

        return event

    def _update_profit_window(self, observed_return: float) -> None:
        """Update current profit window with observed return."""
        if self._current_profit_window is not None:
            self._current_profit_window.update_returns(observed_return)

    def _start_new_profit_window(
        self, regime: PersistentRegime, timestamp: float
    ) -> None:
        """Start a new profit window for a regime."""
        duffing_est = None
        if self._duffing_filter is not None:
            duffing_est = self._duffing_filter.get_estimate()
        self._current_profit_window = ProfitWindow(
            regime=regime,
            duffing_estimate=duffing_est,
            start_time=timestamp,
            min_observations_for_sharpe=self._min_obs_sharpe,
        )

    def get_regime_vector(self) -> Optional[np.ndarray]:
        """(4,) soft regime probabilities if duffing enabled, else None."""
        if not self.enable_duffing or self._duffing_filter is None:
            return None
        regime = self._persistence.current_regime
        if regime is None or regime.duffing_probabilities is None:
            return None
        return regime.duffing_probabilities.copy()

    def get_coherence_summary(self) -> Dict:
        """Summary dict: coherence_energy, regime_id, regime_confidence, etc."""
        regime = self._persistence.current_regime
        result: Dict = {
            "coherence_energy": 0.0,
            "regime_id": None,
            "regime_confidence": 0.0,
            "spectral_entropy": {},
        }

        if regime is not None:
            result["coherence_energy"] = regime.coherence_energy_mean
            result["regime_id"] = regime.regime_id
            result["regime_confidence"] = regime.confidence

        for scale in ["S", "M", "L"]:
            spec = self._spectrum_tracker._cached_spectra.get(scale)
            if spec is not None:
                result["spectral_entropy"][scale] = spec.spectral_entropy

        return result

    @property
    def spectrum_tracker(self) -> BootstrappedSpectrumTracker:
        return self._spectrum_tracker

    @property
    def coherence_scorer(self) -> CoherenceScorer:
        return self._coherence_scorer

    @property
    def persistence_filter(self) -> RegimePersistenceFilter:
        return self._persistence
