"""
Koopman Feature Reducer — windowed EDMD for indicator feature matrices.

Wraps EDMDKoopman to provide:
  1. Windowed fitting with stride (Rule 1-2)
  2. Gram conditioning gate (Rule 3)
  3. Stability enforcement: |λ| > 1+ε → project to unit circle
  4. Cross-validation via eigenvalue drift (Rule 4-5)
  5. Spectral truncation by energy threshold
  6. Spectral robustness score for strategy ranking

Usage:
    reducer = KoopmanFeatureReducer(energy_threshold=0.9)
    result = reducer.fit_windowed(F)           # F: (T, d) feature matrix
    Z = reducer.project(F)                     # Z: (T_out, k) reduced features
    R = reducer.spectral_robustness(F)         # R ∈ [0, 1]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .koopman_edmd import EDMDKoopman, KoopmanResult


@dataclass
class ReductionResult:
    """Result of windowed Koopman feature reduction."""
    Z: np.ndarray                   # (T_out, k) projected features
    eigenvalues: np.ndarray         # (k,) retained eigenvalues (complex)
    eigenvectors: np.ndarray        # (d_obs, k) projection matrix V_k
    energy_retained: float          # Σ|λ_i|² (top-k) / Σ|λ_i|² (all)
    spectral_gap: float             # |λ_1| - |λ_2|
    reconstruction_error: float     # mean MSE across windows
    gram_condition: float           # median κ(Φ) from EDMD across windows
    trust: float                    # Koopman trust score
    n_windows: int                  # number of valid windows used
    rejected_windows: int           # windows rejected by conditioning/stability


class KoopmanFeatureReducer:
    """Windowed EDMD reducer with stability enforcement and spectral truncation.

    Applies EDMDKoopman on sliding windows of a (T, d) feature matrix,
    aggregates eigenvalues across windows (median for robustness),
    truncates to top-k modes by energy, and builds a projection matrix.
    """

    def __init__(
        self,
        energy_threshold: float = 0.9,
        max_condition: float = 1e6,
        observable_degree: int = 2,
        stability_epsilon: float = 0.01,
        spectral_gap_threshold: float = 0.1,
    ):
        """
        Args:
            energy_threshold: η — keep modes until this fraction of total energy.
            max_condition: κ_max — reject window if Gram condition exceeds this.
            observable_degree: polynomial basis degree for EDMD.
            stability_epsilon: project |λ| > 1 + ε to unit circle.
            spectral_gap_threshold: inherited by EDMDKoopman.
        """
        self.energy_threshold = energy_threshold
        self.max_condition = max_condition
        self.observable_degree = observable_degree
        self.stability_epsilon = stability_epsilon
        self.spectral_gap_threshold = spectral_gap_threshold

        # Stored after fitting
        self._V_k: Optional[np.ndarray] = None  # (d_obs, k) projection matrix
        self._eigenvalues: Optional[np.ndarray] = None
        self._edmd: Optional[EDMDKoopman] = None
        self._fitted = False

    def fit_windowed(
        self,
        F: np.ndarray,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> ReductionResult:
        """Fit windowed EDMD on feature matrix F.

        Args:
            F: (T, d) feature matrix.
            window_size: W — samples per window. Default: max(10*d, 60).
            stride: step between windows. Default: W // 4.

        Returns:
            ReductionResult with projected features, eigenvalues, etc.

        Raises:
            ValueError: if W < 10 * d (Rule 1).
        """
        T, d = F.shape

        # Z-score normalize features to prevent Gram ill-conditioning
        # from scale mismatches (e.g., RSI in [0,100] vs Momentum in [-0.1,0.1])
        self._feat_mean = np.mean(F, axis=0)
        self._feat_std = np.std(F, axis=0)
        self._feat_std[self._feat_std < 1e-10] = 1.0  # avoid division by zero
        F = (F - self._feat_mean) / self._feat_std

        # Rule 1: window must be large enough
        if window_size is None:
            window_size = max(10 * d, 60)
        if window_size < 10 * d:
            raise ValueError(
                f"Window size {window_size} < 10 * d = {10 * d}. "
                "Need W >= 10*d for stable EDMD (Rule 1)."
            )

        # Rule 2: stride default
        if stride is None:
            stride = max(window_size // 4, 1)

        # Set up EDMD template
        edmd_template = EDMDKoopman(
            observable_degree=self.observable_degree,
            spectral_gap_threshold=self.spectral_gap_threshold,
        )

        # Determine observable dimension
        d_obs = edmd_template._observable_dim(d)

        # Collect per-window results
        all_eigenvalues = []
        all_eigenvectors = []
        all_recon_errors = []
        all_gram_conds = []
        all_trusts = []
        rejected = 0
        total_windows = 0

        for start in range(0, T - window_size + 1, stride):
            total_windows += 1
            window_data = F[start:start + window_size]

            # Fit EDMD on this window
            edmd = EDMDKoopman(
                observable_degree=self.observable_degree,
                spectral_gap_threshold=self.spectral_gap_threshold,
            )
            edmd.fit_trajectory(window_data)

            # Rule 3: check Gram conditioning
            if edmd._gram_cond > self.max_condition:
                rejected += 1
                continue

            result = edmd.eigendecomposition()

            # Stability enforcement: project unstable eigenvalues to unit circle
            eigvals = result.eigenvalues.copy()
            mags = np.abs(eigvals)
            unstable = mags > (1.0 + self.stability_epsilon)
            if np.any(unstable):
                eigvals[unstable] = eigvals[unstable] / mags[unstable]

            # Rule 5: cross-validate within window (split in half)
            half = window_size // 2
            edmd_a = EDMDKoopman(
                observable_degree=self.observable_degree,
                spectral_gap_threshold=self.spectral_gap_threshold,
            )
            edmd_b = EDMDKoopman(
                observable_degree=self.observable_degree,
                spectral_gap_threshold=self.spectral_gap_threshold,
            )
            edmd_a.fit_trajectory(window_data[:half])
            edmd_b.fit_trajectory(window_data[half:])
            result_a = edmd_a.eigendecomposition()
            result_b = edmd_b.eigendecomposition()
            drift = edmd.eigenfunction_stability(result_a, result_b)

            # Compute trust with drift info
            trust = EDMDKoopman.compute_trust_score(
                gap=result.spectral_gap,
                reconstruction_error=result.reconstruction_error,
                drift=drift,
                gram_cond=edmd._gram_cond,
            )

            all_eigenvalues.append(eigvals)
            all_eigenvectors.append(result.eigenvectors)
            all_recon_errors.append(result.reconstruction_error)
            all_gram_conds.append(edmd._gram_cond)
            all_trusts.append(trust)

        if not all_eigenvalues:
            raise ValueError(
                f"All {total_windows} windows rejected (κ > {self.max_condition}). "
                "Data may be too ill-conditioned for EDMD."
            )

        # Aggregate across valid windows
        n_valid = len(all_eigenvalues)
        eig_matrix = np.array(all_eigenvalues)  # (n_valid, d_obs)

        # Median eigenvalues (by magnitude, preserving complex phase)
        median_mags = np.median(np.abs(eig_matrix), axis=0)
        # Use eigenvectors from window closest to median trust
        median_trust_idx = np.argmin(np.abs(np.array(all_trusts) - np.median(all_trusts)))
        representative_eigvecs = all_eigenvectors[median_trust_idx]

        # Reconstruct eigenvalues with median magnitudes and representative phases
        phases = np.angle(eig_matrix[median_trust_idx])
        median_eigenvalues = median_mags * np.exp(1j * phases)

        mean_recon = float(np.mean(all_recon_errors))
        median_gram = float(np.median(all_gram_conds))
        mean_trust = float(np.mean(all_trusts))

        # Spectral truncation: sort by |λ|, keep top-k until energy ≥ η
        sort_idx = np.argsort(np.abs(median_eigenvalues))[::-1]
        sorted_eigs = median_eigenvalues[sort_idx]
        sorted_vecs = representative_eigvecs[:, sort_idx]

        total_energy = np.sum(np.abs(sorted_eigs) ** 2)
        if total_energy < 1e-15:
            # Degenerate case: all eigenvalues near zero
            k = 1
            energy_retained = 1.0
        else:
            cumulative = np.cumsum(np.abs(sorted_eigs) ** 2) / total_energy
            k = int(np.searchsorted(cumulative, self.energy_threshold) + 1)
            k = min(k, len(sorted_eigs))
            energy_retained = float(cumulative[k - 1])

        retained_eigs = sorted_eigs[:k]
        V_k = sorted_vecs[:, :k]  # (d_obs, k) projection matrix

        # Spectral gap of retained eigenvalues
        mags_sorted = np.abs(sorted_eigs)
        if len(mags_sorted) >= 2:
            spectral_gap = float(mags_sorted[0] - mags_sorted[1])
        else:
            spectral_gap = float(mags_sorted[0])

        # Store for projection
        self._V_k = V_k
        self._eigenvalues = retained_eigs
        self._edmd = EDMDKoopman(
            observable_degree=self.observable_degree,
            spectral_gap_threshold=self.spectral_gap_threshold,
        )
        self._fitted = True

        # Project full F through V_k
        Z = self._project_raw(F, V_k)

        return ReductionResult(
            Z=Z,
            eigenvalues=retained_eigs,
            eigenvectors=V_k,
            energy_retained=energy_retained,
            spectral_gap=spectral_gap,
            reconstruction_error=mean_recon,
            gram_condition=median_gram,
            trust=mean_trust,
            n_windows=n_valid,
            rejected_windows=rejected,
        )

    def project(self, F: np.ndarray) -> np.ndarray:
        """Project feature matrix through stored V_k.

        Args:
            F: (T, d) feature matrix (raw, unnormalized).

        Returns:
            Z: (T, k) projected features.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_windowed() before project()")
        # Apply same normalization used during fitting
        F_norm = (F - self._feat_mean) / self._feat_std
        return self._project_raw(F_norm, self._V_k)

    def _project_raw(self, F: np.ndarray, V_k: np.ndarray) -> np.ndarray:
        """Project F through arbitrary V_k.

        Builds observable basis for each row of F, then projects:
          Z_t = V_k^T · ψ(F_t)
        """
        edmd = EDMDKoopman(
            observable_degree=self.observable_degree,
            spectral_gap_threshold=self.spectral_gap_threshold,
        )
        T = F.shape[0]
        psi = np.array([edmd.build_observable_basis(F[t]) for t in range(T)])  # (T, d_obs)
        Z = psi @ V_k  # (T, k)
        return np.real(Z)  # real part of projection

    def spectral_consistency(self, F: np.ndarray, n_folds: int = 5) -> float:
        """Rule 4: measure eigenvalue drift across rolling windows.

        Returns mean relative drift (lower = more consistent).
        """
        # Normalize features
        feat_mean = np.mean(F, axis=0)
        feat_std = np.std(F, axis=0)
        feat_std[feat_std < 1e-10] = 1.0
        F = (F - feat_mean) / feat_std

        T, d = F.shape
        window_size = max(10 * d, 60)
        if T < window_size * 2:
            return 0.0  # not enough data

        fold_size = T // n_folds
        prev_result = None
        drifts = []

        edmd = EDMDKoopman(
            observable_degree=self.observable_degree,
            spectral_gap_threshold=self.spectral_gap_threshold,
        )

        for i in range(n_folds):
            start = i * fold_size
            end = min(start + fold_size, T)
            if end - start < d + 2:
                continue
            fold_data = F[start:end]

            edmd_fold = EDMDKoopman(
                observable_degree=self.observable_degree,
                spectral_gap_threshold=self.spectral_gap_threshold,
            )
            edmd_fold.fit_trajectory(fold_data)
            curr_result = edmd_fold.eigendecomposition()

            if prev_result is not None:
                drift = edmd.eigenfunction_stability(prev_result, curr_result)
                # Normalize by magnitude
                ref_mag = np.mean(np.abs(prev_result.eigenvalues))
                if ref_mag > 1e-10:
                    drifts.append(drift / ref_mag)
                else:
                    drifts.append(drift)
            prev_result = curr_result

        return float(np.mean(drifts)) if drifts else 0.0

    def spectral_robustness(
        self,
        F: np.ndarray,
        alpha: float = 0.3,
        beta: float = 0.25,
        gamma: float = 0.25,
        delta: float = 0.2,
    ) -> float:
        """Spectral robustness score R ∈ [0, 1].

        R = α·S1 + β·tanh(S2) + γ·S3 + δ·S4

        S1 = spectral stability = 1 - mean eigenvalue drift
        S2 = spectral gap = |λ_1| - |λ_2|
        S3 = reconstruction consistency = 1 - mean_MSE / η_max
        S4 = regime persistence (from BifurcationDetector if available, else 1.0)

        Args:
            F: (T, d) feature matrix.
            alpha, beta, gamma, delta: weights summing to 1.

        Returns:
            R ∈ [0, 1] — spectral robustness score.
        """
        # S1: spectral stability
        consistency = self.spectral_consistency(F)
        S1 = max(0.0, 1.0 - consistency)

        # Fit if needed to get spectral gap and reconstruction error
        result = self.fit_windowed(F)

        # S2: spectral gap
        S2 = result.spectral_gap

        # S3: reconstruction consistency
        eta_max = 1.0
        S3 = max(0.0, 1.0 - result.reconstruction_error / eta_max)

        # S4: regime persistence (use BifurcationDetector if available)
        S4 = self._compute_regime_persistence(F)

        R = alpha * S1 + beta * np.tanh(S2) + gamma * S3 + delta * S4
        return float(np.clip(R, 0.0, 1.0))

    def _compute_regime_persistence(self, F: np.ndarray) -> float:
        """Compute regime persistence via BifurcationDetector.

        Falls back to 1.0 if detector is not available or data is insufficient.
        """
        try:
            from .bifurcation_detector import BifurcationDetector

            # Normalize features
            feat_mean = np.mean(F, axis=0)
            feat_std = np.std(F, axis=0)
            feat_std[feat_std < 1e-10] = 1.0
            F = (F - feat_mean) / feat_std

            detector = BifurcationDetector()
            T, d = F.shape

            # Fit EDMD on windows and check bifurcation status
            window_size = max(10 * d, 60)
            stride = max(window_size // 4, 1)

            stable_count = 0
            total_count = 0

            for start in range(0, T - window_size + 1, stride):
                window_data = F[start:start + window_size]
                edmd = EDMDKoopman(
                    observable_degree=self.observable_degree,
                    spectral_gap_threshold=self.spectral_gap_threshold,
                )
                edmd.fit_trajectory(window_data)
                result = edmd.eigendecomposition()
                status = detector.check(result.eigenvalues)

                if status.status == 'stable':
                    stable_count += 1
                total_count += 1

            return stable_count / max(total_count, 1)

        except ImportError:
            return 1.0
