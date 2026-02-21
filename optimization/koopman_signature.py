"""
Koopman signature system for runtime trace analysis.

Two-level signature architecture:
  FullKoopmanSignature     — full eigendecomposition (KoopmanResult alias)
  KoopmanInvariantDescriptor — compact descriptor for fast first-stage retrieval

Two-stage retrieval:
  coarse filter: L2 on to_query_vector() (3-D domain-invariant dynamical space)
  fine verify:   L2 on sorted |eigenvalue| spectra, threshold 0.25

Domain-invariant dynamical quantities (log_omega0_norm, log_Q_norm, damping_ratio)
replace the old raw param_centroid (log L, log C coordinates).  They encode
WHERE the optimization converged in terms of physical resonance and dissipation,
using the same scale for RLC circuits and spring-mass systems.

Reference normalisation:
  log_omega0_norm = (log(ω₀) − log(2π·1kHz)) / log(10)
    → 1 kHz maps to 0.0; a factor-of-10 change in ω₀ maps to ±1.0
  log_Q_norm     = log(Q) / log(10)
    → Q=1 maps to 0.0; Q=10 maps to +1.0
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass
from typing import List

import numpy as np

# Alias: the full Koopman eigendecomposition IS the full signature.
# Reuses the existing EDMDKoopman infrastructure without duplication.
from tensor.koopman_edmd import KoopmanResult as FullKoopmanSignature  # noqa: F401

# ── Reference constants for log-normalisation ─────────────────────────────────

# log(ω₀ at 1 kHz) = log(2π × 1000)
_LOG_OMEGA0_REF: float = _math.log(2.0 * _math.pi * 1_000.0)

# One decade in ω₀ space
_LOG_OMEGA0_SCALE: float = _math.log(10.0)


@dataclass
class KoopmanInvariantDescriptor:
    """
    Compact, order-invariant descriptor of a Koopman spectrum.

    Used as the fast first-stage filter in KoopmanExperienceMemory retrieval.

    Fields
    ------
    spectral_radius         max |λ|
    slow_mode_count         # eigenvalues with |λ| > 0.9  (persistent modes)
    oscillatory_mode_count  # eigenvalues with |Im(λ)| > 0.1  (oscillatory)
    top_k_real              Re(λ) for top-k by |λ|, shape (k,) — zero-padded
    top_k_imag              Im(λ) for same ordering,  shape (k,) — zero-padded
    dominant_operator_histogram  normalized |eigvec[:,0]|, shape (n_ops,)
    operator_basis_order    ordered list of operator type names

    Domain-invariant dynamical quantities (WHERE the optimization converged):
    log_omega0_norm         (log ω₀ − _LOG_OMEGA0_REF) / _LOG_OMEGA0_SCALE
                            0.0 at 1 kHz; ±1.0 per decade change in ω₀
    log_Q_norm              log(Q) / _LOG_OMEGA0_SCALE
                            0.0 at Q=1; +1.0 at Q=10
    damping_ratio           ζ = 1/(2Q); 0.5 at Q=1, ~0 for high-Q resonators
    """

    spectral_radius: float
    slow_mode_count: int
    oscillatory_mode_count: int
    top_k_real: np.ndarray          # shape (k,)
    top_k_imag: np.ndarray          # shape (k,)
    dominant_operator_histogram: np.ndarray   # shape (n_ops,)
    operator_basis_order: List[str]
    log_omega0_norm: float = 0.0    # (log ω₀ − ref) / scale
    log_Q_norm: float = 0.0         # log(Q) / scale
    damping_ratio: float = 0.5      # ζ = 1/(2Q)

    def to_query_vector(self) -> np.ndarray:
        """
        3-D domain-invariant retrieval key: [log_ω₀_norm, log_Q_norm, ζ].

        This is the primary retrieval signal.  It encodes WHERE the optimization
        converged in terms of physical resonance (ω₀) and dissipation (Q, ζ),
        using the same scale for RLC circuits and spring-mass systems.

        Used in KoopmanExperienceMemory.retrieve_candidates() for distance ranking.
        """
        return np.array([self.log_omega0_norm, self.log_Q_norm, self.damping_ratio])

    def to_retrieval_vector(self) -> np.ndarray:
        """
        Full (2k + 3) retrieval vector: [top_k_real × 0.1, top_k_imag × 0.1,
        log_ω₀_norm, log_Q_norm, ζ].

        Eigenvalue components are dampened (×0.1) so the dynamical quantities
        (which encode physical location) dominate the L2 distance.
        """
        return np.concatenate([
            self.top_k_real * 0.1,
            self.top_k_imag * 0.1,
            self.to_query_vector(),
        ])

    def dynamical_omega0(self) -> float:
        """Recover ω₀ [rad/s] from the normalised log_omega0_norm field."""
        return float(np.exp(self.log_omega0_norm * _LOG_OMEGA0_SCALE + _LOG_OMEGA0_REF))

    def dynamical_Q(self) -> float:
        """Recover Q from the normalised log_Q_norm field."""
        return float(np.exp(self.log_Q_norm * _LOG_OMEGA0_SCALE))


def compute_invariants(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    operator_types: List[str],
    k: int = 5,
    log_omega0_norm: float = 0.0,
    log_Q_norm: float = 0.0,
    damping_ratio: float = 0.5,
) -> KoopmanInvariantDescriptor:
    """
    Compute KoopmanInvariantDescriptor from a Koopman eigendecomposition.

    Eigenvalues are sorted by descending magnitude before extracting top-k,
    so the descriptor is invariant to the ordering returned by np.linalg.eig.

    Args:
        eigenvalues:      complex (d,) Koopman eigenvalues
        eigenvectors:     (d, d) right eigenvectors (columns)
        operator_types:   ordered list of operator type names (one per state dim)
        k:                number of top eigenvalues to include in descriptor
        log_omega0_norm:  (log ω₀ − ref) / scale  — WHERE the optimisation converged
        log_Q_norm:       log(Q) / scale
        damping_ratio:    ζ = 1/(2Q)

    Returns:
        KoopmanInvariantDescriptor
    """
    if len(eigenvalues) == 0:
        zero_k = np.zeros(k)
        return KoopmanInvariantDescriptor(
            spectral_radius=0.0,
            slow_mode_count=0,
            oscillatory_mode_count=0,
            top_k_real=zero_k,
            top_k_imag=zero_k,
            dominant_operator_histogram=np.zeros(max(len(operator_types), 1)),
            operator_basis_order=list(operator_types),
            log_omega0_norm=float(log_omega0_norm),
            log_Q_norm=float(log_Q_norm),
            damping_ratio=float(damping_ratio),
        )

    magnitudes = np.abs(eigenvalues)

    # Sort descending by magnitude
    idx = np.argsort(magnitudes)[::-1]
    sorted_eigs = eigenvalues[idx]
    sorted_vecs = eigenvectors[:, idx]

    # Top-k real and imaginary (zero-padded when fewer than k eigenvalues)
    k_actual = min(k, len(sorted_eigs))
    top_k_real = np.zeros(k)
    top_k_imag = np.zeros(k)
    top_k_real[:k_actual] = np.real(sorted_eigs[:k_actual])
    top_k_imag[:k_actual] = np.imag(sorted_eigs[:k_actual])

    # Dominant operator histogram: normalized absolute components of top eigenvector.
    # Tells you which operator types drive the dominant Koopman mode.
    n_ops = len(operator_types)
    if n_ops > 0 and sorted_vecs.shape[1] > 0:
        top_vec = np.abs(sorted_vecs[:n_ops, 0]).astype(float)
        total = top_vec.sum()
        hist = top_vec / total if total > 1e-12 else top_vec
    else:
        hist = np.zeros(max(n_ops, 1))

    slow_count = int(np.sum(magnitudes > 0.9))
    osc_count = int(np.sum(np.abs(np.imag(eigenvalues)) > 0.1))

    return KoopmanInvariantDescriptor(
        spectral_radius=float(magnitudes.max()),
        slow_mode_count=slow_count,
        oscillatory_mode_count=osc_count,
        top_k_real=top_k_real,
        top_k_imag=top_k_imag,
        dominant_operator_histogram=hist,
        operator_basis_order=list(operator_types),
        log_omega0_norm=float(log_omega0_norm),
        log_Q_norm=float(log_Q_norm),
        damping_ratio=float(damping_ratio),
    )
