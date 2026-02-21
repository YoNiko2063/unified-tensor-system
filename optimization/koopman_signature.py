"""
Koopman signature system for runtime trace analysis.

Two-level signature architecture:
  FullKoopmanSignature     — full eigendecomposition (KoopmanResult alias)
  KoopmanInvariantDescriptor — compact 2k-vector for fast first-stage retrieval

Two-stage retrieval mirrors perceptual hashing:
  coarse filter: L2 on to_retrieval_vector() (top-k real + imag, 2k dims)
  fine verify:   L2 on sorted |eigenvalue| spectra, threshold 0.25

Refinement over naive magnitude-only descriptors:
  Including both Re(λ) and Im(λ) prevents false matches between systems
  with the same magnitude spectrum but different oscillatory structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# Alias: the full Koopman eigendecomposition IS the full signature.
# Reuses the existing EDMDKoopman infrastructure without duplication.
from tensor.koopman_edmd import KoopmanResult as FullKoopmanSignature  # noqa: F401


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
    param_centroid          normalized mean of accepted log-parameter steps,
                            shape (d,); encodes WHERE in parameter space the
                            optimization converged.  Zero-vector if unavailable.
                            This is the physical-location anchor that prevents
                            the purely spectral descriptor from collapsing when
                            different targets produce similar optimization dynamics.
    """

    spectral_radius: float
    slow_mode_count: int
    oscillatory_mode_count: int
    top_k_real: np.ndarray          # shape (k,)
    top_k_imag: np.ndarray          # shape (k,)
    dominant_operator_histogram: np.ndarray   # shape (n_ops,)
    operator_basis_order: List[str]
    param_centroid: np.ndarray = None  # shape (d,); set after __post_init__

    def __post_init__(self):
        if self.param_centroid is None:
            self.param_centroid = np.zeros(3)

    def to_retrieval_vector(self) -> np.ndarray:
        """
        Concatenate [top_k_real, top_k_imag, param_centroid] retrieval key.

        Shape: (2k + d,) where d = len(param_centroid).

        Two-component design:
          - top_k_real / top_k_imag: encodes HOW the optimization ran
            (spectral dynamics, convergence rate, oscillatory modes).
          - param_centroid: encodes WHERE the optimization converged in
            parameter space.  This is the physical anchor that ensures
            descriptors from different physical targets are separated.

        Without param_centroid, different targets can produce similar
        spectral dynamics (all RLC optimizations look similar once near
        the solution) causing the retrieval vector to collapse.
        """
        # Eigenvalue components are dampened (×0.1) so the physically-meaningful
        # param_centroid (location in parameter space) dominates the L2 distance.
        # Without this, near-identical eigenvalues across different targets swamp
        # the centroid signal, collapsing the retrieval metric.
        return np.concatenate([
            self.top_k_real * 0.1,
            self.top_k_imag * 0.1,
            self.param_centroid,
        ])


def compute_invariants(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    operator_types: List[str],
    k: int = 5,
    param_centroid: np.ndarray = None,
) -> KoopmanInvariantDescriptor:
    """
    Compute KoopmanInvariantDescriptor from a Koopman eigendecomposition.

    Eigenvalues are sorted by descending magnitude before extracting top-k,
    so the descriptor is invariant to the ordering returned by np.linalg.eig.

    Args:
        eigenvalues:    complex (d,) Koopman eigenvalues
        eigenvectors:   (d, d) right eigenvectors (columns)
        operator_types: ordered list of operator type names (one per state dim)
        k:              number of top eigenvalues to include in descriptor
        param_centroid: optional (d,) float array encoding WHERE in parameter
                        space the optimization converged (e.g. normalized mean
                        of accepted log-parameter steps).  Zero-vector if None.

    Returns:
        KoopmanInvariantDescriptor
    """
    centroid = np.asarray(param_centroid, dtype=float) if param_centroid is not None else np.zeros(3)

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
            param_centroid=centroid,
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
        param_centroid=centroid,
    )
