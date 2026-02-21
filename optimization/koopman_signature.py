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
    """

    spectral_radius: float
    slow_mode_count: int
    oscillatory_mode_count: int
    top_k_real: np.ndarray          # shape (k,)
    top_k_imag: np.ndarray          # shape (k,)
    dominant_operator_histogram: np.ndarray   # shape (n_ops,)
    operator_basis_order: List[str]

    def to_retrieval_vector(self) -> np.ndarray:
        """
        Concatenate [top_k_real, top_k_imag] → shape (2k,) retrieval key.

        Including imaginary parts means two systems with the same magnitude
        spectrum but different frequencies (Im(λ)) produce distinct vectors,
        preventing false first-stage matches.
        """
        return np.concatenate([self.top_k_real, self.top_k_imag])


def compute_invariants(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    operator_types: List[str],
    k: int = 5,
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
    )
