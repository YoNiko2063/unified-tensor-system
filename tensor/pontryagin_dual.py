"""
Pontryagin Dual Extractor — frequency character extraction from LCA patches.

Mathematical basis (LOGIC_FLOW.md, Sections 0B, 0E):
  In an abelian (LCA) patch where [Aᵢ,Aⱼ] = 0:
    - Group G_R = {Φᵗ | t ∈ ℝ} ≅ ℝʳ (commuting flows)
    - Pontryagin dual Ĝ_R ≅ ℝʳ (frequency space)
    - Characters χ_λ(x) = e^{λx} are Koopman eigenfunctions in this regime
    - Dominant characters = eigenvectors of the operator basis with largest |λ|

  Cross-patch shared characters:
    Two LCA patches sharing characters → algebraically equivalent operator submanifold
    This corresponds to shared Pontryagin dual elements → cross-domain universals in HDV

Reference: LOGIC_FLOW.md Sections 0B, 0C, 0E
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tensor.lca_patch_detector import PatchClassification


@dataclass
class Character:
    """A Pontryagin character = eigenfunction of the LCA group action."""
    eigenvalue: complex        # λ such that χ_λ(Φᵗx) = e^{λt} χ_λ(x)
    eigenvector: np.ndarray    # real part of the character direction in state space
    frequency: float           # |Im(λ)| — oscillatory frequency
    decay_rate: float          # |Re(λ)| — exponential growth/decay rate
    magnitude: float           # significance weight (from SVD basis)

    @property
    def is_oscillatory(self) -> bool:
        """True if character has significant imaginary component."""
        return self.frequency > 1e-6

    @property
    def is_stable(self) -> bool:
        """True if character corresponds to stable mode (Re(λ) < 0)."""
        return np.real(self.eigenvalue) < 0


@dataclass
class SharedCharacterResult:
    """Result of comparing characters between two patches."""
    characters_a: List[Character]
    characters_b: List[Character]
    shared_indices: List[Tuple[int, int]]  # (idx_in_a, idx_in_b) matched pairs
    alignment_scores: List[float]           # cosine similarity per pair
    mean_alignment: float                   # average across matched pairs


class PontryaginDualExtractor:
    """
    Extracts Pontryagin characters (frequency modes) from LCA patches and
    finds shared characters between pairs of patches.

    In an LCA patch, the Pontryagin dual elements are the dominant eigenvalues
    of the operator algebra. Characters that appear in multiple patches represent
    universal cross-domain modes.

    Usage:
        extractor = PontryaginDualExtractor()
        chars_a = extractor.extract_characters(patch_a)
        chars_b = extractor.extract_characters(patch_b)
        result = extractor.shared_characters(chars_a, chars_b)
        # result.shared_indices → matched (universal) modes
    """

    def __init__(
        self,
        frequency_tol: float = 0.1,
        min_magnitude: float = 1e-3,
    ):
        """
        Args:
            frequency_tol: tolerance for eigenvalue matching between patches
            min_magnitude: minimum singular value to include in character set
        """
        self.frequency_tol = frequency_tol
        self.min_magnitude = min_magnitude

    # ------------------------------------------------------------------
    # Character extraction
    # ------------------------------------------------------------------

    def extract_characters(
        self,
        classification: PatchClassification,
    ) -> List[Character]:
        """
        Extract Pontryagin characters from an LCA patch classification.

        Characters are the eigenmodes of the operator basis matrices.
        For LCA patches: these correspond to Pontryagin dual elements.
        For non-abelian patches: these are approximate characters (Tannaka-Krein modes).

        Args:
            classification: output of LCAPatchDetector.classify_region()

        Returns:
            List of Character objects, sorted by magnitude (descending)
        """
        basis = classification.basis_matrices  # r × n × n
        r, n, _ = basis.shape

        characters = []
        for k in range(r):
            A = basis[k]  # n × n operator

            # Eigendecomposition of this basis matrix
            try:
                eigvals, eigvecs = np.linalg.eig(A)
            except np.linalg.LinAlgError:
                continue

            # Each eigenpair is a potential character
            for i, lam in enumerate(eigvals):
                vec = eigvecs[:, i]
                # Magnitude from operator's spectral weight
                magnitude = abs(lam) / (np.sum(np.abs(eigvals)) + 1e-12)

                if magnitude < self.min_magnitude:
                    continue

                char = Character(
                    eigenvalue=complex(lam),
                    eigenvector=np.real(vec),
                    frequency=abs(np.imag(lam)),
                    decay_rate=abs(np.real(lam)),
                    magnitude=float(magnitude),
                )
                characters.append(char)

        # Also include overall patch eigenvalues as characters
        for i, lam in enumerate(classification.eigenvalues):
            magnitude = abs(lam) / (np.sum(np.abs(classification.eigenvalues)) + 1e-12)
            if magnitude >= self.min_magnitude:
                # Use identity direction (centroid eigenvector approximation)
                char = Character(
                    eigenvalue=complex(lam),
                    eigenvector=classification.centroid / (np.linalg.norm(classification.centroid) + 1e-12),
                    frequency=abs(np.imag(lam)),
                    decay_rate=abs(np.real(lam)),
                    magnitude=float(magnitude),
                )
                characters.append(char)

        # Sort by magnitude descending
        characters.sort(key=lambda c: -c.magnitude)
        return characters

    # ------------------------------------------------------------------
    # Shared character detection
    # ------------------------------------------------------------------

    def shared_characters(
        self,
        chars_a: List[Character],
        chars_b: List[Character],
        top_k: Optional[int] = None,
    ) -> SharedCharacterResult:
        """
        Find characters shared between two patches.

        Two characters match if their eigenvalues are within frequency_tol
        and their eigenvectors are approximately aligned (cosine similarity).

        Args:
            chars_a: characters from patch A (from extract_characters)
            chars_b: characters from patch B (from extract_characters)
            top_k: if given, only compare top-k characters from each patch

        Returns:
            SharedCharacterResult with matched pairs and alignment scores
        """
        if top_k is not None:
            chars_a = chars_a[:top_k]
            chars_b = chars_b[:top_k]

        if not chars_a or not chars_b:
            return SharedCharacterResult(
                characters_a=chars_a,
                characters_b=chars_b,
                shared_indices=[],
                alignment_scores=[],
                mean_alignment=0.0,
            )

        shared_indices = []
        alignment_scores = []

        used_b = set()

        for i, ca in enumerate(chars_a):
            best_j = None
            best_score = -1.0

            for j, cb in enumerate(chars_b):
                if j in used_b:
                    continue

                # Eigenvalue proximity (frequency match)
                freq_diff = abs(ca.frequency - cb.frequency)
                decay_diff = abs(ca.decay_rate - cb.decay_rate)
                if freq_diff > self.frequency_tol or decay_diff > self.frequency_tol * 10:
                    continue

                # Eigenvector alignment (direction similarity)
                va = ca.eigenvector
                vb = cb.eigenvector
                na = np.linalg.norm(va)
                nb = np.linalg.norm(vb)
                if na < 1e-10 or nb < 1e-10:
                    cos_sim = 0.0
                else:
                    cos_sim = float(abs(np.dot(va, vb) / (na * nb)))

                if cos_sim > best_score:
                    best_score = cos_sim
                    best_j = j

            if best_j is not None and best_score > 0.5:  # threshold for "shared"
                shared_indices.append((i, best_j))
                alignment_scores.append(best_score)
                used_b.add(best_j)

        mean_align = float(np.mean(alignment_scores)) if alignment_scores else 0.0

        return SharedCharacterResult(
            characters_a=chars_a,
            characters_b=chars_b,
            shared_indices=shared_indices,
            alignment_scores=alignment_scores,
            mean_alignment=mean_align,
        )

    # ------------------------------------------------------------------
    # HDV dimension mapping
    # ------------------------------------------------------------------

    def map_to_hdv_dims(
        self,
        characters: List[Character],
        hdv_dim: int = 10000,
    ) -> List[int]:
        """
        Map Pontryagin characters to HDV dimension indices.

        Characters with similar frequency → similar HDV dimension cluster.
        Uses frequency-based hashing to assign stable dimension indices.

        Args:
            characters: list of Character objects from a patch
            hdv_dim: total HDV dimensionality

        Returns:
            List of HDV dimension indices (one per character, may have duplicates)
        """
        dims = []
        for char in characters:
            # Hash frequency + decay_rate → deterministic HDV dimension
            freq_bucket = int(char.frequency * 1000) % (hdv_dim // 2)
            decay_bucket = int(char.decay_rate * 100) % (hdv_dim // 2)
            dim = (freq_bucket + decay_bucket) % hdv_dim
            dims.append(dim)
        return dims

    def dominant_frequency_vector(
        self,
        characters: List[Character],
        hdv_dim: int = 10000,
    ) -> np.ndarray:
        """
        Construct a sparse frequency signature vector in HDV space.

        Active dimensions correspond to the character frequencies.
        Values are character magnitudes (spectral weights).

        Args:
            characters: list of Character objects
            hdv_dim: total HDV dimensionality

        Returns:
            np.ndarray of shape (hdv_dim,) — the frequency signature
        """
        vec = np.zeros(hdv_dim, dtype=np.float32)
        dims = self.map_to_hdv_dims(characters, hdv_dim)
        for dim, char in zip(dims, characters):
            vec[dim] += char.magnitude
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec
