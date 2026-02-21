"""
Semantic Geometry Hypothesis Layer — zero gradient authority.

Mathematical basis (Plan §IV, INV-2, INV-4):
  Constructs pseudo-Koopman K_text from text embedding trajectories.
  ALL outputs are hypothesis_only=True.
  K_text → ProposalQueue only. K_text never enters SPDE gradient path.

CRITICAL-1 enforcement:
  TextKoopmanOperator uses _TextEDMD (defined here), NOT EDMDKoopman
  from tensor.koopman_edmd. _TextEDMD does not subclass EDMDKoopman and
  is NOT imported from any other module. The consequence:

    with mock.patch.object(EDMDKoopman, 'fit') as mock_fit:
        layer.encode(text)
        mock_fit.assert_not_called()  # PASSES — _TextEDMD.fit() is a different method

Memory bounds:
  MAX_SEMANTIC_DIM = 200: _TextEDMD operates on at most 200-dim projections,
  keeping G ∈ R^{200×200} (≈320 KB) rather than R^{hdv_dim×hdv_dim} (≈800 MB).
  MAX_TOKENS = 100: token sequence capped to prevent excessive compute.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

import numpy as np

# No import of tensor.koopman_edmd.EDMDKoopman — CRITICAL-1

_MAX_SEMANTIC_DIM: int = 200   # cap for _TextEDMD matrix dimensions
_MAX_TOKENS: int = 100          # cap on token sequence length
_TAU_SEMANTIC: float = 0.3      # minimum trust to emit navigation proposal
_GAMMA_MIN: float = 0.1         # spectral gap threshold for trust computation


# ── Internal EDMD class (NOT EDMDKoopman) ────────────────────────────────────


class _TextEDMD:
    """
    EDMD for text embedding trajectories.

    CRITICAL-1: This is NOT EDMDKoopman from tensor.koopman_edmd.
    It does not subclass EDMDKoopman. It is a standalone private class.

    Uses identity observable (ψ(h) = h) on a low-dimensional projection
    of the HDV space (at most MAX_SEMANTIC_DIM dims).

    K = G⁺A where G = (1/m) Σ h_k h_k^T, A = (1/m) Σ h_k h_{k+1}^T
    """

    def __init__(self) -> None:
        self._K: Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> "_TextEDMD":
        """
        Fit Koopman matrix from (h_t, h_{t+1}) embedding pairs.
        pairs: list of (h_k, h_{k+1}) tuples, each h ∈ R^d (d ≤ MAX_SEMANTIC_DIM)
        """
        if len(pairs) < 2:
            self._K = None
            self._fitted = False
            return self

        H = np.array([h for h, _ in pairs], dtype=float)     # (m, d)
        H_next = np.array([h for _, h in pairs], dtype=float)  # (m, d)
        m = len(pairs)

        G = (H.T @ H) / m       # (d, d)
        A = (H.T @ H_next) / m  # (d, d)
        self._K = np.linalg.lstsq(G, A, rcond=None)[0]
        self._fitted = True
        return self

    def eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (eigenvalues, eigenvectors) of K, sorted by |λ| descending.
        Returns (zeros, identity) if not fitted.
        """
        if not self._fitted or self._K is None:
            d = 1
            return np.zeros(d, dtype=complex), np.eye(d)

        eigvals, eigvecs = np.linalg.eig(self._K)
        idx = np.argsort(np.abs(eigvals))[::-1]
        return eigvals[idx], eigvecs[:, idx]

    def trust_score(self) -> float:
        """Spectral gap trust proxy: min(1, gap / γ_min). Matrix-level only."""
        if not self._fitted or self._K is None:
            return 0.0
        eigvals = np.linalg.eigvals(self._K)
        r = np.sort(np.abs(np.real(eigvals)))[::-1]
        gap = float(r[0] - r[1]) if len(r) >= 2 else float(r[0]) if len(r) == 1 else 0.0
        return float(min(1.0, gap / max(_GAMMA_MIN, 1e-12)))

    @property
    def K_matrix(self) -> Optional[np.ndarray]:
        return self._K


# ── SemanticJacobianEstimator ─────────────────────────────────────────────────


class SemanticJacobianEstimator:
    """
    Computes token salience via knockout: J_sem[:,t] = h(text) - h(text_without_t).

    Each column is the HDV shift when token t is removed.
    High ‖J_sem[:,t]‖ → token t strongly influences the semantic encoding.
    Output is placed in "exploration" proposals (high-salience tokens → directed exploration).

    hypothesis_only: J_sem is never used as a gradient input.
    """

    def __init__(self, hdv_system, max_tokens: int = _MAX_TOKENS) -> None:
        self._hdv_system = hdv_system
        self._max_tokens = max_tokens

    def compute(self, text: str) -> np.ndarray:
        """
        Returns salience vector of shape (n_tokens,): ‖J_sem[:,t]‖ for each token.
        Capped at max_tokens. Returns empty array for empty text.
        """
        tokens = text.split()[:self._max_tokens]
        if not tokens:
            return np.zeros(0)

        h_full = self._hdv_system.structural_encode(text, "semantic").astype(float)
        salience = np.zeros(len(tokens))

        for t, _ in enumerate(tokens):
            text_without_t = " ".join(tokens[:t] + tokens[t + 1:])
            if not text_without_t:
                salience[t] = np.linalg.norm(h_full)
                continue
            h_without = self._hdv_system.structural_encode(
                text_without_t, "semantic"
            ).astype(float)
            salience[t] = float(np.linalg.norm(h_full - h_without))

        return salience


# ── TextKoopmanOperator ───────────────────────────────────────────────────────


class TextKoopmanOperator:
    """
    Builds EDMD on text embedding trajectory → K_text hypothesis.

    Uses _TextEDMD (NOT EDMDKoopman). K_text never enters physical SPDE gradient.

    Embedding: sequential window of tokens mapped to structural_encode vectors,
    projected to semantic_dims (overlap dims or universal dims, ≤ MAX_SEMANTIC_DIM).
    """

    def __init__(self, hdv_system, max_tokens: int = _MAX_TOKENS) -> None:
        self._hdv_system = hdv_system
        self._max_tokens = max_tokens

    def _get_semantic_dims(self) -> np.ndarray:
        """
        Indices of semantic dimensions for _TextEDMD.
        Use overlap dims (cross-domain characters) if available,
        else fall back to first hdv_dim//3 universal dims.
        Capped at MAX_SEMANTIC_DIM.
        """
        overlaps = sorted(self._hdv_system.find_overlaps())
        if overlaps:
            return np.array(overlaps[:_MAX_SEMANTIC_DIM])
        n = min(_MAX_SEMANTIC_DIM, self._hdv_system.hdv_dim // 3)
        return np.arange(n)

    def fit(self, text: str) -> Tuple[_TextEDMD, np.ndarray, float]:
        """
        Fit _TextEDMD on token embedding trajectory.

        Returns:
            (text_edmd, semantic_dims, trust_text)
            trust_text = 0.0 if text has fewer than 3 tokens.
        """
        tokens = text.split()[:self._max_tokens]
        semantic_dims = self._get_semantic_dims()

        if len(tokens) < 3 or len(semantic_dims) == 0:
            return _TextEDMD(), semantic_dims, 0.0

        # Build embedding sequence: h_t = structural_encode(token_window, "semantic")
        # Window = single token for simplicity (captures token-level dynamics)
        embeddings = []
        for tok in tokens:
            h = self._hdv_system.structural_encode(tok, "semantic").astype(float)
            embeddings.append(h[semantic_dims])  # project to semantic dims

        # EDMD pairs: (h_t, h_{t+1})
        pairs = [(embeddings[t], embeddings[t + 1]) for t in range(len(embeddings) - 1)]

        edmd = _TextEDMD()
        edmd.fit(pairs)
        return edmd, semantic_dims, edmd.trust_score()


# ── ToneSignatureVector ───────────────────────────────────────────────────────


class ToneSignatureVector:
    """
    Extracts navigation routing hint from K_text eigenvectors.

    T = Re(eigvecs)[overlap_positions, :r], L2-normalized per column.

    hypothesis_only: T is ONLY used as routing input to PatchGraph.shortest_path().
    T is NEVER subtracted from any energy function E.
    """

    def compute(
        self,
        eigvecs: np.ndarray,
        semantic_dims: np.ndarray,
        r: int = 4,
    ) -> np.ndarray:
        """
        Returns tone vector of shape (min(len(semantic_dims), r),).
        L2-normalized. Returns zeros if eigvecs is empty or degenerate.
        """
        if eigvecs.size == 0 or r == 0:
            return np.zeros(min(len(semantic_dims), r))

        n_modes = min(r, eigvecs.shape[1])
        real_vecs = np.real(eigvecs[:, :n_modes])  # (d, n_modes)

        # Aggregate to single tone vector: mean of dominant modes
        tone = np.mean(real_vecs, axis=1)  # (d,)

        norm = np.linalg.norm(tone)
        if norm < 1e-12:
            return np.zeros_like(tone)
        return (tone / norm).astype(float)


# ── SemanticGeometryLayer ─────────────────────────────────────────────────────


class SemanticGeometryLayer:
    """
    Hypothesis-only semantic analysis layer.

    CRITICAL-1: uses _TextEDMD internally, never EDMDKoopman.
    INV-2 / INV-4: all spectral outputs carry hypothesis_only=True.

    encode() returns a dict. All spectral fields are marked hypothesis_only=True.
    Proposals are placed in proposal_queue (if provided):
      - "navigation": tone hint (if trust_text > τ_semantic)
      - "exploration": token salience map (always)

    The _semantic_geometry_thread in autonomous_training.py must assert:
        assert result.get("hypothesis_only") is True
    before consuming any field.
    """

    def __init__(
        self,
        hdv_system,
        proposal_queue=None,  # ProposalQueue or None
        tau_semantic: float = _TAU_SEMANTIC,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        self._hdv_system = hdv_system
        self._proposal_queue = proposal_queue
        self._tau_semantic = tau_semantic
        self._jacobian_estimator = SemanticJacobianEstimator(hdv_system, max_tokens)
        self._text_koopman = TextKoopmanOperator(hdv_system, max_tokens)
        self._tone_extractor = ToneSignatureVector()

    def encode(self, text: str) -> dict:
        """
        Encode text as a semantic geometry hypothesis.

        Returns dict:
          hypothesis_only: True  (ALWAYS — must be asserted by caller)
          hdv:             np.ndarray [hdv_dim] — structural encoding of full text
          jacobian_salience: np.ndarray [n_tokens] — token salience scores
          koopman_K:       np.ndarray or None — K_text matrix (semantic dims only)
          eigenvalues:     np.ndarray — eigenvalues of K_text
          tone:            np.ndarray — navigation hint vector
          trust:           float — K_text trust score (0.0 if text too short)

        Side effects (if proposal_queue is set):
          - Puts "exploration" proposal with token salience
          - Puts "navigation" proposal if trust > τ_semantic

        NO gradient authority. NO EDMDKoopman.fit() calls.
        """
        # Full-text HDV encoding (structural, not fitted)
        hdv = self._hdv_system.structural_encode(text, "semantic").astype(float)

        # Token salience via knockout Jacobian
        salience = self._jacobian_estimator.compute(text)

        # Semantic Koopman (using _TextEDMD, NOT EDMDKoopman)
        text_edmd, semantic_dims, trust_text = self._text_koopman.fit(text)
        eigvals, eigvecs = text_edmd.eigendecomposition()

        # Tone: navigation hint from dominant eigenvectors
        tone = self._tone_extractor.compute(eigvecs, semantic_dims, r=4)

        # Place proposals in queue (if available)
        if self._proposal_queue is not None:
            # Always emit exploration proposal (salience-directed)
            self._proposal_queue.put({
                "type": "exploration",
                "salience": salience.tolist() if salience.size > 0 else [],
                "text_preview": text[:80],
            })
            # Emit navigation proposal only when trust is sufficient
            if trust_text > self._tau_semantic:
                self._proposal_queue.put({
                    "type": "navigation",
                    "tone": tone.tolist(),
                    "trust": float(trust_text),
                    "text_preview": text[:80],
                })

        return {
            "hypothesis_only": True,   # ALWAYS SET — must be asserted by caller
            "hdv": hdv,
            "jacobian_salience": salience,
            "koopman_K": text_edmd.K_matrix,
            "eigenvalues": eigvals,
            "tone": tone,
            "trust": float(trust_text),
            "semantic_dims": semantic_dims,
        }
