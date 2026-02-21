"""
ValidationBridge — pure enforcement gate for cross-domain and symbolic proposals.

Mathematical basis (Plan §II):
  Gates proposals from Layers 3–4 before they reach Layers 1, 2, 7.
  Pure function: no side effects, no EDMD fits, no operator updates.

  "equivalence" proposals must pass BOTH:
    (a) spectral_preservation: spectrum_distance(merged, src) < ε for each source
    (b) predictive_compression: trust(merged) > max(trust_a, trust_b) - δ

  "navigation" and "exploration" proposals always pass (zero gradient authority).

  ProposalQueue: queue.Queue(maxsize=0) — unbounded, put() never blocks.
  process() called exclusively from dedicated _validation_thread; no producer
  thread ever calls it.

CRITICAL invariants enforced here:
  INV-2 (Spectral Authority): symbolic proposals cannot bypass this gate.
  CRITICAL-2: merged_patch must be constructed via make_merged_patch() — algebraic
              only, no EDMDKoopman.fit() permitted.
"""

from __future__ import annotations

import logging
import queue
from typing import List, Optional

import numpy as np

from tensor.patch_graph import Patch
from tensor.operator_equivalence import OperatorEquivalenceDetector

logger = logging.getLogger(__name__)

# ── Algebraic merged-patch construction ───────────────────────────────────────


def _spectral_gap_from_eigvals(eigvals: np.ndarray) -> float:
    """Spectral gap = |Re(λ₁)| - |Re(λ₂)| of dominant eigenvalues. Pure matrix-level."""
    r = np.sort(np.abs(np.real(eigvals)))[::-1]
    if len(r) == 0:
        return 0.0
    if len(r) == 1:
        return float(r[0])
    return float(r[0] - r[1])


def make_merged_patch(
    patch_a: Patch,
    patch_b: Patch,
    trust_a: float,
    trust_b: float,
    K_a: np.ndarray,
    K_b: np.ndarray,
    merged_id: int = -1,
) -> Patch:
    """
    Construct a merged Patch algebraically from two Koopman matrices.

    CRITICAL-2 enforcement: NO EDMDKoopman.fit() call. NO data-level operations.
    Pure matrix algebra only. Callers must hold K matrices from prior EDMD results
    (already in memory) — they must not refit to obtain them.

    Construction:
      K_merged = trust-weighted average of K_a and K_b
      spectrum  = np.linalg.eigvals(K_merged)
      trust     = spectral_gap_score of K_merged (gap / γ_min, capped at 1.0)
                  — matrix-level only, no reconstruction error or gram conditioning
                    (those require stored ψ matrices not available at merge time)
      centroid  = arithmetic mean of patch_a.centroid and patch_b.centroid

    The returned Patch stores trust and K_merged in metadata for use by
    ValidationBridge.validate_equivalence(). The id field is set to merged_id
    (caller assigns the canonical ID before inserting into HarmonicAtlas).

    Args:
        patch_a, patch_b:  source Patches
        trust_a, trust_b:  Koopman trust scores of each source (from prior EDMD result)
        K_a, K_b:          Koopman operator matrices (np.ndarray, same shape)
        merged_id:         ID for the returned Patch (-1 = placeholder)

    Returns:
        Patch with metadata['trust'], metadata['K_matrix'], metadata['merged']=True
    """
    w_total = trust_a + trust_b
    if w_total < 1e-12:
        w_a, w_b = 0.5, 0.5
    else:
        w_a = trust_a / w_total
        w_b = trust_b / w_total

    K_merged = w_a * K_a + w_b * K_b
    eigvals = np.linalg.eigvals(K_merged)
    gap = _spectral_gap_from_eigvals(eigvals)
    gamma_min = 0.1  # matches EDMDKoopman default gamma_min
    trust_merged = float(min(1.0, gap / max(gamma_min, 1e-12)))

    centroid = (patch_a.centroid + patch_b.centroid) / 2.0
    curv = (patch_a.curvature_ratio + patch_b.curvature_ratio) / 2.0
    comm = (patch_a.commutator_norm + patch_b.commutator_norm) / 2.0
    rank = max(patch_a.operator_rank, patch_b.operator_rank)

    return Patch(
        id=merged_id,
        patch_type=patch_a.patch_type,
        operator_basis=patch_a.operator_basis,
        spectrum=eigvals,
        centroid=centroid,
        operator_rank=rank,
        commutator_norm=comm,
        curvature_ratio=curv,
        spectral_gap=gap,
        metadata={
            "trust": trust_merged,
            "K_matrix": K_merged,
            "merged": True,
            "source_ids": (patch_a.id, patch_b.id),
        },
    )


# ── ProposalQueue ──────────────────────────────────────────────────────────────


class ProposalQueue:
    """
    Unbounded thread-safe FIFO from symbolic/semantic layers to operator layers.

    Backed by queue.Queue(maxsize=0) — put() NEVER blocks regardless of consumer
    speed or proposal volume. This prevents any producer thread from stalling while
    holding a lock that a consumer depends on (deadlock prevention — MOD-2).

    Contract:
      put()     — callable by Layers 3, 4 (semantic, discovery threads); never blocks
      process() — callable by ValidationBridge only, from dedicated _validation_thread
                  No producer thread ever calls process().
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=0)

    def put(self, proposal: dict) -> None:
        """Place a proposal in the queue. Never blocks."""
        self._q.put_nowait(proposal)

    def process(self) -> List[dict]:
        """
        Drain all pending proposals and return them as a list.
        Must only be called from the dedicated _validation_thread.
        """
        items: List[dict] = []
        while True:
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items

    def qsize(self) -> int:
        """Approximate queue depth (for monitoring only)."""
        return self._q.qsize()


# ── ValidationBridge ───────────────────────────────────────────────────────────


class ValidationBridge:
    """
    Single enforcement checkpoint for cross-domain and symbolic proposals.

    Pure function: no side effects. No EDMD fits. No operator updates.

    All proposals from Layers 3–4 must pass through process_queue() before
    reaching any Layer 1, 2, or 7 update. There is no fallback path.

    Usage (from _validation_thread only):
        bridge = ValidationBridge()
        accepted = bridge.process_queue(proposal_queue)
        for prop in accepted:
            if prop["type"] == "equivalence":
                atlas.register_validated_equivalence(prop)
            elif prop["type"] in ("navigation", "exploration"):
                patch_graph.update_routing_hint(prop)
    """

    def __init__(
        self,
        spectral_preservation_eps: float = 0.2,
        compression_delta: float = 0.05,
    ) -> None:
        """
        Args:
            spectral_preservation_eps: max allowed spectrum_distance(merged, source)
            compression_delta:         max allowed trust drop: trust_merged > max_trust - delta
        """
        self._eps = spectral_preservation_eps
        self._delta = compression_delta
        # OperatorEquivalenceDetector.spectrum_distance() is the canonical spectral metric.
        # threshold=1.0 so are_equivalent() is never called; we use spectrum_distance() only.
        self._detector = OperatorEquivalenceDetector(threshold=1.0)
        self._n_validated = 0
        self._n_rejected = 0

    # ── Core validation ────────────────────────────────────────────────────────

    def validate_equivalence(
        self,
        patch_a: Patch,
        patch_b: Patch,
        merged_patch: Patch,
        trust_a: float,
        trust_b: float,
    ) -> bool:
        """
        Gate an equivalence proposal. Returns True only if BOTH conditions hold:

          (a) spectral_preservation:
                spectrum_distance(merged, patch_a) < eps_preservation
                AND spectrum_distance(merged, patch_b) < eps_preservation

          (b) predictive_compression:
                trust(merged) > max(trust_a, trust_b) - compression_delta

        trust(merged) is read from merged_patch.metadata['trust'], which must be
        set by make_merged_patch() using matrix-level computation only.

        CRITICAL-2: This method never calls EDMDKoopman.fit() or any method that
        internally triggers a fit. Trust and spectrum come from the pre-constructed
        merged_patch only.
        """
        dist_a = self._detector.spectrum_distance(merged_patch, patch_a)
        dist_b = self._detector.spectrum_distance(merged_patch, patch_b)
        trust_merged = float(merged_patch.metadata.get("trust", 0.0))
        trust_threshold = max(trust_a, trust_b) - self._delta

        spectral_ok = (dist_a < self._eps) and (dist_b < self._eps)
        compression_ok = trust_merged > trust_threshold

        if spectral_ok and compression_ok:
            self._n_validated += 1
            return True

        reasons: List[str] = []
        if not spectral_ok:
            reasons.append(
                f"spectral_preservation failed: dist_a={dist_a:.4f} dist_b={dist_b:.4f} "
                f"eps={self._eps}"
            )
        if not compression_ok:
            reasons.append(
                f"predictive_compression failed: trust_merged={trust_merged:.4f} "
                f"threshold={trust_threshold:.4f}"
            )
        logger.debug("ValidationBridge rejected equivalence proposal: %s", "; ".join(reasons))
        self._n_rejected += 1
        return False

    def validate_navigation(self, proposal: dict) -> bool:  # noqa: ARG002
        """
        Navigation and exploration proposals always pass.

        They carry zero gradient authority — they only influence routing start
        points for PatchGraph.shortest_path(), never operator basis A_k.
        """
        return True

    # ── Queue processing ───────────────────────────────────────────────────────

    def process_queue(self, proposal_queue: ProposalQueue) -> List[dict]:
        """
        Drain ProposalQueue and return all validated proposals.

        Must be called from _validation_thread only (never from producer threads).

        Proposal types:
          "navigation"  → accepted unconditionally
          "exploration" → accepted unconditionally
          "equivalence" → accepted only if validate_equivalence() passes;
                          requires proposal keys: patch_a, patch_b, merged_patch,
                          trust_a, trust_b (all pre-computed by caller)
          other         → logged and discarded

        Returns list of accepted proposals (may be empty).
        """
        raw = proposal_queue.process()
        accepted: List[dict] = []

        for proposal in raw:
            ptype = proposal.get("type", "")

            if ptype in ("navigation", "exploration"):
                accepted.append(proposal)
                continue

            if ptype == "equivalence":
                pa = proposal.get("patch_a")
                pb = proposal.get("patch_b")
                pm = proposal.get("merged_patch")
                trust_a = float(proposal.get("trust_a", 0.0))
                trust_b = float(proposal.get("trust_b", 0.0))

                if pa is None or pb is None or pm is None:
                    logger.debug(
                        "ValidationBridge: equivalence proposal missing patch field(s); "
                        "discarding (patch_a=%s patch_b=%s merged_patch=%s)",
                        pa is not None,
                        pb is not None,
                        pm is not None,
                    )
                    self._n_rejected += 1
                    continue

                if self.validate_equivalence(pa, pb, pm, trust_a, trust_b):
                    accepted.append(proposal)
                # else: already logged and counted inside validate_equivalence
                continue

            logger.debug(
                "ValidationBridge: unknown proposal type '%s'; discarding", ptype
            )
            self._n_rejected += 1

        return accepted

    # ── Monitoring ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return cumulative validation statistics."""
        return {
            "n_validated": self._n_validated,
            "n_rejected": self._n_rejected,
        }
