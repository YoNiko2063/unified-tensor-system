"""
KoopmanExperienceMemory — associative memory indexed by Koopman invariant descriptors.

Two-stage retrieval:
  Stage 1 (fast):   L2 on KoopmanInvariantDescriptor.to_retrieval_vector()
  Stage 2 (verify): L2 on sorted |eigenvalue| spectra, threshold 0.25

Merge policy: if a new experience matches an existing entry's signature, increment
n_observations and update best_params / runtime_improvement if improved.

No hard coupling to AST, semantic layer, or HDV growth machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from optimization.koopman_signature import (
    FullKoopmanSignature,
    KoopmanInvariantDescriptor,
)

# L2 distance on sorted |eigenvalue| spectra below which two signatures
# are considered the same dynamical regime.  Calibrated against the
# existing system's OED threshold (0.3) with a tighter margin.
EIGENVALUE_MATCH_THRESHOLD: float = 0.25


@dataclass
class OptimizationExperience:
    """
    What was learned from one optimization run.

    Fields
    ------
    bottleneck_operator   operator type that dominated the runtime trace
    replacement_applied   implementation substitution applied
    runtime_improvement   fractional speedup (0.38 = 38% faster / closer to target)
    n_observations        independent programs that confirmed this strategy
    hardware_target       "cpu" | "gpu" | "analog"
    best_params           domain-specific parameter dict at the optimum
                          e.g. {"R": 150.0, "L": 0.02, "C": 1.5e-6}
    """

    bottleneck_operator: str
    replacement_applied: str
    runtime_improvement: float
    n_observations: int
    hardware_target: str
    best_params: dict = field(default_factory=dict)


@dataclass
class _MemoryEntry:
    invariant: KoopmanInvariantDescriptor
    signature: FullKoopmanSignature
    experience: OptimizationExperience


class KoopmanExperienceMemory:
    """
    Associative memory: Koopman invariant → stored optimization experience.

    Usage
    -----
        mem = KoopmanExperienceMemory()

        # After an optimization run:
        mem.add(invariant, full_signature, experience)

        # Before the next run:
        candidates = mem.retrieve_candidates(new_invariant, top_n=5)
        for entry in candidates:
            if mem.confirm_match(new_full_sig, entry.signature):
                warm_start_params = entry.experience.best_params
                break
    """

    def __init__(self) -> None:
        self._entries: List[_MemoryEntry] = []

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(
        self,
        invariant: KoopmanInvariantDescriptor,
        signature: FullKoopmanSignature,
        experience: OptimizationExperience,
    ) -> None:
        """
        Store a new experience.

        Merge policy: if an existing entry confirms a signature match,
        increment n_observations and update best_params / runtime_improvement
        if the new run achieved a higher improvement.  Otherwise append fresh.
        """
        for entry in self._entries:
            if self.confirm_match(signature, entry.signature):
                entry.experience.n_observations += 1
                if experience.runtime_improvement > entry.experience.runtime_improvement:
                    entry.experience.runtime_improvement = experience.runtime_improvement
                    entry.experience.best_params = dict(experience.best_params)
                return

        self._entries.append(_MemoryEntry(invariant, signature, experience))

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def retrieve_candidates(
        self,
        invariant: KoopmanInvariantDescriptor,
        top_n: int = 5,
    ) -> List[_MemoryEntry]:
        """
        Stage-1 retrieval: rank entries by L2 distance on to_retrieval_vector().

        Returns up to top_n entries sorted by ascending distance.
        Callers should follow up with confirm_match() for final verification.
        """
        if not self._entries:
            return []

        query_vec = invariant.to_retrieval_vector()
        scored: List[Tuple[float, _MemoryEntry]] = []
        for entry in self._entries:
            d = float(np.linalg.norm(query_vec - entry.invariant.to_retrieval_vector()))
            scored.append((d, entry))

        scored.sort(key=lambda x: x[0])
        return [e for _, e in scored[:top_n]]

    def confirm_match(
        self,
        query_sig: FullKoopmanSignature,
        candidate_sig: FullKoopmanSignature,
        threshold: float = EIGENVALUE_MATCH_THRESHOLD,
    ) -> bool:
        """
        Stage-2 verification: L2 on sorted |eigenvalue| spectra.

        Eigenvalues are sorted by magnitude before comparison so that
        permutation differences from np.linalg.eig do not inflate the distance.
        Comparison uses the shorter spectrum's length to handle dimension mismatches.
        """
        q = np.sort(np.abs(query_sig.eigenvalues))[::-1]
        c = np.sort(np.abs(candidate_sig.eigenvalues))[::-1]
        n = min(len(q), len(c))
        if n == 0:
            return False
        return float(np.linalg.norm(q[:n] - c[:n])) < threshold

    # ── Introspection ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def summary(self) -> dict:
        return {
            "n_entries": len(self._entries),
            "total_observations": sum(
                e.experience.n_observations for e in self._entries
            ),
            "operators_seen": sorted(
                {e.experience.bottleneck_operator for e in self._entries}
            ),
        }
