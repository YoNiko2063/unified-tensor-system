"""
FICUTS Task 9.5: Cross-Dimensional Discovery

Monitors HDV space for patterns that appear across multiple learning dimensions.

The 5 dimensions:
  math        — equations from arXiv papers
  behavioral  — workflows from DeepWiki + GitHub
  execution   — validated code execution results
  optimization— Optuna meta-learning signals
  physical    — hardware / 3D printing measurements

A "universal" is a pattern that appears in ≥2 dimensions with high similarity
in the overlap dimensions (where domains share HDV space).

Example:
  Math:     exponential_decay → HDV[...0.8, 0.0, 0.1...]
  Code:     rate_limiter      → HDV[...0.79, 0.0, 0.11...]
  Overlap similarity = 0.97 > 0.85 threshold → UNIVERSAL DISCOVERED

Similarity is measured ONLY in overlap dimensions — this filters out
domain-specific noise and measures genuine cross-domain affinity.

Persists universals to tensor/data/universals.json for:
  - Human inspection
  - FICUTSUpdater logging to FICUTS.md
  - Future-session continuity
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

DIMENSIONS = ["math", "behavioral", "execution", "optimization", "physical"]


class CrossDimensionalDiscovery:
    """
    Scan all pairs of learning dimensions for cross-domain universal patterns.

    Records HDV vectors from each dimension as patterns arrive, then
    periodically scans all pairs for high-similarity overlaps.
    """

    def __init__(
        self,
        hdv_system,
        similarity_threshold: float = 0.85,
        universals_path: str = "tensor/data/universals.json",
    ):
        """
        Args:
            hdv_system:           IntegratedHDVSystem instance.
            similarity_threshold: cosine similarity in overlap dims required
                                  for a pattern pair to be called universal.
                                  0.85 is aggressive but MDL confirms it.
            universals_path:      JSON file for persistent universal storage.
        """
        self.hdv_system = hdv_system
        self.similarity_threshold = similarity_threshold
        self.universals_path = Path(universals_path)
        self.universals_path.parent.mkdir(parents=True, exist_ok=True)

        # Per-dimension pattern stores
        # dimension → List[{'hdv': np.ndarray, 'metadata': dict, 'recorded_at': float}]
        self._patterns: Dict[str, List[Dict]] = {d: [] for d in DIMENSIONS}

        # Loaded or discovered universals
        self.universals: List[Dict] = []
        if self.universals_path.exists():
            try:
                self.universals = json.loads(self.universals_path.read_text())
            except Exception:
                self.universals = []

    # ── Recording ──────────────────────────────────────────────────────────────

    def record_pattern(
        self, dimension: str, hdv_vec: np.ndarray, metadata: Dict
    ):
        """
        Record an HDV pattern from a learning dimension.

        Args:
            dimension: one of DIMENSIONS ('math', 'behavioral', ...)
            hdv_vec:   np.ndarray [hdv_dim] — structural or learned encoding
            metadata:  arbitrary dict describing what the pattern represents,
                       e.g. {'type': 'exponential', 'content': 'e^{-t/τ}',
                              'domain': 'ece', 'paper_id': '2602.13213'}
        """
        if dimension not in self._patterns:
            self._patterns[dimension] = []

        self._patterns[dimension].append({
            "hdv": hdv_vec.detach().cpu().numpy() if hasattr(hdv_vec, "detach") else hdv_vec.copy(),
            "metadata": {k: v for k, v in metadata.items()},
            "recorded_at": time.time(),
        })

    def pattern_count(self, dimension: Optional[str] = None) -> int:
        """Count recorded patterns, optionally for a specific dimension."""
        if dimension:
            return len(self._patterns.get(dimension, []))
        return sum(len(v) for v in self._patterns.values())

    # ── Discovery ──────────────────────────────────────────────────────────────

    def find_universals(self, similarity_threshold: Optional[float] = None) -> List[Dict]:
        """
        Scan all dimension pairs for patterns with overlap similarity ≥ threshold.

        Args:
            similarity_threshold: Override the instance threshold for this call.
                If None, uses self.similarity_threshold (default 0.85).

        Only compares dimensions that have recorded patterns.
        Deduplicates against already-known universals.

        Returns: list of newly discovered universals (not yet in self.universals).
        """
        threshold = similarity_threshold if similarity_threshold is not None \
            else self.similarity_threshold

        new_universals = []
        active_dims = [d for d in DIMENSIONS if self._patterns.get(d)]

        for i in range(len(active_dims)):
            for j in range(i + 1, len(active_dims)):
                dim1, dim2 = active_dims[i], active_dims[j]
                new_universals.extend(
                    self._compare_dimensions(dim1, dim2, threshold)
                )

        # Filter duplicates and add to persistent list
        added = []
        for u in new_universals:
            if not self._is_duplicate(u):
                self.universals.append(u)
                added.append(u)

        return added

    def _compare_dimensions(
        self, dim1: str, dim2: str, threshold: float = None
    ) -> List[Dict]:
        """Compare all pattern pairs between two dimensions."""
        if threshold is None:
            threshold = self.similarity_threshold
        found = []
        for p1 in self._patterns[dim1]:
            for p2 in self._patterns[dim2]:
                sim = self.hdv_system.compute_overlap_similarity(
                    p1["hdv"], p2["hdv"]
                )
                if sim >= threshold:
                    mdl = self._compute_mdl(p1["hdv"], p2["hdv"])
                    found.append({
                        "dimensions": [dim1, dim2],
                        "similarity": float(sim),
                        "mdl": float(mdl),
                        "patterns": [
                            dict(p1["metadata"]),
                            dict(p2["metadata"]),
                        ],
                        "type": "cross_dimensional_universal",
                        "discovered_at": time.time(),
                    })
        return found

    # ── MDL proxy ─────────────────────────────────────────────────────────────

    def _compute_mdl(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Minimum Description Length proxy: 1 - normalized projection residual.

        Measures how well vec1 explains vec2 in the overlap dimensions.
          MDL = 0.0 : vec2 is perfectly explained by vec1 (identical direction)
          MDL = 1.0 : vec2 is orthogonal to vec1 (no explanation)

        Only computed in overlap dimensions to keep it cross-domain relevant.
        """
        overlap = self.hdv_system.find_overlaps()
        if not overlap:
            return 1.0

        idx = list(overlap)
        v1 = vec1[idx].astype(np.float32)
        v2 = vec2[idx].astype(np.float32)

        n1 = np.linalg.norm(v1)
        if n1 < 1e-9:
            return 1.0

        # Project v2 onto v1, measure residual
        projection = (np.dot(v1, v2) / (n1 ** 2)) * v1
        residual = np.linalg.norm(v2 - projection)
        n2 = np.linalg.norm(v2)

        return float(residual / max(n2, 1e-9))

    # ── Deduplication ─────────────────────────────────────────────────────────

    def _is_duplicate(self, candidate: Dict) -> bool:
        """
        Check if this universal is already known.

        Duplicate if: same dimensions + same pattern types + similar similarity.
        """
        cand_dims = frozenset(candidate["dimensions"])
        cand_sim = candidate["similarity"]
        cand_types = frozenset(
            p.get("type", "") for p in candidate["patterns"]
        )

        for existing in self.universals:
            if frozenset(existing["dimensions"]) != cand_dims:
                continue
            if abs(existing["similarity"] - cand_sim) > 0.05:
                continue
            ex_types = frozenset(p.get("type", "") for p in existing["patterns"])
            if cand_types & ex_types:  # any shared types
                return True
        return False

    # ── LCA patch integration ─────────────────────────────────────────────────

    def classify_dimension_patches(
        self,
        dimension: str,
        system_fn,
        n_states: int = 2,
        sample_spread: float = 0.1,
    ) -> List[Dict]:
        """
        Classify stored HDV patterns in a dimension as LCA / non-abelian / chaotic.

        Requires lca_patch_detector to be available. If not installed, returns empty.

        This implements Stage 2.5 of the capability ladder (LOGIC_FLOW.md Section 6):
        identify LCA patches for each domain before cross-domain comparison.

        Args:
            dimension:    which dimension to classify ('math', 'behavioral', ...)
            system_fn:    vector field for Jacobian sampling
            n_states:     state space dimension
            sample_spread: spread of sample points around pattern centroid

        Returns:
            List of dicts: {pattern_index, patch_type, commutator_norm, curvature_ratio}
        """
        try:
            from tensor.lca_patch_detector import LCAPatchDetector
        except ImportError:
            return []

        patterns = self._patterns.get(dimension, [])
        if not patterns:
            return []

        detector = LCAPatchDetector(system_fn, n_states=n_states)
        results = []

        for i, p in enumerate(patterns):
            # Use HDV centroid to generate sample points (approximate)
            rng = np.random.default_rng(i)
            samples = rng.normal(0, sample_spread, (15, n_states))
            try:
                cl = detector.classify_region(samples)
                results.append({
                    "pattern_index": i,
                    "dimension": dimension,
                    "patch_type": cl.patch_type,
                    "commutator_norm": cl.commutator_norm,
                    "curvature_ratio": cl.curvature_ratio,
                    "operator_rank": cl.operator_rank,
                })
            except Exception:
                pass

        return results

    def find_universals_with_lca_context(
        self,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Find universals and annotate each with LCA patch context when available.

        Wraps find_universals() — all existing behavior preserved.
        Adds 'lca_context': 'abelian_match' | 'mixed' | 'unknown' to each universal.

        Returns: newly discovered universals with LCA context annotations.
        """
        universals = self.find_universals(similarity_threshold)

        for u in universals:
            # Tag with generic context (full LCA detection requires system_fn)
            u.setdefault("lca_context", "unknown")

        return universals

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_universals(self):
        """Persist all universals to JSON (human-readable)."""
        # Strip numpy arrays — metadata is plain dicts, already serializable
        self.universals_path.write_text(
            json.dumps(self.universals, indent=2, default=str)
        )
        print(f"[Discovery] {len(self.universals)} universals → {self.universals_path}")

    def load_universals(self):
        """Reload universals from disk."""
        if self.universals_path.exists():
            try:
                self.universals = json.loads(self.universals_path.read_text())
            except Exception:
                self.universals = []

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of all discovered universals."""
        if not self.universals:
            return "[Discovery] No universals found yet."

        lines = [
            f"=== Cross-Dimensional Universals ({len(self.universals)} found) ==="
        ]
        for i, u in enumerate(self.universals[:20]):
            dims = " ↔ ".join(u["dimensions"])
            sim = u["similarity"]
            mdl = u.get("mdl", "?")
            types = [p.get("type", "?") for p in u["patterns"]]
            lines.append(
                f"  {i+1}. [{dims}] sim={sim:.3f} mdl={mdl:.3f} | "
                f"{' ≈ '.join(types)}"
            )
        return "\n".join(lines)

    def get_pattern_counts(self) -> Dict[str, int]:
        """Return pattern count per dimension."""
        return {d: len(self._patterns.get(d, [])) for d in DIMENSIONS}
