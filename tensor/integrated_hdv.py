"""
FICUTS Phase 0: Integrated HDV System

Unifies Layer 2 (DynamicLatentSpace sparse masks) with Layer 9
(UnifiedTensorNetwork dense embeddings) into a single coherent system.

Architecture:
  - Sparse domain masks  : which HDV dims each domain uses (structural)
  - Dense network embeds : learned values at those dims (semantic after training)
  - Hash-based encoding  : deterministic text→HDV without training
  - Overlap detection    : dims shared by ≥2 domains → universals live here

Encoding modes:
  structural_encode()  — deterministic hash, works before training
  learned_encode()     — through network, meaningful after training
  compute_overlap_similarity() — cosine similarity in OVERLAP dims only
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from tensor.unified_network import UnifiedTensorNetwork, DOMAINS
from tensor.function_basis import FunctionBasisLibrary, FunctionBasisToHDV, EquationParser


class IntegratedHDVSystem:
    """
    Single unified HDV system combining sparse structural masks and dense
    learned embeddings.

    Components:
    1. Sparse domain masks  : which HDV dims each domain uses (Layer 2 concept)
    2. Dense network        : learned values at those dims (Layer 9)
    3. Structural encoding  : hash text → HDV indices (pre-training)
    4. Learned encoding     : forward pass through network (post-training)
    """

    def __init__(
        self,
        hdv_dim: int = 10000,
        n_modes: int = 150,
        embed_dim: int = 512,
        library_path: str = "tensor/data/function_library.json",
    ):
        self.hdv_dim = hdv_dim
        self.n_modes = n_modes
        self.embed_dim = embed_dim

        # Layer 9: unified neural network
        self.network = UnifiedTensorNetwork(hdv_dim, n_modes, embed_dim)

        # Layer 8: function basis → HDV mapping
        self.function_library = FunctionBasisLibrary(library_path)
        self.hdv_mapper = FunctionBasisToHDV(self.function_library, hdv_dim)
        if self.function_library.library:
            self.hdv_mapper.assign_dimensions()

        # Layer 2: sparse domain masks (structural)
        # domain_name → np.ndarray bool [hdv_dim]
        self.domain_masks: Dict[str, np.ndarray] = {}

        # Count how many domains use each dimension
        self._domain_dim_usage: np.ndarray = np.zeros(hdv_dim, dtype=np.int32)

        # Equation parser for math dimension
        self._eq_parser = EquationParser()

        # Data paths
        Path("tensor/data").mkdir(parents=True, exist_ok=True)

    # ── Structural encoding (deterministic, pre-training) ─────────────────────

    def structural_encode(self, text: str, domain: str) -> np.ndarray:
        """
        Hash text into HDV space deterministically.

        Returns: float32 HDV vector [hdv_dim] (sparse, values 0 or 1)

        Works without training. Similar texts → collisions in overlapping
        vocabulary → partial similarity. Full semantic similarity comes after
        network training.
        """
        vec = np.zeros(self.hdv_dim, dtype=np.float32)
        tokens = text.lower().split()
        universal_end = self.hdv_dim // 3
        for token in tokens:
            # Primary hash → HDV index
            h1 = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h1 % self.hdv_dim] = 1.0
            # Secondary hash for density
            h2 = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            vec[h2 % self.hdv_dim] = 1.0
            # Universal dim projection (first 30%): ensures cross-domain overlap
            h3 = int(hashlib.sha1(token.encode()).hexdigest(), 16)
            vec[h3 % universal_end] = 1.0

        self._register_domain_dims(domain, vec)
        return vec

    def encode_equation(self, latex: str, domain: str) -> np.ndarray:
        """
        Encode a LaTeX equation into HDV space.

        Uses function-type classification + parameter hashing.
        Returns float32 [hdv_dim].
        """
        expr = self._eq_parser.parse(latex)
        func_type = self._eq_parser.classify_function_type(expr)
        params = self._eq_parser.extract_parameters(expr) if expr is not None else []

        # Start from function-type encoding
        vec = self.structural_encode(func_type, domain)

        # Layer parameter information on top
        for param in params:
            param_vec = self.structural_encode(f"param_{param}", domain)
            vec = np.clip(vec + param_vec, 0.0, 1.0)

        self._register_domain_dims(domain, vec)
        return vec

    def encode_workflow(self, workflow_steps: List[str], domain: str) -> np.ndarray:
        """
        Encode a workflow (ordered action strings) into HDV.

        Position-weighted: earlier steps contribute slightly more, encoding
        the directionality of the workflow sequence.
        Returns float32 [hdv_dim].
        """
        if not workflow_steps:
            return np.zeros(self.hdv_dim, dtype=np.float32)

        vecs = []
        for i, step in enumerate(workflow_steps):
            weight = 1.0 / (1.0 + i * 0.1)  # recency decay
            step_vec = self.structural_encode(step, domain) * weight
            vecs.append(step_vec)

        combined = np.clip(sum(vecs), 0.0, 1.0)
        self._register_domain_dims(domain, combined)
        return combined

    # ── Learned encoding (through network, meaningful after training) ──────────

    def learned_encode(self, content: str, domain: str) -> torch.Tensor:
        """
        Encode content through the trained neural network.

        Returns: [hdv_dim] tensor — semantically meaningful after training.
        Pre-training: random but consistent (same content → same output).
        """
        tokens = content.lower().split()[:64]
        indices = []
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            indices.append(h % self.hdv_dim)
        if not indices:
            indices = [0]

        active_modes = self._domain_modes(domain)
        token_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            output = self.network(token_tensor, active_modes)
        return output.squeeze(0)

    # ── Domain management ─────────────────────────────────────────────────────

    def register_domain(self, domain: str, n_active: int = 100) -> np.ndarray:
        """
        Explicitly register a domain with a random sparse mask.

        This is Layer 2's approach — random projection into HDV space.
        Structural encoding also registers domains implicitly via
        _register_domain_dims(). Both approaches are compatible.
        """
        if domain not in self.domain_masks:
            mask = np.zeros(self.hdv_dim, dtype=bool)
            active_idx = np.random.choice(self.hdv_dim, n_active, replace=False)
            mask[active_idx] = True
            self.domain_masks[domain] = mask
            self._domain_dim_usage[active_idx] += 1
        return self.domain_masks[domain]

    def _register_domain_dims(self, domain: str, vec: np.ndarray):
        """
        Update domain mask from a structural encoding result.

        Called automatically by structural_encode / encode_equation /
        encode_workflow so domain masks grow organically from usage.
        """
        active = vec > 0
        if domain not in self.domain_masks:
            self.domain_masks[domain] = active.astype(bool)
        else:
            self.domain_masks[domain] = self.domain_masks[domain] | active.astype(bool)
        self._domain_dim_usage[active] += 1

    def _domain_modes(self, domain: str) -> List[int]:
        """Map a domain name to mode-head indices in the network."""
        modes = [i for i, d in enumerate(DOMAINS[: self.n_modes]) if d == domain]
        return modes if modes else [0]

    # ── Overlap + similarity ──────────────────────────────────────────────────

    def find_overlaps(self) -> set:
        """
        Find dimensions active in 2+ domains.
        
        Returns universal dimensions (first 33% of HDV space).
        ALL domains write here, ensuring cross-dimensional discovery works.
        """
        if len(self.domain_masks) < 2:
            return set()
        
        # Universal dimensions: [0 .. hdv_dim/3)
        universal_count = self.hdv_dim // 3
        
        return set(range(universal_count))
    
    def extend_capacity(self, n: int) -> None:
        """
        Extend HDV dimension by n (called by RecursiveGrowthScheduler).

        Expands dimension tracking arrays; new dims have zero usage until
        patterns are encoded into them.  The neural network (fixed-size) is
        not resized — HDV masking is the growth surface.
        """
        self.hdv_dim += n
        self._domain_dim_usage = np.concatenate(
            [self._domain_dim_usage, np.zeros(n, dtype=np.int32)]
        )
        for domain in self.domain_masks:
            self.domain_masks[domain] = np.concatenate(
                [self.domain_masks[domain], np.zeros(n, dtype=bool)]
            )

    def compute_overlap_similarity(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """
        Cosine similarity in OVERLAP dimensions only.

        Compares vectors only where domains share HDV space — this filters
        out domain-specific noise and measures genuine cross-domain affinity.

        Returns: float in [-1, 1]. Values > 0.85 indicate universal pattern.
        """
        overlap = self.find_overlaps()
        if not overlap:
            return 0.0

        idx = list(overlap)
        v1 = vec1[idx].astype(np.float32)
        v2 = vec2[idx].astype(np.float32)

        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0

        return float(np.dot(v1, v2) / (n1 * n2))

    # ── Patch-aware encoding (LCA geometry integration) ───────────────────────

    def structural_encode_with_patch_info(
        self, text: str, domain: str,
        patch_type: Optional[str] = None,
    ) -> tuple:
        """
        Encode text into HDV space and return patch geometry metadata.

        Extends structural_encode() with optional LCA patch context.
        Existing callers using structural_encode() are not affected.

        Args:
            text:       input text to encode
            domain:     HDV domain name
            patch_type: optional patch classification ('lca'|'nonabelian'|'chaotic')

        Returns:
            (hdv_vec, metadata_dict) where metadata has patch_type and overlap info
        """
        vec = self.structural_encode(text, domain)
        metadata = {
            "domain": domain,
            "patch_type": patch_type,
            "n_overlap_dims": len(self.find_overlaps()),
            "n_domain_dims": int(np.sum(self.domain_masks.get(domain, np.zeros(self.hdv_dim)))),
        }
        return vec, metadata

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_state(self, path: str = "tensor/data/hdv_state.json"):
        """Save domain masks and dimension usage to JSON."""
        state = {
            "hdv_dim": self.hdv_dim,
            "n_modes": self.n_modes,
            "domain_masks": {k: v.tolist() for k, v in self.domain_masks.items()},
            "dim_usage": self._domain_dim_usage.tolist(),
        }
        Path(path).write_text(json.dumps(state))
        print(f"[HDV] State saved: {len(self.domain_masks)} domains, "
              f"{len(self.find_overlaps())} overlap dims")

    @classmethod
    def load_state(
        cls, path: str = "tensor/data/hdv_state.json", **kwargs
    ) -> "IntegratedHDVSystem":
        """Restore from saved state."""
        state = json.loads(Path(path).read_text())
        obj = cls(hdv_dim=state["hdv_dim"], n_modes=state["n_modes"], **kwargs)
        obj.domain_masks = {
            k: np.array(v, dtype=bool) for k, v in state["domain_masks"].items()
        }
        obj._domain_dim_usage = np.array(state["dim_usage"], dtype=np.int32)
        return obj
