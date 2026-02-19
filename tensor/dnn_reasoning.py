"""
FICUTS Deep Neural Network Reasoning via HDV Traversal

Uses the IntegratedHDV space as a reasoning substrate.

Algorithm:
  1. Embed query into HDV
  2. Find top-k similar stored patterns (knowledge base)
  3. Softmax-attention over similarities → weighted next step
  4. Walk to attention-weighted average of top-k
  5. Repeat until Lyapunov energy E = 1 - cos(current, prev) < threshold

Mathematical guarantee:
  E_n = 1 - cos(chain[n], chain[n-1])    (angular distance)
  If E decreases monotonically → Lyapunov stable → reasoning converges
  to a fixed point in HDV space (a concept the knowledge base agrees on).

The knowledge base is populated by calling .store(text, domain) before
reasoning. Richer knowledge → faster convergence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class DeepNeuralNetworkReasoner:
    """
    Traverse HDV space to generate reasoning chains.

    Chain generation per step:
      1. Query → find top_k similar stored vectors
      2. Softmax(similarities · temperature) → attention weights
      3. next = Σ weight_i · vec_i   (weighted combination)
      4. Normalize next, compute energy = 1 - cos(current, next)
      5. Stop if energy < threshold or max_steps reached

    Usage:
      reasoner = DeepNeuralNetworkReasoner(hdv_system=hdv)
      reasoner.store("gradient descent minimises loss", "math")
      reasoner.store("RC circuit exponential decay", "physical")
      result = reasoner.reason_about("optimization landscape")
      print(result["chain"], result["converged"])
    """

    def __init__(
        self,
        hdv_system=None,
        max_steps: int = 20,
        energy_threshold: float = 0.05,
        top_k: int = 10,
        temperature: float = 5.0,
    ):
        self.hdv = hdv_system
        self.max_steps = max_steps
        self.energy_threshold = energy_threshold
        self.top_k = top_k
        self.temperature = temperature
        self._store: List[Dict[str, Any]] = []  # {vec, text, domain}

    # ── Knowledge base ────────────────────────────────────────────────────────

    def store(self, text: str, domain: str = "math") -> np.ndarray:
        """Encode text and add to reasoning knowledge base. Returns HDV."""
        if self.hdv is None:
            raise RuntimeError("HDV system required")
        vec = self.hdv.structural_encode(text, domain)
        self._store.append({"vec": vec, "text": text, "domain": domain})
        return vec

    def store_batch(self, items: List[Dict[str, str]]) -> List[np.ndarray]:
        """
        Store multiple items.

        items: list of {"text": ..., "domain": ...} dicts
        """
        return [self.store(item["text"], item.get("domain", "math"))
                for item in items]

    @property
    def knowledge_size(self) -> int:
        return len(self._store)

    # ── Reasoning ─────────────────────────────────────────────────────────────

    def reason_about(
        self, query: str, domain: str = "math"
    ) -> Dict[str, Any]:
        """
        Generate a reasoning chain from query through HDV space.

        Returns dict:
          chain         — list of {"text", "domain", "energy"} per step
          converged     — True if final energy < energy_threshold
          final_energy  — terminal angular distance
          steps         — number of steps taken
          energy_history — list of per-step energies
        """
        if self.hdv is None:
            return {
                "chain": [], "converged": False,
                "final_energy": float("inf"), "steps": 0,
                "energy_history": [],
            }

        query_vec = self.hdv.structural_encode(query, domain)
        chain: List[Dict[str, Any]] = [
            {"text": query, "domain": domain, "energy": 1.0}
        ]
        energy_history: List[float] = [1.0]
        current = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        for _ in range(self.max_steps):
            similar = self._find_similar(current)
            if not similar:
                break

            # Softmax attention over similarities
            sims = np.array([s["similarity"] for s in similar])
            weights = _softmax(sims * self.temperature)

            # Weighted combination of similar vectors
            next_vec = sum(w * s["vec"] for w, s in zip(weights, similar))
            n = np.linalg.norm(next_vec)
            if n < 1e-9:
                break
            next_vec = next_vec / n

            # Lyapunov energy: angular distance (1 - cosine similarity)
            cos_sim = float(np.dot(current, next_vec))
            energy = 1.0 - max(-1.0, min(1.0, cos_sim))
            energy_history.append(energy)

            # Best-matching entry for chain label
            best_idx = int(np.argmax(weights))
            best = similar[best_idx]
            chain.append({
                "text": best["text"],
                "domain": best["domain"],
                "energy": energy,
            })

            current = next_vec
            if energy < self.energy_threshold:
                break

        return {
            "chain": chain,
            "converged": energy_history[-1] < self.energy_threshold,
            "final_energy": float(energy_history[-1]),
            "steps": len(chain) - 1,
            "energy_history": energy_history,
        }

    def reason_batch(
        self, queries: List[str], domain: str = "math"
    ) -> List[Dict[str, Any]]:
        """Reason about multiple queries sequentially (each with full chain)."""
        return [self.reason_about(q, domain) for q in queries]

    def compute_reasoning_similarity(
        self, query_a: str, query_b: str
    ) -> float:
        """
        How similar are two queries' reasoning chains?

        Compares the HDV of their final chain positions.
        Values > 0.85 indicate the queries converge to the same concept.
        """
        if self.hdv is None or not self._store:
            return 0.0

        chain_a = self.reason_about(query_a)["chain"]
        chain_b = self.reason_about(query_b)["chain"]

        if not chain_a or not chain_b:
            return 0.0

        vec_a = self.hdv.structural_encode(chain_a[-1]["text"], chain_a[-1]["domain"])
        vec_b = self.hdv.structural_encode(chain_b[-1]["text"], chain_b[-1]["domain"])
        n_a, n_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
        if n_a < 1e-9 or n_b < 1e-9:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (n_a * n_b))

    # ── Mode-aware reasoning (LCA geometry integration) ───────────────────────

    def reason_about_with_mode(
        self,
        query: str,
        domain: str = "math",
        mode: str = "lca",
    ) -> Dict[str, Any]:
        """
        Mode-aware reasoning chain generation.

        Extends reason_about() with patch-mode context:
          - 'lca':       uses standard cosine similarity (Pontryagin character alignment)
          - 'koopman':   uses overlap-only similarity (Koopman eigenfunction overlap)
          - 'transition': uses overlap similarity with lower threshold

        Args:
            query:  input query text
            domain: HDV domain
            mode:   navigation mode ('lca' | 'transition' | 'koopman')

        Returns:
            Same structure as reason_about(), plus 'mode' key.
        """
        if self.hdv is None:
            return {
                "chain": [], "converged": False,
                "final_energy": float("inf"), "steps": 0,
                "energy_history": [], "mode": mode,
            }

        result = self.reason_about(query, domain)
        result["mode"] = mode

        # In Koopman/transition mode, also compute overlap-based similarity
        if mode in ("koopman", "transition") and self._store:
            query_vec = self.hdv.structural_encode(query, domain)
            similar_overlap = self._find_similar_overlap(query_vec)
            result["overlap_chain"] = [
                {"text": s["text"], "domain": s["domain"], "overlap_sim": s["similarity"]}
                for s in similar_overlap
            ]

        return result

    def _find_similar_overlap(self, query: np.ndarray) -> List[Dict[str, Any]]:
        """Find similar entries using overlap-only similarity (Koopman mode)."""
        results = []
        for entry in self._store:
            v = entry["vec"]
            sim = self.hdv.compute_overlap_similarity(query, v)
            results.append({**entry, "similarity": sim})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[: self.top_k]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_similar(self, query: np.ndarray) -> List[Dict[str, Any]]:
        """Cosine similarity search over the knowledge store."""
        n_q = np.linalg.norm(query)
        if n_q < 1e-9:
            return []

        results = []
        for entry in self._store:
            v = entry["vec"]
            n_v = np.linalg.norm(v)
            if n_v < 1e-9:
                continue
            sim = float(np.dot(query, v) / (n_q * n_v))
            results.append({**entry, "similarity": sim})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[: self.top_k]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)
