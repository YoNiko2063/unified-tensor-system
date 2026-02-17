"""DomainFibers: ECE, finance, biology, hardware as fiber bundle subspaces.

Each domain is a projection of the same underlying dynamics onto a different
physical substrate. The base space contains universal mathematical patterns;
each fiber is a domain-specific projection. Cross-domain resonance near the
golden angle means the system found a genuine universal principle.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PHI = 1.6180339887
GOLDEN_ANGLE_COS = 1.0 / PHI  # 0.618

DOMAINS = {
    "ece":      {"vars": ["voltage", "current", "impedance", "phase"],
                 "level": "L2",
                 "model": "deepseek-coder:6.7b"},
    "finance":  {"vars": ["price", "volume", "volatility", "correlation"],
                 "level": "L0",
                 "model": "qwen3:8b"},
    "biology":  {"vars": ["potential", "conductance", "frequency", "coupling"],
                 "level": "L1",
                 "model": "qwen3:8b"},
    "hardware": {"vars": ["latency", "bandwidth", "thermal", "power"],
                 "level": "L3",
                 "model": "qwen2.5:1.5b"},
}


@dataclass
class DomainFiber:
    domain: str
    subspace_basis: np.ndarray  # from assign_subspace()

    def pattern_signature(self) -> np.ndarray:
        """Eigenvalue ratios of this fiber's subspace — the mathematical fingerprint."""
        if self.subspace_basis.size == 0:
            return np.array([1.0])
        ata = self.subspace_basis.T @ self.subspace_basis
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(ata)))[::-1]
        if len(eigvals) < 2 or eigvals[0] < 1e-30:
            return eigvals if len(eigvals) > 0 else np.array([1.0])
        return eigvals / eigvals[0]


class FiberBundle:
    def __init__(self):
        self.fibers: Dict[str, DomainFiber] = {}

    def add_fiber(self, domain: str, basis: np.ndarray):
        self.fibers[domain] = DomainFiber(domain=domain, subspace_basis=basis)

    def cross_domain_resonance(self, d1: str, d2: str) -> float:
        """Cosine similarity of pattern signatures. Near GOLDEN_ANGLE_COS = universal."""
        f1 = self.fibers.get(d1)
        f2 = self.fibers.get(d2)
        if f1 is None or f2 is None:
            return 0.0
        s1 = f1.pattern_signature()
        s2 = f2.pattern_signature()
        n = min(len(s1), len(s2))
        if n == 0:
            return 0.0
        a, b = s1[:n], s2[:n]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-30 or nb < 1e-30:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def universal_patterns(self) -> List[np.ndarray]:
        """Directions in base space shared across all fibers.

        Returns bases that appear (high cosine similarity) in every fiber.
        These are the deepest mathematical invariants.
        """
        if len(self.fibers) < 2:
            return []
        fiber_list = list(self.fibers.values())
        ref = fiber_list[0].subspace_basis
        if ref.size == 0:
            return []
        # For each column in the reference basis, check if a similar
        # direction exists in all other fibers
        universals = []
        for col_idx in range(ref.shape[1]):
            direction = ref[:, col_idx]
            dn = np.linalg.norm(direction)
            if dn < 1e-30:
                continue
            direction = direction / dn
            is_universal = True
            for other in fiber_list[1:]:
                if other.subspace_basis.size == 0:
                    is_universal = False
                    break
                # Project direction onto other's subspace
                n_common = min(len(direction), other.subspace_basis.shape[0])
                proj = other.subspace_basis[:n_common, :].T @ direction[:n_common]
                if np.linalg.norm(proj) < 0.5:
                    is_universal = False
                    break
            if is_universal:
                universals.append(direction)
        return universals

    def fiber_resonance_matrix(self) -> Dict[str, Dict[str, float]]:
        """All pairwise cross-domain resonance scores."""
        domains = sorted(self.fibers.keys())
        result = {}
        for d1 in domains:
            result[d1] = {}
            for d2 in domains:
                if d1 == d2:
                    result[d1][d2] = 1.0
                else:
                    result[d1][d2] = round(
                        self.cross_domain_resonance(d1, d2), 4)
        return result
