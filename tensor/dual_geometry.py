"""
FICUTS Layer 9: Dual Geometry (FIM + IRMF)

Classes:
  - FisherInformationManifold   : statistical manifold for experimental patterns  (Task 9.1)
  - IsometricFunctionManifold   : deterministic manifold for foundational patterns (Task 9.2)
  - DualGeometrySystem          : routes + promotes patterns                       (Task 9.3)

Environment: tensor env (NumPy/SciPy) for FIM + DualGeometrySystem.
             torch only used inside IsometricFunctionManifold (lazy import).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ── Task 9.1: Fisher Information Manifold ────────────────────────────────────

class FisherInformationManifold:
    """
    Statistical manifold for experimental (uncertain) patterns.

    Fits p(x | θ) = N(μ, σ²) via MLE, computes FIM, tracks uncertainty.
    Cramér-Rao: var(θ_i) ≥ (FIM⁻¹)_ii
    """

    def __init__(self):
        self.patterns: Dict[str, dict] = {}

    def learn_distribution(self, pattern_id: str,
                           data: np.ndarray,
                           initial_theta: np.ndarray) -> dict:
        def neg_ll(theta):
            return -self._log_likelihood(data, theta)

        result = minimize(neg_ll, initial_theta, method='BFGS',
                          options={'gtol': 1e-6, 'maxiter': 500})
        theta_mle = result.x

        fim = self._compute_fim(theta_mle, data)

        try:
            fim_inv = np.linalg.inv(fim)
            uncertainty = np.sqrt(np.abs(np.diag(fim_inv)))
        except np.linalg.LinAlgError:
            uncertainty = np.full(len(theta_mle), np.inf)

        self.patterns[pattern_id] = {
            'theta': theta_mle,
            'FIM': fim,
            'uncertainty': uncertainty,
            'data_size': len(data),
        }
        return self.patterns[pattern_id]

    def _log_likelihood(self, data: np.ndarray, theta: np.ndarray) -> float:
        mu = theta[0]
        sigma = max(abs(theta[1]), 1e-6)
        log_p = -0.5 * np.log(2 * np.pi * sigma ** 2) - (data - mu) ** 2 / (2 * sigma ** 2)
        return float(np.sum(log_p))

    def _compute_fim(self, theta: np.ndarray, data: np.ndarray) -> np.ndarray:
        d = len(theta)
        fim = np.zeros((d, d))
        eps = 1e-5
        for i in range(d):
            for j in range(d):
                tp = theta.copy(); tp[i] += eps; tp[j] += eps
                tm = theta.copy(); tm[i] += eps; tm[j] -= eps
                mp = theta.copy(); mp[i] -= eps; mp[j] += eps
                mm = theta.copy(); mm[i] -= eps; mm[j] -= eps
                d2L = (
                    self._log_likelihood(data, tp)
                    - self._log_likelihood(data, tm)
                    - self._log_likelihood(data, mp)
                    + self._log_likelihood(data, mm)
                ) / (4 * eps ** 2)
                fim[i, j] = -d2L
        return fim

    def is_ready_for_promotion(self, pattern_id: str, threshold: float = 0.01) -> bool:
        if pattern_id not in self.patterns:
            return False
        return bool(np.all(self.patterns[pattern_id]['uncertainty'] < threshold))

    def get_most_informative_parameters(self, pattern_id: str) -> np.ndarray:
        fim = self.patterns[pattern_id]['FIM']
        return np.argsort(np.diag(fim))[::-1]


# ── Task 9.2: Isometric Function Manifold ────────────────────────────────────

class IsometricFunctionManifold:
    """
    Deterministic manifold: latent z → f(x; z) with distance preservation.

    Requires torch (dev-agent env). Falls back gracefully if torch missing.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.patterns: Dict[str, object] = {}  # pattern_id → latent tensor
        self._decoder = None
        self._torch = None
        self._nn = None
        self._init_torch()

    def _init_torch(self):
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._nn = nn
            self._decoder = nn.Sequential(
                nn.Linear(self.latent_dim + 1, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            ).to(torch.device('cpu'))
        except ImportError:
            pass  # torch not available; methods will raise

    @property
    def _available(self) -> bool:
        return self._torch is not None

    def learn_function(self, pattern_id: str,
                       data_points: List[Tuple[float, float]],
                       n_epochs: int = 1000):
        if not self._available:
            raise RuntimeError("torch not available — switch to dev-agent env")
        torch = self._torch

        x = torch.tensor([p[0] for p in data_points], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor([p[1] for p in data_points], dtype=torch.float32).unsqueeze(1)

        z = torch.randn(self.latent_dim, requires_grad=True,
                        device=torch.device('cpu'))
        opt = torch.optim.Adam([z], lr=0.01)

        for epoch in range(n_epochs):
            opt.zero_grad()
            z_exp = z.unsqueeze(0).expand(x.shape[0], -1)
            inp = torch.cat([z_exp, x], dim=1)
            pred = self._decoder(inp)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()

        self.patterns[pattern_id] = z.detach().clone()
        return z.detach()

    def generate_function_values(self, pattern_id: str,
                                  x_values: np.ndarray) -> np.ndarray:
        if not self._available:
            raise RuntimeError("torch not available")
        torch = self._torch
        z = self.patterns[pattern_id]
        x = torch.tensor(x_values, dtype=torch.float32).unsqueeze(1)
        z_exp = z.unsqueeze(0).expand(x.shape[0], -1)
        inp = torch.cat([z_exp, x], dim=1)
        with torch.no_grad():
            out = self._decoder(inp)
        return out.numpy().flatten()

    def compute_isometric_loss(self, latent_vectors, n_pairs: int = 100):
        if not self._available:
            raise RuntimeError("torch not available")
        torch = self._torch
        n = latent_vectors.shape[0]
        if n < 2:
            return torch.tensor(0.0)

        idx = torch.randint(0, n, (n_pairs, 2))
        z1 = latent_vectors[idx[:, 0]]
        z2 = latent_vectors[idx[:, 1]]
        d_latent = torch.norm(z1 - z2, dim=1)

        x_samp = torch.linspace(0, 1, 50).unsqueeze(1)
        d_fn = torch.zeros(n_pairs)
        for i in range(n_pairs):
            z1e = z1[i].unsqueeze(0).expand(x_samp.shape[0], -1)
            z2e = z2[i].unsqueeze(0).expand(x_samp.shape[0], -1)
            with torch.no_grad():
                f1 = self._decoder(torch.cat([z1e, x_samp], dim=1))
                f2 = self._decoder(torch.cat([z2e, x_samp], dim=1))
            d_fn[i] = torch.norm(f1 - f2)

        return torch.mean((d_latent - d_fn) ** 2)


# ── Task 9.3: Dual Geometry System ───────────────────────────────────────────

class DualGeometrySystem:
    """
    Routes patterns to FIM (statistical) or IRMF (deterministic).
    Promotes experimental → foundational when criteria met.
    """

    def __init__(self):
        self.fisher = FisherInformationManifold()
        self.isometric = IsometricFunctionManifold()

    def classify_pattern(self, pattern: dict) -> str:
        variance = float(np.var(pattern.get('observations', [])))
        domain_count = len(pattern.get('domains', []))
        has_conservation = bool(pattern.get('conserved_quantity', False))

        if variance > 0.1 or domain_count < 3 or not has_conservation:
            return 'statistical'
        return 'deterministic'

    def learn_pattern(self, pattern_id: str, pattern: dict, data: np.ndarray):
        classification = self.classify_pattern(pattern)
        if classification == 'statistical':
            print(f"[DualGeometry] Learning {pattern_id} on FIM (experimental)")
            initial = np.array([np.mean(data), max(np.std(data), 1e-6)])
            self.fisher.learn_distribution(pattern_id, data, initial)
        else:
            print(f"[DualGeometry] Learning {pattern_id} on IRMF (foundational)")
            x = np.linspace(0, 1, len(data))
            pts = [(x[i], float(data[i])) for i in range(len(data))]
            self.isometric.learn_function(pattern_id, pts)

    def promote_pattern(self, pattern_id: str, function_library=None,
                        threshold: float = 0.01) -> bool:
        if pattern_id not in self.fisher.patterns:
            return False
        if not self.fisher.is_ready_for_promotion(pattern_id, threshold=threshold):
            print(f"[DualGeometry] {pattern_id} not ready — uncertainty too high")
            return False
        print(f"[DualGeometry] PROMOTING {pattern_id}: FIM → IRMF")
        if function_library is not None:
            function_library.promote_to_foundational(pattern_id)
        return True
