"""
ParameterSpaceWalker — DNN harmonic path navigator.

Learns to predict Δθ that moves from the current parameter point toward
the next LCA region with minimum accumulated dissonance along the path.

Architecture: MLP(input_dim → hidden → hidden → theta_dim)

Input features (concatenated):
  Re(λ₁..₄)       — real parts of top-4 eigenvalues by magnitude   (4-dim)
  Im(λ₁..₄)       — imaginary parts                                (4-dim)
  spectral_gap     — |Re(λ₁)| − |Re(λ₂)|                           (1-dim)
  regime_onehot    — [lca, nonabelian, chaotic]                     (3-dim)
  θ_normalized     — theta values scaled to [0,1] per param         (theta_dim)
  ──────────────────────────────────────────────────────────────────
  total:  12 + theta_dim

Output: Δθ (theta_dim) — predicted step in parameter space

Training signal: imitation on the lowest-dissonance transitions recorded
in the experience buffer. Only the bottom-quartile experiences (by τ) are
used as positive examples. Higher-dissonance transitions are discarded.

Usage:
    walker = ParameterSpaceWalker(
        theta_keys=['R', 'L', 'C'],
        param_bounds={'R': (1.0, 1000.0), 'L': (1e-6, 1e-2), 'C': (1e-9, 1e-6)},
    )

    # Record observed transitions from a scan:
    for result_before, result_after in zip(scan_results[:-1], scan_results[1:]):
        walker.record(WalkerExperience(
            theta_before=result_before.theta,
            eigvals_before=result_before.classification.eigenvalues,
            theta_after=result_after.theta,
            eigvals_after=result_after.classification.eigenvalues,
            dissonance=tau_metric.compute(
                omega_before, omega_after
            ),
        ))

    walker.train(n_epochs=20)
    delta = walker.predict_step(current_theta, current_eigvals)
    next_theta = {k: current_theta[k] + delta[i] for i, k in enumerate(keys)}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_N_SPECTRAL = 4  # eigenvalues kept for feature vector


# ── Experience dataclass ──────────────────────────────────────────────────────

@dataclass
class WalkerExperience:
    """One observed parameter-space transition."""
    theta_before: dict
    eigvals_before: np.ndarray
    theta_after: dict
    eigvals_after: np.ndarray
    dissonance: float   # τ(ω_before, ω_after), lower = more harmonic


# ── Feature engineering ───────────────────────────────────────────────────────

def _eigval_features(eigvals: np.ndarray, n: int = _N_SPECTRAL) -> np.ndarray:
    """
    Extract 2n spectral features from eigenvalue array.

    Sorts by descending magnitude, pads/truncates to exactly n values.
    Returns [Re(λ₁..ₙ)/‖λ‖_max, Im(λ₁..ₙ)/‖λ‖_max] normalised to [-1, 1].

    Normalisation by max magnitude is critical for physical systems where
    eigenvalues can span many orders of magnitude (e.g. RLC circuits where
    Re(λ) = -R/(2L) can reach -5×10⁸).  Without it, MLP training is
    numerically unstable regardless of learning rate.
    """
    if len(eigvals) == 0:
        return np.zeros(2 * n)
    idx = np.argsort(np.abs(eigvals))[::-1]
    ev = eigvals[idx]
    # Normalise by maximum magnitude so features lie in [-1, 1]
    max_mag = float(np.max(np.abs(ev)))
    scale = max_mag if max_mag > 1e-12 else 1.0
    re = np.real(ev) / scale
    im = np.imag(ev) / scale
    # Pad or truncate to exactly n
    re = np.pad(re[:n], (0, max(0, n - len(re))))
    im = np.pad(im[:n], (0, max(0, n - len(im))))
    return np.concatenate([re, im])


def _spectral_gap(eigvals: np.ndarray) -> float:
    """|Re(λ₁)| − |Re(λ₂)| for top two eigenvalues by |Re|."""
    if len(eigvals) < 2:
        return 0.0
    real_abs = np.sort(np.abs(np.real(eigvals)))[::-1]
    return float(real_abs[0] - real_abs[1])


def _regime_onehot(patch_type: str) -> np.ndarray:
    return {
        'lca':        np.array([1.0, 0.0, 0.0]),
        'nonabelian': np.array([0.0, 1.0, 0.0]),
        'chaotic':    np.array([0.0, 0.0, 1.0]),
    }.get(patch_type, np.array([0.0, 0.0, 0.0]))


# ── MLP (numpy-only, no torch dependency) ────────────────────────────────────

class _MLP:
    """
    Lightweight 2-hidden-layer MLP with ReLU activations.

    Pure NumPy — no PyTorch dependency. Sufficient for the small input dims
    (< 30) and training sets (< 10 000) typical of parameter space scans.

    Training via SGD with Adam-style moment estimates.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Xavier init
        s1 = np.sqrt(2.0 / in_dim)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.standard_normal((hidden, in_dim)) * s1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, hidden)) * s2
        self.b2 = np.zeros(hidden)
        self.W3 = rng.standard_normal((out_dim, hidden)) * s2
        self.b3 = np.zeros(out_dim)
        # Adam moments
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)
        h2 = np.maximum(0.0, self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """X: (N, in_dim) → (N, out_dim)"""
        H1 = np.maximum(0.0, X @ self.W1.T + self.b1)
        H2 = np.maximum(0.0, H1 @ self.W2.T + self.b2)
        return H2 @ self.W3.T + self.b3

    def mse_loss_and_grads(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[float, list]:
        """
        Compute MSE loss and parameter gradients via backprop.

        X: (N, in_dim), Y: (N, out_dim)
        Returns: (loss, [dW1, db1, dW2, db2, dW3, db3])
        """
        N = X.shape[0]
        # Forward
        H1_pre = X @ self.W1.T + self.b1          # (N, hidden)
        H1 = np.maximum(0.0, H1_pre)
        H2_pre = H1 @ self.W2.T + self.b2          # (N, hidden)
        H2 = np.maximum(0.0, H2_pre)
        pred = H2 @ self.W3.T + self.b3            # (N, out_dim)

        # Loss
        diff = pred - Y
        loss = float(np.mean(diff ** 2))

        # Backward
        dL_dpred = 2.0 * diff / N                  # (N, out_dim)
        dW3 = dL_dpred.T @ H2
        db3 = dL_dpred.sum(axis=0)

        dH2 = dL_dpred @ self.W3                   # (N, hidden)
        dH2_pre = dH2 * (H2_pre > 0)
        dW2 = dH2_pre.T @ H1
        db2 = dH2_pre.sum(axis=0)

        dH1 = dH2_pre @ self.W2                    # (N, hidden)
        dH1_pre = dH1 * (H1_pre > 0)
        dW1 = dH1_pre.T @ X
        db1 = dH1_pre.sum(axis=0)

        return loss, [dW1, db1, dW2, db2, dW3, db3]

    def adam_step(
        self, grads: list, lr: float = 1e-3, beta1: float = 0.9,
        beta2: float = 0.999, eps: float = 1e-8,
    ):
        self._t += 1
        params = self._params()
        for i, (g, p) in enumerate(zip(grads, params)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * (g ** 2)
            m_hat = self._m[i] / (1 - beta1 ** self._t)
            v_hat = self._v[i] / (1 - beta2 ** self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ── ParameterSpaceWalker ──────────────────────────────────────────────────────

class ParameterSpaceWalker:
    """
    DNN harmonic navigator in parameter space.

    Learns which Δθ moves toward the next LCA region with minimal dissonance.
    Trains on the lowest-dissonance transitions observed during scans.
    """

    def __init__(
        self,
        theta_keys: List[str],
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        hidden: int = 128,
        dissonance_quantile: float = 0.25,
        seed: int = 0,
    ):
        """
        Args:
            theta_keys:          ordered list of parameter names
            param_bounds:        {'R': (lo, hi), ...} for normalizing θ to [0,1].
                                 If None, θ values are used unnormalized.
            hidden:              MLP hidden layer width
            dissonance_quantile: only the bottom-Q fraction of experiences
                                 (by dissonance) are used as training signal.
            seed:                for reproducibility
        """
        self.theta_keys = list(theta_keys)
        self.theta_dim = len(theta_keys)
        self.param_bounds = param_bounds or {}
        self.dissonance_quantile = dissonance_quantile

        in_dim = 2 * _N_SPECTRAL + 1 + 3 + self.theta_dim  # 12 + theta_dim
        self._mlp = _MLP(in_dim, hidden, self.theta_dim, seed=seed)
        self._buffer: List[WalkerExperience] = []

    # ── Experience recording ──────────────────────────────────────────────────

    def record(self, experience: WalkerExperience) -> None:
        """Add one observed transition to the replay buffer."""
        self._buffer.append(experience)

    def record_from_scan(
        self,
        results: list,   # List[MapResult] from EigenspaceMapper
        dissonance_fn,   # callable(eigvals_a, eigvals_b) -> float
    ) -> None:
        """
        Convenience: record consecutive transitions from a scan result list.

        Args:
            results:        list of MapResult in traversal order
            dissonance_fn:  callable computing τ between two eigenvalue arrays
        """
        for i in range(len(results) - 1):
            r0 = results[i]
            r1 = results[i + 1]
            tau = dissonance_fn(
                r0.classification.eigenvalues,
                r1.classification.eigenvalues,
            )
            self._buffer.append(WalkerExperience(
                theta_before=r0.theta,
                eigvals_before=r0.classification.eigenvalues,
                theta_after=r1.theta,
                eigvals_after=r1.classification.eigenvalues,
                dissonance=tau,
            ))

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_step(
        self,
        theta: dict,
        eigvals: np.ndarray,
        regime: str = 'lca',
    ) -> np.ndarray:
        """
        Predict Δθ to move toward the next harmonic LCA region.

        Args:
            theta:   current parameter dict
            eigvals: current eigenvalue array (complex)
            regime:  current regime classification string

        Returns:
            Δθ as np.ndarray of shape (theta_dim,)
        """
        x = self._build_feature(theta, eigvals, regime)
        return self._mlp.forward(x)

    def predict_next_theta(
        self,
        theta: dict,
        eigvals: np.ndarray,
        regime: str = 'lca',
    ) -> dict:
        """
        Return next_theta = theta + predict_step as a dict.
        Values are NOT clamped to param_bounds here — caller clips if needed.
        """
        delta = self.predict_step(theta, eigvals, regime)
        # Denormalize delta (it's in normalized space)
        delta_physical = self._denormalize_delta(delta)
        return {
            k: theta[k] + delta_physical[i]
            for i, k in enumerate(self.theta_keys)
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        n_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        rng_seed: Optional[int] = None,
    ) -> float:
        """
        Train on lowest-dissonance experiences in replay buffer.

        Args:
            n_epochs:   training epochs
            lr:         Adam learning rate
            batch_size: mini-batch size
            rng_seed:   for reproducible shuffling

        Returns:
            Final epoch MSE loss, or 0.0 if buffer too small.
        """
        buf = self._buffer
        if len(buf) < 4:
            return 0.0

        # Select lowest-dissonance quartile
        taus = np.array([e.dissonance for e in buf])
        threshold = np.quantile(taus, self.dissonance_quantile)
        positive = [e for e in buf if e.dissonance <= threshold]
        if not positive:
            positive = buf

        # Build feature / target arrays
        X_list, Y_list = [], []
        for exp in positive:
            x = self._build_feature(
                exp.theta_before, exp.eigvals_before, regime='lca'
            )
            delta_raw = np.array([
                exp.theta_after.get(k, 0.0) - exp.theta_before.get(k, 0.0)
                for k in self.theta_keys
            ])
            delta_norm = self._normalize_delta(delta_raw)
            X_list.append(x)
            Y_list.append(delta_norm)

        X = np.array(X_list, dtype=np.float64)
        Y = np.array(Y_list, dtype=np.float64)

        rng = np.random.default_rng(rng_seed)
        N = len(X)
        final_loss = 0.0

        for epoch in range(n_epochs):
            idx = rng.permutation(N)
            X, Y = X[idx], Y[idx]
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                Xb = X[start:start + batch_size]
                Yb = Y[start:start + batch_size]
                loss, grads = self._mlp.mse_loss_and_grads(Xb, Yb)
                self._mlp.adam_step(grads, lr=lr)
                epoch_loss += loss
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)

        return final_loss

    # ── Buffer utilities ──────────────────────────────────────────────────────

    def buffer_size(self) -> int:
        return len(self._buffer)

    def clear_buffer(self) -> None:
        self._buffer.clear()

    def buffer_dissonance_stats(self) -> dict:
        if not self._buffer:
            return {}
        taus = np.array([e.dissonance for e in self._buffer])
        return {
            'count': len(taus),
            'mean': float(np.mean(taus)),
            'min': float(np.min(taus)),
            'max': float(np.max(taus)),
            'q25': float(np.quantile(taus, 0.25)),
            'q75': float(np.quantile(taus, 0.75)),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _theta_to_array(self, theta: dict) -> np.ndarray:
        """Extract theta values in key order."""
        return np.array([theta.get(k, 0.0) for k in self.theta_keys])

    def _normalize_theta(self, theta: dict) -> np.ndarray:
        """Normalize theta to [0,1] per param using param_bounds."""
        arr = self._theta_to_array(theta)
        for i, k in enumerate(self.theta_keys):
            if k in self.param_bounds:
                lo, hi = self.param_bounds[k]
                rng = hi - lo
                if abs(rng) > 1e-12:
                    arr[i] = (arr[i] - lo) / rng
        return arr

    def _normalize_delta(self, delta: np.ndarray) -> np.ndarray:
        """Normalize a Δθ vector by the range of each parameter."""
        result = delta.copy()
        for i, k in enumerate(self.theta_keys):
            if k in self.param_bounds:
                lo, hi = self.param_bounds[k]
                rng = hi - lo
                if abs(rng) > 1e-12:
                    result[i] = result[i] / rng
        return result

    def _denormalize_delta(self, delta_norm: np.ndarray) -> np.ndarray:
        """Convert normalized Δθ back to physical units."""
        result = delta_norm.copy()
        for i, k in enumerate(self.theta_keys):
            if k in self.param_bounds:
                lo, hi = self.param_bounds[k]
                rng = hi - lo
                result[i] = result[i] * rng
        return result

    def _build_feature(
        self,
        theta: dict,
        eigvals: np.ndarray,
        regime: str,
    ) -> np.ndarray:
        """Concatenate all input features into a single vector."""
        # Normalise eigenvalues once so both spec_feats and gap use the same scale.
        # _eigval_features also normalises internally (idempotent), but doing it
        # here ensures _spectral_gap sees the same bounded [-1, 1] values rather
        # than raw physical magnitudes that can reach 1e9 for typical RLC ranges.
        if len(eigvals) > 0:
            _max_mag = float(np.max(np.abs(eigvals)))
            _scale   = _max_mag if _max_mag > 1e-12 else 1.0
            eigvals_n = eigvals / _scale
        else:
            eigvals_n = eigvals
        spec_feats = _eigval_features(eigvals_n, _N_SPECTRAL)
        gap = np.array([_spectral_gap(eigvals_n)])
        regime_enc = _regime_onehot(regime)
        theta_norm = self._normalize_theta(theta)
        return np.concatenate([spec_feats, gap, regime_enc, theta_norm])
