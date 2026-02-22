"""Source Spectral Profile — per-source reliability model.

Each information source s gets a spectral profile:
  - Mean μ_s, covariance Σ_s from many article embeddings
  - Eigen-spectrum: large dominant eigenvalue = repetitive tone
  - Sentiment amplitude distribution, surprise/novelty distribution

SourceTransferOperator learns: Δr = W_s · x + ε
  - bias b_s, amplification_factor, time_decay_constant, volatility_inflation

DynamicReliabilityUpdater maintains online trust ellipsoids
  that widen for unreliable sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SourceSpectralProfile:
    """Spectral profile for a single information source.

    Built from embeddings X_s = {x_1, ..., x_n} of articles from source s.
    """
    source_id: str
    mean: np.ndarray = field(default_factory=lambda: np.zeros(200))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(200))
    eigenvalues: np.ndarray = field(default_factory=lambda: np.ones(10))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.eye(200, 10))
    n_samples: int = 0
    sentiment_mean: float = 0.0
    sentiment_std: float = 0.0
    novelty_mean: float = 0.0
    novelty_std: float = 0.0

    @property
    def spectral_concentration(self) -> float:
        """Ratio of dominant eigenvalue to total variance.

        High concentration → repetitive, predictable tone.
        Low concentration → diverse, varied content.
        """
        total = self.eigenvalues.sum()
        if total < 1e-15:
            return 0.0
        return float(self.eigenvalues[0] / total)

    @property
    def effective_rank(self) -> float:
        """Effective rank: exp(entropy of normalized eigenvalues).

        High rank → diverse content. Low rank → narrow/repetitive.
        """
        total = self.eigenvalues.sum()
        if total < 1e-15:
            return 1.0
        p = self.eigenvalues / total
        p = p[p > 1e-15]
        return float(np.exp(-np.sum(p * np.log(p))))


@dataclass
class SourceTransferOperator:
    """Linear operator: semantic_state → market_state_change.

    Learned from (article_embedding, price_change) pairs.
    Models: Δr = W_s · x + b_s + ε
    """
    source_id: str
    W: np.ndarray = field(default_factory=lambda: np.zeros((1, 200)))
    bias: float = 0.0
    amplification: float = 1.0
    time_decay_constant: float = 1.0  # hours
    volatility_inflation: float = 0.0
    n_observations: int = 0
    r_squared: float = 0.0

    def predict_impact(self, embedding: np.ndarray, hours_since: float = 0.0) -> float:
        """Predict market impact of article.

        Returns estimated price change with time decay.
        """
        raw = float((self.W @ embedding).item()) + self.bias
        decay = np.exp(-hours_since / max(self.time_decay_constant, 0.01))
        return raw * self.amplification * decay


class SourceProfileBuilder:
    """Builds SourceSpectralProfile from article embeddings."""

    def __init__(self, max_rank: int = 20) -> None:
        self._max_rank = max_rank

    def build(
        self,
        source_id: str,
        embeddings: np.ndarray,
        sentiments: Optional[np.ndarray] = None,
        novelties: Optional[np.ndarray] = None,
    ) -> SourceSpectralProfile:
        """Build profile from (n_articles, dim) embedding matrix.

        Args:
            source_id: Unique source identifier.
            embeddings: (N, D) matrix of article embeddings.
            sentiments: Optional (N,) sentiment scores.
            novelties: Optional (N,) novelty scores.
        """
        n, dim = embeddings.shape
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean

        # Covariance (regularized for small samples)
        if n > dim:
            cov = centered.T @ centered / (n - 1)
        else:
            cov = centered.T @ centered / max(n - 1, 1)
            cov += 1e-6 * np.eye(dim)  # regularize

        # Eigen-decomposition (top-k)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        k = min(self._max_rank, dim, n)
        eigvals = np.maximum(eigvals[:k], 0)  # clip negative eigenvalues
        eigvecs = eigvecs[:, :k]

        profile = SourceSpectralProfile(
            source_id=source_id,
            mean=mean,
            covariance=cov,
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            n_samples=n,
        )

        if sentiments is not None and len(sentiments) > 0:
            profile.sentiment_mean = float(np.mean(sentiments))
            profile.sentiment_std = float(np.std(sentiments))

        if novelties is not None and len(novelties) > 0:
            profile.novelty_mean = float(np.mean(novelties))
            profile.novelty_std = float(np.std(novelties))

        return profile


class TransferOperatorLearner:
    """Learn SourceTransferOperator from (embedding, price_change) pairs."""

    def learn(
        self,
        source_id: str,
        embeddings: np.ndarray,
        price_changes: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> SourceTransferOperator:
        """Fit linear transfer operator via ridge regression.

        Args:
            source_id: Source identifier.
            embeddings: (N, D) article embeddings.
            price_changes: (N,) subsequent price changes.
            timestamps: Optional (N,) hours-since-publication at observation.
        """
        n, dim = embeddings.shape
        op = SourceTransferOperator(source_id=source_id, n_observations=n)

        if n < 3:
            return op

        # Ridge regression: W = (X^T X + λI)^{-1} X^T y
        lam = 1.0  # regularization
        XtX = embeddings.T @ embeddings + lam * np.eye(dim)
        Xty = embeddings.T @ price_changes

        try:
            W = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            return op

        op.W = W.reshape(1, -1)
        op.bias = float(price_changes.mean() - W @ embeddings.mean(axis=0))

        # R-squared
        predictions = embeddings @ W + op.bias
        ss_res = np.sum((price_changes - predictions) ** 2)
        ss_tot = np.sum((price_changes - price_changes.mean()) ** 2)
        op.r_squared = float(1.0 - ss_res / max(ss_tot, 1e-15))

        # Amplification = std of predictions / std of actual
        pred_std = np.std(predictions)
        actual_std = np.std(price_changes)
        op.amplification = float(pred_std / max(actual_std, 1e-15))

        # Volatility inflation = excess variance from source
        op.volatility_inflation = float(max(0, pred_std - actual_std))

        # Time decay constant (if timestamps provided)
        if timestamps is not None and len(timestamps) > 3:
            # Fit exponential decay to |prediction - actual| vs time
            errors = np.abs(predictions - price_changes)
            # Simple: half-life = median time where error exceeds mean
            median_time = np.median(timestamps)
            op.time_decay_constant = float(max(median_time, 0.1))

        return op


class DynamicReliabilityUpdater:
    """Online updater for source profiles as new articles arrive.

    Maintains trust ellipsoid per source: unreliable sources → wider uncertainty.
    """

    def __init__(self, decay_rate: float = 0.95) -> None:
        self._decay_rate = decay_rate
        self._profiles: Dict[str, SourceSpectralProfile] = {}
        self._operators: Dict[str, SourceTransferOperator] = {}
        self._trust_scores: Dict[str, float] = {}

    def register_profile(self, profile: SourceSpectralProfile) -> None:
        """Register a source profile."""
        self._profiles[profile.source_id] = profile
        if profile.source_id not in self._trust_scores:
            self._trust_scores[profile.source_id] = 0.5  # neutral prior

    def register_operator(self, operator: SourceTransferOperator) -> None:
        """Register a transfer operator."""
        self._operators[operator.source_id] = operator

    def update(
        self,
        source_id: str,
        new_embedding: np.ndarray,
        actual_impact: Optional[float] = None,
    ) -> float:
        """Update source profile with new article, return trust score.

        If actual_impact provided, also update transfer operator accuracy.
        """
        profile = self._profiles.get(source_id)
        if profile is None:
            return 0.5

        # Online mean/covariance update (exponential decay)
        alpha = 1.0 - self._decay_rate
        profile.mean = self._decay_rate * profile.mean + alpha * new_embedding
        delta = new_embedding - profile.mean
        profile.covariance = (
            self._decay_rate * profile.covariance
            + alpha * np.outer(delta, delta)
        )
        profile.n_samples += 1

        # Update trust based on prediction accuracy
        trust = self._trust_scores.get(source_id, 0.5)
        if actual_impact is not None and source_id in self._operators:
            op = self._operators[source_id]
            predicted = op.predict_impact(new_embedding)
            error = abs(predicted - actual_impact)
            # Trust increases when predictions are accurate
            accuracy = 1.0 / (1.0 + error)
            trust = self._decay_rate * trust + alpha * accuracy

        self._trust_scores[source_id] = trust
        return trust

    def trust_ellipsoid(self, source_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (center, covariance) of trust ellipsoid.

        Unreliable sources get inflated covariance (wider uncertainty).
        """
        profile = self._profiles.get(source_id)
        if profile is None:
            return np.zeros(200), np.eye(200) * 10.0

        trust = self._trust_scores.get(source_id, 0.5)
        # Inflate covariance inversely with trust
        inflation = 1.0 / max(trust, 0.01)
        return profile.mean, profile.covariance * inflation

    def adjust_encoding(
        self,
        encoding: np.ndarray,
        source_id: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust article encoding by source profile.

        Returns (debiased_encoding, inflated_covariance).
        Integration: α' = α - b_s, Σ' = Σ + Σ_s
        """
        profile = self._profiles.get(source_id)
        op = self._operators.get(source_id)

        debiased = encoding.copy()
        cov = np.zeros_like(np.outer(encoding, encoding))

        if op is not None:
            debiased = encoding - op.bias  # debias

        if profile is not None:
            cov = profile.covariance.copy()

        return debiased, cov

    def get_trust(self, source_id: str) -> float:
        """Current trust score for a source."""
        return self._trust_scores.get(source_id, 0.5)
