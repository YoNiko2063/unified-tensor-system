"""SemanticFlowEncoder — temporal HDV encoding for articles, market windows, and code intents.

Unifies three information streams into a common HDV space:
  1. Articles: epistemic geometry → source adjustment → temporal decay
  2. Market windows: OHLCV → Duffing param mapping → resonance proximity
  3. Code intents: domain + operation + complexity + borrow profile

Each encoding records trajectory for Koopman observer state via _TextEDMD.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ObserverState:
    """Koopman observer state for a domain.

    Eigenvalues: dominant mode = fundamental drift, secondary = technical oscillations.
    Trust: spectral gap proxy — high gap = clear dynamics.
    """
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    trust: float = 0.0
    n_observations: int = 0
    domain: str = ""


class SemanticFlowEncoder:
    """Temporal HDV encoding for multi-source information streams.

    Integrates:
      - EpistemicGeometryLayer for article analysis
      - SourceSpectralProfile for source trust adjustment
      - Duffing parameter mapping for market data
      - IntentSpec encoding for code generation
    """

    def __init__(
        self,
        hdv_system=None,
        epistemic_layer=None,
        reliability_updater=None,
        hdv_dim: int = 10000,
        temporal_half_life_days: float = 7.0,
    ) -> None:
        self._hdv = hdv_system
        self._epistemic = epistemic_layer
        self._reliability = reliability_updater
        self._hdv_dim = hdv_dim
        self._half_life = temporal_half_life_days
        self._decay_rate = np.log(2) / temporal_half_life_days

        # Trajectory storage per domain for Koopman observer
        self._trajectories: Dict[str, List[np.ndarray]] = {}
        self._timestamps: Dict[str, List[float]] = {}

    def encode_article(
        self,
        content: str,
        source: str = "",
        date: float = 0.0,
        topic: str = "general",
        tickers: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Encode article into HDV space with epistemic + source adjustments.

        Flow:
          1. Epistemic geometry → section weights
          2. Source spectral profile → trust-adjusted encoding
          3. Temporal decay (7-day half-life)
          4. Record trajectory for Koopman observer
        """
        # Base encoding
        if self._hdv is not None:
            base_vec = self._hdv.structural_encode(content, topic)
            if not isinstance(base_vec, np.ndarray):
                base_vec = self._hash_encode(content)
        else:
            base_vec = self._hash_encode(content)

        # Ensure consistent dimension
        vec = self._project_dim(base_vec)

        # Epistemic geometry weighting
        if self._epistemic is not None:
            profile = self._epistemic.analyze(content, topic)
            if profile.section_weights is not None and len(profile.section_weights) > 0:
                # Weight by epistemic quality (validity score modulates magnitude)
                validity_scale = max(0.1, 0.5 + profile.overall_validity)
                vec = vec * validity_scale

        # Source adjustment
        if self._reliability is not None and source:
            debiased, source_cov = self._reliability.adjust_encoding(vec, source)
            vec = debiased
            # Record for trust update
            self._reliability.update(source, vec)

        # Temporal decay
        if date > 0:
            now = time.time() / 86400  # days
            age_days = max(now - date, 0)
            decay = np.exp(-self._decay_rate * age_days)
            vec = vec * decay

        # Record trajectory
        domain_key = f"article_{topic}"
        self._record_trajectory(domain_key, vec, date)

        return vec

    def encode_market_window(
        self,
        ohlcv: np.ndarray,
        ticker: str = "",
        duffing_params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Encode OHLCV market window into HDV space.

        Candle features → structural HDV.
        Duffing parameter mapping:
          autocorrelation → alpha (mean reversion)
          kurtosis → beta (nonlinearity)
          spread → delta (friction)

        Resonance proximity: distance to backbone curve fold.
        """
        if ohlcv.ndim == 1:
            ohlcv = ohlcv.reshape(1, -1)

        n_candles = ohlcv.shape[0]

        # Extract candle features
        if ohlcv.shape[1] >= 5:
            opens = ohlcv[:, 0]
            highs = ohlcv[:, 1]
            lows = ohlcv[:, 2]
            closes = ohlcv[:, 3]
            volumes = ohlcv[:, 4]
        else:
            closes = ohlcv[:, 0]
            opens = highs = lows = closes
            volumes = np.ones(n_candles)

        # Returns
        if n_candles > 1:
            returns = np.diff(closes) / (closes[:-1] + 1e-10)
        else:
            returns = np.array([0.0])

        # Duffing parameter mapping from data
        if duffing_params is None:
            duffing_params = self._fit_duffing_params(returns, closes, volumes)

        # Build feature vector
        features = np.array([
            np.mean(returns),                           # mean return
            np.std(returns),                            # volatility
            self._autocorrelation(returns, 1),          # lag-1 AC → alpha proxy
            float(self._kurtosis(returns)),             # kurtosis → beta proxy
            np.mean(highs - lows) / (np.mean(closes) + 1e-10),  # spread → delta proxy
            np.mean(volumes),                           # avg volume
            duffing_params.get("alpha", 1.0),
            duffing_params.get("beta", 0.1),
            duffing_params.get("delta", 0.1),
            duffing_params.get("resonance_proximity", 0.0),
        ])

        # Encode into HDV
        if self._hdv is not None:
            # Use structural_encode on string repr of features
            feature_str = " ".join(f"{f:.4f}" for f in features)
            base_vec = self._hdv.structural_encode(feature_str, "market")
            vec = self._project_dim(base_vec) if isinstance(base_vec, np.ndarray) else self._hash_encode(feature_str)
        else:
            vec = self._deterministic_market_encode(features)

        # Record trajectory
        self._record_trajectory(f"market_{ticker}", vec)

        return vec

    def encode_code_intent(self, intent) -> np.ndarray:
        """Encode IntentSpec into HDV space.

        Features: domain + operation + complexity + borrow profile.
        """
        # Build text representation of intent
        parts = [
            intent.domain,
            intent.operation,
            intent.complexity_class,
            intent.estimated_borrow_profile.value,
        ]
        for trait in intent.required_traits:
            parts.append(trait)
        text = " ".join(parts)

        if self._hdv is not None:
            base_vec = self._hdv.structural_encode(text, "code")
            vec = self._project_dim(base_vec) if isinstance(base_vec, np.ndarray) else self._hash_encode(text)
        else:
            vec = self._hash_encode(text)

        # Add borrow vector signature to specific dimensions
        bv = intent.expected_borrow_vector()
        for i, b in enumerate(bv):
            if i < len(vec):
                vec[i] += b * 0.5  # blend BV into first 6 dims

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        self._record_trajectory("code_intent", vec)
        return vec

    def compute_observer_state(self, domain: str) -> Optional[ObserverState]:
        """Compute Koopman observer state for a domain.

        Uses _TextEDMD from semantic_geometry.py (NOT EDMDKoopman — preserves CRITICAL-1).
        Requires ≥5 trajectory points. Returns None if trust < 0.3.
        """
        traj = self._trajectories.get(domain, [])
        if len(traj) < 5:
            return None

        # Import _TextEDMD (internal, not EDMDKoopman)
        try:
            from tensor.semantic_geometry import _TextEDMD
        except ImportError:
            return None

        # Build (x_t, x_{t+1}) pairs
        traj_arr = np.array(traj[-50:])  # last 50 points
        dim = min(traj_arr.shape[1], 200)  # cap dimension
        traj_proj = traj_arr[:, :dim]

        pairs = [(traj_proj[i], traj_proj[i + 1]) for i in range(len(traj_proj) - 1)]

        edmd = _TextEDMD()
        edmd.fit(pairs)

        eigvals, eigvecs = edmd.eigendecomposition()
        trust = edmd.trust_score()

        if trust < 0.3:
            return None

        return ObserverState(
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            trust=trust,
            n_observations=len(traj),
            domain=domain,
        )

    def detect_resonance(
        self,
        market_domain: str,
        article_domain: str,
    ) -> float:
        """Detect resonance: market eigenvalues approaching article eigenvalues.

        Returns resonance proximity score [0, 1].
        High score → news forcing frequency matches market natural frequency.
        """
        market_obs = self.compute_observer_state(market_domain)
        article_obs = self.compute_observer_state(article_domain)

        if market_obs is None or article_obs is None:
            return 0.0

        # Compare eigenvalue spectra
        m_eig = np.abs(market_obs.eigenvalues)
        a_eig = np.abs(article_obs.eigenvalues)

        if len(m_eig) == 0 or len(a_eig) == 0:
            return 0.0

        # Minimum distance between any pair of eigenvalues
        min_dist = float("inf")
        for me in m_eig[:5]:
            for ae in a_eig[:5]:
                dist = abs(me - ae)
                if dist < min_dist:
                    min_dist = dist

        # Convert to proximity score: 1/(1+dist)
        return float(1.0 / (1.0 + min_dist))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _record_trajectory(
        self,
        domain: str,
        vec: np.ndarray,
        timestamp: float = 0.0,
    ) -> None:
        """Record vector in domain trajectory."""
        if domain not in self._trajectories:
            self._trajectories[domain] = []
            self._timestamps[domain] = []
        self._trajectories[domain].append(vec.copy())
        self._timestamps[domain].append(timestamp or time.time())
        # Cap at 200 points
        if len(self._trajectories[domain]) > 200:
            self._trajectories[domain] = self._trajectories[domain][-200:]
            self._timestamps[domain] = self._timestamps[domain][-200:]

    def _project_dim(self, vec: np.ndarray) -> np.ndarray:
        """Project vector to standard HDV dimension."""
        if len(vec) >= self._hdv_dim:
            return vec[:self._hdv_dim]
        padded = np.zeros(self._hdv_dim)
        padded[:len(vec)] = vec
        return padded

    def _hash_encode(self, text: str) -> np.ndarray:
        """Deterministic hash embedding (fallback)."""
        import hashlib
        vec = np.zeros(self._hdv_dim)
        for i, ch in enumerate(text.encode("utf-8")):
            h = int(hashlib.md5(f"{ch}_{i}".encode()).hexdigest(), 16)
            idx = h % self._hdv_dim
            vec[idx] += 1.0 if (h // self._hdv_dim) % 2 == 0 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _deterministic_market_encode(self, features: np.ndarray) -> np.ndarray:
        """Deterministic market feature encoding into HDV."""
        vec = np.zeros(self._hdv_dim)
        for i, f in enumerate(features):
            # Spread each feature across multiple dimensions
            base = i * (self._hdv_dim // (len(features) + 1))
            spread = self._hdv_dim // (len(features) * 3 + 1)
            for j in range(spread):
                idx = (base + j) % self._hdv_dim
                vec[idx] = f * np.cos(j * 0.1)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _fit_duffing_params(
        returns: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> Dict[str, float]:
        """Fit Duffing parameters from market data.

        autocorrelation → alpha (mean reversion)
        kurtosis → beta (nonlinearity)
        spread → delta (friction)
        """
        if len(returns) < 5:
            return {"alpha": 1.0, "beta": 0.1, "delta": 0.1, "resonance_proximity": 0.0}

        # Alpha from autocorrelation (negative AC → mean reversion)
        ac1 = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        alpha = max(0.1, 1.0 - ac1)  # negative AC → higher alpha

        # Beta from kurtosis (fat tails → nonlinearity)
        kurt = float(np.mean(returns ** 4) / (np.std(returns) ** 4 + 1e-15) - 3.0)
        beta = max(0.01, 0.1 * abs(kurt) / 3.0)

        # Delta from volatility (high vol → friction)
        vol = float(np.std(returns))
        delta = max(0.01, vol * 5.0)

        # Resonance proximity: how close is driving frequency to natural frequency?
        # Natural frequency ≈ sqrt(alpha), driving from volume periodicity
        nat_freq = np.sqrt(alpha)
        if len(volumes) > 10:
            vol_fft = np.abs(np.fft.rfft(volumes - volumes.mean()))
            if len(vol_fft) > 1:
                peak_idx = np.argmax(vol_fft[1:]) + 1
                drive_freq = peak_idx / len(volumes)
                resonance = 1.0 / (1.0 + abs(nat_freq - drive_freq))
            else:
                resonance = 0.0
        else:
            resonance = 0.0

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "delta": float(delta),
            "resonance_proximity": float(resonance),
        }

    @staticmethod
    def _autocorrelation(x: np.ndarray, lag: int) -> float:
        """Lag-k autocorrelation."""
        if len(x) <= lag:
            return 0.0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Excess kurtosis."""
        if len(x) < 4:
            return 0.0
        std = np.std(x)
        if std < 1e-15:
            return 0.0
        return float(np.mean(((x - np.mean(x)) / std) ** 4) - 3.0)
