"""Three-Timeframe State Spaces + Cross-Timescale Lifting Operators.

Three state spaces:
  x(L) ∈ R^dL — Fundamentals (months-years)
  x(M) ∈ R^dM — Regime/Technical (days-weeks)
  x(S) ∈ R^dS — News/Shock (minutes-days)

Cross-timescale lifting operators:
  Φ_S→M: news events → regime perturbation
  Φ_M→L: regime state → fundamental expectation shift

All operators are linear in Koopman observable space with
Jacobian regularization for bounded spectral structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Fundamental State (Long timescale) ────────────────────────────────────────

@dataclass
class FundamentalState:
    """x(L) — months-to-years fundamental state.

    Features: revenue growth, margins, FCF, leverage, valuation,
    ROIC, sector macro proxies.
    """
    # Feature vector
    features: np.ndarray = field(default_factory=lambda: np.zeros(12))
    timestamp: float = 0.0

    # Feature names for interpretability
    FEATURE_NAMES = [
        "revenue_growth", "gross_margin", "operating_margin", "fcf_yield",
        "debt_to_equity", "pe_ratio", "pb_ratio", "roic",
        "sector_gdp_beta", "sector_rate_beta", "sector_inflation_beta",
        "quality_score",
    ]

    @property
    def linearity_score(self) -> float:
        """ρ_L(x) = ‖∇J_L(x)‖ / λ_max(J_L).

        Low ρ → linear/Laplace-valid (fundamentals evolve slowly).
        Computed from feature variance as a proxy.
        """
        var = np.var(self.features)
        return float(min(var / (np.max(np.abs(self.features)) + 1e-10), 1.0))


@dataclass
class RegimeState:
    """x(M) — days-to-weeks regime/technical state.

    Features: realized vol, trend strength, drawdown, liquidity,
    factor exposures, sector rotation, breadth, technicals + microstructure.

    Duffing mapping:
      alpha → mean reversion strength in this regime
      beta  → nonlinear reaction (kurtosis-driven)
    """
    features: np.ndarray = field(default_factory=lambda: np.zeros(16))
    regime_id: int = 0  # 0=stable, 1=transitioning, 2=crisis
    timestamp: float = 0.0

    FEATURE_NAMES = [
        "realized_vol_5d", "realized_vol_20d", "trend_strength",
        "max_drawdown_20d", "liquidity_score", "momentum_factor",
        "value_factor", "size_factor", "sector_rotation_speed",
        "market_breadth", "rsi_14", "macd_signal",
        "volume_ratio", "bid_ask_spread", "order_imbalance",
        "regime_duration_days",
    ]

    @property
    def duffing_alpha(self) -> float:
        """Mean reversion strength from realized vol and trend."""
        vol = self.features[0] if len(self.features) > 0 else 0.1
        trend = abs(self.features[2]) if len(self.features) > 2 else 0.0
        return float(max(0.1, 1.0 - trend + vol))

    @property
    def duffing_beta(self) -> float:
        """Nonlinear reaction from kurtosis proxy."""
        vol_5 = self.features[0] if len(self.features) > 0 else 0.1
        vol_20 = self.features[1] if len(self.features) > 1 else 0.1
        # High vol ratio → fat tails → larger beta
        ratio = vol_5 / max(vol_20, 1e-10)
        return float(max(0.01, 0.1 * ratio))

    @property
    def stability(self) -> float:
        """Regime stability score [0, 1]. High = stable."""
        duration = self.features[15] if len(self.features) > 15 else 1.0
        vol = self.features[0] if len(self.features) > 0 else 0.1
        return float(min(1.0, duration / (30.0 + 100.0 * vol)))


@dataclass
class ShockState:
    """x(S) — minutes-to-days news/shock state.

    Event-driven with exponential time decay:
      x_t(S) ∋ s · exp(-(t - t_0) / τ) per event

    Epistemic geometry feeds directly: article section weights,
    source reliability → x(S) components.
    """
    features: np.ndarray = field(default_factory=lambda: np.zeros(12))
    active_events: List[Dict] = field(default_factory=list)
    timestamp: float = 0.0

    FEATURE_NAMES = [
        "sentiment_score", "sentiment_confidence", "novelty_score",
        "entity_count", "event_type_code", "time_since_event_hours",
        "source_trust", "epistemic_validity", "section_tech_density",
        "section_cite_density", "event_decay_factor", "surprise_magnitude",
    ]

    # Event type codes
    EVENT_TYPES = {
        "earnings": 1, "guidance": 2, "lawsuit": 3, "fda": 4,
        "merger_acquisition": 5, "macro": 6, "product": 7, "management": 8,
    }

    def apply_decay(self, current_time: float, tau: float = 24.0) -> None:
        """Apply exponential time decay to event impacts.

        tau: decay time constant in hours (default 24h).
        """
        for event in self.active_events:
            t0 = event.get("timestamp", current_time)
            dt = max(current_time - t0, 0)
            event["decay_factor"] = float(np.exp(-dt / tau))
            event["decayed_impact"] = event.get("raw_impact", 0.0) * event["decay_factor"]

        # Update features from decayed events
        if self.active_events:
            total_impact = sum(e.get("decayed_impact", 0.0) for e in self.active_events)
            max_decay = max(e.get("decay_factor", 0.0) for e in self.active_events)
            self.features[10] = max_decay
            self.features[11] = abs(total_impact)

    def inject_event(
        self,
        event_type: str,
        sentiment: float,
        confidence: float,
        novelty: float,
        source_trust: float,
        epistemic_validity: float = 0.0,
        tech_density: float = 0.0,
        cite_density: float = 0.0,
        timestamp: float = 0.0,
    ) -> None:
        """Inject a new news event into the shock state."""
        event = {
            "event_type": event_type,
            "raw_impact": sentiment * confidence * source_trust,
            "timestamp": timestamp,
            "decay_factor": 1.0,
            "decayed_impact": sentiment * confidence * source_trust,
        }
        self.active_events.append(event)

        # Update feature vector
        self.features[0] = sentiment
        self.features[1] = confidence
        self.features[2] = novelty
        self.features[4] = float(self.EVENT_TYPES.get(event_type, 0))
        self.features[5] = 0.0  # just injected
        self.features[6] = source_trust
        self.features[7] = epistemic_validity
        self.features[8] = tech_density
        self.features[9] = cite_density
        self.features[10] = 1.0  # fresh event
        self.features[11] = abs(event["raw_impact"])
        self.timestamp = timestamp


# ── State Builders ────────────────────────────────────────────────────────────

class FundamentalStateBuilder:
    """Build x(L) from raw financial data."""

    def build(self, data: Dict[str, float]) -> FundamentalState:
        """Build fundamental state from key-value financial metrics."""
        features = np.zeros(12)
        for i, name in enumerate(FundamentalState.FEATURE_NAMES):
            if name in data:
                features[i] = data[name]
        return FundamentalState(
            features=features,
            timestamp=data.get("timestamp", 0.0),
        )


class RegimeStateBuilder:
    """Build x(M) from OHLCV + technical indicator data."""

    def build(self, data: Dict[str, float]) -> RegimeState:
        """Build regime state from technical/market data."""
        features = np.zeros(16)
        for i, name in enumerate(RegimeState.FEATURE_NAMES):
            if name in data:
                features[i] = data[name]

        # Regime classification from volatility
        vol = features[0]
        if vol < 0.02:
            regime_id = 0  # stable
        elif vol < 0.05:
            regime_id = 1  # transitioning
        else:
            regime_id = 2  # crisis

        return RegimeState(
            features=features,
            regime_id=regime_id,
            timestamp=data.get("timestamp", 0.0),
        )


class ShockStateBuilder:
    """Build x(S) from news event data + epistemic profiles."""

    def build(
        self,
        events: List[Dict],
        current_time: float = 0.0,
        tau: float = 24.0,
    ) -> ShockState:
        """Build shock state from list of event dicts."""
        state = ShockState(timestamp=current_time)
        for event in events:
            state.inject_event(
                event_type=event.get("event_type", "macro"),
                sentiment=event.get("sentiment", 0.0),
                confidence=event.get("confidence", 0.5),
                novelty=event.get("novelty", 0.0),
                source_trust=event.get("source_trust", 0.5),
                epistemic_validity=event.get("epistemic_validity", 0.0),
                tech_density=event.get("tech_density", 0.0),
                cite_density=event.get("cite_density", 0.0),
                timestamp=event.get("timestamp", current_time),
            )
        state.apply_decay(current_time, tau)
        return state


# ── Cross-Timescale Lifting Operators ────────────────────────────────────────

class LiftingOperator:
    """Linear operator in Koopman observable space for cross-timescale coupling.

    Φ: x(source) ↦ Δx(target) — maps state from one timescale
    to perturbation on the next slower timescale.

    Regularized for:
      - Stable spectrum (spectral radius < 1)
      - Bounded curvature (Jacobian Frobenius norm)
      - Low-rank constraint (small intrinsic operator dimension)
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        max_rank: int = 10,
        spectral_radius_bound: float = 0.95,
    ) -> None:
        self._source_dim = source_dim
        self._target_dim = target_dim
        self._max_rank = max_rank
        self._sr_bound = spectral_radius_bound
        # Low-rank factorization: Φ = U @ V^T, where U∈R^{target×r}, V∈R^{source×r}
        self._U: Optional[np.ndarray] = None
        self._V: Optional[np.ndarray] = None
        self._rank: int = 0
        self._is_fitted = False

    def fit(
        self,
        source_states: np.ndarray,
        target_deltas: np.ndarray,
        regularization: float = 1.0,
    ) -> None:
        """Fit lifting operator from paired observations.

        Args:
            source_states: (N, source_dim) source timescale states.
            target_deltas:  (N, target_dim) target timescale changes.
            regularization: Ridge regularization strength.
        """
        n = source_states.shape[0]
        if n < 3:
            return

        # Ridge regression: Φ = (X^T X + λI)^{-1} X^T Y
        XtX = source_states.T @ source_states + regularization * np.eye(self._source_dim)
        XtY = source_states.T @ target_deltas

        try:
            Phi = np.linalg.solve(XtX, XtY).T  # (target_dim, source_dim)
        except np.linalg.LinAlgError:
            return

        # Low-rank truncation via SVD
        U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
        rank = min(self._max_rank, len(S), np.sum(S > 1e-10 * S[0]))
        rank = max(rank, 1)

        # Spectral radius enforcement
        S_truncated = S[:rank].copy()
        scale = self._sr_bound / max(S_truncated[0], 1e-15)
        if scale < 1.0:
            S_truncated *= scale

        self._U = U[:, :rank] * S_truncated[np.newaxis, :]
        self._V = Vt[:rank, :].T
        self._rank = int(rank)
        self._is_fitted = True

    def lift(self, source_state: np.ndarray) -> np.ndarray:
        """Apply lifting operator: Φ(x_source) → Δx_target."""
        if not self._is_fitted:
            return np.zeros(self._target_dim)
        return self._U @ (self._V.T @ source_state)

    @property
    def spectral_radius(self) -> float:
        """Spectral radius of the operator."""
        if not self._is_fitted:
            return 0.0
        Phi = self._U @ self._V.T
        eigvals = np.linalg.eigvals(Phi @ Phi.T)
        return float(np.sqrt(np.max(np.abs(eigvals))))

    @property
    def frobenius_norm(self) -> float:
        """Frobenius norm — bounded curvature proxy."""
        if not self._is_fitted:
            return 0.0
        return float(np.linalg.norm(self._U @ self._V.T, "fro"))

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def eigenspectrum(self) -> np.ndarray:
        """Singular values of the operator (eigenspectrum proxy)."""
        if not self._is_fitted:
            return np.array([])
        Phi = self._U @ self._V.T
        return np.linalg.svd(Phi, compute_uv=False)


class CrossTimescaleSystem:
    """Complete three-timescale system with lifting operators.

    Manages:
      Φ_S→M: ShockState → RegimeState perturbation
      Φ_M→L: RegimeState → FundamentalState perturbation
    """

    def __init__(
        self,
        shock_dim: int = 12,
        regime_dim: int = 16,
        fundamental_dim: int = 12,
        max_rank: int = 10,
        calendar_encoder=None,
    ) -> None:
        self._calendar_encoder = calendar_encoder

        if calendar_encoder is not None:
            from tensor.frequency_dependent_lifter import FrequencyDependentLifter
            self.phi_s_to_m = FrequencyDependentLifter(
                source_dim=shock_dim,
                target_dim=regime_dim,
                max_rank=max_rank,
            )
            self.phi_m_to_l = FrequencyDependentLifter(
                source_dim=regime_dim,
                target_dim=fundamental_dim,
                max_rank=max_rank,
            )
        else:
            self.phi_s_to_m = LiftingOperator(
                source_dim=shock_dim,
                target_dim=regime_dim,
                max_rank=max_rank,
            )
            self.phi_m_to_l = LiftingOperator(
                source_dim=regime_dim,
                target_dim=fundamental_dim,
                max_rank=max_rank,
            )

    def fit_s_to_m(
        self,
        shock_states: np.ndarray,
        regime_deltas: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit Φ_S→M from (shock_state, regime_change) pairs."""
        self.phi_s_to_m.fit(shock_states, regime_deltas, **kwargs)

    def fit_m_to_l(
        self,
        regime_states: np.ndarray,
        fundamental_deltas: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit Φ_M→L from (regime_state, fundamental_change) pairs."""
        self.phi_m_to_l.fit(regime_states, fundamental_deltas, **kwargs)

    def propagate_shock(
        self,
        shock: ShockState,
        regime: RegimeState,
        fundamental: FundamentalState,
        event_date: Optional[date] = None,
        ticker: Optional[str] = None,
    ) -> Tuple[RegimeState, FundamentalState]:
        """Propagate a shock through all timescales.

        shock → Δregime via Φ_S→M
        (regime + Δregime) → Δfundamental via Φ_M→L

        When event_date and ticker are provided and calendar_encoder is active,
        uses frequency-dependent lift_at() instead of static lift().
        """
        use_calendar = (
            self._calendar_encoder is not None
            and event_date is not None
        )

        if use_calendar:
            phase = self._calendar_encoder.encode(event_date, ticker)
            # Shock → Regime (calendar-aware)
            result_sm = self.phi_s_to_m.lift_at(shock.features, phase)
            delta_m = result_sm.delta
        else:
            # Shock → Regime (static)
            delta_m = self.phi_s_to_m.lift(shock.features)

        new_regime = RegimeState(
            features=regime.features + delta_m,
            regime_id=regime.regime_id,
            timestamp=shock.timestamp,
        )

        if use_calendar:
            # Regime → Fundamental (calendar-aware)
            result_ml = self.phi_m_to_l.lift_at(new_regime.features, phase)
            delta_l = result_ml.delta
        else:
            # Regime → Fundamental (static)
            delta_l = self.phi_m_to_l.lift(new_regime.features)

        new_fundamental = FundamentalState(
            features=fundamental.features + delta_l,
            timestamp=shock.timestamp,
        )

        return new_regime, new_fundamental


# ── CrossTimescaleLifter (calendar_lifter-aware wrapper) ──────────────────────

class CrossTimescaleLifter:
    """Cross-timescale lifting operator with optional calendar-aware lifter.

    Manages:
      Φ_S→M: ShockState features → RegimeState perturbation
      Φ_M→L: RegimeState features → FundamentalState perturbation

    When calendar_lifter is provided, uses FrequencyDependentLifter for
    both operators. Otherwise falls back to static LiftingOperator.

    Parameters:
        shock_dim: Dimension of shock (S) state features.
        regime_dim: Dimension of regime (M) state features.
        fundamental_dim: Dimension of fundamental (L) state features.
        max_rank: Maximum rank for lifting operators.
        calendar_lifter: Optional FrequencyDependentLifter for calendar-aware lifting.
            When provided, it is used for both Φ_S→M and Φ_M→L.
    """

    def __init__(
        self,
        shock_dim: int = 12,
        regime_dim: int = 16,
        fundamental_dim: int = 12,
        max_rank: int = 10,
        calendar_lifter: Optional["FrequencyDependentLifter"] = None,
    ) -> None:
        self._shock_dim = shock_dim
        self._regime_dim = regime_dim
        self._fundamental_dim = fundamental_dim
        self._calendar_lifter = calendar_lifter

        if calendar_lifter is not None:
            # Use the provided calendar_lifter for S→M
            # Create a separate one for M→L with correct dims
            from tensor.frequency_dependent_lifter import FrequencyDependentLifter
            self._phi_s_to_m = calendar_lifter
            self._phi_m_to_l = FrequencyDependentLifter(
                source_dim=regime_dim,
                target_dim=fundamental_dim,
                max_rank=max_rank,
            )
        else:
            self._phi_s_to_m = LiftingOperator(
                source_dim=shock_dim,
                target_dim=regime_dim,
                max_rank=max_rank,
            )
            self._phi_m_to_l = LiftingOperator(
                source_dim=regime_dim,
                target_dim=fundamental_dim,
                max_rank=max_rank,
            )

    def lift_shock_to_regime(
        self,
        shock_features: np.ndarray,
        event_date: Optional[date] = None,
        encoder=None,
    ) -> np.ndarray:
        """Lift shock features to regime perturbation.

        Args:
            shock_features: (shock_dim,) shock feature vector.
            event_date: Optional calendar date (used when calendar_lifter is set).
            encoder: Optional CalendarRegimeEncoder instance.

        Returns:
            (regime_dim,) perturbation vector Δx_M.
        """
        if self._calendar_lifter is not None and event_date is not None:
            return self._phi_s_to_m.lift_shock_to_regime(
                shock_features, event_date, encoder=encoder
            )
        return self._phi_s_to_m.lift(shock_features)

    def lift_regime_to_fundamental(
        self,
        regime_features: np.ndarray,
        event_date: Optional[date] = None,
        encoder=None,
    ) -> np.ndarray:
        """Lift regime features to fundamental perturbation.

        Args:
            regime_features: (regime_dim,) regime feature vector.
            event_date: Optional calendar date (used when calendar_lifter is set).
            encoder: Optional CalendarRegimeEncoder instance.

        Returns:
            (fundamental_dim,) perturbation vector Δx_L.
        """
        if self._calendar_lifter is not None and event_date is not None:
            return self._phi_m_to_l.lift_regime_to_fundamental(
                regime_features, event_date, encoder=encoder
            )
        return self._phi_m_to_l.lift(regime_features)

    @property
    def uses_calendar(self) -> bool:
        """True when a calendar_lifter is active."""
        return self._calendar_lifter is not None
