"""Multi-Horizon Mixer — geometric gating across three timeframes.

Combines per-timeframe predictions via confidence-weighted softmax:

  r̂(H) = w_L · r̂_L + w_M · r̂_M + w_S · r̂_S

  w_k(t) = softmax(α·confidence_k - β·ρ_k + γ·spectral_gap_k)

Gating behavior:
  - Stable fundamentals patch  → w_L dominates
  - Regime instability          → w_M dominates
  - Fresh news shock            → w_S dominates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from tensor.timescale_state import (
    CrossTimescaleSystem,
    FundamentalState,
    RegimeState,
    ShockState,
)


@dataclass
class TimeframePrediction:
    """Prediction from a single timeframe."""
    return_estimate: float = 0.0
    confidence: float = 0.5
    linearity_score: float = 0.5  # ρ_k — lower = more linear/trustworthy
    spectral_gap: float = 0.0     # Koopman spectral gap
    label: str = ""


@dataclass
class MixedPrediction:
    """Combined multi-horizon prediction."""
    blended_return: float = 0.0
    weights: np.ndarray = field(default_factory=lambda: np.array([1/3, 1/3, 1/3]))
    predictions: Dict[str, TimeframePrediction] = field(default_factory=dict)
    dominant_timeframe: str = ""
    resonance_flag: bool = False


class PerTimeframePredictor:
    """Base class for single-timeframe return prediction.

    Subclass and implement predict() for each timeframe.
    """

    def predict(self, state: np.ndarray) -> TimeframePrediction:
        """Predict return from state vector."""
        raise NotImplementedError


class FundamentalPredictor(PerTimeframePredictor):
    """Long-horizon return prediction from fundamentals."""

    def __init__(self, weights: Optional[np.ndarray] = None) -> None:
        # Default: value composite weights
        self._weights = weights if weights is not None else np.array([
            0.15,  # revenue_growth
            0.10,  # gross_margin
            0.10,  # operating_margin
            0.15,  # fcf_yield
            -0.10, # debt_to_equity (negative = high debt bad)
            -0.10, # pe_ratio (negative = high PE risky)
            0.05,  # pb_ratio
            0.15,  # roic
            0.05,  # sector_gdp_beta
            0.05,  # sector_rate_beta
            0.05,  # sector_inflation_beta
            0.15,  # quality_score
        ])

    def predict(self, state: np.ndarray) -> TimeframePrediction:
        """Value-composite fundamental return estimate."""
        raw = float(self._weights[:len(state)] @ state[:len(self._weights)])
        # Sigmoid confidence from absolute signal strength
        conf = float(1.0 / (1.0 + np.exp(-2.0 * abs(raw))))
        return TimeframePrediction(
            return_estimate=raw,
            confidence=conf,
            linearity_score=float(np.var(state) / (np.max(np.abs(state)) + 1e-10)),
            label="fundamental",
        )


class RegimePredictor(PerTimeframePredictor):
    """Medium-horizon return prediction from regime/technical state."""

    def predict(self, state: np.ndarray) -> TimeframePrediction:
        """Momentum + mean-reversion blend."""
        trend = state[2] if len(state) > 2 else 0.0
        vol = state[0] if len(state) > 0 else 0.1
        rsi = state[10] if len(state) > 10 else 50.0

        # Trend following + RSI mean reversion
        momentum_signal = trend * 0.5
        mean_reversion = (50.0 - rsi) / 100.0 * 0.3

        raw = momentum_signal + mean_reversion
        conf = float(1.0 / (1.0 + 2.0 * vol))  # high vol → low confidence

        return TimeframePrediction(
            return_estimate=float(raw),
            confidence=conf,
            linearity_score=float(min(vol * 10, 1.0)),  # high vol → high ρ
            label="regime",
        )


class ShockPredictor(PerTimeframePredictor):
    """Short-horizon return prediction from news/shock state."""

    def predict(self, state: np.ndarray) -> TimeframePrediction:
        """Event-driven return estimate with decay weighting."""
        sentiment = state[0] if len(state) > 0 else 0.0
        confidence = state[1] if len(state) > 1 else 0.5
        source_trust = state[6] if len(state) > 6 else 0.5
        epistemic_validity = state[7] if len(state) > 7 else 0.0
        decay_factor = state[10] if len(state) > 10 else 1.0

        raw = sentiment * confidence * source_trust * decay_factor
        # Trust-adjusted confidence
        conf = float(confidence * source_trust * max(epistemic_validity + 0.5, 0.0))
        conf = min(conf, 1.0)

        return TimeframePrediction(
            return_estimate=float(raw),
            confidence=conf,
            linearity_score=0.8,  # shocks are inherently nonlinear
            spectral_gap=float(abs(sentiment) * decay_factor),
            label="shock",
        )


class MultiHorizonMixer:
    """Geometric gating mixer for three-timeframe predictions.

    w_k(t) = softmax(α·confidence_k - β·ρ_k + γ·spectral_gap_k)

    Parameters:
      alpha: confidence weight (higher = trust high-confidence predictors)
      beta:  linearity penalty (higher = penalize nonlinear regimes)
      gamma: spectral gap bonus (higher = reward clear eigenstructure)
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        fundamental_predictor: Optional[PerTimeframePredictor] = None,
        regime_predictor: Optional[PerTimeframePredictor] = None,
        shock_predictor: Optional[PerTimeframePredictor] = None,
        calendar_alpha_modulation: Optional[np.ndarray] = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._pred_l = fundamental_predictor or FundamentalPredictor()
        self._pred_m = regime_predictor or RegimePredictor()
        self._pred_s = shock_predictor or ShockPredictor()
        self._calendar_mod = calendar_alpha_modulation

    def mix(
        self,
        fundamental: FundamentalState,
        regime: RegimeState,
        shock: ShockState,
        calendar_phase=None,
    ) -> MixedPrediction:
        """Compute blended prediction with geometric gating.

        Args:
            fundamental: Long-timescale state.
            regime: Medium-timescale state.
            shock: Short-timescale state.
            calendar_phase: Optional CalendarPhase for calendar-modulated gating.

        Returns MixedPrediction with per-timeframe predictions and weights.
        """
        # Per-timeframe predictions
        pred_l = self._pred_l.predict(fundamental.features)
        pred_m = self._pred_m.predict(regime.features)
        pred_s = self._pred_s.predict(shock.features)

        predictions = {"fundamental": pred_l, "regime": pred_m, "shock": pred_s}

        # Effective alpha: modulate by calendar if available
        alpha_eff = self.alpha
        resonance_flag = False
        if calendar_phase is not None and self._calendar_mod is not None:
            from tensor.frequency_dependent_lifter import von_mises_vector, detect_resonance
            basis_weights = von_mises_vector(calendar_phase)
            alpha_adjustment = float(self._calendar_mod @ basis_weights)
            alpha_eff = self.alpha + alpha_adjustment

            # Check resonance
            report = detect_resonance(calendar_phase)
            resonance_flag = report.is_resonant

        # Geometric gating logits
        logits = np.array([
            alpha_eff * pred_l.confidence - self.beta * pred_l.linearity_score + self.gamma * pred_l.spectral_gap,
            alpha_eff * pred_m.confidence - self.beta * pred_m.linearity_score + self.gamma * pred_m.spectral_gap,
            alpha_eff * pred_s.confidence - self.beta * pred_s.linearity_score + self.gamma * pred_s.spectral_gap,
        ])

        # Softmax
        logits -= logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        weights = exp_logits / exp_logits.sum()

        # Blended return
        returns = np.array([pred_l.return_estimate, pred_m.return_estimate, pred_s.return_estimate])
        blended = float(weights @ returns)

        # Dominant timeframe
        labels = ["fundamental", "regime", "shock"]
        dominant = labels[int(np.argmax(weights))]

        return MixedPrediction(
            blended_return=blended,
            weights=weights,
            predictions=predictions,
            dominant_timeframe=dominant,
            resonance_flag=resonance_flag,
        )

    def mix_from_states(
        self,
        fundamental_features: np.ndarray,
        regime_features: np.ndarray,
        shock_features: np.ndarray,
    ) -> MixedPrediction:
        """Convenience: mix from raw feature arrays."""
        return self.mix(
            FundamentalState(features=fundamental_features),
            RegimeState(features=regime_features),
            ShockState(features=shock_features),
        )
