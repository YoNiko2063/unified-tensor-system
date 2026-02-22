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
    confidence: float = 1.0  # reduced when Arnold tongue resonance is active


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


# Default (5, 3) calendar modulation matrix.
# Rows: [earnings, fed, options_expiry, rebalance, holiday]
# Cols: [logit_L (fundamental), logit_M (regime), logit_S (shock)]
DEFAULT_CALENDAR_MODULATION = np.array([
    [-0.5,  0.3,  0.8],   # earnings: suppress L, boost M, strongly boost S
    [-0.3,  0.5,  0.4],   # fed: suppress L, boost M+S equally
    [ 0.0,  0.3,  0.2],   # options expiry: mild M+S boost
    [ 0.2,  0.0, -0.2],   # rebalance: mild L boost, suppress S
    [ 0.3, -0.2, -0.3],   # holiday: boost L (stable), suppress M+S
])


class MultiHorizonMixer:
    """Geometric gating mixer for three-timeframe predictions.

    w_k(t) = softmax(α·confidence_k - β·ρ_k + γ·spectral_gap_k
                     + (calendar_alpha_modulation.T @ phase_vec)_k)

    The calendar_alpha_modulation is a (5, 3) matrix mapping the 5-channel
    CalendarPhase amplitude vector to per-timeframe logit adjustments:

        delta_logits = calendar_alpha_modulation.T @ phase_vec   # (3,)
        adjusted_logits = logits + delta_logits
        weights = softmax(adjusted_logits)

    Unlike the old scalar approach where softmax(x + c) == softmax(x) for
    scalar c, the matrix form shifts each timeframe logit independently,
    allowing weights to change with calendar state.

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

        # Resolve calendar modulation matrix.
        # Accepts:
        #   None        → use DEFAULT_CALENDAR_MODULATION (5, 3)
        #   (5, 3) ndarray → used directly as matrix modulation
        #   (5,) vector → treated as OLD scalar API (broadcast: each col is
        #                 the same vector, i.e., col = vec).  Kept for backward
        #                 compat; callers should migrate to (5, 3).
        if calendar_alpha_modulation is None:
            self._calendar_mod: Optional[np.ndarray] = None  # no-modulation (disabled)
        else:
            mod = np.asarray(calendar_alpha_modulation, dtype=float)
            if mod.ndim == 1 and mod.shape == (5,):
                # Legacy scalar-per-channel API: treat as (5, 1) broadcast → (5, 3)
                # Each timeframe gets the same per-channel scalar shift.
                self._calendar_mod = np.column_stack([mod, mod, mod])  # (5, 3)
            elif mod.ndim == 2 and mod.shape == (5, 3):
                self._calendar_mod = mod
            else:
                raise ValueError(
                    f"calendar_alpha_modulation must be None, shape (5,) or (5, 3), "
                    f"got shape {mod.shape}"
                )

    def mix(
        self,
        fundamental: FundamentalState,
        regime: RegimeState,
        shock: ShockState,
        calendar_phase: Optional["CalendarPhase"] = None,
        regime_vector: Optional[np.ndarray] = None,
    ) -> MixedPrediction:
        """Compute blended prediction with geometric gating.

        The calendar_alpha_modulation (5, 3) matrix shifts per-timeframe logits
        independently based on the 5-channel calendar phase amplitude vector:

            phase_vec = [earnings_amp, fed_amp, options_amp, rebalance_amp, holiday_amp]
            delta_logits = calendar_alpha_modulation.T @ phase_vec   # (3,)
            adjusted_logits = base_logits + delta_logits
            weights = softmax(adjusted_logits)

        This differs from the old scalar approach (softmax(x + c) = softmax(x))
        because delta_logits has different components per timeframe.

        Args:
            fundamental: Long-timescale state.
            regime: Medium-timescale state.
            shock: Short-timescale state.
            calendar_phase: Optional CalendarPhase for calendar-modulated gating.
                When provided, the (5,3) modulation matrix shifts per-timeframe
                logits; Arnold tongue resonance detection also runs.

        Returns MixedPrediction with per-timeframe predictions, weights,
            resonance_flag, and confidence (reduced under resonance).
        """
        from tensor.frequency_dependent_lifter import detect_resonance

        # Per-timeframe predictions
        pred_l = self._pred_l.predict(fundamental.features)
        pred_m = self._pred_m.predict(regime.features)
        pred_s = self._pred_s.predict(shock.features)

        predictions = {"fundamental": pred_l, "regime": pred_m, "shock": pred_s}

        # Resonance detection (always run when calendar_phase is provided)
        resonance_flag = False
        resonance_info: dict = {}
        if calendar_phase is not None:
            report = detect_resonance(calendar_phase)
            resonance_flag = report.is_resonant
            resonance_info = {
                "active_tongues": report.resonant_pairs,
                "n_tongues": len(report.resonant_pairs),
            }

        # Base logits
        logits = np.array([
            self.alpha * pred_l.confidence - self.beta * pred_l.linearity_score + self.gamma * pred_l.spectral_gap,
            self.alpha * pred_m.confidence - self.beta * pred_m.linearity_score + self.gamma * pred_m.spectral_gap,
            self.alpha * pred_s.confidence - self.beta * pred_s.linearity_score + self.gamma * pred_s.spectral_gap,
        ])

        # Calendar matrix modulation (Gap 3 fix).
        # The (5, 3) matrix maps 5-channel amplitude vector → 3 per-timeframe
        # logit deltas, so softmax can actually shift the weight distribution.
        if calendar_phase is not None and self._calendar_mod is not None:
            # phase_vec: (5,) calendar amplitude vector
            phase_vec = np.array([
                calendar_phase.amplitudes[0],  # earnings
                calendar_phase.amplitudes[1],  # fed
                calendar_phase.amplitudes[2],  # options_expiry
                calendar_phase.amplitudes[3],  # rebalance
                calendar_phase.amplitudes[4],  # holiday
            ])
            # delta_logits: (3,) — different shift for each timeframe
            delta_logits = self._calendar_mod.T @ phase_vec  # (5,3).T @ (5,) → (3,)
            logits = logits + delta_logits

        # Regime vector modulation (from ReharmonizationTracker Layer 4)
        # Maps (4,) soft Duffing regime probabilities to per-timeframe logit shifts
        if regime_vector is not None:
            rv = np.asarray(regime_vector, dtype=float)
            if rv.shape == (4,):
                REGIME_TO_LOGIT = np.array([
                    [ 0.0,  0.5, -0.3],  # mean_rev: boost M
                    [-0.3,  0.5,  0.3],  # breakout: boost M+S
                    [-0.2,  0.0,  0.5],  # momentum: boost S
                    [ 0.3,  0.3, -0.2],  # anticipatory: boost L+M
                ])
                regime_delta = rv @ REGIME_TO_LOGIT  # (4,) @ (4,3) -> (3,)
                logits = logits + regime_delta

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

        # Gap 7: resonance penalty — Arnold tongue resonance widens uncertainty
        result_confidence = 1.0
        if resonance_flag:
            n_tongues = len(resonance_info.get("active_tongues", []))
            resonance_penalty = 0.15 * n_tongues  # 15% vol multiplier per active tongue
            blended *= (1.0 + resonance_penalty)
            result_confidence = max(0.3, 1.0 - 0.1 * n_tongues)

        return MixedPrediction(
            blended_return=blended,
            weights=weights,
            predictions=predictions,
            dominant_timeframe=dominant,
            resonance_flag=resonance_flag,
            confidence=result_confidence,
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
