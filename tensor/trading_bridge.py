"""Bridge between UnifiedTensor and trading bot pipeline.

Tensor-enhanced predictions feed back into trading decisions.
Registers as pipeline step 'tensor_score_enhancement'.
"""
import sys
import os
import numpy as np
from typing import Dict, Optional

_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tensor.core import UnifiedTensor
from tensor.market_graph import MarketGraph
from core.sparse_solver import compute_harmonic_signature, HarmonicSignature


class TradingBridge:
    """Connects UnifiedTensor to trading bot pipeline."""

    def __init__(self, tensor: UnifiedTensor, market_graph: MarketGraph,
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.tensor = tensor
        self.mg = market_graph
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _tensor_signal(self, ticker: str) -> float:
        """Free energy at ticker's L0 node (high tension = high signal)."""
        fe = self.tensor.free_energy_map(0)
        idx = self.mg.node_ids.get(ticker)
        if idx is not None and idx < len(fe):
            return float(fe[idx])
        return 0.0

    def enhance_scores(self, pipeline_state: dict) -> dict:
        """Enhance FinBERT sentiment scores with tensor intelligence.

        Returns dict with enhanced scores, signals, and signature.
        """
        finbert = pipeline_state.get('sentiment_scores', {})
        regime, regime_conf = self.mg.regime_detection()
        regime_dir = {0: 1.0, 1: 0.0, 2: -1.0}.get(regime, 0.0)

        gap = self.tensor.eigenvalue_gap(0)
        sig = self.tensor.harmonic_signature(0)

        cross_res = 0.5
        try:
            if self.tensor._mna.get(2) is not None:
                cross_res = self.tensor.cross_level_resonance(0, 2)
        except Exception:
            pass

        enhanced: Dict[str, float] = {}
        for ticker, fb_score in finbert.items():
            ts = self._tensor_signal(ticker)
            # Normalize tensor signal to [-1,1]
            ts_norm = np.tanh(ts)
            ra = regime_dir * regime_conf

            score = self.alpha * fb_score + self.beta * ts_norm + self.gamma * ra
            enhanced[ticker] = float(np.clip(score, -1.0, 1.0))

        # Position size modifier
        position_modifier = 1.0
        if gap < 0.2:
            position_modifier = 0.5  # Reduce exposure near regime shift
        confidence_window = cross_res > 0.9

        return {
            'enhanced_scores': enhanced,
            'original_scores': dict(finbert),
            'position_modifier': position_modifier,
            'high_confidence_window': confidence_window,
            'eigenvalue_gap': round(gap, 4),
            'cross_level_resonance': round(cross_res, 4),
            'harmonic_signature': {
                'dominant_interval': sig.dominant_interval,
                'consonance_score': round(sig.consonance_score, 4),
                'stability_verdict': sig.stability_verdict,
            },
        }

    def regime_signal(self) -> dict:
        """Current market regime from tensor perspective."""
        regime, conf = self.mg.regime_detection()
        risk = self.tensor.phase_transition_risk(0)
        sig = self.tensor.harmonic_signature(0)

        if risk > 0.8:
            rec = 'exit'
        elif risk > 0.6:
            rec = 'reduce'
        elif sig.consonance_score > 0.7:
            rec = 'increase'
        else:
            rec = 'hold'

        return {
            'regime': regime,
            'confidence': round(float(conf), 4),
            'harmonic_key': sig.dominant_interval,
            'phase_risk': round(float(risk), 4),
            'recommendation': rec,
            'consonance': round(sig.consonance_score, 4),
        }
