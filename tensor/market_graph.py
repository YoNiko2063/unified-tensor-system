"""Market graph: represents market as MNA-equivalent circuit.

Tickers           = nodes
Sector membership = capacitive edges (correlated momentum storage)
Correlation       = resistive edges (signal flow)
Sentiment score   = external current source u(t)
Price momentum    = node voltage v(t)
Volatility        = diffusion coefficient sigma
Regime            = Markov chain state index

Physical interpretation:
  High correlation = low resistance = fast signal propagation
  Sector rotation = current flowing between sector nodes
  Sentiment shock = impulsive current injection
  Volatility spike = increased diffusion
  Market regime shift = Markov chain transition
"""
import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

_ECEMATH_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

from core.matrix import MNASystem


@dataclass
class TickerNode:
    """A single ticker in the market graph."""
    symbol: str
    sector: str = ''
    price: float = 0.0
    momentum: float = 0.0
    volatility: float = 0.01
    sentiment: float = 0.0  # [-1, 1]


class MarketGraph:
    """Represents market as MNA-equivalent circuit graph."""

    def __init__(self):
        self.tickers: Dict[str, TickerNode] = {}
        self._node_ids: Dict[str, int] = {}
        self._correlations: Dict[Tuple[str, str], float] = {}
        self._sectors: Dict[str, List[str]] = {}
        # Regime parameters
        self._regime_Q: Optional[np.ndarray] = None
        self._current_regime: int = 0

    def add_ticker(self, symbol: str, sector: str = '',
                   price: float = 100.0, momentum: float = 0.0,
                   volatility: float = 0.01, sentiment: float = 0.0):
        """Add a ticker node."""
        self.tickers[symbol] = TickerNode(
            symbol=symbol, sector=sector, price=price,
            momentum=momentum, volatility=volatility, sentiment=sentiment,
        )
        if symbol not in self._node_ids:
            self._node_ids[symbol] = len(self._node_ids)
        if sector:
            self._sectors.setdefault(sector, []).append(symbol)

    def set_correlation(self, ticker_a: str, ticker_b: str, corr: float):
        """Set pairwise correlation between tickers."""
        key = tuple(sorted([ticker_a, ticker_b]))
        self._correlations[key] = np.clip(corr, -1.0, 1.0)

    def update_from_pipeline(self, pipeline_output: dict):
        """Ingest trading bot pipeline output into the graph.

        Expected keys:
          'tickers': list of {symbol, sector, price, momentum, volatility}
          'correlations': list of {ticker_a, ticker_b, correlation}
          'sentiment_scores': dict of {symbol: score}
          'sector_weights': dict of {sector: weight}
        """
        # Update tickers
        for t in pipeline_output.get('tickers', []):
            self.add_ticker(
                symbol=t['symbol'],
                sector=t.get('sector', ''),
                price=t.get('price', 100.0),
                momentum=t.get('momentum', 0.0),
                volatility=t.get('volatility', 0.01),
            )

        # Update correlations
        for c in pipeline_output.get('correlations', []):
            self.set_correlation(c['ticker_a'], c['ticker_b'], c['correlation'])

        # Update sentiment
        for symbol, score in pipeline_output.get('sentiment_scores', {}).items():
            if symbol in self.tickers:
                self.tickers[symbol].sentiment = float(np.clip(score, -1.0, 1.0))

    def to_mna(self) -> MNASystem:
        """Build MNA from current market state.

        Correlations → G matrix (resistive)
        Sector membership → C matrix (capacitive)
        """
        n = len(self.tickers)
        if n == 0:
            return MNASystem(
                C=np.zeros((1, 1)), G=np.zeros((1, 1)),
                n_nodes=1, n_branches=0, n_total=1,
                node_map={0: 0}, branch_map={}, branch_info=[],
            )

        G = np.zeros((n, n))
        C = np.zeros((n, n))

        # Correlations → conductance (high corr = low resistance)
        for (ta, tb), corr in self._correlations.items():
            i = self._node_ids.get(ta)
            j = self._node_ids.get(tb)
            if i is None or j is None or i == j:
                continue
            # Conductance proportional to |correlation|
            # Positive corr = positive conductance, negative = negative
            g = abs(corr) * 10.0  # Scale to reasonable conductance values
            G[i, i] += g
            G[j, j] += g
            G[i, j] -= g * np.sign(corr)
            G[j, i] -= g * np.sign(corr)

        # Sector membership → capacitive coupling
        for sector, members in self._sectors.items():
            for mi in range(len(members)):
                for mj in range(mi + 1, len(members)):
                    i = self._node_ids.get(members[mi])
                    j = self._node_ids.get(members[mj])
                    if i is None or j is None:
                        continue
                    c = 1.0  # Uniform sector coupling
                    C[i, i] += c
                    C[j, j] += c
                    C[i, j] -= c
                    C[j, i] -= c

        # Diagonal loading for stability
        diag_load = 1e-6
        G += diag_load * np.eye(n)
        C += diag_load * np.eye(n)

        # Add volatility to diagonal of G (higher vol = more dissipation)
        symbols_sorted = sorted(self.tickers.keys(), key=lambda s: self._node_ids[s])
        for sym in symbols_sorted:
            idx = self._node_ids[sym]
            vol = self.tickers[sym].volatility
            G[idx, idx] += vol * 100.0

        node_map = {i: i for i in range(n)}
        return MNASystem(
            C=C, G=G, n_nodes=n, n_branches=0, n_total=n,
            node_map=node_map, branch_map={}, branch_info=[],
        )

    def sentiment_injection(self, ticker: str, score: float) -> np.ndarray:
        """Convert sentiment score to u(t) current injection vector.

        score in [-1, 1]: positive = bullish (current in), negative = bearish.
        Returns (n,) vector with injection at ticker's node.
        """
        n = len(self.tickers)
        u = np.zeros(n)
        idx = self._node_ids.get(ticker)
        if idx is not None:
            u[idx] = score  # Direct injection
        return u

    def momentum_vector(self) -> np.ndarray:
        """State vector v from price momentum."""
        n = len(self.tickers)
        v = np.zeros(n)
        for sym, node in self.tickers.items():
            idx = self._node_ids[sym]
            v[idx] = node.momentum
        return v

    def regime_detection(self) -> Tuple[int, float]:
        """Detect current market regime from volatility distribution.

        Regime 0 = low volatility (stable)
        Regime 1 = high volatility (transitioning)
        Regime 2 = crisis (bifurcating)

        Returns (regime_index, confidence).
        """
        if not self.tickers:
            return 0, 1.0

        vols = [t.volatility for t in self.tickers.values()]
        mean_vol = np.mean(vols)

        # Simple threshold-based regime detection
        if mean_vol < 0.02:
            regime = 0
            conf = 1.0 - mean_vol / 0.02
        elif mean_vol < 0.05:
            regime = 1
            conf = 1.0 - abs(mean_vol - 0.035) / 0.015
        else:
            regime = 2
            conf = min(1.0, mean_vol / 0.05 - 1.0)

        return regime, float(np.clip(conf, 0.1, 1.0))

    def setup_regime_transitions(self, Q: np.ndarray):
        """Set Markov chain transition rate matrix for regime switching."""
        self._regime_Q = Q.copy()

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)

    @property
    def node_ids(self) -> Dict[str, int]:
        return dict(self._node_ids)
