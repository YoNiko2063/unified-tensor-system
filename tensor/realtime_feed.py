"""Real-time data feed: streams market data into tensor L0.

Supports:
  - Yahoo Finance WebSocket (wss://streamer.finance.yahoo.com/)
  - Coinbase Advanced Trade WebSocket (wss://advanced-trade-ws.coinbase.com)
  - Mock feed for testing without network

Runs as background daemon threads. Updates MarketGraph on each tick.
"""
import os
import sys
import json
import time
import threading
import logging
from typing import Dict, List, Optional, Callable

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tensor.core import UnifiedTensor
from tensor.market_graph import MarketGraph

logger = logging.getLogger(__name__)


class RealtimeFeed:
    """Streams real-time market data into tensor L0.

    Runs as background threads. Updates MarketGraph on each tick.
    Triggers tensor.update_level(0, ...) on each update.
    """

    def __init__(self, tensor: UnifiedTensor,
                 market_graph: Optional[MarketGraph] = None,
                 sources: Optional[List[str]] = None,
                 update_interval: float = 5.0):
        self.tensor = tensor
        self.market_graph = market_graph or MarketGraph.mock_live(n_tickers=10)
        self.sources = sources or ['mock']
        self.update_interval = update_interval

        self._threads: Dict[str, threading.Thread] = {}
        self._running = False
        self._ticks_received = 0
        self._last_update = 0.0
        self._lock = threading.Lock()
        self._connected_sources: List[str] = []
        self._callbacks: List[Callable] = []

    def _update_l0(self):
        """Push current market graph to tensor L0."""
        with self._lock:
            mna = self.market_graph.to_mna()
            self.tensor.update_level(0, mna, t=time.time())
            self._last_update = time.time()
            self._ticks_received += 1

        for cb in self._callbacks:
            try:
                cb(self.market_graph)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def on_tick(self, callback: Callable):
        """Register callback for each market tick."""
        self._callbacks.append(callback)

    def connect_yahoo(self):
        """Connect to Yahoo Finance WebSocket.

        Subscribes to: SPY, QQQ, sector ETFs.
        On each tick: update market_graph node for that ticker.
        """
        try:
            import websocket
        except ImportError:
            logger.warning("websocket-client not installed, using mock yahoo feed")
            self._run_mock_yahoo()
            return

        url = 'wss://streamer.finance.yahoo.com/'
        tickers = ['SPY', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV',
                    'XLC', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE']

        def on_message(ws, message):
            try:
                data = json.loads(message)
                symbol = data.get('id', '')
                price = data.get('price', 0.0)
                if symbol and symbol in self.market_graph.tickers:
                    node = self.market_graph.tickers[symbol]
                    if node.price > 0:
                        node.momentum = (price - node.price) / node.price
                    node.price = price
                    self._update_l0()
            except (json.JSONDecodeError, KeyError):
                pass

        def on_open(ws):
            sub = json.dumps({'subscribe': tickers})
            ws.send(sub)
            with self._lock:
                self._connected_sources.append('yahoo')
            logger.info("Yahoo Finance WebSocket connected")

        def on_error(ws, error):
            logger.warning(f"Yahoo WS error: {error}")

        def run():
            while self._running:
                try:
                    ws = websocket.WebSocketApp(
                        url, on_message=on_message,
                        on_open=on_open, on_error=on_error)
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    logger.warning(f"Yahoo reconnecting: {e}")
                    time.sleep(5)

        t = threading.Thread(target=run, daemon=True, name='yahoo-feed')
        self._threads['yahoo'] = t

    def connect_coinbase(self):
        """Connect to Coinbase Advanced Trade WebSocket.

        Subscribes to BTC-USD, ETH-USD level2.
        Derives sentiment from order imbalance.
        """
        try:
            import websocket
        except ImportError:
            logger.warning("websocket-client not installed, using mock coinbase feed")
            self._run_mock_coinbase()
            return

        url = 'wss://advanced-trade-ws.coinbase.com'

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get('channel') == 'l2_data':
                    events = data.get('events', [])
                    for event in events:
                        product_id = event.get('product_id', '')
                        ticker = product_id.replace('-', '')
                        updates = event.get('updates', [])
                        bid_vol = sum(float(u.get('qty', 0))
                                      for u in updates if u.get('side') == 'bid')
                        ask_vol = sum(float(u.get('qty', 0))
                                      for u in updates if u.get('side') == 'offer')
                        total = bid_vol + ask_vol
                        if total > 0 and ticker in self.market_graph.tickers:
                            imbalance = (bid_vol - ask_vol) / total
                            self.market_graph.tickers[ticker].sentiment = float(imbalance)
                            self._update_l0()
            except (json.JSONDecodeError, KeyError):
                pass

        def on_open(ws):
            sub = json.dumps({
                'type': 'subscribe',
                'product_ids': ['BTC-USD', 'ETH-USD'],
                'channel': 'level2',
            })
            ws.send(sub)
            with self._lock:
                self._connected_sources.append('coinbase')
            logger.info("Coinbase WebSocket connected")

        def on_error(ws, error):
            logger.warning(f"Coinbase WS error: {error}")

        def run():
            while self._running:
                try:
                    ws = websocket.WebSocketApp(
                        url, on_message=on_message,
                        on_open=on_open, on_error=on_error)
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    logger.warning(f"Coinbase reconnecting: {e}")
                    time.sleep(5)

        t = threading.Thread(target=run, daemon=True, name='coinbase-feed')
        self._threads['coinbase'] = t

    def _run_mock_yahoo(self):
        """Mock Yahoo feed for testing."""
        import numpy as np

        def run():
            rng = np.random.default_rng(42)
            symbols = list(self.market_graph.tickers.keys())
            with self._lock:
                self._connected_sources.append('yahoo_mock')

            while self._running:
                sym = symbols[rng.integers(len(symbols))]
                node = self.market_graph.tickers[sym]
                delta = rng.standard_normal() * 0.001
                node.momentum = delta
                node.price *= (1 + delta)
                self._update_l0()
                time.sleep(self.update_interval)

        t = threading.Thread(target=run, daemon=True, name='yahoo-mock')
        self._threads['yahoo_mock'] = t

    def _run_mock_coinbase(self):
        """Mock Coinbase feed for testing."""
        import numpy as np

        def run():
            rng = np.random.default_rng(123)
            with self._lock:
                self._connected_sources.append('coinbase_mock')

            while self._running:
                for sym in list(self.market_graph.tickers.keys())[:2]:
                    node = self.market_graph.tickers[sym]
                    node.sentiment = float(rng.standard_normal() * 0.3)
                self._update_l0()
                time.sleep(self.update_interval)

        t = threading.Thread(target=run, daemon=True, name='coinbase-mock')
        self._threads['coinbase_mock'] = t

    def connect_mock(self):
        """Start mock feeds for both Yahoo and Coinbase."""
        self._run_mock_yahoo()
        self._run_mock_coinbase()

    def start(self):
        """Start all feeds as daemon threads."""
        self._running = True

        for source in self.sources:
            if source == 'yahoo':
                self.connect_yahoo()
            elif source == 'coinbase':
                self.connect_coinbase()
            elif source == 'mock':
                self.connect_mock()

        for t in self._threads.values():
            t.start()

        # Initial L0 update
        self._update_l0()

    def stop(self):
        """Stop all feeds."""
        self._running = False
        for t in self._threads.values():
            t.join(timeout=2)
        self._threads.clear()

    def status(self) -> dict:
        """Current feed status."""
        regime, _ = self.market_graph.regime_detection()
        return {
            'connected_sources': list(self._connected_sources),
            'ticks_received': self._ticks_received,
            'last_update': self._last_update,
            'l0_node_count': self.market_graph.n_tickers,
            'current_regime': regime,
            'running': self._running,
        }

    def inject_tick(self, symbol: str, price: float = 0.0,
                    momentum: float = 0.0, sentiment: float = 0.0):
        """Manually inject a tick for testing."""
        if symbol in self.market_graph.tickers:
            node = self.market_graph.tickers[symbol]
            if price > 0:
                if node.price > 0:
                    node.momentum = (price - node.price) / node.price
                node.price = price
            if momentum != 0:
                node.momentum = momentum
            if sentiment != 0:
                node.sentiment = sentiment
            self._update_l0()
