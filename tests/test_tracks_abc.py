"""Tests for Tracks A, B, C: NeuralBridge, MarketGraph extensions, ScraperBridge.

Run: python tests/test_tracks_abc.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))
sys.path.insert(0, _ROOT)

import numpy as np


# ═══════════════════════════════════════════════════════════
# A1: NeuralBridge builds L1 MNA and updates tensor
# ═══════════════════════════════════════════════════════════

def test_A1_neural_bridge_builds_l1():
    from tensor.core import UnifiedTensor
    from tensor.neural_bridge import NeuralBridge

    T = UnifiedTensor(max_nodes=128, n_levels=4)
    nb = NeuralBridge(T, n_neurons=16, seed=42)

    # Build MNA
    mna = nb.to_mna()
    assert mna.n_total == 16
    assert mna.C.shape == (16, 16)
    assert mna.G.shape == (16, 16)

    # Update tensor
    nb.update_tensor(t=0.0)
    assert T._mna[1] is not None
    assert T._mna[1].n_total == 16
    print(f"  L1 MNA: {mna.n_total} neurons, G diag sum={np.trace(mna.G):.2f}")
    print(f"  PASS: A1")


# ═══════════════════════════════════════════════════════════
# A2: Forward pass produces finite state vector
# ═══════════════════════════════════════════════════════════

def test_A2_forward_pass():
    from tensor.core import UnifiedTensor
    from tensor.neural_bridge import NeuralBridge

    T = UnifiedTensor(max_nodes=128, n_levels=4)
    nb = NeuralBridge(T, n_neurons=16, seed=42)

    v0 = np.ones(16) * 0.1
    v_final = nb.forward(v0=v0, dt=0.01, steps=20)

    assert v_final.shape == (16,)
    assert np.all(np.isfinite(v_final))
    assert not np.allclose(v_final, v0), "State should evolve"
    print(f"  v0 norm={np.linalg.norm(v0):.4f}, v_final norm={np.linalg.norm(v_final):.4f}")
    print(f"  PASS: A2")


# ═══════════════════════════════════════════════════════════
# A3: Coarsen chain L2→L1→L0 and lift chain L0→L1→L2
# ═══════════════════════════════════════════════════════════

def test_A3_coarsen_lift_chain():
    from tensor.core import UnifiedTensor
    from tensor.neural_bridge import NeuralBridge
    from tensor.market_graph import MarketGraph
    from tensor.code_graph import CodeGraph

    T = UnifiedTensor(max_nodes=128, n_levels=4)

    # L0: Market
    mg = MarketGraph.mock_live(n_tickers=8, seed=42)
    mna_market = mg.to_mna()
    T.update_level(0, mna_market, t=0.0)

    # L1: Neural
    nb = NeuralBridge(T, n_neurons=16, seed=42)
    nb.update_tensor(t=0.0)

    # L2: Code
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')
    cg = CodeGraph.from_directory(dev_path, max_files=50)
    mna_code = cg.to_mna()
    T.update_level(2, mna_code, t=0.0)

    # Coarsen L1→L0 and lift back (before L2→L1 overwrites L1)
    result_10 = T.coarsen_to(1, 0)
    k_coarse = result_10.projection.shape[1]
    x_coarse = np.ones(k_coarse) * 0.1
    x_L1_lifted = T.lift_from(0, 1, x_coarse)
    assert np.all(np.isfinite(x_L1_lifted))
    assert len(x_L1_lifted) == 16  # back to L1 neuron count
    print(f"  Lift L0({k_coarse})→L1({len(x_L1_lifted)})")

    # Re-populate L1 (coarsen_to overwrote it via update_level)
    nb.update_tensor(t=0.1)

    # Coarsen chain L2→L1
    results = nb.coarsen_chain([2, 1])
    assert len(results) == 1
    assert results[0].mna_coarse is not None
    print(f"  Coarsen L2({mna_code.n_total})→L1: {results[0].mna_coarse.n_total} nodes")
    print(f"  PASS: A3")


# ═══════════════════════════════════════════════════════════
# B1: from_trading_pipeline builds valid MarketGraph
# ═══════════════════════════════════════════════════════════

def test_B1_from_trading_pipeline():
    from tensor.market_graph import MarketGraph

    pipeline_out = {
        'tickers': [
            {'symbol': 'AAPL', 'sector': 'tech', 'price': 180, 'momentum': 0.02, 'volatility': 0.015},
            {'symbol': 'MSFT', 'sector': 'tech', 'price': 420, 'momentum': 0.01, 'volatility': 0.012},
            {'symbol': 'JPM', 'sector': 'fin', 'price': 200, 'momentum': 0.008, 'volatility': 0.02},
        ],
        'correlations': [
            {'ticker_a': 'AAPL', 'ticker_b': 'MSFT', 'correlation': 0.85},
        ],
        'sentiment_scores': {'AAPL': 0.7, 'MSFT': 0.3, 'JPM': -0.2},
        'articles': [
            {'ticker': 'AAPL', 'sentiment': 0.5},
            {'ticker': 'AAPL', 'sentiment': 0.9},
        ],
    }

    mg = MarketGraph.from_trading_pipeline(pipeline_out)
    assert mg.n_tickers == 3
    assert 'AAPL' in mg.tickers
    assert 'MSFT' in mg.tickers
    assert 'JPM' in mg.tickers

    mna = mg.to_mna()
    assert mna.n_total == 3
    assert mna.G.shape == (3, 3)

    # Article sentiment should have overridden sentiment_scores for AAPL
    # (0.5 + 0.9) / 2 = 0.7
    assert abs(mg.tickers['AAPL'].sentiment - 0.7) < 0.01

    print(f"  from_trading_pipeline: {mg.n_tickers} tickers, MNA {mna.n_total}x{mna.n_total}")
    print(f"  PASS: B1")


# ═══════════════════════════════════════════════════════════
# B2: mock_live is deterministic and produces valid graph
# ═══════════════════════════════════════════════════════════

def test_B2_mock_live():
    from tensor.market_graph import MarketGraph

    mg1 = MarketGraph.mock_live(n_tickers=10, seed=42)
    mg2 = MarketGraph.mock_live(n_tickers=10, seed=42)

    # Deterministic
    assert mg1.n_tickers == mg2.n_tickers == 10
    for sym in mg1.tickers:
        assert sym in mg2.tickers
        assert mg1.tickers[sym].price == mg2.tickers[sym].price
        assert mg1.tickers[sym].momentum == mg2.tickers[sym].momentum

    # Valid MNA
    mna = mg1.to_mna()
    assert mna.n_total == 10
    assert np.all(np.isfinite(mna.G))
    assert np.all(np.isfinite(mna.C))

    # Has correlations
    assert len(mg1._correlations) > 0

    print(f"  mock_live: {mg1.n_tickers} tickers, {len(mg1._correlations)} correlations")
    print(f"  PASS: B2")


# ═══════════════════════════════════════════════════════════
# B3: mock_live → to_mna → tensor update works
# ═══════════════════════════════════════════════════════════

def test_B3_mock_live_tensor():
    from tensor.core import UnifiedTensor
    from tensor.market_graph import MarketGraph

    T = UnifiedTensor(max_nodes=128, n_levels=4)
    mg = MarketGraph.mock_live(n_tickers=10, seed=42)
    mna = mg.to_mna()
    T.update_level(0, mna, t=0.0)

    assert T._mna[0] is not None
    assert T._mna[0].n_total == 10

    # Harmonic signature should be valid
    sig = T.harmonic_signature(0)
    assert sig.consonance_score >= 0
    assert sig.stability_verdict in ('stable', 'transitioning', 'bifurcating')

    # Regime detection
    regime, conf = mg.regime_detection()
    assert 0 <= regime <= 2
    assert 0.0 <= conf <= 1.0

    print(f"  mock_live → tensor: {mna.n_total} nodes, "
          f"key={sig.dominant_interval}, regime={regime}")
    print(f"  PASS: B3")


# ═══════════════════════════════════════════════════════════
# C1: parse_article extracts tickers and sentiment
# ═══════════════════════════════════════════════════════════

def test_C1_parse_article():
    from tensor.core import UnifiedTensor
    from tensor.market_graph import MarketGraph
    from tensor.scraper_bridge import ScraperBridge

    mg = MarketGraph.mock_live(n_tickers=10, seed=42)
    T = UnifiedTensor(max_nodes=128, n_levels=4)
    sb = ScraperBridge(T, mg)

    html = """
    <html><body>
    <h1>Tech Stocks Rally</h1>
    <p>$AAPL surged 5% today on strong earnings beat. MSFT also rallied
    after positive guidance. Analysts upgrade both stocks to buy.</p>
    <p>Meanwhile JPM declined on weak trading revenue. GS also fell.</p>
    </body></html>
    """

    result = sb.parse_article(html)
    assert 'AAPL' in result['tickers']
    assert 'MSFT' in result['tickers']
    assert 'JPM' in result['tickers']
    assert isinstance(result['sentiment'], float)
    assert -1.0 <= result['sentiment'] <= 1.0
    assert result['text_length'] > 0

    # Per-ticker sentiments should exist
    assert 'AAPL' in result['ticker_sentiments']
    print(f"  Tickers: {result['tickers']}")
    print(f"  Overall sentiment: {result['sentiment']:.4f}")
    print(f"  Ticker sentiments: {result['ticker_sentiments']}")
    print(f"  PASS: C1")


# ═══════════════════════════════════════════════════════════
# C2: inject updates MarketGraph sentiment
# ═══════════════════════════════════════════════════════════

def test_C2_inject():
    from tensor.core import UnifiedTensor
    from tensor.market_graph import MarketGraph
    from tensor.scraper_bridge import ScraperBridge

    mg = MarketGraph.mock_live(n_tickers=10, seed=42)
    T = UnifiedTensor(max_nodes=128, n_levels=4)
    sb = ScraperBridge(T, mg)

    old_sent = mg.tickers['AAPL'].sentiment

    article = {
        'tickers': ['AAPL'],
        'sentiment': 0.8,
        'ticker_sentiments': {'AAPL': 0.9},
    }
    sb.inject(article)

    new_sent = mg.tickers['AAPL'].sentiment
    # Should be blended: 0.7 * old + 0.3 * 0.9
    expected = 0.7 * old_sent + 0.3 * 0.9
    assert abs(new_sent - expected) < 0.01, f"Expected {expected}, got {new_sent}"
    print(f"  AAPL sentiment: {old_sent:.4f} → {new_sent:.4f}")
    print(f"  PASS: C2")


# ═══════════════════════════════════════════════════════════
# C3: batch_inject updates tensor L0 once
# ═══════════════════════════════════════════════════════════

def test_C3_batch_inject():
    from tensor.core import UnifiedTensor
    from tensor.market_graph import MarketGraph
    from tensor.scraper_bridge import ScraperBridge

    mg = MarketGraph.mock_live(n_tickers=10, seed=42)
    T = UnifiedTensor(max_nodes=128, n_levels=4)
    sb = ScraperBridge(T, mg)

    articles_html = [
        "<p>AAPL stock surged on bullish earnings beat.</p>",
        "<p>MSFT rallied after strong cloud growth revenue.</p>",
        "<p>JPM declined amid bearish outlook and risk concerns.</p>",
    ]

    result = sb.batch_inject(articles_html, t=1.0)
    assert result['n_articles'] == 3
    assert len(result['tickers_found']) > 0
    assert isinstance(result['avg_sentiment'], float)

    # Tensor L0 should be populated
    assert T._mna[0] is not None
    assert T._mna[0].n_total == 10

    print(f"  Batch: {result['n_articles']} articles, "
          f"tickers={result['tickers_found']}, avg_sent={result['avg_sentiment']:.4f}")
    print(f"  PASS: C3")


# ═══════════════════════════════════════════════════════════
# INTEGRATION: L0 + L1 + L2 all populated in snapshot
# ═══════════════════════════════════════════════════════════

def test_integration_all_levels():
    from tensor.core import UnifiedTensor
    from tensor.neural_bridge import NeuralBridge
    from tensor.market_graph import MarketGraph
    from tensor.code_graph import CodeGraph
    from tensor.scraper_bridge import ScraperBridge
    from tensor.observer import TensorObserver

    T = UnifiedTensor(max_nodes=128, n_levels=4)

    # L0: Market via mock_live + scraper
    mg = MarketGraph.mock_live(n_tickers=10, seed=42)
    mna_market = mg.to_mna()
    T.update_level(0, mna_market, t=0.0)

    sb = ScraperBridge(T, mg)
    sb.batch_inject([
        "<p>AAPL bullish surge rally strong buy upgrade.</p>",
        "<p>MSFT positive growth momentum breakout.</p>",
    ], t=0.1)

    # L1: Neural
    nb = NeuralBridge(T, n_neurons=16, seed=42)
    nb.update_tensor(t=0.0)

    # L2: Code
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')
    cg = CodeGraph.from_directory(dev_path, max_files=50)
    mna_code = cg.to_mna()
    T.update_level(2, mna_code, t=0.0)

    # All three levels populated
    assert T._mna[0] is not None, "L0 not populated"
    assert T._mna[1] is not None, "L1 not populated"
    assert T._mna[2] is not None, "L2 not populated"

    # Observer snapshot
    obs = TensorObserver(T)
    md = obs.snapshot_markdown()
    ctx = obs.to_agent_context()

    assert '| L0' in md
    assert '| L1' in md
    assert '| L2' in md
    assert 'L0' in ctx
    assert 'L1' in ctx
    assert 'L2' in ctx

    # Cross-level resonance should exist between all populated pairs
    res_01 = T.cross_level_resonance(0, 1)
    res_02 = T.cross_level_resonance(0, 2)
    res_12 = T.cross_level_resonance(1, 2)
    assert 0.0 <= res_01 <= 1.0
    assert 0.0 <= res_02 <= 1.0
    assert 0.0 <= res_12 <= 1.0

    print(f"\n{'='*60}")
    print(f"  FULL SNAPSHOT (L0+L1+L2):")
    print(f"{'='*60}")
    print(md)
    print(f"{'='*60}")
    print(f"  AGENT CONTEXT:")
    print(f"{'='*60}")
    print(ctx)
    print(f"\n  Resonance: L0↔L1={res_01:.4f}, L0↔L2={res_02:.4f}, L1↔L2={res_12:.4f}")
    print(f"  PASS: integration (all levels)")


if __name__ == '__main__':
    tests = [
        ("A1_neural_bridge_builds_l1", test_A1_neural_bridge_builds_l1),
        ("A2_forward_pass", test_A2_forward_pass),
        ("A3_coarsen_lift_chain", test_A3_coarsen_lift_chain),
        ("B1_from_trading_pipeline", test_B1_from_trading_pipeline),
        ("B2_mock_live", test_B2_mock_live),
        ("B3_mock_live_tensor", test_B3_mock_live_tensor),
        ("C1_parse_article", test_C1_parse_article),
        ("C2_inject", test_C2_inject),
        ("C3_batch_inject", test_C3_batch_inject),
        ("integration_all_levels", test_integration_all_levels),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"--- {name} ---")
        print(f"{'='*60}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
    print("All track A/B/C tests passed.")
