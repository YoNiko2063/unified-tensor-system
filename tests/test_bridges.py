"""Tests for Steps 6-8: DevAgentBridge, TradingBridge, TensorObserver.

Run: python tests/test_bridges.py
"""
import os
import sys
import tempfile
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))
sys.path.insert(0, _ROOT)

import numpy as np


def _make_tensor_with_market():
    """Build a tensor with L0 populated from market data."""
    from tensor.core import UnifiedTensor
    from tensor.market_graph import MarketGraph

    T = UnifiedTensor(max_nodes=128, n_levels=4)
    mg = MarketGraph()
    for sym, sec, mom, vol in [
        ('AAPL', 'tech', 0.02, 0.015), ('MSFT', 'tech', 0.01, 0.012),
        ('GOOGL', 'tech', -0.005, 0.018), ('JPM', 'fin', 0.008, 0.02),
        ('GS', 'fin', 0.012, 0.022), ('XOM', 'energy', -0.01, 0.025),
    ]:
        mg.add_ticker(sym, sec, price=100, momentum=mom, volatility=vol)
    mg.set_correlation('AAPL', 'MSFT', 0.85)
    mg.set_correlation('JPM', 'GS', 0.90)
    mg.set_correlation('AAPL', 'JPM', 0.25)

    mna = mg.to_mna()
    T.update_level(0, mna, t=0.0)
    return T, mg


# ═══════════════════════════════════════════════════════════
# TEST 1: DEV AGENT BRIDGE
# ═══════════════════════════════════════════════════════════

def test_dev_agent_bridge():
    from tensor.dev_agent_bridge import DevAgentBridge

    T, _ = _make_tensor_with_market()
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')

    tmplog = tempfile.mkdtemp()
    try:
        bridge = DevAgentBridge(T, dev_path, max_files=100, log_dir=tmplog)

        # proposal_weights: hotspots should have higher weight
        hotspots = bridge._hotspots[:3]
        cold = [n for n in bridge._code_graph.node_names
                if n not in hotspots][:3]
        all_targets = hotspots + cold
        weights = bridge.proposal_weights(all_targets)

        assert len(weights) == len(all_targets)
        hot_avg = np.mean([weights[h] for h in hotspots])
        cold_avg = np.mean([weights[c] for c in cold])
        assert hot_avg > cold_avg, (
            f"Hotspot avg weight ({hot_avg:.4f}) should be > cold ({cold_avg:.4f})")
        print(f"  Hotspot avg={hot_avg:.4f}, cold avg={cold_avg:.4f}")

        # improvement_priority_map: all modules, sorted desc
        pmap = bridge.improvement_priority_map()
        n_mods = bridge._code_graph.n_modules
        assert len(pmap) == n_mods, f"Expected {n_mods} items, got {len(pmap)}"
        # Verify sorted desc
        for i in range(len(pmap) - 1):
            assert pmap[i]['weight'] >= pmap[i + 1]['weight']
        # Verify fields
        for item in pmap[:3]:
            assert 'module' in item
            assert 'free_energy' in item
            assert 'harmonic_tension' in item
            assert 'weight' in item
            assert 'reason' in item
            assert isinstance(item['reason'], str) and len(item['reason']) > 0
        print(f"  Priority map: {len(pmap)} modules, top={pmap[0]['module']} "
              f"(w={pmap[0]['weight']:.4f})")
        print(f"  Top reason: {pmap[0]['reason']}")

        # on_improvement_applied: updates tensor and logs
        entry = bridge.on_improvement_applied(pmap[0]['module'], 'success')
        assert entry['outcome'] == 'success'
        log_path = os.path.join(tmplog, 'improvement_history.jsonl')
        assert os.path.exists(log_path)
        print(f"  on_improvement_applied logged: delta={entry['delta']:.4f}")

        # refresh detects state
        bridge.refresh()
        assert bridge._code_graph is not None
        print(f"  PASS: refresh works, {bridge._code_graph.n_modules} modules")

    finally:
        shutil.rmtree(tmplog, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 2: TRADING BRIDGE
# ═══════════════════════════════════════════════════════════

def test_trading_bridge():
    from tensor.trading_bridge import TradingBridge

    T, mg = _make_tensor_with_market()
    tb = TradingBridge(T, mg)

    pipeline = {
        'sentiment_scores': {
            'AAPL': 0.7, 'MSFT': 0.5, 'GOOGL': 0.3,
            'JPM': -0.2, 'GS': -0.1, 'XOM': -0.5,
        }
    }

    result = tb.enhance_scores(pipeline)

    # Enhanced scores in [-1, 1]
    for ticker, score in result['enhanced_scores'].items():
        assert -1.0 <= score <= 1.0, f"{ticker} score {score} out of range"
    print(f"  Enhanced scores: {result['enhanced_scores']}")

    # Tensor signal adds meaningful adjustment
    # At least one ticker should differ from original
    diffs = [abs(result['enhanced_scores'][t] - result['original_scores'][t])
             for t in result['original_scores']]
    assert max(diffs) > 0.001, "Tensor should adjust at least one score"
    print(f"  Max adjustment: {max(diffs):.4f}")

    # Harmonic signature included
    assert 'harmonic_signature' in result
    assert 'dominant_interval' in result['harmonic_signature']
    print(f"  Harmonic sig: {result['harmonic_signature']}")

    # regime_signal returns valid
    sig = tb.regime_signal()
    assert isinstance(sig['regime'], int)
    assert 0 <= sig['regime'] <= 2
    assert sig['recommendation'] in ('hold', 'reduce', 'increase', 'exit')
    assert 0.0 <= sig['phase_risk'] <= 1.0
    print(f"  Regime: {sig['regime']}, rec={sig['recommendation']}, "
          f"risk={sig['phase_risk']:.4f}, key={sig['harmonic_key']}")

    # Phase risk > 0.8 → recommendation is exit or reduce
    if sig['phase_risk'] > 0.8:
        assert sig['recommendation'] in ('exit', 'reduce')
    print(f"  PASS: trading bridge complete")


# ═══════════════════════════════════════════════════════════
# TEST 3: OBSERVER
# ═══════════════════════════════════════════════════════════

def test_observer():
    from tensor.observer import TensorObserver
    from tensor.core import UnifiedTensor
    from tensor.code_graph import CodeGraph

    T, _ = _make_tensor_with_market()

    # Also populate L2
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')
    cg = CodeGraph.from_directory(dev_path, max_files=50)
    mna_code = cg.to_mna()
    T.update_level(2, mna_code, t=0.0)

    tmplog = tempfile.mkdtemp()
    try:
        obs = TensorObserver(T, log_dir=tmplog)

        # snapshot_markdown is valid markdown
        md = obs.snapshot_markdown()
        assert isinstance(md, str)
        assert '# Tensor State' in md
        assert '## System Health' in md
        assert '## Level Status' in md
        assert '| L0' in md
        assert '| L2' in md
        assert '## Active Signals' in md
        print(f"  Markdown length: {len(md)} chars")

        # to_agent_context under 500 tokens (~2000 chars)
        ctx = obs.to_agent_context()
        assert isinstance(ctx, str)
        assert len(ctx) < 2000, f"Context too long: {len(ctx)} chars"
        assert 'TENSOR STATE' in ctx
        assert 'L0' in ctx
        assert 'L2' in ctx
        assert 'SIGNALS' in ctx
        print(f"  Agent context: {len(ctx)} chars")

        # Log snapshot
        obs.log_snapshot()
        log_path = os.path.join(tmplog, 'tensor_state.jsonl')
        assert os.path.exists(log_path)
        print(f"  PASS: observer complete")

    finally:
        shutil.rmtree(tmplog, ignore_errors=True)

    return md  # Return for printing in full_system test


# ═══════════════════════════════════════════════════════════
# TEST 4: FULL SYSTEM END-TO-END
# ═══════════════════════════════════════════════════════════

def test_full_system():
    from tensor.core import UnifiedTensor
    from tensor.code_graph import CodeGraph
    from tensor.market_graph import MarketGraph
    from tensor.dev_agent_bridge import DevAgentBridge
    from tensor.trading_bridge import TradingBridge
    from tensor.observer import TensorObserver

    T = UnifiedTensor(max_nodes=128, n_levels=4, history_len=50)

    # --- L0: Market ---
    mg = MarketGraph()
    pipeline = {
        'tickers': [
            {'symbol': 'AAPL', 'sector': 'tech', 'price': 180, 'momentum': 0.02, 'volatility': 0.015},
            {'symbol': 'MSFT', 'sector': 'tech', 'price': 420, 'momentum': 0.01, 'volatility': 0.012},
            {'symbol': 'JPM', 'sector': 'fin', 'price': 200, 'momentum': 0.008, 'volatility': 0.02},
            {'symbol': 'XOM', 'sector': 'energy', 'price': 105, 'momentum': -0.01, 'volatility': 0.025},
        ],
        'correlations': [
            {'ticker_a': 'AAPL', 'ticker_b': 'MSFT', 'correlation': 0.85},
            {'ticker_a': 'JPM', 'ticker_b': 'XOM', 'correlation': 0.20},
        ],
        'sentiment_scores': {'AAPL': 0.7, 'MSFT': 0.5, 'JPM': -0.2, 'XOM': -0.5},
    }
    mg.update_from_pipeline(pipeline)
    mna_market = mg.to_mna()
    T.update_level(0, mna_market, t=0.0)
    print(f"  L0: {mna_market.n_total} market nodes")

    # --- L2: Code ---
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')
    tmplog = tempfile.mkdtemp()
    try:
        bridge = DevAgentBridge(T, dev_path, max_files=100, log_dir=tmplog)
        print(f"  L2: {bridge._code_graph.n_modules} code nodes")

        # Bridge proposal weights — hotspots weighted higher
        hotspots = bridge._hotspots[:3]
        cold = [n for n in bridge._code_graph.node_names if n not in hotspots][:3]
        weights = bridge.proposal_weights(hotspots + cold)
        hot_avg = np.mean([weights[h] for h in hotspots])
        cold_avg = np.mean([weights[c] for c in cold])
        assert hot_avg > cold_avg
        print(f"  Proposal weights: hot={hot_avg:.4f} > cold={cold_avg:.4f}")

        # Trading bridge
        tb = TradingBridge(T, mg)
        enhanced = tb.enhance_scores(pipeline)
        assert all(-1 <= v <= 1 for v in enhanced['enhanced_scores'].values())
        regime = tb.regime_signal()
        print(f"  Trading: regime={regime['regime']}, rec={regime['recommendation']}")

        # Observer
        obs = TensorObserver(T, log_dir=tmplog)
        pmap = bridge.improvement_priority_map()
        md = obs.snapshot_markdown(priority_map=pmap)
        ctx = obs.to_agent_context()

        assert '# Tensor State' in md
        assert len(ctx) < 2000
        assert 'Improvement Priorities' in md

        # Mock agent prompt injection
        agent_prompt = f"You are a dev-agent. Current system state:\n{ctx}\n\nWhat should you work on next?"
        assert 'L2' in agent_prompt
        assert 'SIGNALS' in agent_prompt
        print(f"  Agent context injected ({len(ctx)} chars)")

        print(f"\n{'='*60}")
        print(f"  FULL SNAPSHOT MARKDOWN:")
        print(f"{'='*60}")
        print(md)
        print(f"{'='*60}")
        print(f"  AGENT CONTEXT:")
        print(f"{'='*60}")
        print(ctx)
        print(f"\n  PASS: full system end-to-end")

    finally:
        shutil.rmtree(tmplog, ignore_errors=True)


if __name__ == '__main__':
    tests = [
        ("dev_agent_bridge", test_dev_agent_bridge),
        ("trading_bridge", test_trading_bridge),
        ("observer", test_observer),
        ("full_system", test_full_system),
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
    print("All bridge tests passed.")
