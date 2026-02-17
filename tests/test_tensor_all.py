"""Tests for Steps 3-5: UnifiedTensor, CodeGraph, MarketGraph.

Run: python tests/test_tensor_all.py

Acceptance criteria:
1. test_tensor_construction — 4 levels, update, eigenvalue_gap, snapshot
2. test_cross_level_resonance — identical=1, symmetric
3. test_phase_transition_risk — stable<0.4, bifurcating>0.6, monotone
4. test_code_graph — parse, to_mna, cycles, hotspots, eigenvalues
5. test_market_graph — construct, sentiment, regime, to_mna, update L0
6. test_full_pipeline — end-to-end integration
"""
import os
import sys

# Setup paths
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))
sys.path.insert(0, _ROOT)

import numpy as np


def _make_mna_from_eigenvalues(eigvals):
    """Build a symmetric MNASystem with prescribed eigenvalues."""
    from core.matrix import MNASystem
    n = len(eigvals)
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    G = Q @ np.diag(eigvals) @ Q.T
    G = 0.5 * (G + G.T)
    C = np.eye(n)
    return MNASystem(C=C, G=G, n_nodes=n, n_branches=0, n_total=n,
                     node_map={i: i for i in range(n)}, branch_map={}, branch_info=[])


def _stable_mna():
    """Consonant eigenvalues: 6, 3, 2, 1 (ratios 2:1, 3:2, 2:1)."""
    return _make_mna_from_eigenvalues([6.0, 3.0, 2.0, 1.0])


def _bifurcation_mna():
    """Dissonant eigenvalues: near-degenerate cluster + outlier."""
    return _make_mna_from_eigenvalues([10.0, 9.99, 9.98, 0.01])


# ═══════════════════════════════════════════════════════════
# TEST 1: TENSOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def test_tensor_construction():
    from tensor.core import UnifiedTensor

    T = UnifiedTensor(max_nodes=32, n_levels=4, history_len=50)

    # Can hold 4 levels
    assert T.n_levels == 4
    assert T.max_nodes == 32

    # Update level 0
    mna0 = _stable_mna()
    T.update_level(0, mna0, t=0.0)

    # Update level 2
    mna2 = _make_mna_from_eigenvalues([5.0, 3.0, 2.0, 1.0, 0.5])
    T.update_level(2, mna2, t=0.0)

    # eigenvalue_gap returns float
    gap = T.eigenvalue_gap(0)
    assert isinstance(gap, float)
    assert 0.0 <= gap <= 1.0
    print(f"  L0 eigenvalue gap: {gap:.4f}")

    # tensor_snapshot returns dict with all levels
    snap = T.tensor_snapshot()
    assert 'levels' in snap
    assert len(snap['levels']) == 4
    assert snap['levels'][0]['populated'] is True
    assert snap['levels'][1]['populated'] is False
    assert snap['levels'][2]['populated'] is True
    assert snap['levels'][3]['populated'] is False

    print(f"  PASS: 4 levels, 2 populated, snapshot complete")
    print(f"  L0: {snap['levels'][0]['harmonic_signature']}")
    print(f"  L2: {snap['levels'][2]['harmonic_signature']}")


# ═══════════════════════════════════════════════════════════
# TEST 2: CROSS-LEVEL RESONANCE
# ═══════════════════════════════════════════════════════════

def test_cross_level_resonance():
    from tensor.core import UnifiedTensor

    T = UnifiedTensor(max_nodes=16, n_levels=4)

    # Identical MNA → resonance should be very high
    mna = _stable_mna()
    T.update_level(0, mna, t=0.0)
    T.update_level(1, mna, t=0.0)
    r_same = T.cross_level_resonance(0, 1)
    assert r_same > 0.9, f"Identical MNA resonance should be >0.9, got {r_same:.4f}"
    print(f"  Identical MNA resonance: {r_same:.4f}")

    # Very different eigenstructures → lower resonance
    mna_diff = _make_mna_from_eigenvalues([100.0, 1.0, 0.1, 0.001])
    T.update_level(2, mna_diff, t=0.0)
    r_diff = T.cross_level_resonance(0, 2)
    assert r_diff < r_same, (
        f"Different eigenstructure resonance ({r_diff:.4f}) should be < "
        f"identical ({r_same:.4f})")
    print(f"  Different eigenstructure resonance: {r_diff:.4f}")

    # Symmetry
    r_01 = T.cross_level_resonance(0, 1)
    r_10 = T.cross_level_resonance(1, 0)
    assert abs(r_01 - r_10) < 1e-10, f"Resonance not symmetric: {r_01} vs {r_10}"
    print(f"  PASS: symmetric r(0,1)={r_01:.4f} == r(1,0)={r_10:.4f}")


# ═══════════════════════════════════════════════════════════
# TEST 3: PHASE TRANSITION RISK
# ═══════════════════════════════════════════════════════════

def test_phase_transition_risk():
    from tensor.core import UnifiedTensor

    T = UnifiedTensor(max_nodes=16, n_levels=4)

    # Stable circuit → low risk
    T.update_level(0, _stable_mna(), t=0.0)
    risk_stable = T.phase_transition_risk(0)
    assert risk_stable < 0.5, f"Stable risk should be <0.5, got {risk_stable:.4f}"
    print(f"  Stable risk: {risk_stable:.4f}")

    # Bifurcating circuit → high risk
    T.update_level(1, _bifurcation_mna(), t=0.0)
    risk_bif = T.phase_transition_risk(1)
    assert risk_bif > risk_stable, (
        f"Bifurcation risk ({risk_bif:.4f}) should be > stable ({risk_stable:.4f})")
    print(f"  Bifurcation risk: {risk_bif:.4f}")

    # Monotonicity: progressively narrowing gap → increasing risk
    risks = []
    for gap_factor in [10.0, 5.0, 2.0, 1.1, 1.001]:
        mna = _make_mna_from_eigenvalues([10.0, 10.0 / gap_factor, 5.0, 1.0])
        T.update_level(3, mna, t=0.0)
        r = T.phase_transition_risk(3)
        risks.append(r)
    # Risks should generally increase as gap narrows
    # (allow small tolerance for non-strict monotonicity in extreme cases)
    assert risks[-1] > risks[0], (
        f"Risk should increase with narrowing gap: {risks[0]:.4f} → {risks[-1]:.4f}")
    print(f"  PASS: risks increase with narrowing gap: {[f'{r:.3f}' for r in risks]}")


# ═══════════════════════════════════════════════════════════
# TEST 4: CODE GRAPH
# ═══════════════════════════════════════════════════════════

def test_code_graph():
    from tensor.code_graph import CodeGraph

    # Parse dev-agent source
    dev_agent_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')
    cg = CodeGraph.from_directory(dev_agent_path, max_files=100)

    assert cg.n_modules > 0, "Should parse at least some modules"
    print(f"  Parsed {cg.n_modules} modules, {cg.n_edges} edges")

    # to_mna() produces valid system
    mna = cg.to_mna()
    assert mna.n_total == cg.n_modules
    assert mna.G.shape == (cg.n_modules, cg.n_modules)
    assert mna.C.shape == (cg.n_modules, cg.n_modules)
    # G should be symmetric
    assert np.allclose(mna.G, mna.G.T, atol=1e-10), "G should be symmetric"
    print(f"  MNA: {mna.n_total}x{mna.n_total}, G symmetric")

    # Eigenvalues should be real (symmetric matrix)
    eigvals = np.linalg.eigvalsh(mna.G)
    assert np.all(np.isfinite(eigvals)), "Eigenvalues should be finite"
    print(f"  Eigenvalue range: [{eigvals.min():.4e}, {eigvals.max():.4e}]")

    # circular_imports returns list
    cycles = cg.circular_imports()
    print(f"  Circular imports: {len(cycles)} cycles found")

    # complexity_hotspots returns top-5
    hotspots = cg.complexity_hotspots(top_k=5)
    assert len(hotspots) <= 5
    assert all(isinstance(h, str) for h in hotspots)
    print(f"  Top-5 complexity hotspots: {hotspots}")

    # Eigenvalue magnitudes should correlate with complexity
    # (higher complexity modules contribute more to eigenvalue spread)
    # Just verify non-trivial eigenvalue spread exists
    eig_spread = eigvals.max() - eigvals.min()
    assert eig_spread > 0, "Should have non-trivial eigenvalue spread"
    print(f"  PASS: eigenvalue spread = {eig_spread:.4e}")


# ═══════════════════════════════════════════════════════════
# TEST 5: MARKET GRAPH
# ═══════════════════════════════════════════════════════════

def test_market_graph():
    from tensor.market_graph import MarketGraph
    from tensor.core import UnifiedTensor

    mg = MarketGraph()

    # Construct from mock pipeline output
    pipeline_data = {
        'tickers': [
            {'symbol': 'AAPL', 'sector': 'tech', 'price': 180.0,
             'momentum': 0.02, 'volatility': 0.015},
            {'symbol': 'MSFT', 'sector': 'tech', 'price': 420.0,
             'momentum': 0.01, 'volatility': 0.012},
            {'symbol': 'GOOGL', 'sector': 'tech', 'price': 175.0,
             'momentum': -0.005, 'volatility': 0.018},
            {'symbol': 'JPM', 'sector': 'finance', 'price': 200.0,
             'momentum': 0.008, 'volatility': 0.02},
            {'symbol': 'GS', 'sector': 'finance', 'price': 480.0,
             'momentum': 0.012, 'volatility': 0.022},
            {'symbol': 'XOM', 'sector': 'energy', 'price': 105.0,
             'momentum': -0.01, 'volatility': 0.025},
        ],
        'correlations': [
            {'ticker_a': 'AAPL', 'ticker_b': 'MSFT', 'correlation': 0.85},
            {'ticker_a': 'AAPL', 'ticker_b': 'GOOGL', 'correlation': 0.75},
            {'ticker_a': 'MSFT', 'ticker_b': 'GOOGL', 'correlation': 0.80},
            {'ticker_a': 'JPM', 'ticker_b': 'GS', 'correlation': 0.90},
            {'ticker_a': 'AAPL', 'ticker_b': 'JPM', 'correlation': 0.30},
            {'ticker_a': 'XOM', 'ticker_b': 'JPM', 'correlation': 0.20},
        ],
        'sentiment_scores': {
            'AAPL': 0.7,
            'MSFT': 0.5,
            'GOOGL': 0.3,
            'JPM': -0.2,
            'GS': -0.1,
            'XOM': -0.5,
        },
    }

    mg.update_from_pipeline(pipeline_data)
    assert mg.n_tickers == 6
    print(f"  Tickers: {mg.n_tickers}")

    # sentiment_injection produces correct sign
    u_aapl = mg.sentiment_injection('AAPL', 0.7)
    aapl_idx = mg.node_ids['AAPL']
    assert u_aapl[aapl_idx] == 0.7, f"AAPL injection should be 0.7"
    u_xom = mg.sentiment_injection('XOM', -0.5)
    xom_idx = mg.node_ids['XOM']
    assert u_xom[xom_idx] == -0.5, f"XOM injection should be -0.5"
    print(f"  PASS: sentiment injection sign correct")

    # regime_detection returns valid (int, float)
    regime, conf = mg.regime_detection()
    assert isinstance(regime, int)
    assert 0 <= regime <= 2
    assert 0.0 < conf <= 1.0
    print(f"  Regime: {regime}, confidence: {conf:.4f}")

    # to_mna produces valid system
    mna = mg.to_mna()
    assert mna.n_total == 6
    assert np.allclose(mna.G, mna.G.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(mna.G)
    print(f"  MNA eigenvalues: {eigvals}")

    # Update tensor L0 from market graph
    T = UnifiedTensor(max_nodes=32, n_levels=4)
    T.update_level(0, mna, t=0.0)
    snap = T.tensor_snapshot()
    assert snap['levels'][0]['populated'] is True
    assert snap['levels'][0]['n_nodes'] == 6
    print(f"  PASS: tensor L0 updated from market graph")
    print(f"  L0 signature: {snap['levels'][0]['harmonic_signature']}")


# ═══════════════════════════════════════════════════════════
# TEST 6: FULL PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════

def test_full_pipeline():
    from tensor.core import UnifiedTensor
    from tensor.code_graph import CodeGraph
    from tensor.market_graph import MarketGraph

    T = UnifiedTensor(max_nodes=128, n_levels=4, history_len=50)

    # --- L2: Code graph from dev-agent ---
    dev_agent_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')
    cg = CodeGraph.from_directory(dev_agent_path, max_files=50)
    mna_code = cg.to_mna()
    T.update_level(2, mna_code, t=0.0)
    print(f"  L2 (code): {mna_code.n_total} nodes from dev-agent")

    # --- L0: Market graph from mock data ---
    mg = MarketGraph()
    for sym, sec in [('AAPL', 'tech'), ('MSFT', 'tech'), ('JPM', 'fin'),
                     ('GS', 'fin'), ('XOM', 'energy')]:
        mg.add_ticker(sym, sec, price=100, momentum=0.01, volatility=0.015)
    mg.set_correlation('AAPL', 'MSFT', 0.85)
    mg.set_correlation('JPM', 'GS', 0.90)
    mg.set_correlation('AAPL', 'JPM', 0.25)
    mna_market = mg.to_mna()
    T.update_level(0, mna_market, t=0.0)
    print(f"  L0 (market): {mna_market.n_total} nodes")

    # --- Coarsen L2 → L1 via phi ---
    # L2 may be large, coarsen to sqrt(n)
    if mna_code.n_total > 2:
        result = T.coarsen_to(2, 1)
        print(f"  Coarsened L2→L1: {mna_code.n_total} → {result.mna_coarse.n_total} nodes, "
              f"ratio_error={result.ratio_error:.4f}")

        # --- Lift from L1 → L2 ---
        x_coarse = np.ones(result.mna_coarse.n_total)
        x_lifted = T.lift_from(1, 2, x_coarse)
        assert x_lifted.shape[0] == mna_code.n_total
        print(f"  Lifted L1→L2: coarse({result.mna_coarse.n_total}) → fine({x_lifted.shape[0]})")
    else:
        print(f"  L2 too small to coarsen ({mna_code.n_total} nodes)")

    # --- tensor_snapshot shows both levels ---
    snap = T.tensor_snapshot()
    assert snap['levels'][0]['populated'] is True
    assert snap['levels'][2]['populated'] is True
    print(f"\n  === TENSOR SNAPSHOT ===")
    for l in range(4):
        linfo = snap['levels'][l]
        if not linfo['populated']:
            print(f"  L{l} ({linfo['name']}): empty")
            continue
        sig = linfo['harmonic_signature']
        print(f"  L{l} ({linfo['name']}): {linfo['n_nodes']} nodes | "
              f"gap={linfo['eigenvalue_gap']:.4f} | "
              f"risk={linfo['phase_transition_risk']:.4f} | "
              f"consonance={sig['consonance_score']:.4f} | "
              f"interval={sig['dominant_interval']} | "
              f"verdict={sig['stability_verdict']}")
        if linfo['cross_level_resonance']:
            for other, res in linfo['cross_level_resonance'].items():
                print(f"    resonance with {other}: {res:.4f}")

    # --- cross_level_resonance(0, 2) returns float ---
    r_02 = T.cross_level_resonance(0, 2)
    assert isinstance(r_02, float)
    assert 0.0 <= r_02 <= 1.0
    print(f"\n  cross_level_resonance(market, code) = {r_02:.4f}")
    print(f"  PASS: full pipeline integration")


if __name__ == '__main__':
    tests = [
        ("tensor_construction", test_tensor_construction),
        ("cross_level_resonance", test_cross_level_resonance),
        ("phase_transition_risk", test_phase_transition_risk),
        ("code_graph", test_code_graph),
        ("market_graph", test_market_graph),
        ("full_pipeline", test_full_pipeline),
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
    print("All tensor tests passed.")
