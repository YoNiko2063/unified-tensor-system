"""Tests for math connections: Fisher, regime, stochastic, neural error,
SNN firing, ground truth, feed health, and full math loop."""
import os
import sys
import time
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))

from tensor.core import UnifiedTensor
from tensor.market_graph import MarketGraph
from tensor.neural_bridge import NeuralBridge
from tensor.math_connections import (
    fisher_guided_planning, FisherGuidance,
    detect_regime, RegimeStatus,
    stochastic_robustness_check, RobustnessResult,
    neural_prediction_error, PredictionErrorReport,
    snn_firing_activation, FiringActivation,
    ground_truth_pytest, TestJumpEvent,
    check_feed_health, FeedHealth,
)


@pytest.fixture
def populated_tensor():
    """Tensor with L0 (market), L1 (neural), L2 (code-like), L3 (hardware)."""
    t = UnifiedTensor(n_levels=4, max_nodes='auto')

    # L0: market
    mg = MarketGraph.mock_live(n_tickers=5)
    mna0 = mg.to_mna()
    t.update_level(0, mna0, t=time.time())
    t.set_state(0, mg.momentum_vector())

    # L1: neural
    bridge = NeuralBridge(t, n_neurons=8, seed=42)
    bridge.update_tensor(t=time.time())

    # L2: synthetic code graph
    from tensor.code_graph import CodeGraph
    tensor_dir = os.path.join(_ROOT, 'tensor')
    if os.path.isdir(tensor_dir):
        cg = CodeGraph.from_directory(tensor_dir, max_files=20)
        mna2 = cg.to_mna()
        t.update_level(2, mna2, t=time.time())

    # L3: hardware
    from tensor.hardware_profiler import HardwareProfiler
    profiler = HardwareProfiler()
    profile = profiler.profile()
    mna3 = profiler.to_mna(profile)
    t.update_level(3, mna3, t=time.time())

    return t


# ─── Test 1: Fisher → GSD planning ───
def test_fisher_to_gsd(populated_tensor):
    guidance = fisher_guided_planning(populated_tensor, level=2, top_k=3)

    assert isinstance(guidance, FisherGuidance)
    assert len(guidance.eigenvalues) > 0
    assert len(guidance.priority_indices) <= 3
    assert guidance.condition_number >= 1.0
    # Eigenvalues should be non-negative (FIM is PSD)
    assert all(e >= -1e-10 for e in guidance.eigenvalues)


# ─── Test 2: Regime detection → monitoring ───
def test_regime_monitor(populated_tensor):
    status = detect_regime(populated_tensor, level=2)

    assert isinstance(status, RegimeStatus)
    assert status.current_regime in (0, 1, 2)
    assert status.n_regimes == 3
    assert status.regime_duration > 0
    assert 0.0 <= status.transition_probability <= 1.0
    assert isinstance(status.should_pause, bool)


# ─── Test 3: Stochastic robustness check ───
def test_stochastic_robustness(populated_tensor):
    mna = populated_tensor._mna.get(2)
    if mna is None:
        pytest.skip("L2 not populated")

    G = mna.G[:mna.n_total, :mna.n_total]

    def simple_score(eigvals, cons, dom):
        return float(cons)

    result = stochastic_robustness_check(G, simple_score, n_paths=10, seed=42)

    assert isinstance(result, RobustnessResult)
    assert result.n_paths == 10
    assert result.mean_score >= 0.0
    assert result.std_score >= 0.0
    assert result.min_score >= 0.0
    assert isinstance(result.robust, bool)


# ─── Test 4: Neural prediction error → GSD weights ───
def test_neural_prediction_error(populated_tensor):
    # Create a known actual state different from predicted
    actual = np.random.default_rng(42).standard_normal(8) * 0.1
    report = neural_prediction_error(populated_tensor, actual, level=1)

    assert isinstance(report, PredictionErrorReport)
    assert report.mean_error >= 0.0
    assert len(report.errors) > 0
    assert isinstance(report.high_error_modules, list)


# ─── Test 5: SNN free energy firing → L1 activation ───
def test_snn_firing(populated_tensor):
    activation = snn_firing_activation(populated_tensor, tau=1.0, gamma=0.5)

    assert isinstance(activation, FiringActivation)
    assert len(activation.firing_mask) > 0
    assert len(activation.free_energies) > 0
    assert activation.n_firing >= 0
    assert len(activation.activation_vector) == len(activation.firing_mask)
    # Activation vector should be 0 or 1
    for v in activation.activation_vector:
        assert v in (0.0, 1.0)


# ─── Test 6: Ground truth pytest → jump events ───
def test_ground_truth():
    event = ground_truth_pytest(test_dir='tests', baseline_pass_rate=1.0)

    assert isinstance(event, TestJumpEvent)
    assert event.tests_total >= 0
    assert 0.0 <= event.success_rate <= 1.0
    assert isinstance(event.is_jump, bool)
    assert event.jump_magnitude >= 0.0


# ─── Test 7: Feed health monitoring ───
def test_feed_health(populated_tensor):
    # Simulate a feed status dict
    feed_status = {
        'connected_sources': ['mock'],
        'ticks_received': 10,
        'last_update': time.time() - 5.0,  # 5 seconds ago
        'l0_node_count': 5,
        'current_regime': 'calm',
        'running': True,
    }
    health = check_feed_health(populated_tensor, feed_status, max_staleness=60.0)

    assert isinstance(health, FeedHealth)
    assert health.is_healthy is True
    assert health.l0_populated is True
    assert health.staleness_seconds < 60.0
    assert health.regime == 'calm'
    assert len(health.warnings) == 0

    # Test unhealthy: stale feed
    stale_status = dict(feed_status, last_update=time.time() - 120.0)
    health_stale = check_feed_health(populated_tensor, stale_status, max_staleness=60.0)
    assert health_stale.is_healthy is False
    assert len(health_stale.warnings) > 0


# ─── Test 8: Full math loop ───
def test_full_math_loop(populated_tensor):
    """Integration: all 7 math connections work together on one tensor."""
    # 1. Fisher guidance
    guidance = fisher_guided_planning(populated_tensor, level=2, top_k=3)
    assert len(guidance.priority_indices) > 0

    # 2. Regime detection
    regime = detect_regime(populated_tensor, level=2)
    assert regime.current_regime >= 0

    # 3. SNN firing
    firing = snn_firing_activation(populated_tensor)
    assert firing.n_firing >= 0

    # 4. Neural prediction error
    actual = np.zeros(8)
    error = neural_prediction_error(populated_tensor, actual, level=1)
    assert error.mean_error >= 0.0

    # 5. Feed health
    health = check_feed_health(populated_tensor)
    assert isinstance(health.is_healthy, bool)

    # 6. All produce consistent tensor state
    snap = populated_tensor.tensor_snapshot()
    assert snap['n_levels'] == 4
    # At least L1, L2, L3 should be populated
    populated_count = sum(1 for l in snap['levels'].values()
                          if isinstance(l, dict) and l.get('populated', False))
    assert populated_count >= 3
