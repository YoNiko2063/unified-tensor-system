"""Tests for Layers 1-5: trajectory, agent_network, predictive, domain_fibers, meta_loss.

Run: pytest tests/test_layers.py -q
"""
import os
import sys
import json
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))

import numpy as np


# ═══════════════════════════════════════════════════════════
# LAYER 1: LearningTrajectory
# ═══════════════════════════════════════════════════════════

def _make_trajectory_points(n=20, trend='up'):
    """Generate n trajectory points with known consonance series."""
    points = []
    for i in range(n):
        t = float(i)
        if trend == 'up':
            cons = 0.5 + 0.01 * i  # linearly increasing
        elif trend == 'accelerating':
            cons = 0.5 + 0.001 * i * i  # quadratically increasing
        elif trend == 'flat':
            cons = 0.6
        else:
            cons = 0.5
        points.append({
            'timestamp': t,
            'consonance': {'code': cons, 'market': 0.5},
            'eigenvalue_gaps': {'code': 0.1 + 0.005 * i},
            'growth_nodes': [{'count': 1}],
            'golden_resonance_matrix': [[1.0, 0.8], [0.8, 1.0]],
        })
    return points


def test_trajectory_record():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory(window=100)
    for p in _make_trajectory_points(20):
        traj.record(p)
    assert len(traj.points) == 20


def test_trajectory_velocity_positive():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'up'):
        traj.record(p)
    vel = traj.consonance_velocity('code')
    assert vel > 0, f"Expected positive velocity, got {vel}"


def test_trajectory_velocity_flat():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'flat'):
        traj.record(p)
    vel = traj.consonance_velocity('code')
    assert abs(vel) < 1e-6


def test_trajectory_acceleration_positive():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'accelerating'):
        traj.record(p)
    acc = traj.consonance_acceleration('code')
    assert acc > 0, f"Expected positive acceleration, got {acc}"


def test_trajectory_compounding_subspaces():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'accelerating'):
        traj.record(p)
    cs = traj.compounding_subspaces()
    assert 'code' in cs


def test_trajectory_stagnant_subspaces():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'flat'):
        traj.record(p)
    ss = traj.stagnant_subspaces(threshold=10)
    assert 'code' in ss


def test_trajectory_phi_conjugate():
    from tensor.trajectory import LearningTrajectory, PHI
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'flat'):
        traj.record(p)
    target = traj.phi_conjugate_target('code')
    # Should be near equilibrium since flat
    assert target >= 0.6


def test_trajectory_meta_loss():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'accelerating'):
        traj.record(p)
    ml = traj.meta_loss()
    # Negative of positive acceleration = negative meta_loss
    assert ml < 0


def test_trajectory_save_load():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(5):
        traj.record(p)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name
    try:
        traj.save(path)
        traj2 = LearningTrajectory()
        traj2.load(path)
        assert len(traj2.points) == 5
        assert traj2.points[0].consonance == traj.points[0].consonance
    finally:
        os.unlink(path)


def test_trajectory_window_trim():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory(window=10)
    for p in _make_trajectory_points(20):
        traj.record(p)
    assert len(traj.points) == 10


# ═══════════════════════════════════════════════════════════
# LAYER 2: AgentNetwork
# ═══════════════════════════════════════════════════════════

def test_agent_node_firing():
    from tensor.agent_network import AgentNode
    agent = AgentNode(role='test', model='test', level='code')
    ctx_low = {'consonance': {'code': 0.3}}
    ctx_high = {'consonance': {'code': 0.95}}
    assert agent.should_fire(ctx_low)
    assert not agent.should_fire(ctx_high)


def test_agent_influence_update():
    from tensor.agent_network import AgentNode, PHI
    agent = AgentNode(role='test', model='test', level='code', influence=1.0)
    # Accurate prediction
    agent.update_influence(0.01, 0.01)
    assert agent.influence > 1.0
    assert agent.correct_predictions == 1
    # Inaccurate prediction
    agent.update_influence(0.05, -0.05)
    prev = agent.influence
    assert agent.correct_predictions == 1
    assert agent.total_predictions == 2


def test_agent_hebbian_decay():
    from tensor.agent_network import AgentNode, PHI
    agent = AgentNode(role='test', model='test', level='code', influence=1.0)
    agent.update_influence(0.1, -0.1)  # Wrong prediction
    assert abs(agent.influence - 1.0 / PHI) < 0.01


def test_agent_network_arbitration():
    from tensor.agent_network import AgentNetwork, AgentNode
    net = AgentNetwork()
    a1 = AgentNode(role='a1', model='m', level='code', influence=2.0)
    a2 = AgentNode(role='a2', model='m', level='code', influence=0.5)
    net.add_agent(a1)
    net.add_agent(a2)
    # Both should fire on low consonance
    ctx = {'consonance': {'code': 0.3}}
    assert a1.should_fire(ctx)
    assert a2.should_fire(ctx)
    # a1 has higher influence -> should win arbitration


# ═══════════════════════════════════════════════════════════
# LAYER 3: PredictiveLayer
# ═══════════════════════════════════════════════════════════

def test_predictive_layer_ignorance():
    from tensor.agent_network import PredictiveLayer
    pl = PredictiveLayer()
    for i in range(20):
        pl.record_prediction('code', 0.01, 0.01 + 0.005 * i)
    ig = pl.ignorance_map()
    assert 'code' in ig
    assert ig['code'] > 0


def test_predictive_layer_learning_priority():
    from tensor.agent_network import PredictiveLayer
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'accelerating'):
        traj.record(p)
    pl = PredictiveLayer(trajectory=traj)
    pl.record_prediction('code', 0.01, 0.05)
    pl.record_prediction('market', 0.01, 0.01)
    prio = pl.learning_priority()
    # 'code' has higher ignorance * acceleration
    assert prio[0] == 'code'


# ═══════════════════════════════════════════════════════════
# LAYER 4: DomainFibers
# ═══════════════════════════════════════════════════════════

def test_domain_fiber_signature():
    from tensor.domain_fibers import DomainFiber
    basis = np.random.randn(10, 3)
    fiber = DomainFiber(domain='ece', subspace_basis=basis)
    sig = fiber.pattern_signature()
    assert len(sig) == 3
    assert sig[0] >= sig[-1]  # sorted descending


def test_fiber_bundle_resonance():
    from tensor.domain_fibers import FiberBundle
    fb = FiberBundle()
    # Same basis -> resonance should be 1.0
    basis = np.eye(5, 3)
    fb.add_fiber('ece', basis)
    fb.add_fiber('finance', basis)
    res = fb.cross_domain_resonance('ece', 'finance')
    assert abs(res - 1.0) < 0.01


def test_fiber_bundle_different_bases():
    from tensor.domain_fibers import FiberBundle
    fb = FiberBundle()
    fb.add_fiber('ece', np.eye(5, 3))
    fb.add_fiber('finance', np.random.randn(5, 3))
    res = fb.cross_domain_resonance('ece', 'finance')
    assert 0.0 <= res <= 1.0


def test_fiber_resonance_matrix():
    from tensor.domain_fibers import FiberBundle
    fb = FiberBundle()
    fb.add_fiber('ece', np.eye(5, 3))
    fb.add_fiber('finance', np.eye(5, 3))
    mat = fb.fiber_resonance_matrix()
    assert mat['ece']['finance'] > 0.9
    assert mat['ece']['ece'] == 1.0


def test_universal_patterns():
    from tensor.domain_fibers import FiberBundle
    fb = FiberBundle()
    # Same basis across all -> should find universal patterns
    basis = np.eye(5, 3)
    fb.add_fiber('ece', basis)
    fb.add_fiber('finance', basis)
    fb.add_fiber('biology', basis)
    ups = fb.universal_patterns()
    assert len(ups) > 0


# ═══════════════════════════════════════════════════════════
# LAYER 5: MetaLoss
# ═══════════════════════════════════════════════════════════

def test_meta_loss_accelerating():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'accelerating'):
        traj.record(p)
    # Meta-loss = -mean(acceleration). Accelerating -> negative meta_loss
    assert traj.meta_loss() < 0


def test_meta_loss_flat():
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'flat'):
        traj.record(p)
    assert abs(traj.meta_loss()) < 1e-4


def test_agent_predict_meta_delta():
    from tensor.agent_network import AgentNode, AgentProposal
    from tensor.trajectory import LearningTrajectory
    traj = LearningTrajectory()
    for p in _make_trajectory_points(20, 'accelerating'):
        traj.record(p)
    agent = AgentNode(role='test', model='m', level='code')
    proposal = AgentProposal(
        agent_role='test', target_level='code',
        description='test', predicted_delta=0.05)
    md = agent.predict_meta_delta(proposal, traj)
    assert md > 0


# ═══════════════════════════════════════════════════════════
# AGENT ROSTER
# ═══════════════════════════════════════════════════════════

def test_structural_agent():
    from tensor.agents.structural_agent import StructuralAgent
    agent = StructuralAgent()
    assert agent.role == 'structural'
    assert agent.should_fire({'consonance': {'code': 0.5}})
    assert not agent.should_fire({'consonance': {'code': 0.9}})


def test_resonance_agent():
    from tensor.agents.resonance_agent import ResonanceAgent
    agent = ResonanceAgent()
    assert agent.role == 'resonance'
    ctx = {'golden_resonance_matrix': [[1.0, 0.5], [0.5, 1.0]]}
    assert agent.should_fire(ctx)
    ctx_ok = {'golden_resonance_matrix': [[1.0, 0.9], [0.9, 1.0]]}
    assert not agent.should_fire(ctx_ok)


def test_validity_agent():
    from tensor.agents.validity_agent import ValidityAgent
    agent = ValidityAgent()
    assert agent.role == 'validity'
    ctx = {'stress_nodes': [{'level': 'market', 'risk': 0.8}]}
    assert agent.should_fire(ctx)


def test_validator_agent():
    from tensor.agents.validator_agent import ValidatorAgent
    agent = ValidatorAgent()
    assert agent.role == 'validator'
    assert agent.should_fire({})  # Always fires


def test_hardware_agent():
    from tensor.agents.hardware_agent import HardwareAgent
    agent = HardwareAgent()
    assert agent.role == 'hardware'
    assert agent.should_fire({'consonance': {'hardware': 0.3}})
    assert not agent.should_fire({'consonance': {'hardware': 0.8}})
