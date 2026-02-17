"""Integration tests for Phase 2: GSD bridge, skill writer, realtime feed,
neural continuous, and system runner."""
import os
import sys
import json
import time
import shutil
import tempfile
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))

from tensor.core import UnifiedTensor
from tensor.code_graph import CodeGraph
from tensor.market_graph import MarketGraph
from tensor.hardware_profiler import HardwareProfiler
from tensor.neural_bridge import NeuralBridge
from tensor.bootstrap import BootstrapResult
from tensor.gsd_bridge import GSDBridge, PhaseResult
from tensor.skill_writer import SkillWriter
from tensor.realtime_feed import RealtimeFeed


@pytest.fixture
def tensor_with_l2():
    """Tensor with L2 populated from dev-agent codebase."""
    t = UnifiedTensor(n_levels=4, max_nodes='auto')
    dev_root = os.path.join(_ROOT, 'dev-agent')
    if os.path.isdir(dev_root):
        cg = CodeGraph.from_directory(dev_root, max_files=200)
        mna = cg.to_mna()
        t.update_level(2, mna, t=time.time())
    else:
        # Fallback: use tensor/ directory itself
        cg = CodeGraph.from_directory(os.path.join(_ROOT, 'tensor'), max_files=50)
        mna = cg.to_mna()
        t.update_level(2, mna, t=time.time())
    return t


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp(prefix='tensor_test_')
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ─── Test 1: GSD Bridge creates project files ───
def test_gsd_bridge_project(tensor_with_l2, tmpdir):
    dev_root = os.path.join(_ROOT, 'dev-agent')
    if not os.path.isdir(dev_root):
        dev_root = os.path.join(_ROOT, 'tensor')

    planning_dir = os.path.join(tmpdir, '.planning')
    gsd = GSDBridge(tensor_with_l2, _ROOT, dev_root, planning_dir=planning_dir)
    result_dir = gsd.create_improvement_project()

    assert os.path.exists(os.path.join(result_dir, 'PROJECT.md'))
    assert os.path.exists(os.path.join(result_dir, 'REQUIREMENTS.md'))
    assert os.path.exists(os.path.join(result_dir, 'ROADMAP.md'))

    with open(os.path.join(result_dir, 'PROJECT.md')) as f:
        content = f.read()
    assert 'consonance' in content.lower()
    assert 'free_energy' in content or 'free energy' in content.lower()


# ─── Test 2: GSD plan phase generates XML tasks ───
def test_gsd_plan_phase(tensor_with_l2, tmpdir):
    dev_root = os.path.join(_ROOT, 'dev-agent')
    if not os.path.isdir(dev_root):
        dev_root = os.path.join(_ROOT, 'tensor')

    planning_dir = os.path.join(tmpdir, '.planning')
    gsd = GSDBridge(tensor_with_l2, _ROOT, dev_root, planning_dir=planning_dir)
    tasks = gsd.plan_phase(1)

    assert len(tasks) >= 1
    assert '<task type="auto">' in tasks[0]
    assert '<verify>' in tasks[0]
    assert '<done>' in tasks[0]


# ─── Test 3: GSD execute phase completes ───
def test_gsd_execute_phase(tensor_with_l2, tmpdir):
    dev_root = os.path.join(_ROOT, 'dev-agent')
    if not os.path.isdir(dev_root):
        dev_root = os.path.join(_ROOT, 'tensor')

    planning_dir = os.path.join(tmpdir, '.planning')
    gsd = GSDBridge(tensor_with_l2, _ROOT, dev_root, planning_dir=planning_dir)
    gsd.plan_phase(1)
    result = gsd.execute_phase(1)

    assert isinstance(result, PhaseResult)
    assert result.phase == 1
    assert result.consonance_before >= 0
    assert result.consonance_after >= 0
    assert isinstance(result.tasks_completed, int)


# ─── Test 4: GSD verify phase returns bool ───
def test_gsd_verify_phase(tensor_with_l2, tmpdir):
    dev_root = os.path.join(_ROOT, 'dev-agent')
    if not os.path.isdir(dev_root):
        dev_root = os.path.join(_ROOT, 'tensor')

    planning_dir = os.path.join(tmpdir, '.planning')
    gsd = GSDBridge(tensor_with_l2, _ROOT, dev_root, planning_dir=planning_dir)
    gsd.plan_phase(1)
    gsd.execute_phase(1)
    verified = gsd.verify_phase(1)

    assert isinstance(verified, bool)


# ─── Test 5: Skill writer creates valid SKILL.md ───
def test_skill_writer(tmpdir):
    skills_dir = os.path.join(tmpdir, 'skills', 'tensor-learned')
    writer = SkillWriter(skills_dir=skills_dir, log_dir=os.path.join(tmpdir, 'logs'))

    improvement = BootstrapResult(
        improved=True,
        consonance_before=0.45,
        consonance_after=0.52,
        consonance_delta=0.07,
        files_changed=['agent/core.py', 'agent/runner.py'],
        high_tension_nodes=['agent', 'utils', 'config'],
        step=1,
    )
    path = writer.write_skill(improvement)

    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    # Check obsidian-skills format: YAML frontmatter
    assert content.startswith('---')
    assert 'name:' in content
    assert 'description:' in content
    assert '---' in content[3:]  # closing frontmatter
    # Check measurements
    assert '0.4500' in content
    assert '0.5200' in content


# ─── Test 6: Skill library lists skills with success rates ───
def test_skill_library(tmpdir):
    skills_dir = os.path.join(tmpdir, 'skills', 'tensor-learned')
    writer = SkillWriter(skills_dir=skills_dir, log_dir=os.path.join(tmpdir, 'logs'))

    # Write two skills
    for step in [1, 2]:
        improvement = BootstrapResult(
            improved=True,
            consonance_before=0.4,
            consonance_after=0.5,
            consonance_delta=0.1,
            files_changed=[f'file_{step}.py'],
            high_tension_nodes=[f'module_{step}'],
            step=step,
        )
        writer.write_skill(improvement)

    # Record a failure for first
    writer.record_failure('module_1')

    library = writer.skill_library()
    assert len(library) >= 2

    # Find module_1 skill
    m1 = [s for s in library if s['pattern'] == 'module_1']
    assert len(m1) == 1
    assert m1[0]['total_applications'] == 2  # 1 success + 1 failure
    assert m1[0]['success_rate'] == 0.5
    assert m1[0]['flagged'] is False  # 0.5 is not < 0.5


# ─── Test 7: RealtimeFeed init and status ───
def test_realtime_feed_init():
    t = UnifiedTensor(n_levels=4, max_nodes='auto')
    mg = MarketGraph.mock_live(n_tickers=5)
    feed = RealtimeFeed(t, mg, sources=['mock'], update_interval=1.0)

    status = feed.status()
    assert isinstance(status, dict)
    assert 'connected_sources' in status
    assert 'ticks_received' in status
    assert 'last_update' in status
    assert 'l0_node_count' in status
    assert status['l0_node_count'] == 5
    assert 'current_regime' in status
    assert status['running'] is False


# ─── Test 8: Mock Yahoo tick updates L0 MNA ───
def test_yahoo_mock():
    t = UnifiedTensor(n_levels=4, max_nodes='auto')
    mg = MarketGraph.mock_live(n_tickers=5)
    feed = RealtimeFeed(t, mg, sources=['mock'], update_interval=0.1)

    # Inject a tick manually
    symbols = list(mg.tickers.keys())
    feed.inject_tick(symbols[0], price=150.0)

    # Check L0 was updated
    mna0 = t._mna.get(0)
    assert mna0 is not None
    assert mna0.n_total == 5

    # Check the tick was registered
    assert mg.tickers[symbols[0]].price == 150.0


# ─── Test 9: Neural continuous one iteration ───
def test_neural_continuous():
    t = UnifiedTensor(n_levels=4, max_nodes='auto')

    # Populate L0 to drive neural input
    mg = MarketGraph.mock_live(n_tickers=5)
    mna0 = mg.to_mna()
    t.update_level(0, mna0, t=time.time())
    t.set_state(0, mg.momentum_vector())

    bridge = NeuralBridge(t, n_neurons=8)
    log_dir = tempfile.mkdtemp(prefix='neural_test_')

    try:
        thread = bridge.run_continuous(
            interval_seconds=0.1, max_iterations=3, log_dir=log_dir)
        thread.join(timeout=5)

        # L1 should be populated
        mna1 = t._mna.get(1)
        assert mna1 is not None
        assert mna1.n_total == 8

        # Log file should exist
        log_path = os.path.join(log_dir, 'neural_state.jsonl')
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) >= 1

        entry = json.loads(lines[0])
        assert 'iteration' in entry
        assert 'state_norm' in entry
    finally:
        bridge.stop_continuous()
        shutil.rmtree(log_dir, ignore_errors=True)


# ─── Test 10: run_system startup with --improve ───
def test_run_system_startup(tmpdir):
    """Test that SystemRunner initializes and prints first snapshot."""
    # Import here to avoid side effects
    sys.path.insert(0, _ROOT)
    from run_system import SystemRunner

    runner = SystemRunner(snapshot_interval=1, dev_agent_root='dev-agent')

    # Setup L3
    runner._setup_l3_hardware()

    # Check L3 is populated
    mna3 = runner.tensor._mna.get(3)
    assert mna3 is not None
    assert mna3.n_total == 8  # 8 functional units

    # Print snapshot should not error
    runner.print_snapshot()

    # Observer should produce markdown
    md = runner.observer.snapshot_markdown()
    assert '# Tensor State' in md
    assert 'hardware' in md.lower() or 'L3' in md

    # Shutdown should be clean
    runner.shutdown()
