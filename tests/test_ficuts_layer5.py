"""FICUTS Layer 5 tests: FICUTSUpdater + run_system wiring.

Tasks:
  5.1 — FICUTSUpdater (mark complete, log discovery, append hypothesis, update field)
  5.2 — SystemRunner wires FICUTSUpdater (status on start/shutdown, uptime)
"""
import os
import sys
import re
import tempfile
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ── fixtures ───────────────────────────────────────────────────────────────

MINIMAL_FICUTS = """\
**Version:** 1.0.0
**Status:** INITIALIZATION
**System Uptime:** 0h 0m
**Universals Discovered:** 0

## Current Hypothesis (Updated by System)

**Hypothesis 1:** (awaiting first discovery)

---

## Task List (Execute in Order, Update Status as You Go)

### LAYER 1: Lyapunov Energy

#### Task 1.1: Add Lyapunov Energy Functional

**Status:** `[ ]`
**Notes:**

---

#### Task 1.2: Replace meta_loss with Damped Acceleration

**Status:** `[ ]`
**Notes:**

---

## Discoveries (Logged by System)

---

## Success Criteria

Some criteria here.
"""


def _make_ficuts(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, 'FICUTS.md')
    with open(path, 'w') as f:
        f.write(MINIMAL_FICUTS)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5.1 — FICUTSUpdater
# ══════════════════════════════════════════════════════════════════════════════

def test_mark_task_complete_changes_status():
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.mark_task_complete('1.1')
        content = open(path).read()
        assert '[✓]' in content, "Task 1.1 not marked complete"
        # Task 1.2 should still be [ ]
        assert '`[ ]`' in content, "Task 1.2 unexpectedly changed"


def test_mark_task_complete_idempotent():
    """Marking twice doesn't duplicate [✓]."""
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.mark_task_complete('1.1')
        updater.mark_task_complete('1.1')
        content = open(path).read()
        assert content.count('[✓]') == 1, \
            f"Expected 1 [✓], got {content.count('[✓]')}"


def test_mark_task_in_progress():
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.mark_task_in_progress('1.2')
        content = open(path).read()
        assert '[~]' in content


def test_update_field_status():
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.update_system_status('RUNNING')
        content = open(path).read()
        assert '**Status:** RUNNING' in content


def test_update_field_uptime():
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.update_field('System Uptime', '2.5h')
        content = open(path).read()
        assert '**System Uptime:** 2.5h' in content


def test_update_field_universals():
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.update_field('Universals Discovered', '3')
        content = open(path).read()
        assert '**Universals Discovered:** 3' in content


def test_log_discovery_appears_in_content():
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.log_discovery({
            'type': 'Exponential Decay',
            'timestamp': 1700000000.0,
            'domains': ['ece', 'biology'],
            'pattern_summary': 'RC-like decay in both domains',
            'mdl_scores': {'ece': 0.3, 'biology': 0.4},
        })
        content = open(path).read()
        assert '### Discovery 1: Exponential Decay' in content
        assert 'ece, biology' in content
        assert 'RC-like decay' in content
        assert 'Confirmed ✓' in content


def test_log_discovery_before_success_criteria():
    """Discovery is inserted before ## Success Criteria."""
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.log_discovery({
            'type': 'Test',
            'timestamp': 0.0,
            'domains': ['a', 'b'],
            'pattern_summary': 'test',
            'mdl_scores': {},
        })
        content = open(path).read()
        disc_pos = content.index('### Discovery 1')
        crit_pos = content.index('## Success Criteria')
        assert disc_pos < crit_pos, "Discovery not before Success Criteria"


def test_log_discovery_count_increments():
    """Second discovery gets number 2."""
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        for i in range(3):
            updater.log_discovery({
                'type': f'Pattern {i}',
                'timestamp': float(i),
                'domains': ['a'],
                'pattern_summary': '',
                'mdl_scores': {},
            })
        content = open(path).read()
        assert '### Discovery 3:' in content


def test_append_hypothesis_replaces_placeholder():
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.append_hypothesis('**Hypothesis 1:** Exponential decay is universal.')
        content = open(path).read()
        assert 'Exponential decay is universal' in content
        assert '(awaiting first discovery)' not in content


def test_append_hypothesis_second_goes_before_task_list():
    """Second hypothesis appended before Task List."""
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        updater.append_hypothesis('**Hypothesis 1:** First.')
        updater.append_hypothesis('**Hypothesis 2:** Second.')
        content = open(path).read()
        assert 'Second.' in content
        hyp2_pos = content.index('Second.')
        task_pos = content.index('## Task List')
        assert hyp2_pos < task_pos


def test_atomic_write_no_partial_content():
    """File is never half-written: content is always valid after update_field."""
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        # Rapid sequential updates
        for i in range(20):
            updater.update_field('System Uptime', f'{i}h')
            content = open(path).read()
            # File should always be readable (not empty, not partial)
            assert len(content) > 100
            assert '**Status:**' in content


def test_thread_safety_concurrent_updates():
    """Multiple threads updating different fields simultaneously — no corruption."""
    from tensor.ficuts_updater import FICUTSUpdater
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        updater = FICUTSUpdater(path)
        errors = []

        def update_status():
            for i in range(20):
                try:
                    updater.update_system_status(f'STATE_{i}')
                except Exception as e:
                    errors.append(e)

        def update_uptime():
            for i in range(20):
                try:
                    updater.update_field('System Uptime', f'{i}h')
                except Exception as e:
                    errors.append(e)

        threads = ([threading.Thread(target=update_status) for _ in range(3)] +
                   [threading.Thread(target=update_uptime) for _ in range(3)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # File still readable and non-empty
        content = open(path).read()
        assert '**Status:**' in content
        assert len(content) > 100


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5.2 — SystemRunner wiring
# ══════════════════════════════════════════════════════════════════════════════

def test_system_runner_creates_ficuts_updater():
    """SystemRunner instantiates FICUTSUpdater when FICUTS.md exists."""
    from run_system import SystemRunner
    with tempfile.TemporaryDirectory() as d:
        path = _make_ficuts(d)
        runner = SystemRunner(ficuts_path=path)
        assert runner.ficuts is not None
        from tensor.ficuts_updater import FICUTSUpdater
        assert isinstance(runner.ficuts, FICUTSUpdater)


def test_system_runner_no_ficuts_when_missing():
    """SystemRunner.ficuts is None when FICUTS.md doesn't exist."""
    from run_system import SystemRunner
    runner = SystemRunner(ficuts_path='/tmp/nonexistent_ficuts_xyz.md')
    assert runner.ficuts is None
