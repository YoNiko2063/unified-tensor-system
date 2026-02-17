"""Tests for full-stack optimization: bootstrap, compiler stack, hardware profiler, validator.

Run: python tests/test_full_stack.py

Tests 1-12 validate the complete optimization loop from Python source
down to physical hardware.
"""
import os
import sys
import json
import tempfile
import shutil
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))
sys.path.insert(0, _ROOT)

import numpy as np


# ═══════════════════════════════════════════════════════════
# TEST 1: bootstrap_step
# ═══════════════════════════════════════════════════════════

def test_bootstrap_step():
    """Runs one bootstrap cycle on dev-agent/src/dev_agent, produces BootstrapResult."""
    from tensor.core import UnifiedTensor
    from tensor.bootstrap import BootstrapOrchestrator, BootstrapResult

    T = UnifiedTensor(max_nodes='auto', n_levels=4)
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')

    tmplog = tempfile.mkdtemp()
    try:
        orch = BootstrapOrchestrator(T, dev_path, target_consonance=0.75,
                                      log_dir=tmplog)
        result = orch.run_bootstrap_step()

        assert isinstance(result, BootstrapResult)
        assert isinstance(result.improved, bool)
        assert isinstance(result.consonance_delta, float)
        assert isinstance(result.files_changed, list)
        assert isinstance(result.high_tension_nodes, list)
        assert len(result.high_tension_nodes) > 0
        assert result.step == 1
        assert result.consonance_before >= 0
        assert result.consonance_after >= 0
        print(f"  Bootstrap step 1: consonance {result.consonance_before:.4f}"
              f"→{result.consonance_after:.4f}")
        print(f"  High tension nodes: {result.high_tension_nodes[:3]}")
        print(f"  PASS")
    finally:
        shutil.rmtree(tmplog, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 2: functionality_database
# ═══════════════════════════════════════════════════════════

def test_functionality_database():
    """Emits valid JSON with correct fields for all parsed modules."""
    from tensor.core import UnifiedTensor
    from tensor.bootstrap import BootstrapOrchestrator

    T = UnifiedTensor(max_nodes='auto', n_levels=4)
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')

    tmplog = tempfile.mkdtemp()
    try:
        orch = BootstrapOrchestrator(T, dev_path, log_dir=tmplog)
        db = orch.functionality_database()

        assert isinstance(db, dict)
        assert len(db) > 0

        # Check required fields
        for name, entry in list(db.items())[:5]:
            assert 'file' in entry, f"Missing 'file' in {name}"
            assert 'responsibility' in entry, f"Missing 'responsibility' in {name}"
            assert 'tensor_node' in entry, f"Missing 'tensor_node' in {name}"
            assert 'free_energy' in entry, f"Missing 'free_energy' in {name}"
            assert 'eigenvalue' in entry, f"Missing 'eigenvalue' in {name}"
            assert 'calls' in entry, f"Missing 'calls' in {name}"
            assert 'called_by' in entry, f"Missing 'called_by' in {name}"
            assert 'language_level' in entry, f"Missing 'language_level' in {name}"
            assert entry['language_level'] == 'python'

        # Verify saved to disk
        db_path = os.path.join(tmplog, 'functionality_db.json')
        assert os.path.exists(db_path)
        with open(db_path) as f:
            saved = json.load(f)
        assert len(saved) == len(db)

        print(f"  Functionality DB: {len(db)} modules")
        sample = list(db.values())[0]
        print(f"  Sample: {sample['responsibility']}")
        print(f"  PASS")
    finally:
        shutil.rmtree(tmplog, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 3: python_to_mna
# ═══════════════════════════════════════════════════════════

def test_python_to_mna():
    """Wraps code_graph correctly."""
    from tensor.compiler_stack import CompilerStack
    from core.matrix import MNASystem

    cs = CompilerStack()
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')
    mna = cs.python_to_mna(dev_path)

    assert isinstance(mna, MNASystem)
    assert mna.n_total > 0
    assert mna.G.shape[0] == mna.n_total
    assert np.allclose(mna.G, mna.G.T, atol=1e-10)
    print(f"  Python→MNA: {mna.n_total} nodes")
    print(f"  PASS")


# ═══════════════════════════════════════════════════════════
# TEST 4: bytecode_to_mna
# ═══════════════════════════════════════════════════════════

def test_bytecode_to_mna():
    """Parses a simple Python file, produces valid MNASystem with n_nodes > 0."""
    from tensor.compiler_stack import CompilerStack
    from core.matrix import MNASystem

    # Create a simple test file
    tmpdir = tempfile.mkdtemp()
    try:
        test_file = os.path.join(tmpdir, 'test_bc.py')
        with open(test_file, 'w') as f:
            f.write('''
def add(a, b):
    return a + b

def multiply(a, b):
    result = 0
    for i in range(b):
        result += a
    return result

class Calculator:
    def __init__(self):
        self.history = []

    def compute(self, op, a, b):
        if op == 'add':
            r = add(a, b)
        elif op == 'mul':
            r = multiply(a, b)
        else:
            r = 0
        self.history.append(r)
        return r

x = Calculator()
print(x.compute('add', 3, 4))
''')
        cs = CompilerStack()
        mna = cs.bytecode_to_mna(test_file)

        assert isinstance(mna, MNASystem)
        assert mna.n_total > 0, f"Expected n_nodes > 0, got {mna.n_total}"
        assert mna.G.shape == (mna.n_total, mna.n_total)
        assert np.allclose(mna.G, mna.G.T, atol=1e-10)
        assert np.all(np.isfinite(mna.G))

        print(f"  Bytecode→MNA: {mna.n_total} opcode nodes")
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(mna.G)))[::-1]
        print(f"  Top eigenvalues: {eigvals[:5]}")
        print(f"  PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 5: phi_between_levels
# ═══════════════════════════════════════════════════════════

def test_phi_between_levels():
    """Python→bytecode coarsening preserves eigenvalue ratios within tolerance."""
    from tensor.compiler_stack import CompilerStack

    cs = CompilerStack()
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')
    python_mna = cs.python_to_mna(dev_path)

    result = cs.phi_between('python', 'bytecode', python_mna)

    assert result.level_high == 'python'
    assert result.level_low == 'bytecode'
    assert result.projection is not None
    assert isinstance(result.ratio_error, float)
    assert result.ratio_error >= 0

    # Eigenvalue ratios should be preserved within tolerance
    assert result.ratio_error <= 1.0, (
        f"Ratio error {result.ratio_error:.4f} exceeds tolerance")

    print(f"  φ(python→bytecode): ratio_error={result.ratio_error:.4f}")
    print(f"  Projection shape: {result.projection.shape}")
    print(f"  High ratios: {result.eigenvalue_ratios_high[:5]}")
    print(f"  Low ratios: {result.eigenvalue_ratios_low[:5]}")
    print(f"  PASS")


# ═══════════════════════════════════════════════════════════
# TEST 6: hardware_profiler
# ═══════════════════════════════════════════════════════════

def test_hardware_profiler():
    """Reads THIS machine, produces HardwareProfile with cpu_cores > 0."""
    from tensor.hardware_profiler import HardwareProfiler, HardwareProfile

    profiler = HardwareProfiler()
    profile = profiler.profile()

    assert isinstance(profile, HardwareProfile)
    assert profile.cpu_cores > 0, f"cpu_cores should be >0, got {profile.cpu_cores}"
    assert profile.cpu_threads > 0
    assert profile.ram_total_gb > 0

    print(f"  CPU: {profile.cpu_model}")
    print(f"  Cores: {profile.cpu_cores}, Threads: {profile.cpu_threads}")
    print(f"  RAM: {profile.ram_total_gb:.1f} GB")
    print(f"  L1/L2/L3: {profile.l1_cache_kb:.0f}K/{profile.l2_cache_kb:.0f}K/"
          f"{profile.l3_cache_kb:.0f}K")
    print(f"  SIMD: {profile.simd_capabilities}")
    print(f"  PASS")


# ═══════════════════════════════════════════════════════════
# TEST 7: hardware_to_mna
# ═══════════════════════════════════════════════════════════

def test_hardware_to_mna():
    """Produces valid 8-node MNASystem."""
    from tensor.hardware_profiler import HardwareProfiler, FUNCTIONAL_UNITS
    from core.matrix import MNASystem

    profiler = HardwareProfiler()
    profile = profiler.profile()
    mna = profiler.to_mna(profile)

    assert isinstance(mna, MNASystem)
    assert mna.n_total == 8, f"Expected 8 nodes, got {mna.n_total}"
    assert mna.G.shape == (8, 8)
    assert mna.C.shape == (8, 8)
    assert np.allclose(mna.G, mna.G.T, atol=1e-10)
    assert np.all(np.isfinite(mna.G))
    assert np.all(np.isfinite(mna.C))

    eigvals = np.sort(np.abs(np.linalg.eigvalsh(mna.G)))[::-1]
    print(f"  Hardware MNA: {mna.n_total} nodes ({', '.join(FUNCTIONAL_UNITS)})")
    print(f"  Eigenvalues: {eigvals}")
    print(f"  PASS")


# ═══════════════════════════════════════════════════════════
# TEST 8: l3_populated
# ═══════════════════════════════════════════════════════════

def test_l3_populated():
    """After hardware_profiler.profile(), tensor L3 is populated and snapshot shows it."""
    from tensor.core import UnifiedTensor
    from tensor.hardware_profiler import HardwareProfiler
    from tensor.observer import TensorObserver

    T = UnifiedTensor(max_nodes='auto', n_levels=4)
    profiler = HardwareProfiler()
    profile = profiler.profile()
    mna = profiler.to_mna(profile)
    T.update_level(3, mna, t=time.time())

    snap = T.tensor_snapshot()
    assert snap['levels'][3]['populated'] is True
    assert snap['levels'][3]['n_nodes'] == 8
    assert snap['levels'][3]['name'] == 'hardware'

    sig = snap['levels'][3]['harmonic_signature']
    assert 'consonance_score' in sig
    assert 'dominant_interval' in sig

    # Observer should show L3
    obs = TensorObserver(T)
    md = obs.snapshot_markdown()
    assert '| L3 hardware' in md
    ctx = obs.to_agent_context()
    assert 'L3' in ctx

    print(f"  L3 hardware: {snap['levels'][3]['n_nodes']} nodes")
    print(f"  Consonance: {sig['consonance_score']:.4f}")
    print(f"  Interval: {sig['dominant_interval']}")
    print(f"  Verdict: {sig['stability_verdict']}")
    print(f"  Snapshot shows L3: yes")
    print(f"  PASS")


# ═══════════════════════════════════════════════════════════
# TEST 9: code_validator_approve
# ═══════════════════════════════════════════════════════════

def test_code_validator_approve():
    """Valid simple file approved."""
    from tensor.core import UnifiedTensor
    from tensor.code_graph import CodeGraph
    from tensor.code_validator import CodeValidator, ValidationResult

    T = UnifiedTensor(max_nodes='auto', n_levels=4)
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')

    # Populate L2
    cg = CodeGraph.from_directory(dev_path, max_files=100)
    mna = cg.to_mna()
    T.update_level(2, mna, t=time.time())

    validator = CodeValidator(T, dev_path, max_files=100)

    # Simple, well-structured code should be approved
    proposed = '''"""A simple utility module."""

def hello():
    return "world"

def add(a, b):
    return a + b
'''
    tmpdir = tempfile.mkdtemp()
    try:
        test_file = os.path.join(tmpdir, 'test_simple.py')
        result = validator.validate(test_file, proposed)

        assert isinstance(result, ValidationResult)
        assert result.approved is True, f"Should be approved, got: {result.reason}"
        assert isinstance(result.consonance_delta, float)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

        print(f"  Validation: approved={result.approved}")
        print(f"  Consonance delta: {result.consonance_delta:.4f}")
        print(f"  Reason: {result.reason}")
        print(f"  PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 10: code_validator_reject
# ═══════════════════════════════════════════════════════════

def test_code_validator_reject():
    """File with syntax error or massive complexity is rejected with reason."""
    from tensor.core import UnifiedTensor
    from tensor.code_graph import CodeGraph
    from tensor.code_validator import CodeValidator, ValidationResult

    T = UnifiedTensor(max_nodes='auto', n_levels=4)
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')

    cg = CodeGraph.from_directory(dev_path, max_files=100)
    mna = cg.to_mna()
    T.update_level(2, mna, t=time.time())

    validator = CodeValidator(T, dev_path, max_files=100)

    # Invalid Python — syntax error should be rejected
    bad_code = 'def broken(\n  this is not valid python'
    tmpdir = tempfile.mkdtemp()
    try:
        test_file = os.path.join(tmpdir, 'bad_code.py')
        result = validator.validate(test_file, bad_code)

        assert isinstance(result, ValidationResult)
        assert result.approved is False, "Syntax error should be rejected"
        assert 'Syntax error' in result.reason
        assert len(result.suggestions) > 0

        print(f"  Rejection: approved={result.approved}")
        print(f"  Reason: {result.reason}")
        print(f"  Suggestions: {result.suggestions}")
        print(f"  PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 11: suggest_structure
# ═══════════════════════════════════════════════════════════

def test_suggest_structure():
    """Returns dict with file suggestions."""
    from tensor.core import UnifiedTensor
    from tensor.code_graph import CodeGraph
    from tensor.code_validator import CodeValidator

    T = UnifiedTensor(max_nodes='auto', n_levels=4)
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')

    cg = CodeGraph.from_directory(dev_path, max_files=100)
    mna = cg.to_mna()
    T.update_level(2, mna, t=time.time())

    validator = CodeValidator(T, dev_path, max_files=100)
    suggestion = validator.suggest_structure('Add a new caching layer')

    assert isinstance(suggestion, dict)
    assert 'behavior' in suggestion
    assert 'recommended_files' in suggestion
    assert 'max_lines_per_file' in suggestion
    assert 'structure_advice' in suggestion
    assert isinstance(suggestion['structure_advice'], list)
    assert suggestion['recommended_files'] >= 1
    assert suggestion['max_lines_per_file'] == 200

    print(f"  Suggestion for 'caching layer':")
    print(f"    Recommended files: {suggestion['recommended_files']}")
    print(f"    Current modules: {suggestion['current_modules']}")
    print(f"    Dominant interval: {suggestion['dominant_interval']}")
    print(f"    Advice: {suggestion['structure_advice'][:2]}")
    print(f"  PASS")


# ═══════════════════════════════════════════════════════════
# TEST 12: full_loop
# ═══════════════════════════════════════════════════════════

def test_full_loop():
    """Full optimization loop: parse→profile→resonance→bootstrap→validate→snapshot."""
    from tensor.core import UnifiedTensor
    from tensor.code_graph import CodeGraph
    from tensor.market_graph import MarketGraph
    from tensor.hardware_profiler import HardwareProfiler
    from tensor.bootstrap import BootstrapOrchestrator
    from tensor.code_validator import CodeValidator, ValidationResult
    from tensor.observer import TensorObserver

    T = UnifiedTensor(max_nodes='auto', n_levels=4)
    dev_path = os.path.join(_ROOT, 'dev-agent', 'src', 'dev_agent')

    # --- L0: Market (mock) ---
    mg = MarketGraph()
    for sym, sec in [('AAPL', 'tech'), ('MSFT', 'tech'), ('JPM', 'fin')]:
        mg.add_ticker(sym, sec, price=100, momentum=0.01, volatility=0.015)
    mg.set_correlation('AAPL', 'MSFT', 0.85)
    mna_market = mg.to_mna()
    T.update_level(0, mna_market, t=time.time())
    print(f"  L0 market: {mna_market.n_total} nodes")

    # --- L2: Code structure ---
    cg = CodeGraph.from_directory(dev_path, max_files=100)
    mna_code = cg.to_mna()
    T.update_level(2, mna_code, t=time.time())
    print(f"  L2 code: {mna_code.n_total} nodes")

    # --- L3: Hardware ---
    profiler = HardwareProfiler()
    profile = profiler.profile()
    mna_hw = profiler.to_mna(profile)
    T.update_level(3, mna_hw, t=time.time())
    print(f"  L3 hardware: {mna_hw.n_total} nodes")

    # --- Resonance(L2, L3) ---
    res_before = T.cross_level_resonance(2, 3)
    assert isinstance(res_before, float)
    assert 0.0 <= res_before <= 1.0
    print(f"  resonance(code, hardware) = {res_before:.4f}")

    # --- Bootstrap one step ---
    tmplog = tempfile.mkdtemp()
    try:
        orch = BootstrapOrchestrator(T, dev_path, log_dir=tmplog)
        result = orch.run_bootstrap_step()
        print(f"  Bootstrap: consonance {result.consonance_before:.4f}"
              f"→{result.consonance_after:.4f}")

        # Re-measure resonance
        res_after = T.cross_level_resonance(2, 3)
        print(f"  resonance after bootstrap: {res_after:.4f}")

        # --- Validator approves bootstrap output ---
        validator = CodeValidator(T, dev_path, max_files=100)
        simple_code = '"""Module."""\ndef foo(): return 42\n'
        tmpfile = os.path.join(tmplog, 'test_valid.py')
        val_result = validator.validate(tmpfile, simple_code)
        assert isinstance(val_result, ValidationResult)
        print(f"  Validator: approved={val_result.approved}, "
              f"reason={val_result.reason}")

        # --- Snapshot shows all 4 levels ---
        obs = TensorObserver(T, log_dir=tmplog)
        snap = T.tensor_snapshot()
        populated_levels = [l for l in range(4) if snap['levels'][l].get('populated')]
        assert 0 in populated_levels, "L0 should be populated"
        assert 2 in populated_levels, "L2 should be populated"
        assert 3 in populated_levels, "L3 should be populated"

        md = obs.snapshot_markdown()
        assert '| L0' in md
        assert '| L2' in md
        assert '| L3 hardware' in md

        # --- Hardware report ---
        hw_report = profiler.hardware_report()
        assert 'Computational Geometry' in hw_report
        print(f"\n  === HARDWARE REPORT ===")
        print(hw_report)

        # --- Functionality DB ---
        db = orch.functionality_database()
        print(f"  Functionality DB: {len(db)} entries")

        # --- Example ValidationResults ---
        print(f"\n  === VALIDATION EXAMPLES ===")
        print(f"  Approve: approved={val_result.approved}, "
              f"cons_delta={val_result.consonance_delta:.4f}")

        bad_result = validator.validate(tmpfile, 'def broken(\n  not valid')
        print(f"  Reject:  approved={bad_result.approved}, "
              f"reason={bad_result.reason}")

        print(f"\n  === TENSOR SNAPSHOT ===")
        for l in range(4):
            linfo = snap['levels'][l]
            if not linfo.get('populated'):
                print(f"  L{l} ({linfo['name']}): empty")
                continue
            sig = linfo['harmonic_signature']
            print(f"  L{l} ({linfo['name']}): {linfo['n_nodes']} nodes | "
                  f"gap={linfo['eigenvalue_gap']:.4f} | "
                  f"key={sig['dominant_interval']} | "
                  f"verdict={sig['stability_verdict']}")
            for other, res in linfo.get('cross_level_resonance', {}).items():
                print(f"    resonance with {other}: {res:.4f}")

        print(f"\n  Final outputs:")
        print(f"  1. Tensor snapshot: {len(populated_levels)} levels populated")
        print(f"  2. Hardware report: {len(hw_report)} chars")
        print(f"  3. Functionality DB: {len(db)} entries")
        print(f"  4. resonance(L2,L3): {res_after:.4f}")
        print(f"  5. Approve: {val_result.approved} | Reject: {bad_result.approved}")
        print(f"  PASS")

    finally:
        shutil.rmtree(tmplog, ignore_errors=True)


if __name__ == '__main__':
    tests = [
        ("bootstrap_step", test_bootstrap_step),
        ("functionality_database", test_functionality_database),
        ("python_to_mna", test_python_to_mna),
        ("bytecode_to_mna", test_bytecode_to_mna),
        ("phi_between_levels", test_phi_between_levels),
        ("hardware_profiler", test_hardware_profiler),
        ("hardware_to_mna", test_hardware_to_mna),
        ("l3_populated", test_l3_populated),
        ("code_validator_approve", test_code_validator_approve),
        ("code_validator_reject", test_code_validator_reject),
        ("suggest_structure", test_suggest_structure),
        ("full_loop", test_full_loop),
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
    print("All full-stack tests passed.")
