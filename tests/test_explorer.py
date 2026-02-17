"""Tests for ConfigurationExplorer: precompute, batch score, RAM buffer, checkpoint.

Run: python tests/test_explorer.py
"""
import os
import sys
import time
import tempfile
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))
sys.path.insert(0, _ROOT)

# Set BLAS threads before numpy
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import numpy as np
from tensor.explorer import (
    ConfigurationExplorer, ExplorerConfig, PrecomputedManifold,
    ResultsBuffer, score_batch, _score_bandpass, _score_snn,
    ExplorationTarget, _skip_zero_eigenvalues,
)


# ═══════════════════════════════════════════════════════════
# TEST 1: Precompute 1000 configs
# ═══════════════════════════════════════════════════════════

def test_precompute_1000():
    manifold = PrecomputedManifold(max_n=32)
    t0 = time.time()
    manifold.build(1000, n_workers=2, progress_cb=None)
    dt = time.time() - t0

    assert manifold.n_configs == 1000
    assert manifold._G is not None
    assert manifold._G.shape == (1000, 32, 32)
    assert manifold._eigvals.shape == (1000, 32)

    # All eigenvalues should be finite
    assert np.all(np.isfinite(manifold._eigvals))

    # All cons scores in [0, 1]
    assert np.all(manifold._cons_scores >= 0)
    assert np.all(manifold._cons_scores <= 1)

    # Multiple node sizes should be present
    sizes = set(manifold._n_nodes.tolist())
    assert len(sizes) >= 3, f"Expected >=3 node sizes, got {sizes}"

    ram = manifold.ram_usage_mb()
    print(f"  Precomputed 1000 configs in {dt:.1f}s, {ram:.0f}MB RAM")
    print(f"  Node sizes: {sorted(sizes)}")
    print(f"  Mean consonance: {np.mean(manifold._cons_scores):.4f}")
    print(f"  PASS: precompute_1000")


# ═══════════════════════════════════════════════════════════
# TEST 2: Batch score 256 configs in one step
# ═══════════════════════════════════════════════════════════

def test_batch_score_256():
    batch = 256
    max_n = 16
    rng = np.random.default_rng(42)

    # Generate batch of random G matrices
    G_stack = np.zeros((batch, max_n, max_n))
    n_nodes = np.full(batch, 8, dtype=np.int32)

    for b in range(batch):
        G = np.zeros((max_n, max_n))
        for i in range(7):
            g = 0.1 + rng.random() * 5.0
            G[i, i] += g; G[i+1, i+1] += g
            G[i, i+1] -= g; G[i+1, i] -= g
        G[:8, :8] += 1e-6 * np.eye(8)
        G_stack[b] = G

    t0 = time.time()
    scores, eigvals = score_batch(G_stack, n_nodes, _score_bandpass)
    dt = time.time() - t0

    assert scores.shape == (batch,)
    assert eigvals.shape == (batch, max_n)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)

    print(f"  Batch scored {batch} configs in {dt*1000:.1f}ms")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Mean score: {scores.mean():.4f}")
    print(f"  PASS: batch_score_256")


# ═══════════════════════════════════════════════════════════
# TEST 3: RAM buffer holds 10k results without leak
# ═══════════════════════════════════════════════════════════

def test_ram_buffer_10k():
    buf = ResultsBuffer(max_results=20000, max_n=16)

    rng = np.random.default_rng(42)
    for i in range(10000):
        eigvals = np.sort(rng.random(16))[::-1]
        score = rng.random()
        params = rng.random(16)
        buf.add(eigvals, score, params, config_idx=i % 100, step=i)

    assert buf.count == 10000
    assert buf.best_score > 0
    ram = buf.ram_usage_mb()

    # No leak: buffer size should be fixed
    assert ram < 100, f"Buffer RAM too high: {ram:.0f}MB"

    # Best configs should return valid data
    best = buf.best_configurations(5)
    assert len(best) == 5
    assert best[0]['score'] >= best[1]['score']
    assert all(b['score'] > 0 for b in best)

    # Scores above 0.5
    scores = buf.scores()
    above_05 = int(np.sum(scores > 0.5))
    assert above_05 > 0, "Should have some scores > 0.5"

    print(f"  Buffer: {buf.count} results, {ram:.1f}MB RAM")
    print(f"  Best score: {buf.best_score:.4f}")
    print(f"  Above 0.5: {above_05}")
    print(f"  PASS: ram_buffer_10k")


# ═══════════════════════════════════════════════════════════
# TEST 4: Checkpoint saves/loads correctly via npz
# ═══════════════════════════════════════════════════════════

def test_checkpoint():
    tmpdir = tempfile.mkdtemp()
    try:
        buf = ResultsBuffer(max_results=5000, max_n=16)
        rng = np.random.default_rng(42)
        for i in range(1000):
            eigvals = np.sort(rng.random(16))[::-1]
            score = rng.random()
            params = rng.random(16)
            buf.add(eigvals, score, params, config_idx=i, step=i)

        path = os.path.join(tmpdir, 'test_checkpoint.npz')
        buf.save_checkpoint(path)
        assert os.path.exists(path)

        # Load into fresh buffer
        buf2 = ResultsBuffer(max_results=5000, max_n=16)
        buf2.load_checkpoint(path)

        assert buf2.count == buf.count
        assert abs(buf2.best_score - buf.best_score) < 1e-10

        # Scores should match
        s1 = buf.scores()
        s2 = buf2.scores()
        assert np.allclose(s1, s2)

        fsize = os.path.getsize(path)
        print(f"  Checkpoint: {fsize / 1024:.0f}KB for {buf.count} results")
        print(f"  Load verified: {buf2.count} results, best={buf2.best_score:.4f}")
        print(f"  PASS: checkpoint")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 5: Rate test — 100 steps in <10 seconds
# ═══════════════════════════════════════════════════════════

def test_rate_100_steps():
    tmpdir = tempfile.mkdtemp()
    try:
        config = ExplorerConfig(
            n_precompute=200,  # small for test speed
            batch_size=64,
            max_results=50000,
            log_dir=tmpdir,
            target='bandpass',
        )
        explorer = ConfigurationExplorer(config)
        explorer.precompute(progress=False)

        t0 = time.time()
        stats = explorer.run(100, progress=False)
        dt = time.time() - t0

        assert dt < 10.0, f"100 steps took {dt:.1f}s, expected <10s"
        assert stats['total_configs_scored'] == 100 * 64
        assert stats['best_score'] > 0

        rate = stats['configs_per_second']
        print(f"  100 steps in {dt:.2f}s ({rate:.0f} configs/s)")
        print(f"  Total scored: {stats['total_configs_scored']:,}")
        print(f"  Best: {stats['best_score']:.4f}")
        print(f"  PASS: rate_100_steps")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 6: After 1000 steps — >100 configs with score>0.5
# ═══════════════════════════════════════════════════════════

def test_1000_steps_quality():
    tmpdir = tempfile.mkdtemp()
    try:
        config = ExplorerConfig(
            n_precompute=500,
            batch_size=128,
            max_results=200000,
            log_dir=tmpdir,
            target='bandpass',
        )
        explorer = ConfigurationExplorer(config)
        explorer.precompute(progress=False)

        stats = explorer.run(1000, progress=False)

        scores = explorer.results.scores()
        above_05 = int(np.sum(scores > 0.5))
        assert above_05 > 100, (
            f"Expected >100 configs with score>0.5, got {above_05}")

        best = explorer.best_configurations(5)
        assert len(best) == 5
        assert best[0]['score'] > 0.5

        print(f"  1000 steps: {stats['total_configs_scored']:,} scored")
        print(f"  Above 0.5: {above_05}")
        print(f"  Best score: {best[0]['score']:.4f}")
        print(f"  Rate: {stats['configs_per_second']:.0f} configs/s")
        print(f"  PASS: 1000_steps_quality")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 7: Zero eigenvalue fix
# ═══════════════════════════════════════════════════════════

def test_zero_eigenvalue_fix():
    # Create eigenvalues with near-zero ground node
    ev_with_zero = np.array([1e-15, 0.5, 1.0, 2.0, 5.0])
    clean = _skip_zero_eigenvalues(np.sort(np.abs(ev_with_zero))[::-1])

    # Should skip the near-zero eigenvalue
    assert len(clean) == 4, f"Expected 4 eigenvalues, got {len(clean)}"
    assert clean[0] == 5.0
    assert np.all(clean > 1e-10)

    # All-zero case returns fallback
    all_zero = _skip_zero_eigenvalues(np.array([1e-20, 1e-30, 0.0]))
    assert len(all_zero) == 1

    # Score functions should handle zero eigenvalues gracefully
    ev = np.array([1e-15, 0.001, 0.5, 1.0, 2.0])
    s_bp = _score_bandpass(ev, 0.5, 'octave')
    s_snn = _score_snn(ev, 0.5, 'octave')
    assert s_bp > 0, f"Bandpass score should be > 0, got {s_bp}"
    assert s_snn > 0, f"SNN score should be > 0, got {s_snn}"

    # Run explorer with small manifold, check top configs have non-zero λ₁
    tmpdir = tempfile.mkdtemp()
    try:
        config = ExplorerConfig(
            n_precompute=100, batch_size=32, max_results=10000,
            log_dir=tmpdir, target='bandpass')
        explorer = ConfigurationExplorer(config)
        explorer.precompute(progress=False)
        explorer.run(50, progress=False)

        best = explorer.best_configurations(10)
        zero_count = sum(1 for b in best
                         if len(b['eigenvalues']) > 0 and abs(b['eigenvalues'][0]) < 1e-10)
        assert zero_count < 10, f"Top-10 all have λ₁≈0 ({zero_count}/10)"
        print(f"  Top-10: {zero_count}/10 have λ₁≈0 (should be <10)")
        print(f"  PASS: zero_eigenvalue_fix")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 8: Score calibration — ≥1% of configs score > 0.5
# ═══════════════════════════════════════════════════════════

def test_score_calibration():
    tmpdir = tempfile.mkdtemp()
    try:
        config = ExplorerConfig(
            n_precompute=500, batch_size=128, max_results=200000,
            log_dir=tmpdir, target='bandpass')
        explorer = ConfigurationExplorer(config)
        explorer.precompute(progress=False)
        explorer.run(500, progress=False)

        scores = explorer.results.scores()
        above_05 = int(np.sum(scores > 0.5))
        total = len(scores)
        pct = 100.0 * above_05 / max(total, 1)

        assert pct >= 1.0, (
            f"Expected >=1% above 0.5, got {pct:.1f}% ({above_05}/{total})")

        print(f"  {above_05}/{total} ({pct:.1f}%) scored > 0.5")
        print(f"  PASS: score_calibration")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 9: Navigation diversity — top-10 show ≥5 different configs
# ═══════════════════════════════════════════════════════════

def test_navigation_diversity():
    tmpdir = tempfile.mkdtemp()
    try:
        config = ExplorerConfig(
            n_precompute=200, batch_size=64, max_results=50000,
            log_dir=tmpdir, target='bandpass')
        explorer = ConfigurationExplorer(config)
        explorer.precompute(progress=False)
        explorer.run(200, progress=False)

        best = explorer.best_configurations(10)
        unique_configs = set(b['config_idx'] for b in best)
        n_unique = len(unique_configs)

        assert n_unique >= 5, (
            f"Expected >=5 unique configs in top-10, got {n_unique}: {unique_configs}")

        print(f"  Top-10 configs: {n_unique} unique out of 10")
        print(f"  Config indices: {sorted(unique_configs)}")
        print(f"  PASS: navigation_diversity")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 10: RAM scaling — scale_to_ram produces >100MB
# ═══════════════════════════════════════════════════════════

def test_ram_scaling():
    manifold = PrecomputedManifold(max_n=32)
    # Scale to 0.5GB (should produce ~30k configs)
    manifold.scale_to_ram(0.5, n_workers=2)

    ram = manifold.ram_usage_mb()
    assert ram > 100, f"Expected >100MB RAM, got {ram:.0f}MB"
    assert manifold.n_configs > 5000, (
        f"Expected >5000 configs, got {manifold.n_configs}")

    print(f"  Scaled to {manifold.n_configs:,} configs, {ram:.0f}MB RAM")
    print(f"  PASS: ram_scaling")


# ═══════════════════════════════════════════════════════════
# TEST 11: All 5 logic gates construct without error
# ═══════════════════════════════════════════════════════════

def test_logic_gate_construct():
    gates = ['NOT', 'AND', 'OR', 'NAND', 'XOR']
    for gate_type in gates:
        target = ExplorationTarget.logic_gate(gate_type)
        assert target.name == f'logic_{gate_type}'
        assert target.tensor_level == 1
        assert target.level_name == 'neural'
        assert target.eigenvalue_semantics == 'switching_threshold'
        assert len(target.target_ratios) >= 2
        assert callable(target)

        # Test scoring with dummy eigenvalues
        ev = np.array([5.0, 3.0, 1.0, 0.5])
        score = target(ev, 0.5, 'octave')
        assert 0 <= score <= 2.0, f"{gate_type} score out of range: {score}"

    # Invalid gate should raise
    try:
        ExplorationTarget.logic_gate('BUFFER')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print(f"  All 5 gates construct and score correctly")
    print(f"  PASS: logic_gate_construct")


# ═══════════════════════════════════════════════════════════
# TEST 12: NAND target scores tritone-signature configs
# ═══════════════════════════════════════════════════════════

def test_nand_scoring():
    nand = ExplorationTarget.logic_gate('NAND')
    # Target ratios: [1.0, 1.414, 2.0, 2.828] (powers of sqrt(2))

    # Perfect tritone signature should score well
    ev_tritone = np.array([2.828, 2.0, 1.414, 1.0])
    score_good = nand(ev_tritone, 0.5, 'tritone')

    # Random eigenvalues should score worse
    ev_random = np.array([10.0, 3.0, 0.5, 0.1])
    score_bad = nand(ev_random, 0.5, 'octave')

    assert score_good > score_bad, (
        f"Tritone score ({score_good:.4f}) should beat random ({score_bad:.4f})")
    assert score_good > 0.4, f"NAND tritone score should be > 0.4, got {score_good:.4f}"

    print(f"  NAND tritone score: {score_good:.4f}")
    print(f"  NAND random score: {score_bad:.4f}")
    print(f"  PASS: nand_scoring")


# ═══════════════════════════════════════════════════════════
# TEST 13: Diagnose mode returns histogram
# ═══════════════════════════════════════════════════════════

def test_diagnose_mode():
    tmpdir = tempfile.mkdtemp()
    try:
        config = ExplorerConfig(
            n_precompute=200, batch_size=64, max_results=10000,
            log_dir=tmpdir, target='bandpass')
        explorer = ConfigurationExplorer(config)
        explorer.precompute(progress=False)

        diag = explorer.diagnose(n_samples=200)

        assert 'histogram' in diag
        hist, bin_edges = diag['histogram']
        assert len(hist) == 10, f"Expected 10 bins, got {len(hist)}"
        assert len(bin_edges) == 11
        assert sum(hist) == 200

        assert 'top_patterns' in diag
        assert len(diag['top_patterns']) == 10

        assert 'recommended_ratios' in diag
        assert len(diag['recommended_ratios']) > 0

        assert 'mean_score' in diag
        assert 0 <= diag['mean_score'] <= 1

        print(f"  Histogram bins: {hist}")
        print(f"  Mean score: {diag['mean_score']:.4f}")
        print(f"  Max score: {diag['max_score']:.4f}")
        print(f"  PASS: diagnose_mode")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# TEST 14: Level semantics — all factory methods set correctly
# ═══════════════════════════════════════════════════════════

def test_level_semantics():
    targets = [
        ('bandpass', ExplorationTarget.bandpass_filter(1000, 10),
         0, 'market', 'squared_frequency', 'frequency_spacing', 'Hz\u00b2'),
        ('snn', ExplorationTarget.snn_configuration(16, 0.8),
         1, 'neural', 'synaptic_time_constant', 'firing_rate_ratio', 'ms\u207b\u00b9'),
        ('logic_NAND', ExplorationTarget.logic_gate('NAND'),
         1, 'neural', 'switching_threshold', 'logic_voltage_ratio', 'V\u00b2'),
        ('code', ExplorationTarget.code_structure(5.0, 0.3),
         2, 'code', 'module_coupling_strength', 'dependency_depth_ratio', 'complexity'),
        ('cross', ExplorationTarget.cross_level_resonance(0, 2, 0.8),
         -1, 'resonance_market_code', 'cross_level_resonance', 'harmonic_alignment',
         'dimensionless'),
    ]

    for label, t, level, level_name, ev_sem, ratio_sem, unit in targets:
        assert t.tensor_level == level, (
            f"{label}: expected level {level}, got {t.tensor_level}")
        assert t.level_name == level_name, (
            f"{label}: expected level_name {level_name!r}, got {t.level_name!r}")
        assert t.eigenvalue_semantics == ev_sem, (
            f"{label}: expected eigenvalue_semantics {ev_sem!r}, got {t.eigenvalue_semantics!r}")
        assert t.ratio_semantics == ratio_sem, (
            f"{label}: expected ratio_semantics {ratio_sem!r}, got {t.ratio_semantics!r}")
        assert t.physical_unit == unit, (
            f"{label}: expected physical_unit {unit!r}, got {t.physical_unit!r}")

    print(f"  All 5 factory methods set level semantics correctly")
    print(f"  PASS: level_semantics")


# ═══════════════════════════════════════════════════════════
# TEST 15: Cross-level target constructs with tensor_level == -1
# ═══════════════════════════════════════════════════════════

def test_cross_level_target():
    target = ExplorationTarget.cross_level_resonance(
        level_a=0, level_b=2, target_resonance=0.8)

    assert target.tensor_level == -1
    assert 'resonance' in target.level_name
    assert target.eigenvalue_semantics == 'cross_level_resonance'
    assert target.physical_unit == 'dimensionless'
    assert target.physical_constraints['level_a'] == 0
    assert target.physical_constraints['level_b'] == 2
    assert target.physical_constraints['target_resonance'] == 0.8
    assert len(target.target_ratios) == 6

    # Should be callable and score
    ev = np.array([5.0, 3.0, 2.0, 1.0])
    score = target(ev, 0.5, 'octave')
    assert 0 <= score <= 2.0, f"Cross-level score out of range: {score}"

    print(f"  Cross-level target: {target}")
    print(f"  Score on dummy ev: {score:.4f}")
    print(f"  PASS: cross_level_target")


# ═══════════════════════════════════════════════════════════
# TEST 16: Code structure target
# ═══════════════════════════════════════════════════════════

def test_code_structure_target():
    target = ExplorationTarget.code_structure(
        target_complexity=5.0, target_coupling=0.3)

    assert target.tensor_level == 2
    assert target.level_name == 'code'
    assert target.eigenvalue_semantics == 'module_coupling_strength'
    assert target.physical_unit == 'complexity'

    # target_ratios should have ceil(5.0) = 5 elements
    assert len(target.target_ratios) == 5, (
        f"Expected 5 ratios, got {len(target.target_ratios)}")

    # Ratios should decay: 1.0 / (1 + k*0.3)
    expected = [1.0 / (1.0 + k * 0.3) for k in range(5)]
    assert np.allclose(target.target_ratios, expected), (
        f"Ratios mismatch: {target.target_ratios} vs {expected}")

    # Different params should produce different ratios
    t2 = ExplorationTarget.code_structure(
        target_complexity=3.0, target_coupling=0.5)
    assert len(t2.target_ratios) == 3
    assert not np.allclose(t2.target_ratios[:3], target.target_ratios[:3])

    # Should be callable
    ev = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    score = target(ev, 0.5, 'octave')
    assert 0 <= score <= 2.0, f"Code structure score out of range: {score}"

    print(f"  Code structure target: {target}")
    print(f"  Ratios: {list(np.round(target.target_ratios, 3))}")
    print(f"  Score: {score:.4f}")
    print(f"  PASS: code_structure_target")


if __name__ == '__main__':
    tests = [
        ("precompute_1000", test_precompute_1000),
        ("batch_score_256", test_batch_score_256),
        ("ram_buffer_10k", test_ram_buffer_10k),
        ("checkpoint", test_checkpoint),
        ("rate_100_steps", test_rate_100_steps),
        ("1000_steps_quality", test_1000_steps_quality),
        ("zero_eigenvalue_fix", test_zero_eigenvalue_fix),
        ("score_calibration", test_score_calibration),
        ("navigation_diversity", test_navigation_diversity),
        ("ram_scaling", test_ram_scaling),
        ("logic_gate_construct", test_logic_gate_construct),
        ("nand_scoring", test_nand_scoring),
        ("diagnose_mode", test_diagnose_mode),
        ("level_semantics", test_level_semantics),
        ("cross_level_target", test_cross_level_target),
        ("code_structure_target", test_code_structure_target),
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
    print("All explorer tests passed.")
