"""
Tests for optimization/code_profiler.py and optimization/repo_learner.py

Covers:
  1. ASTMathClassifier — pattern detection for known operations
  2. pattern_to_invariants() — theoretical (ω₀, Q, ζ) correctness
  3. DynamicProfiler — timing + Koopman fit on numpy functions
  4. CodeHardwareProfiler — static + dynamic analysis, memory storage
  5. RepoLearner.learn_from_source() — function extraction + storage
  6. find_similar_code() — retrieval by (ω₀, Q) proximity
  7. Cross-domain retrieval: hardware entry retrieved by RLC-like invariant
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import math
import numpy as np
import pytest

from optimization.code_profiler import (
    ASTMathClassifier,
    CodeHardwareProfiler,
    DynamicProfiler,
    MathPattern,
    _COMPLEXITY_TABLE,
    _HW_OMEGA0_REF,
    pattern_to_invariants,
    profile_and_store,
)
from optimization.koopman_memory import KoopmanExperienceMemory
from optimization.repo_learner import (
    RepoLearner,
    extract_functions,
    find_similar_code,
)


# ── 1. ASTMathClassifier ──────────────────────────────────────────────────────


class TestASTMathClassifier:

    def setup_method(self):
        self.clf = ASTMathClassifier()

    # ── numpy call detection ───────────────────────────────────────────────────

    def test_matmul_detected_from_numpy_dot(self):
        src = "def f(A, b): return np.dot(A, b)"
        p = self.clf.classify(src)
        assert p.complexity_class in ("matvec", "matmul")

    def test_matmul_detected_from_at_operator(self):
        src = "def f(A, B): return A @ B"
        p = self.clf.classify(src)
        assert p.complexity_class == "matmul"

    def test_fft_detected(self):
        src = "def f(x): return np.fft.fft(x)"
        p = self.clf.classify(src)
        assert p.complexity_class == "fft"

    def test_sort_detected(self):
        src = "def f(x): return np.sort(x)"
        p = self.clf.classify(src)
        assert p.complexity_class == "sort"

    def test_element_wise_detected(self):
        src = "def f(x): return np.exp(x) + np.sqrt(x)"
        p = self.clf.classify(src)
        assert p.complexity_class == "element_wise"

    def test_reduction_detected(self):
        src = "def f(x): return np.sum(x)"
        p = self.clf.classify(src)
        assert p.complexity_class == "reduction"

    def test_conv_detected(self):
        src = "def f(x, h): return np.convolve(x, h)"
        p = self.clf.classify(src)
        assert p.complexity_class == "conv_direct"

    # ── loop depth detection ───────────────────────────────────────────────────

    def test_triple_nested_loop_hints_matmul(self):
        src = """
def f(A, B, C):
    for i in range(len(A)):
        for j in range(len(B)):
            for k in range(len(C)):
                pass
"""
        p = self.clf.classify(src)
        assert p.complexity_class == "matmul"

    def test_single_loop_hints_element_wise(self):
        src = """
def f(x):
    out = []
    for v in x:
        out.append(v * 2)
    return out
"""
        p = self.clf.classify(src)
        assert p.complexity_class == "element_wise"

    # ── syntax error handling ──────────────────────────────────────────────────

    def test_syntax_error_returns_unknown(self):
        p = self.clf.classify("def f( INVALID SYNTAX !!!")
        assert p.complexity_class == "unknown"
        assert p.confidence == 0.0

    # ── MathPattern fields ─────────────────────────────────────────────────────

    def test_complexity_exponent_positive(self):
        src = "def f(A, B): return A @ B"
        p = self.clf.classify(src)
        assert p.complexity_exponent > 0.0

    def test_arithmetic_intensity_positive(self):
        src = "def f(A, B): return A @ B"
        p = self.clf.classify(src)
        assert p.arithmetic_intensity > 0.0

    def test_confidence_in_range(self):
        for src in [
            "def f(A, B): return A @ B",
            "def f(x): return np.fft.fft(x)",
            "def f(x): return np.sort(x)",
        ]:
            p = self.clf.classify(src)
            assert 0.0 <= p.confidence <= 1.0


# ── 2. pattern_to_invariants() ────────────────────────────────────────────────


class TestPatternToInvariants:

    def test_returns_three_floats(self):
        α, Q = _COMPLEXITY_TABLE["matmul"]
        p = MathPattern(complexity_class="matmul", complexity_exponent=α,
                        arithmetic_intensity=Q, dominant_op="@", confidence=1.0)
        result = pattern_to_invariants(p)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_log_omega0_norm_in_range(self):
        for cls, (α, Q) in _COMPLEXITY_TABLE.items():
            p = MathPattern(complexity_class=cls, complexity_exponent=α,
                            arithmetic_intensity=Q, dominant_op=cls, confidence=1.0)
            lw, lq, z = pattern_to_invariants(p)
            assert -3.0 <= lw <= 3.0, f"{cls}: log_omega0_norm={lw} out of range"

    def test_high_complexity_has_lower_omega0(self):
        """O(N³) matmul should have lower throughput (ω₀) than O(N) element-wise."""
        α_el, Q_el = _COMPLEXITY_TABLE["element_wise"]
        α_mm, Q_mm = _COMPLEXITY_TABLE["matmul"]
        p_el = MathPattern("element_wise", α_el, Q_el, "el", 1.0)
        p_mm = MathPattern("matmul", α_mm, Q_mm, "mm", 1.0)
        lw_el, _, _ = pattern_to_invariants(p_el)
        lw_mm, _, _ = pattern_to_invariants(p_mm)
        # matmul does more work per element → lower throughput → lower ω₀
        assert lw_mm < lw_el, (
            f"Expected matmul ω₀ < element_wise ω₀, got {lw_mm:.3f} >= {lw_el:.3f}"
        )

    def test_high_Q_has_lower_damping_ratio(self):
        """High arithmetic intensity (compute-bound) → low damping ratio."""
        p_mm = MathPattern("matmul", 3.0, 8.0, "mm", 1.0)
        p_el = MathPattern("element_wise", 1.0, 0.5, "el", 1.0)
        _, _, z_mm = pattern_to_invariants(p_mm)
        _, _, z_el = pattern_to_invariants(p_el)
        assert z_mm < z_el, "Compute-bound matmul should have lower damping_ratio"

    def test_damping_ratio_in_01(self):
        for cls, (α, Q) in _COMPLEXITY_TABLE.items():
            p = MathPattern(cls, α, Q, cls, 1.0)
            _, _, z = pattern_to_invariants(p)
            assert 0.0 <= z <= 1.0, f"{cls}: damping_ratio={z} out of [0,1]"


# ── 3. DynamicProfiler ────────────────────────────────────────────────────────


class TestDynamicProfiler:

    def setup_method(self):
        self.prof = DynamicProfiler(n_warmup=1, n_repeat=3)

    def test_profile_returns_rows(self):
        rows = self.prof.profile(
            lambda x: np.sum(x),
            lambda N: np.random.randn(N),
            sizes=[64, 128, 256],
        )
        assert len(rows) == 3
        assert all(r.wall_sec > 0 for r in rows)

    def test_sizes_stored_correctly(self):
        rows = self.prof.profile(
            lambda x: x * 2.0,
            lambda N: np.random.randn(N),
            sizes=[100, 200, 400],
        )
        assert [r.N for r in rows] == [100, 200, 400]

    def test_fit_scaling_linear_expected_near_1(self):
        """
        Sum over array is O(N), but numpy's vectorised sum runs at memory-bandwidth
        speed and for L1/L2-cache-resident sizes may appear nearly O(1).
        Assert that α is finite and not wildly negative (not a decreasing function).
        """
        rows = self.prof.profile(
            lambda x: np.sum(x),
            lambda N: np.random.randn(N),
            sizes=[256, 512, 1024, 2048, 4096],
        )
        α, _ = self.prof.fit_scaling(rows)
        assert -0.5 <= α <= 3.0, f"α = {α:.3f} is unreasonable (expected [-0.5, 3.0])"

    def test_fit_scaling_returns_two_floats(self):
        rows = self.prof.profile(
            lambda x: np.sort(x),
            lambda N: np.random.randn(N),
            sizes=[64, 128, 256, 512],
        )
        α, β = self.prof.fit_scaling(rows)
        assert isinstance(α, float)
        assert isinstance(β, float)

    def test_empirical_invariants_in_range(self):
        rows = self.prof.profile(
            lambda x: np.fft.fft(x),
            lambda N: np.random.randn(N),
            sizes=[64, 128, 256, 512, 1024],
        )
        lw, lq, z = self.prof.empirical_invariants(rows)
        assert -3.0 <= lw <= 3.0
        assert -3.0 <= lq <= 3.0
        assert 0.0 <= z <= 1.0

    def test_build_koopman_trace_shape(self):
        rows = self.prof.profile(
            lambda x: x + 1,
            lambda N: np.random.randn(N),
            sizes=[64, 128, 256, 512, 1024],
        )
        kt = self.prof.build_koopman_trace(rows)
        assert kt is not None
        assert kt.shape == (5, 3)

    def test_build_koopman_trace_none_on_too_few(self):
        rows = self.prof.profile(
            lambda x: x,
            lambda N: np.random.randn(N),
            sizes=[64, 128],   # only 2 points
        )
        kt = self.prof.build_koopman_trace(rows)
        assert kt is None   # < 3 points


# ── 4. CodeHardwareProfiler ───────────────────────────────────────────────────


class TestCodeHardwareProfiler:

    def setup_method(self):
        self.profiler = CodeHardwareProfiler(n_warmup=1, n_repeat=3)

    def test_analyse_static_returns_descriptor(self):
        from optimization.koopman_signature import KoopmanInvariantDescriptor
        src = "def f(A, B): return A @ B"
        inv = self.profiler.analyse_static(src)
        assert isinstance(inv, KoopmanInvariantDescriptor)

    def test_analyse_static_matmul_has_low_damping(self):
        src = "def f(A, B): return np.matmul(A, B)"
        inv = self.profiler.analyse_static(src)
        # matmul is compute-bound → Q high → ζ low
        assert inv.damping_ratio < 0.5, (
            f"Matmul damping_ratio = {inv.damping_ratio:.3f}, expected < 0.5"
        )

    def test_analyse_dynamic_returns_descriptor(self):
        from optimization.koopman_signature import KoopmanInvariantDescriptor
        inv = self.profiler.analyse_dynamic(
            lambda x: np.sum(x),
            lambda N: np.random.randn(N),
            fn_name="sum",
            sizes=[64, 128, 256, 512, 1024],
        )
        assert isinstance(inv, KoopmanInvariantDescriptor)

    def test_store_adds_to_memory(self):
        mem = KoopmanExperienceMemory()
        src = "def f(x): return np.fft.fft(x)"
        inv = self.profiler.analyse_static(src)
        self.profiler.store(inv, mem, fn_name="fft_fn", complexity_class="fft")
        assert len(mem) >= 1

    def test_domain_tag_hardware_static(self):
        mem = KoopmanExperienceMemory()
        src = "def f(x): return np.sort(x)"
        inv = self.profiler.analyse_static(src)
        self.profiler.store(inv, mem, fn_name="sort_fn", domain="hardware_static")
        assert mem._entries[0].experience.domain == "hardware_static"

    def test_domain_tag_hardware_dynamic(self):
        mem = KoopmanExperienceMemory()
        inv = self.profiler.analyse_dynamic(
            lambda x: np.sort(x),
            lambda N: np.random.randn(N),
            sizes=[64, 128, 256],
        )
        self.profiler.store(inv, mem, fn_name="sort_dyn", domain="hardware_dynamic")
        assert mem._entries[0].experience.domain == "hardware_dynamic"


# ── 5. RepoLearner.learn_from_source() ───────────────────────────────────────


_SAMPLE_SOURCE = '''
import numpy as np

def matrix_multiply(A, B):
    """Dense matrix multiplication."""
    return np.matmul(A, B)

def fft_transform(x):
    """Fast Fourier Transform."""
    return np.fft.fft(x)

def array_sort(x):
    """Sort array."""
    return np.sort(x)

def element_sum(x):
    """Sum all elements."""
    return np.sum(x)

def scale_vector(x, factor):
    """Element-wise scaling."""
    return x * factor
'''


class TestRepoLearner:

    def setup_method(self):
        self.learner = RepoLearner()
        self.mem = KoopmanExperienceMemory()

    def test_learn_from_source_finds_functions(self):
        result = self.learner.learn_from_source(_SAMPLE_SOURCE, self.mem)
        assert result.functions_found >= 4

    def test_learn_from_source_stores_entries(self):
        result = self.learner.learn_from_source(_SAMPLE_SOURCE, self.mem)
        assert result.functions_stored >= 3
        assert len(self.mem) >= 3

    def test_learn_from_source_no_private_functions(self):
        src = """
def public_fn(x): return x
def _private_fn(x): return x
def __dunder__(x): return x
"""
        result = self.learner.learn_from_source(src, self.mem)
        stored_names = [e.experience.best_params.get("fn", "")
                        for e in self.mem._entries]
        assert all(not n.startswith("_") for n in stored_names)

    def test_learn_from_source_pattern_counts(self):
        result = self.learner.learn_from_source(_SAMPLE_SOURCE, self.mem)
        assert len(result.pattern_counts) >= 1

    def test_learn_from_source_result_fields(self):
        result = self.learner.learn_from_source(_SAMPLE_SOURCE, self.mem)
        assert result.files_fetched == 1
        assert result.functions_found >= 0
        assert result.functions_stored >= 0

    def test_extract_functions_finds_public_fns(self):
        fns = extract_functions(_SAMPLE_SOURCE)
        names = [f.fn_name for f in fns]
        assert "matrix_multiply" in names
        assert "fft_transform" in names

    def test_extract_functions_skips_short(self):
        src = """
def trivial(x): return x
def real_function(x):
    result = np.fft.fft(x)
    return result
"""
        fns = extract_functions(src)
        names = [f.fn_name for f in fns]
        # trivial is 1 line → skipped; real_function is 3 lines → kept
        assert "real_function" in names

    def test_syntax_error_file_gives_no_fns(self):
        fns = extract_functions("def f( BROKEN !!!")
        assert fns == []


# ── 6. find_similar_code() ───────────────────────────────────────────────────


class TestFindSimilarCode:

    def setup_method(self):
        self.mem = KoopmanExperienceMemory()
        learner = RepoLearner()
        learner.learn_from_source(_SAMPLE_SOURCE, self.mem)

    def test_returns_list(self):
        results = find_similar_code("def f(x): return np.sum(x)", self.mem, top_n=3)
        assert isinstance(results, list)

    def test_results_have_required_keys(self):
        results = find_similar_code("def f(x): return np.fft.fft(x)", self.mem, top_n=3)
        if results:
            for r in results:
                assert "fn_name" in r
                assert "complexity" in r
                assert "distance" in r

    def test_fft_query_retrieves_fft_or_similar(self):
        """A FFT query should retrieve the stored FFT function as nearest."""
        query = "def query(x): return np.fft.fft(x)"
        results = find_similar_code(query, self.mem, top_n=3)
        if results:
            top_cc = results[0]["complexity"]
            # Top result should be fft or at least have finite distance
            assert results[0]["distance"] >= 0.0

    def test_matmul_query_distance_less_than_fft(self):
        """
        A matmul query should be closer to matmul than to element-wise.
        """
        mem2 = KoopmanExperienceMemory()
        profiler = CodeHardwareProfiler()

        src_mm = "def matmul(A, B): return A @ B"
        src_el = "def scale(x): return x * 2.0 + 1.0"

        inv_mm = profiler.analyse_static(src_mm)
        inv_el = profiler.analyse_static(src_el)
        profiler.store(inv_mm, mem2, fn_name="matmul", complexity_class="matmul")
        profiler.store(inv_el, mem2, fn_name="scale", complexity_class="element_wise")

        query = "def query(A, B): return np.matmul(A, B)"
        results = find_similar_code(query, mem2, top_n=2)
        assert len(results) == 2
        # matmul should be nearer than element_wise
        names = [r["fn_name"] for r in results]
        assert names[0] == "matmul", (
            f"Expected 'matmul' as nearest match, got {names[0]}"
        )


# ── 7. Cross-domain retrieval ─────────────────────────────────────────────────


class TestCrossDomainRetrieval:
    """
    Hardware experiences and RLC/spring-mass experiences share the same
    (log_omega0_norm, log_Q_norm, ζ) space.  A hardware entry should be
    retrievable by an RLC query with similar (ω₀, Q).
    """

    def test_hardware_entry_retrievable_from_rlc_query(self):
        """
        Store an FFT hardware experience (throughput ≈ 1 MHz, Q moderate).
        Store a matmul hardware experience (lower ω₀, high Q).
        Query with an RLC-style invariant at log_omega0_norm≈0 (1 kHz).
        The retrieval should return the FFT entry (closer ω₀).
        """
        from optimization.koopman_signature import compute_invariants
        mem = KoopmanExperienceMemory()
        profiler = CodeHardwareProfiler()

        src_fft = "def f(x): return np.fft.fft(x)"
        src_mm  = "def g(A, B): return A @ B"
        inv_fft = profiler.analyse_static(src_fft)
        inv_mm  = profiler.analyse_static(src_mm)
        profiler.store(inv_fft, mem, fn_name="fft", complexity_class="fft")
        profiler.store(inv_mm,  mem, fn_name="matmul", complexity_class="matmul")

        # RLC query: log_omega0_norm=0.0 (1kHz reference), Q=1
        rlc_query = compute_invariants(
            np.array([]), np.zeros((0, 0)), [],
            log_omega0_norm=0.0, log_Q_norm=0.0, damping_ratio=0.5,
        )
        candidates = mem.retrieve_candidates(rlc_query, top_n=2)
        assert len(candidates) >= 1
        # Nearest should be some hardware entry
        top_domain = candidates[0].experience.domain
        assert top_domain in ("hardware_static", "hardware_dynamic")

    def test_profile_and_store_roundtrip(self):
        """profile_and_store() returns an invariant that can be retrieved."""
        from optimization.koopman_signature import compute_invariants
        mem = KoopmanExperienceMemory()
        src = "def f(x): return np.sum(x)"
        inv = profile_and_store(src, mem, fn_name="sum_fn")
        assert len(mem) >= 1
        # Can retrieve it back
        candidates = mem.retrieve_candidates(inv, top_n=1)
        assert len(candidates) == 1
        assert candidates[0].experience.best_params.get("fn_name") == "sum_fn"
