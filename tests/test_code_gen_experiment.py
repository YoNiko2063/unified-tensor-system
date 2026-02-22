"""
Tests for code_gen_experiment.py — invariant-guided code generation gate.

Gates:
  1. Classifier accuracy  = 100% on the 3 designed templates
  2. Passing templates stay in safe zone (ΔE_total < D_sep = 0.43)
  3. Template C (mutable aliasing) is blocked with P(ok) < 0.05
  4. Templates A and B pass with P(ok) > 0.90
  5. ΔE_total for passing templates is positive and ordered A < B
  6. Summary dict fields are complete and consistent
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import pytest
from optimization.code_gen_experiment import (
    METRICS_JSONL,
    TEMPLATES,
    D_SEP,
    E_PYTHON,
    compute_summary,
    delta_e_total,
    e_borrow,
    feature_vec,
    load_classifier,
    predict,
    run_experiment,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clf_scaler():
    return load_classifier(METRICS_JSONL)


@pytest.fixture(scope="module")
def results():
    return run_experiment(METRICS_JSONL)


@pytest.fixture(scope="module")
def summary(results):
    return compute_summary(results)


# ── 1. Template definitions ───────────────────────────────────────────────────

class TestTemplateDefinitions:
    def test_three_templates(self):
        assert len(TEMPLATES) == 3

    def test_template_names(self):
        names = [t.name for t in TEMPLATES]
        assert "A_pure_functional"  in names
        assert "B_shared_reference" in names
        assert "C_mutable_aliasing" in names

    def test_e_borrow_ordering(self):
        """A < B < C in E_borrow."""
        ea, eb, ec = [e_borrow(t.bv) for t in TEMPLATES]
        assert ea < eb < ec

    def test_template_c_above_dsep(self):
        """Template C must be above D_sep to test the gate meaningfully."""
        ec = e_borrow(TEMPLATES[2].bv)
        assert ec > D_SEP, f"C E_borrow={ec:.3f} not above D_sep={D_SEP}"

    def test_templates_a_b_below_dsep(self):
        """Templates A and B must be in the safe zone."""
        for tmpl in TEMPLATES[:2]:
            eb = e_borrow(tmpl.bv)
            assert eb < D_SEP, f"{tmpl.name} E_borrow={eb:.3f} above D_sep"

    def test_expected_compile_flags(self):
        a, b, c = TEMPLATES
        assert a.expected_compile is True
        assert b.expected_compile is True
        assert c.expected_compile is False


# ── 2. Classifier predictions ─────────────────────────────────────────────────

class TestClassifierPredictions:
    def test_template_a_predicted_ok(self, clf_scaler):
        clf, scaler = clf_scaler
        pred, prob = predict(clf, scaler, TEMPLATES[0].bv)
        assert bool(pred) is True
        assert prob > 0.90, f"A P(ok)={prob:.3f} < 0.90"

    def test_template_b_predicted_ok(self, clf_scaler):
        clf, scaler = clf_scaler
        pred, prob = predict(clf, scaler, TEMPLATES[1].bv)
        assert bool(pred) is True
        assert prob > 0.90, f"B P(ok)={prob:.3f} < 0.90"

    def test_template_c_predicted_fail(self, clf_scaler):
        clf, scaler = clf_scaler
        pred, prob = predict(clf, scaler, TEMPLATES[2].bv)
        assert bool(pred) is False
        assert prob < 0.05, f"C P(ok)={prob:.3f} >= 0.05 (should be near 0)"

    def test_probability_ordering(self, clf_scaler):
        clf, scaler = clf_scaler
        probs = [predict(clf, scaler, t.bv)[1] for t in TEMPLATES]
        assert probs[0] > probs[1] > probs[2]


# ── 3. Full experiment results ────────────────────────────────────────────────

class TestExperimentResults:
    def test_accuracy_100_pct(self, results):
        n_correct = sum(r.predicted_ok == r.actual_ok for r in results)
        assert n_correct == 3, f"Accuracy {n_correct}/3"

    def test_two_passing_one_blocked(self, results):
        passing = [r for r in results if r.actual_ok]
        blocked = [r for r in results if not r.actual_ok]
        assert len(passing) == 2
        assert len(blocked) == 1

    def test_blocked_is_template_c(self, results):
        blocked = [r for r in results if not r.actual_ok]
        assert blocked[0].template.name == "C_mutable_aliasing"

    def test_delta_e_defined_for_passing(self, results):
        for r in results:
            if r.actual_ok:
                assert r.delta_e is not None

    def test_delta_e_none_for_blocked(self, results):
        for r in results:
            if not r.actual_ok:
                assert r.delta_e is None


# ── 4. Safe zone (ΔE_total) ───────────────────────────────────────────────────

class TestSafeZone:
    def test_all_passing_below_dsep(self, results):
        for r in results:
            if r.actual_ok:
                assert r.delta_e < D_SEP, \
                    f"{r.template.name} ΔE={r.delta_e:.3f} >= D_sep={D_SEP}"

    def test_delta_e_positive(self, results):
        """Rust adds borrow structure relative to Python (E_python=0)."""
        for r in results:
            if r.actual_ok:
                assert r.delta_e >= 0.0

    def test_delta_e_ordered_a_lt_b(self, results):
        by_name = {r.template.name: r for r in results}
        de_a = by_name["A_pure_functional"].delta_e
        de_b = by_name["B_shared_reference"].delta_e
        assert de_a < de_b, f"Expected ΔE_A < ΔE_B, got {de_a:.3f} vs {de_b:.3f}"

    def test_max_delta_e_below_dsep(self, summary):
        assert summary["max_delta_e"] < D_SEP


# ── 5. Summary dict ───────────────────────────────────────────────────────────

class TestSummary:
    def test_accuracy_1(self, summary):
        assert summary["accuracy"] == 1.0

    def test_n_passing_2(self, summary):
        assert summary["n_passing"] == 2

    def test_safe_zone_flag(self, summary):
        assert summary["safe_zone"] is True

    def test_d_sep_correct(self, summary):
        assert summary["d_sep"] == D_SEP


# ── 6. Unit: e_borrow and feature_vec ─────────────────────────────────────────

class TestHelpers:
    def test_e_borrow_pure_functional(self):
        bv = (0.10, 0.00, 0.00, 0.00, 0.00)
        assert abs(e_borrow(bv) - 0.025) < 1e-9

    def test_feature_vec_length(self):
        bv = (0.1, 0.1, 0.0, 0.1, 0.0)
        fv = feature_vec(bv)
        assert fv.shape == (6,)
        assert abs(fv[5] - e_borrow(bv)) < 1e-9

    def test_delta_e_formula(self):
        assert abs(delta_e_total(0.0, 0.025) - 0.025) < 1e-9
        assert abs(delta_e_total(0.0, 0.080) - 0.080) < 1e-9
