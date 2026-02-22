"""Tests for html_navigator and physics_sim template extensions.

All templates must satisfy:
  - E_borrow < D_SEP = 0.43
  - render() returns a non-empty string containing 'pub fn'
  - Template lookup by (domain, operation) succeeds in TemplateRegistry

Additional content-level checks:
  - link_extractor output contains 'href'
  - duffing_sim output contains 'alpha' and 'beta'
  - CodeGenPipeline can generate from IntentSpec("html", "link_extractor")
"""

import sys
sys.path.insert(0, '/home/nyoo/projects/unified-tensor-system')

import pytest

from codegen.intent_spec import IntentSpec, BorrowProfile, e_borrow
from codegen.template_registry import TemplateRegistry, register_all_templates
from codegen.templates.html_navigator import (
    link_extractor,
    text_content_extractor,
    structured_table_extractor,
    url_canonicalizer,
    LINK_EXTRACTOR_TEMPLATE,
    TEXT_CONTENT_EXTRACTOR_TEMPLATE,
    STRUCTURED_TABLE_EXTRACTOR_TEMPLATE,
    URL_CANONICALIZER_TEMPLATE,
    ALL_HTML_TEMPLATES,
    register_all as html_register_all,
)
from codegen.templates.physics_sim import (
    rk4_integrator,
    harmonic_oscillator,
    duffing_sim,
    rlc_circuit_sim,
    koopman_edmd_kernel,
    RK4_INTEGRATOR_TEMPLATE,
    HARMONIC_OSCILLATOR_TEMPLATE,
    DUFFING_SIM_TEMPLATE,
    RLC_CIRCUIT_SIM_TEMPLATE,
    KOOPMAN_EDMD_TEMPLATE,
    ALL_PHYSICS_TEMPLATES,
    register_all as physics_register_all,
)

D_SEP = 0.43


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_registry():
    """Registry with all templates registered."""
    reg = TemplateRegistry()
    register_all_templates(reg)
    return reg


# ── HTML Navigator: E_borrow < D_SEP ─────────────────────────────────────────

class TestHTMLNavigatorEBorrow:
    def test_link_extractor_e_borrow(self):
        assert LINK_EXTRACTOR_TEMPLATE.e_borrow < D_SEP

    def test_text_content_extractor_e_borrow(self):
        assert TEXT_CONTENT_EXTRACTOR_TEMPLATE.e_borrow < D_SEP

    def test_structured_table_extractor_e_borrow(self):
        assert STRUCTURED_TABLE_EXTRACTOR_TEMPLATE.e_borrow < D_SEP

    def test_url_canonicalizer_e_borrow(self):
        assert URL_CANONICALIZER_TEMPLATE.e_borrow < D_SEP

    def test_all_html_templates_e_borrow(self):
        for t in ALL_HTML_TEMPLATES:
            assert t.e_borrow < D_SEP, (
                f"{t.name}: E_borrow={t.e_borrow:.3f} >= D_SEP={D_SEP}"
            )


# ── HTML Navigator: render output validity ────────────────────────────────────

class TestHTMLNavigatorRender:
    def test_link_extractor_non_empty(self):
        src = link_extractor({})
        assert isinstance(src, str)
        assert len(src) > 0

    def test_link_extractor_pub_fn(self):
        src = link_extractor({})
        assert "pub fn" in src

    def test_link_extractor_contains_href(self):
        src = link_extractor({})
        assert "href" in src

    def test_text_content_extractor_non_empty(self):
        src = text_content_extractor({"preserve_whitespace": False})
        assert isinstance(src, str)
        assert len(src) > 0

    def test_text_content_extractor_pub_fn(self):
        src = text_content_extractor({})
        assert "pub fn" in src

    def test_text_content_extractor_preserve_whitespace_param(self):
        src_false = text_content_extractor({"preserve_whitespace": False})
        src_true = text_content_extractor({"preserve_whitespace": True})
        assert "false" in src_false or "true" in src_false
        assert "true" in src_true

    def test_structured_table_extractor_non_empty(self):
        src = structured_table_extractor({"max_rows": 100, "header_row": True})
        assert isinstance(src, str)
        assert len(src) > 0

    def test_structured_table_extractor_pub_fn(self):
        src = structured_table_extractor({})
        assert "pub fn" in src

    def test_structured_table_extractor_max_rows_in_output(self):
        src = structured_table_extractor({"max_rows": 42, "header_row": True})
        assert "42" in src

    def test_url_canonicalizer_non_empty(self):
        src = url_canonicalizer({"strip_fragments": True})
        assert isinstance(src, str)
        assert len(src) > 0

    def test_url_canonicalizer_pub_fn(self):
        src = url_canonicalizer({})
        assert "pub fn" in src

    def test_url_canonicalizer_contains_utm(self):
        # Should have utm_ filtering logic in the source
        src = url_canonicalizer({})
        assert "utm_" in src


# ── Physics Sim: E_borrow < D_SEP ────────────────────────────────────────────

class TestPhysicsSimEBorrow:
    def test_rk4_integrator_e_borrow(self):
        assert RK4_INTEGRATOR_TEMPLATE.e_borrow < D_SEP

    def test_harmonic_oscillator_e_borrow(self):
        assert HARMONIC_OSCILLATOR_TEMPLATE.e_borrow < D_SEP

    def test_duffing_sim_e_borrow(self):
        assert DUFFING_SIM_TEMPLATE.e_borrow < D_SEP

    def test_rlc_circuit_sim_e_borrow(self):
        assert RLC_CIRCUIT_SIM_TEMPLATE.e_borrow < D_SEP

    def test_koopman_edmd_kernel_e_borrow(self):
        assert KOOPMAN_EDMD_TEMPLATE.e_borrow < D_SEP

    def test_all_physics_templates_e_borrow(self):
        for t in ALL_PHYSICS_TEMPLATES:
            assert t.e_borrow < D_SEP, (
                f"{t.name}: E_borrow={t.e_borrow:.3f} >= D_SEP={D_SEP}"
            )


# ── Physics Sim: render output validity ──────────────────────────────────────

class TestPhysicsSimRender:
    def test_rk4_integrator_non_empty(self):
        src = rk4_integrator({"state_dim": 2, "dt": 0.01})
        assert isinstance(src, str)
        assert len(src) > 0

    def test_rk4_integrator_pub_fn(self):
        src = rk4_integrator({})
        assert "pub fn" in src

    def test_rk4_integrator_state_dim_in_output(self):
        src = rk4_integrator({"state_dim": 4, "dt": 0.005})
        assert "4" in src

    def test_harmonic_oscillator_non_empty(self):
        src = harmonic_oscillator({"omega0": 2.0, "zeta": 0.05, "dt": 0.001})
        assert isinstance(src, str)
        assert len(src) > 0

    def test_harmonic_oscillator_pub_fn(self):
        src = harmonic_oscillator({})
        assert "pub fn" in src

    def test_harmonic_oscillator_omega0_in_output(self):
        src = harmonic_oscillator({"omega0": 3.14, "zeta": 0.1, "dt": 0.01})
        assert "3.14" in src

    def test_duffing_sim_non_empty(self):
        src = duffing_sim({
            "alpha": 1.0, "beta": 0.5, "delta": 0.3,
            "F": 0.5, "omega": 1.0, "dt": 0.001
        })
        assert isinstance(src, str)
        assert len(src) > 0

    def test_duffing_sim_pub_fn(self):
        src = duffing_sim({})
        assert "pub fn" in src

    def test_duffing_sim_contains_alpha(self):
        src = duffing_sim({
            "alpha": 1.0, "beta": 0.5, "delta": 0.3,
            "F": 0.5, "omega": 1.0, "dt": 0.001
        })
        assert "alpha" in src

    def test_duffing_sim_contains_beta(self):
        src = duffing_sim({
            "alpha": 1.0, "beta": 0.5, "delta": 0.3,
            "F": 0.5, "omega": 1.0, "dt": 0.001
        })
        assert "beta" in src

    def test_rlc_circuit_sim_non_empty(self):
        src = rlc_circuit_sim({"R": 10.0, "L": 0.1, "C": 0.001, "dt": 0.0001})
        assert isinstance(src, str)
        assert len(src) > 0

    def test_rlc_circuit_sim_pub_fn(self):
        src = rlc_circuit_sim({})
        assert "pub fn" in src

    def test_rlc_circuit_sim_contains_r_param(self):
        src = rlc_circuit_sim({"R": 5.0, "L": 1.0, "C": 1.0, "dt": 0.001})
        # R value should appear in the default_params
        assert "5.0" in src

    def test_koopman_edmd_non_empty(self):
        src = koopman_edmd_kernel({"state_dim": 2, "n_observables": 6})
        assert isinstance(src, str)
        assert len(src) > 0

    def test_koopman_edmd_pub_fn(self):
        src = koopman_edmd_kernel({})
        assert "pub fn" in src

    def test_koopman_edmd_state_dim_in_output(self):
        src = koopman_edmd_kernel({"state_dim": 3, "n_observables": 10})
        assert "3" in src

    def test_koopman_edmd_n_observables_in_output(self):
        src = koopman_edmd_kernel({"state_dim": 2, "n_observables": 8})
        assert "8" in src


# ── TemplateRegistry lookup ───────────────────────────────────────────────────

class TestTemplateRegistryLookup:
    """All new templates must be findable by (domain, operation) pair."""

    # HTML navigator templates
    def test_lookup_link_extractor(self, full_registry):
        t = full_registry.lookup("html", "link_extractor")
        assert t is not None
        assert t.name == "link_extractor"

    def test_lookup_text_content_extractor(self, full_registry):
        t = full_registry.lookup("html", "text_content_extractor")
        assert t is not None
        assert t.name == "text_content_extractor"

    def test_lookup_structured_table_extractor(self, full_registry):
        t = full_registry.lookup("html", "structured_table_extractor")
        assert t is not None
        assert t.name == "structured_table_extractor"

    def test_lookup_url_canonicalizer(self, full_registry):
        t = full_registry.lookup("html", "url_canonicalizer")
        assert t is not None
        assert t.name == "url_canonicalizer"

    # Physics sim templates
    def test_lookup_rk4_integrator(self, full_registry):
        t = full_registry.lookup("physics", "rk4_integrator")
        assert t is not None
        assert t.name == "rk4_integrator"

    def test_lookup_harmonic_oscillator(self, full_registry):
        t = full_registry.lookup("physics", "harmonic_oscillator")
        assert t is not None
        assert t.name == "harmonic_oscillator"

    def test_lookup_duffing_sim(self, full_registry):
        t = full_registry.lookup("physics", "duffing_sim")
        assert t is not None
        assert t.name == "duffing_sim"

    def test_lookup_rlc_circuit_sim(self, full_registry):
        t = full_registry.lookup("physics", "rlc_circuit_sim")
        assert t is not None
        assert t.name == "rlc_circuit_sim"

    def test_lookup_koopman_edmd_kernel(self, full_registry):
        t = full_registry.lookup("physics", "koopman_edmd_kernel")
        assert t is not None
        assert t.name == "koopman_edmd_kernel"

    def test_register_all_templates_count(self, full_registry):
        # Should have at least 5 (numeric) + 3 (market) + 4 (html) + 5 (physics) = 17+
        # plus api and text_parser templates from existing code
        assert full_registry.count >= 17


# ── CodeGenPipeline integration ───────────────────────────────────────────────

class TestCodeGenPipelineIntegration:
    """CodeGenPipeline.generate() must succeed (or gracefully degrade) for new templates."""

    def test_pipeline_html_link_extractor(self):
        from codegen.pipeline import CodeGenPipeline

        intent = IntentSpec(
            domain="html",
            operation="link_extractor",
            estimated_borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
            parameters={},
        )
        pipeline = CodeGenPipeline()
        result = pipeline.generate(intent)

        # Pre-gate must pass (E_borrow well below D_SEP)
        assert result.pre_gate is not None
        assert result.pre_gate.predicted_compile == True

        # Template must be found
        assert result.template_name == "link_extractor"

        # Rust source must be non-empty
        assert len(result.rust_source) > 0
        assert "pub fn" in result.rust_source
        assert "href" in result.rust_source

    def test_pipeline_physics_duffing_sim(self):
        from codegen.pipeline import CodeGenPipeline

        intent = IntentSpec(
            domain="physics",
            operation="duffing_sim",
            estimated_borrow_profile=BorrowProfile.SHARED_REFERENCE,
            parameters={
                "alpha": 1.0,
                "beta": 0.5,
                "delta": 0.3,
                "F": 0.5,
                "omega": 1.0,
                "dt": 0.001,
            },
        )
        pipeline = CodeGenPipeline()
        result = pipeline.generate(intent)

        assert result.pre_gate is not None
        assert result.pre_gate.predicted_compile == True
        assert result.template_name == "duffing_sim"
        assert len(result.rust_source) > 0
        assert "alpha" in result.rust_source
        assert "beta" in result.rust_source

    def test_pipeline_physics_rk4_integrator(self):
        from codegen.pipeline import CodeGenPipeline

        intent = IntentSpec(
            domain="physics",
            operation="rk4_integrator",
            estimated_borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
            parameters={"state_dim": 2, "dt": 0.01},
        )
        pipeline = CodeGenPipeline()
        result = pipeline.generate(intent)

        assert result.pre_gate.predicted_compile == True
        assert result.template_name == "rk4_integrator"
        assert "pub fn" in result.rust_source

    def test_pipeline_html_url_canonicalizer(self):
        from codegen.pipeline import CodeGenPipeline

        intent = IntentSpec(
            domain="html",
            operation="url_canonicalizer",
            estimated_borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
            parameters={"strip_fragments": True},
        )
        pipeline = CodeGenPipeline()
        result = pipeline.generate(intent)

        assert result.pre_gate.predicted_compile == True
        assert result.template_name == "url_canonicalizer"
        assert "utm_" in result.rust_source


# ── Design BV exact values ────────────────────────────────────────────────────

class TestDesignBorrowVectors:
    """Spot-check that design BVs match the spec."""

    def test_link_extractor_bv(self):
        bv = LINK_EXTRACTOR_TEMPLATE.design_bv
        assert bv == (0.10, 0.00, 0.00, 0.00, 0.00, 0.00)
        assert abs(e_borrow(bv) - 0.025) < 1e-10

    def test_text_content_extractor_bv(self):
        bv = TEXT_CONTENT_EXTRACTOR_TEMPLATE.design_bv
        assert bv == (0.20, 0.10, 0.00, 0.00, 0.00, 0.00)

    def test_structured_table_extractor_bv(self):
        bv = STRUCTURED_TABLE_EXTRACTOR_TEMPLATE.design_bv
        assert bv == (0.20, 0.00, 0.00, 0.30, 0.00, 0.00)

    def test_url_canonicalizer_bv(self):
        bv = URL_CANONICALIZER_TEMPLATE.design_bv
        assert bv == (0.10, 0.00, 0.00, 0.00, 0.00, 0.00)

    def test_rk4_integrator_bv(self):
        bv = RK4_INTEGRATOR_TEMPLATE.design_bv
        assert bv == (0.10, 0.00, 0.00, 0.00, 0.00, 0.00)

    def test_harmonic_oscillator_bv(self):
        bv = HARMONIC_OSCILLATOR_TEMPLATE.design_bv
        assert bv == (0.10, 0.00, 0.00, 0.00, 0.00, 0.00)

    def test_duffing_sim_bv(self):
        bv = DUFFING_SIM_TEMPLATE.design_bv
        assert bv == (0.15, 0.10, 0.00, 0.00, 0.00, 0.00)

    def test_rlc_circuit_sim_bv(self):
        bv = RLC_CIRCUIT_SIM_TEMPLATE.design_bv
        assert bv == (0.15, 0.10, 0.00, 0.00, 0.00, 0.00)

    def test_koopman_edmd_bv(self):
        bv = KOOPMAN_EDMD_TEMPLATE.design_bv
        assert bv == (0.20, 0.10, 0.00, 0.20, 0.00, 0.00)
