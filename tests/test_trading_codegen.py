"""
Tests for codegen/templates/trading_sim.py

All 4 trading templates:
  - multi_asset_price_sim
  - agent_response_kernel
  - crowd_behavior_kernel
  - regime_transition_detector
"""

import sys

sys.path.insert(0, '/home/nyoo/projects/unified-tensor-system')

import pytest

from codegen.intent_spec import BorrowProfile, IntentSpec, e_borrow
from codegen.template_registry import TemplateRegistry
from codegen.templates.trading_sim import (
    ALL_TRADING_TEMPLATES,
    AGENT_RESPONSE_KERNEL_TEMPLATE,
    CROWD_BEHAVIOR_KERNEL_TEMPLATE,
    MULTI_ASSET_PRICE_SIM_TEMPLATE,
    REGIME_TRANSITION_DETECTOR_TEMPLATE,
    register_all,
)
from codegen.pipeline import CodeGenPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry() -> TemplateRegistry:
    reg = TemplateRegistry()
    register_all(reg)
    return reg


@pytest.fixture(scope="module")
def pipeline() -> CodeGenPipeline:
    return CodeGenPipeline()


# ---------------------------------------------------------------------------
# E_borrow safety gate — all 4 templates must be < 0.43 (D_sep)
# ---------------------------------------------------------------------------

D_SEP = 0.43


class TestEBorrowSafetyGate:
    @pytest.mark.parametrize("template", ALL_TRADING_TEMPLATES, ids=lambda t: t.name)
    def test_e_borrow_below_d_sep(self, template):
        eb = e_borrow(template.design_bv)
        assert eb < D_SEP, (
            f"{template.name}: E_borrow={eb:.4f} >= D_sep={D_SEP}"
        )

    def test_multi_asset_e_borrow(self):
        assert e_borrow(MULTI_ASSET_PRICE_SIM_TEMPLATE.design_bv) == pytest.approx(0.025, abs=0.001)

    def test_agent_response_e_borrow(self):
        assert e_borrow(AGENT_RESPONSE_KERNEL_TEMPLATE.design_bv) == pytest.approx(0.068, abs=0.005)

    def test_crowd_behavior_e_borrow(self):
        assert e_borrow(CROWD_BEHAVIOR_KERNEL_TEMPLATE.design_bv) == pytest.approx(0.025, abs=0.001)

    def test_regime_detector_e_borrow(self):
        assert e_borrow(REGIME_TRANSITION_DETECTOR_TEMPLATE.design_bv) == pytest.approx(0.101, abs=0.005)


# ---------------------------------------------------------------------------
# Render: non-empty source with `pub fn`
# ---------------------------------------------------------------------------

class TestRenderOutput:
    @pytest.mark.parametrize("template", ALL_TRADING_TEMPLATES, ids=lambda t: t.name)
    def test_render_is_non_empty(self, template):
        source = template.render({})
        assert source, f"{template.name}: render() returned empty string"

    @pytest.mark.parametrize("template", ALL_TRADING_TEMPLATES, ids=lambda t: t.name)
    def test_render_has_pub_fn(self, template):
        source = template.render({})
        assert "pub fn" in source, f"{template.name}: rendered source has no 'pub fn'"


# ---------------------------------------------------------------------------
# Template-specific content checks
# ---------------------------------------------------------------------------

class TestMultiAssetPriceSim:
    def test_contains_tanh(self):
        source = MULTI_ASSET_PRICE_SIM_TEMPLATE.render(
            {"n_assets": 3, "dt": 0.01, "n_steps": 100}
        )
        assert "tanh" in source, "multi_asset_price_sim must use tanh for crowd behavior"

    def test_param_n_assets_reflected(self):
        source = MULTI_ASSET_PRICE_SIM_TEMPLATE.render(
            {"n_assets": 5, "dt": 0.01, "n_steps": 50}
        )
        assert "5" in source

    def test_uses_euler_integration(self):
        source = MULTI_ASSET_PRICE_SIM_TEMPLATE.render({})
        # Euler step keywords
        assert "dt" in source.lower() or "n_steps" in source

    def test_no_ndarray_crate(self):
        source = MULTI_ASSET_PRICE_SIM_TEMPLATE.render({})
        assert "use ndarray" not in source and "extern crate ndarray" not in source, "Template must not import ndarray crate"


class TestCrowdBehaviorKernel:
    def test_contains_tanh(self):
        source = CROWD_BEHAVIOR_KERNEL_TEMPLATE.render(
            {"amplitude": 1.0, "threshold": 10.0}
        )
        assert "tanh" in source, "crowd_behavior_kernel must contain tanh"

    def test_amplitude_reflected_in_source(self):
        source = CROWD_BEHAVIOR_KERNEL_TEMPLATE.render(
            {"amplitude": 2.5, "threshold": 5.0}
        )
        assert "2.5" in source or "2.500000" in source

    def test_threshold_reflected_in_source(self):
        source = CROWD_BEHAVIOR_KERNEL_TEMPLATE.render(
            {"amplitude": 1.0, "threshold": 20.0}
        )
        assert "20.0" in source or "20.000000" in source


class TestAgentResponseKernel:
    def test_contains_response_time(self):
        source = AGENT_RESPONSE_KERNEL_TEMPLATE.render(
            {"n_agents": 3, "n_assets": 3}
        )
        assert "response_time" in source or "weight" in source

    def test_n_agents_reflected(self):
        source = AGENT_RESPONSE_KERNEL_TEMPLATE.render(
            {"n_agents": 4, "n_assets": 3}
        )
        assert "4" in source

    def test_shared_reference_in_source(self):
        source = AGENT_RESPONSE_KERNEL_TEMPLATE.render({})
        # Shared reference: &[f64] or &Vec<f64>
        assert "&[f64]" in source or "&Vec<f64>" in source or "&[" in source


class TestRegimeTransitionDetector:
    def test_contains_eigenvalue_logic(self):
        source = REGIME_TRANSITION_DETECTOR_TEMPLATE.render(
            {"n_eigenvalues": 4, "stability_threshold": 0.0}
        )
        assert "eigenvalue" in source or "re_" in source or "sign" in source.lower()

    def test_returns_crossed_bool(self):
        source = REGIME_TRANSITION_DETECTOR_TEMPLATE.render(
            {"n_eigenvalues": 4, "stability_threshold": 0.0}
        )
        assert "crossed" in source or "bool" in source

    def test_mutable_output_present(self):
        source = REGIME_TRANSITION_DETECTOR_TEMPLATE.render({})
        # Mutable output: &mut bool or &mut Vec<> or &mut usize
        assert "&mut" in source


# ---------------------------------------------------------------------------
# TemplateRegistry lookup
# ---------------------------------------------------------------------------

class TestTemplateRegistryLookup:
    def test_lookup_multi_asset(self, registry):
        t = registry.lookup("trading", "multi_asset_price_sim")
        assert t is not None, "multi_asset_price_sim not found in registry"
        assert t.name == "multi_asset_price_sim"

    def test_lookup_agent_response(self, registry):
        t = registry.lookup("trading", "agent_response_kernel")
        assert t is not None, "agent_response_kernel not found in registry"

    def test_lookup_crowd_behavior(self, registry):
        t = registry.lookup("trading", "crowd_behavior_kernel")
        assert t is not None, "crowd_behavior_kernel not found in registry"

    def test_lookup_regime_detector(self, registry):
        t = registry.lookup("trading", "regime_transition_detector")
        assert t is not None, "regime_transition_detector not found in registry"

    def test_all_four_registered(self, registry):
        names = {t.name for t in registry.all_templates()}
        expected = {
            "multi_asset_price_sim",
            "agent_response_kernel",
            "crowd_behavior_kernel",
            "regime_transition_detector",
        }
        assert expected.issubset(names), f"Missing templates: {expected - names}"

    def test_domain_is_trading_for_all(self, registry):
        trading_templates = [
            t for t in registry.all_templates() if t.domain == "trading"
        ]
        assert len(trading_templates) == 4, (
            f"Expected 4 trading templates, got {len(trading_templates)}"
        )


# ---------------------------------------------------------------------------
# CodeGenPipeline integration — pipeline registers trading templates
# ---------------------------------------------------------------------------

class TestCodeGenPipelineIntegration:
    def test_pipeline_has_trading_templates(self, pipeline):
        reg = pipeline.registry
        t = reg.lookup("trading", "multi_asset_price_sim")
        assert t is not None, "Pipeline registry must include trading templates"

    def test_generate_multi_asset_price_sim(self, pipeline):
        """Pipeline.generate() with trading intent must complete without error."""
        intent = IntentSpec(
            domain="trading",
            operation="multi_asset_price_sim",
            parameters={"n_assets": 3, "dt": 0.01, "n_steps": 100},
            estimated_borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
        )
        result = pipeline.generate(intent)
        # Success or rustc-unavailable are both acceptable
        assert result.template_name == "multi_asset_price_sim", (
            f"Wrong template: {result.template_name}; error: {result.error}"
        )
        assert result.rust_source, "rust_source must not be empty"
        assert "tanh" in result.rust_source

    def test_generate_crowd_behavior_kernel(self, pipeline):
        intent = IntentSpec(
            domain="trading",
            operation="crowd_behavior_kernel",
            parameters={"amplitude": 1.0, "threshold": 10.0},
            estimated_borrow_profile=BorrowProfile.PURE_FUNCTIONAL,
        )
        result = pipeline.generate(intent)
        assert result.template_name == "crowd_behavior_kernel", (
            f"Wrong template: {result.template_name}; error: {result.error}"
        )
        assert "tanh" in result.rust_source

    def test_generate_agent_response_kernel(self, pipeline):
        intent = IntentSpec(
            domain="trading",
            operation="agent_response_kernel",
            parameters={"n_agents": 3, "n_assets": 3},
            estimated_borrow_profile=BorrowProfile.SHARED_REFERENCE,
        )
        result = pipeline.generate(intent)
        assert result.template_name == "agent_response_kernel", (
            f"Wrong template: {result.template_name}; error: {result.error}"
        )
        assert result.rust_source

    def test_generate_regime_transition_detector(self, pipeline):
        intent = IntentSpec(
            domain="trading",
            operation="regime_transition_detector",
            parameters={"n_eigenvalues": 4, "stability_threshold": 0.0},
            estimated_borrow_profile=BorrowProfile.MUTABLE_OUTPUT,
        )
        result = pipeline.generate(intent)
        assert result.template_name == "regime_transition_detector", (
            f"Wrong template: {result.template_name}; error: {result.error}"
        )
        assert result.rust_source


# ---------------------------------------------------------------------------
# BorrowProfile assignment checks
# ---------------------------------------------------------------------------

class TestBorrowProfiles:
    def test_multi_asset_is_pure_functional(self):
        assert MULTI_ASSET_PRICE_SIM_TEMPLATE.borrow_profile == BorrowProfile.PURE_FUNCTIONAL

    def test_agent_response_is_shared_reference(self):
        assert AGENT_RESPONSE_KERNEL_TEMPLATE.borrow_profile == BorrowProfile.SHARED_REFERENCE

    def test_crowd_behavior_is_pure_functional(self):
        assert CROWD_BEHAVIOR_KERNEL_TEMPLATE.borrow_profile == BorrowProfile.PURE_FUNCTIONAL

    def test_regime_detector_is_mutable_output(self):
        assert REGIME_TRANSITION_DETECTOR_TEMPLATE.borrow_profile == BorrowProfile.MUTABLE_OUTPUT
