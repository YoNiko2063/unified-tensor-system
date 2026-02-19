"""Tests for tensor/growing_network.py — self-expanding DNN framework."""

import tempfile

import pytest
import torch

from tensor.domain_registry import DomainRegistry
from tensor.growing_network import GrowingNeuralNetwork, add_mode_head_to_network
from tensor.unified_network import UnifiedTensorNetwork


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_net():
    """Small network for fast tests: 16-dim embed, 8 modes."""
    return UnifiedTensorNetwork(hdv_dim=256, n_modes=8, embed_dim=16)


@pytest.fixture
def registry(tmp_path):
    return DomainRegistry(persist_path=str(tmp_path / "active.json"))


@pytest.fixture
def grower(small_net, registry):
    return GrowingNeuralNetwork(
        network=small_net,
        domain_registry=registry,
        growth_error_threshold=0.01,  # low threshold so tests trigger growth easily
        max_growth_per_session=10,
    )


# ── Head error computation ─────────────────────────────────────────────────────

class TestHeadErrors:
    def test_compute_head_errors_returns_dict(self, grower, small_net):
        x = torch.randn(4, 16)
        errors = grower.compute_head_errors(x)
        assert isinstance(errors, dict)
        assert len(errors) == small_net.n_modes

    def test_errors_are_non_negative(self, grower):
        x = torch.randn(4, 16)
        errors = grower.compute_head_errors(x)
        for err in errors.values():
            assert err >= 0.0

    def test_errors_with_targets(self, grower):
        x = torch.randn(4, 16)
        targets = torch.randn(4, 16)
        errors = grower.compute_head_errors(x, targets)
        assert len(errors) > 0

    def test_ema_update(self, grower):
        x = torch.randn(4, 16)
        errors = grower.compute_head_errors(x)
        grower.update_ema_errors(errors)
        assert len(grower._head_error_ema) > 0

    def test_worst_head_returns_valid_index(self, grower):
        x = torch.randn(4, 16)
        errors = grower.compute_head_errors(x)
        grower.update_ema_errors(errors)
        idx, err = grower.worst_head()
        assert 0 <= idx < grower.net.n_modes
        assert err >= 0.0


# ── Should grow ────────────────────────────────────────────────────────────────

class TestShouldGrow:
    def test_should_grow_false_when_no_errors(self, grower):
        # No errors tracked yet → EMA is empty or all zero → should not grow
        # (depends on whether default error is 0 or threshold)
        # With empty EMA, worst_head returns err=0 < threshold → False
        result = grower.should_grow()
        assert not result  # No errors tracked yet

    def test_should_grow_true_when_error_high(self, grower, registry):
        # Force high error on head 0
        grower._head_error_ema[0] = 5.0
        # Registry has 150 domains, all inactive initially → room to grow
        result = grower.should_grow()
        assert result

    def test_should_grow_false_when_budget_exhausted(self, small_net, registry):
        g = GrowingNeuralNetwork(small_net, registry, max_growth_per_session=0)
        g._head_error_ema[0] = 5.0
        assert not g.should_grow()


# ── Growth ─────────────────────────────────────────────────────────────────────

class TestGrowth:
    def test_grow_returns_head_index(self, grower, registry):
        grower._head_error_ema[0] = 5.0
        new_head = grower.grow(from_head_idx=0)
        assert new_head is not None
        assert 0 <= new_head < grower.net.n_modes

    def test_grow_activates_domain(self, grower, registry):
        before = registry.n_active()
        grower._head_error_ema[0] = 5.0
        grower.grow(from_head_idx=0)
        assert registry.n_active() > before

    def test_grow_records_history(self, grower):
        grower._head_error_ema[0] = 5.0
        grower.grow(from_head_idx=0)
        assert len(grower.growth_history) == 1
        event = grower.growth_history[0]
        assert "from_head" in event
        assert "new_domain" in event
        assert "new_domain_display" in event

    def test_weight_transfer_happens(self, grower, small_net):
        grower._head_error_ema[0] = 5.0
        new_head = grower.grow(from_head_idx=0)
        if new_head is None:
            return
        # New head weights should be close (but not identical) to source head
        src_params = list(small_net.mode_heads[0].parameters())
        dst_params = list(small_net.mode_heads[new_head].parameters())
        for sp, dp in zip(src_params, dst_params):
            diff = (sp.data - dp.data).abs().mean().item()
            # Should be close (noise_scale=0.01) but not exact
            assert diff < 0.1

    def test_grow_with_domain_hint(self, grower, registry):
        grower._head_error_ema[0] = 5.0
        new_head = grower.grow(from_head_idx=0, domain_hint="weather forecast atmospheric")
        if new_head is not None:
            # Domain should be related to weather
            activated = registry.active_domains()
            assert len(activated) >= 1

    def test_grow_returns_none_when_no_inactive(self, small_net, tmp_path):
        reg = DomainRegistry(persist_path=str(tmp_path / "full.json"))
        # Activate all 150 domains manually
        from tensor.domain_registry import DOMAIN_MANIFEST
        for _, domain_id, _, _ in DOMAIN_MANIFEST:
            reg.activate_domain(domain_id)
        g = GrowingNeuralNetwork(small_net, reg, growth_error_threshold=0.01)
        g._head_error_ema[0] = 5.0
        result = g.grow()
        assert result is None


# ── Self-directed grow ─────────────────────────────────────────────────────────

class TestSelfDirectedGrow:
    def test_self_directed_grow_with_x(self, grower):
        x = torch.randn(4, 16)
        result = grower.self_directed_grow(x=x)
        # May or may not grow depending on error level; just check no crash
        assert result is None or isinstance(result, int)

    def test_self_directed_grow_triggers_on_high_error(self, grower):
        # Force high error so growth triggers
        grower._head_error_ema[0] = 5.0
        result = grower.self_directed_grow()
        assert result is not None
        assert isinstance(result, int)


# ── Status ─────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_dict_has_required_keys(self, grower):
        s = grower.status()
        required = {"total_heads", "promoted_heads", "growth_events",
                    "worst_head", "worst_head_error", "should_grow",
                    "active_domains", "inactive_domains"}
        assert required.issubset(s.keys())

    def test_status_total_heads_matches_network(self, grower, small_net):
        assert grower.status()["total_heads"] == small_net.n_modes


# ── add_mode_head_to_network ───────────────────────────────────────────────────

class TestAddModeHead:
    def test_adds_head_increases_n_modes(self, small_net):
        before = small_net.n_modes
        new_idx = add_mode_head_to_network(small_net, domain="extra_domain")
        assert small_net.n_modes == before + 1
        assert new_idx == before

    def test_new_head_is_callable(self, small_net):
        add_mode_head_to_network(small_net, domain="test_domain")
        new_head = small_net.mode_heads[-1]
        x = torch.randn(2, 16)
        out = new_head(x)
        assert out.shape == (2, 16)
