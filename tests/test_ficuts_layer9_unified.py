"""Tests for FICUTS Layer 9 (v2.1): Unified Continuous Network"""

import torch
import pytest

from tensor.unified_network import (
    DOMAINS,
    ModeHead,
    UnifiedTensorNetwork,
    UnifiedNetworkTrainer,
    compute_isometric_loss,
    compute_lyapunov_energy,
)

# Small dims so tests run fast on CPU
HDV = 200
MODES = 10
EMBED = 64


@pytest.fixture
def net():
    return UnifiedTensorNetwork(hdv_dim=HDV, n_modes=MODES, embed_dim=EMBED)


@pytest.fixture
def trainer(net):
    return UnifiedNetworkTrainer(net, learning_rate=1e-3)


# ── ModeHead ──────────────────────────────────────────────────────────────────

def test_mode_head_output_shape():
    head = ModeHead(0, 'ece', embed_dim=EMBED)
    x = torch.randn(4, EMBED)
    out = head(x)
    assert out.shape == (4, EMBED)


def test_mode_head_domain_bias_learnable():
    head = ModeHead(0, 'ece', embed_dim=EMBED)
    assert head.domain_bias.requires_grad


# ── UnifiedTensorNetwork ──────────────────────────────────────────────────────

def test_forward_output_shape(net):
    x = torch.randint(0, HDV, (4, 10))
    out = net(x, active_modes=[0, 1, 2])
    assert out.shape == (4, HDV)


def test_forward_all_modes_default(net):
    x = torch.randint(0, HDV, (2, 5))
    out = net(x)
    assert out.shape == (2, HDV)


def test_passive_learning_gradients(net):
    """Inactive modes must receive gradients (passive learning)."""
    x = torch.randint(0, HDV, (4, 10))
    out = net(x, active_modes=[0, 1, 2])
    out.sum().backward()
    # Mode 7 was not in active_modes — must still have grad
    assert net.mode_heads[7].domain_bias.grad is not None


def test_output_differentiable(net):
    x = torch.randint(0, HDV, (2, 8))
    out = net(x, active_modes=[0])
    assert out.requires_grad


def test_n_mode_heads(net):
    assert len(net.mode_heads) == MODES


def test_hdv_embedding_size(net):
    assert net.hdv_embedding.num_embeddings == HDV


# ── Isometric Loss ────────────────────────────────────────────────────────────

def test_isometric_loss_non_negative(net):
    x1 = torch.randint(0, HDV, (10,))
    x2 = torch.randint(0, HDV, (10,))
    loss = compute_isometric_loss(net, [(x1, x2)], active_modes=[0, 1])
    assert float(loss) >= 0.0


def test_isometric_loss_requires_grad(net):
    x1 = torch.randint(0, HDV, (10,))
    x2 = torch.randint(0, HDV, (10,))
    loss = compute_isometric_loss(net, [(x1, x2)], active_modes=[0])
    assert loss.requires_grad


def test_isometric_loss_backward(net):
    x1 = torch.randint(0, HDV, (10,))
    x2 = torch.randint(0, HDV, (10,))
    loss = compute_isometric_loss(net, [(x1, x2)], active_modes=[0, 1])
    loss.backward()
    assert net.mode_heads[0].domain_bias.grad is not None


def test_isometric_loss_empty_pairs(net):
    loss = compute_isometric_loss(net, [], active_modes=[0])
    assert float(loss) == 0.0


# ── Lyapunov Energy ───────────────────────────────────────────────────────────

def test_lyapunov_energy_positive(net):
    state = torch.randn(EMBED)
    E = compute_lyapunov_energy(net, state)
    assert float(E) > 0.0


def test_lyapunov_energy_scalar(net):
    state = torch.randn(EMBED)
    E = compute_lyapunov_energy(net, state)
    assert E.shape == ()


# ── UnifiedNetworkTrainer ─────────────────────────────────────────────────────

def test_train_step_returns_float(trainer, net):
    batch = torch.randint(0, HDV, (4, 10))
    target = torch.randn(4, HDV)
    loss = trainer.train_step(batch, 'ece', target)
    assert isinstance(loss, float)


def test_train_step_unknown_domain_fallback(trainer, net):
    batch = torch.randint(0, HDV, (2, 5))
    target = torch.randn(2, HDV)
    loss = trainer.train_step(batch, 'nonexistent_domain', target)
    assert isinstance(loss, float)


def test_passive_learning_verified(trainer):
    assert trainer.verify_passive_learning('ece', 'biology')


def test_passive_learning_params_change(trainer, net):
    """Inactive domain's params differ after one step."""
    inactive_id = DOMAINS.index('biology')
    before = [p.clone().detach() for p in net.mode_heads[inactive_id].parameters()]
    batch = torch.randint(0, HDV, (4, 10))
    target = torch.randn(4, HDV)
    trainer.train_step(batch, 'ece', target)
    changed = any(
        not torch.equal(b, a)
        for b, a in zip(before, net.mode_heads[inactive_id].parameters())
    )
    assert changed
