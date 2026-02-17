"""
Tests for FICUTS Layer 9: Dual Geometry (FIM + IRMF)

FIM tests run in tensor env (NumPy/SciPy only).
IRMF tests are skipped if torch is not available.
"""

import numpy as np
import pytest

from tensor.dual_geometry import (
    FisherInformationManifold,
    IsometricFunctionManifold,
    DualGeometrySystem,
)

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

SKIP_TORCH = pytest.mark.skipif(not _TORCH, reason="torch not available")


# ── Task 9.1: FisherInformationManifold ───────────────────────────────────────

def test_fim_mle_close_to_true():
    rng = np.random.default_rng(42)
    true_mu, true_sigma = 5.0, 2.0
    data = rng.normal(true_mu, true_sigma, size=1000)
    fim = FisherInformationManifold()
    res = fim.learn_distribution('p', data, np.array([0.0, 1.0]))
    assert abs(res['theta'][0] - true_mu) < 0.15
    assert abs(res['theta'][1] - true_sigma) < 0.3

def test_fim_uncertainty_small_for_large_n():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 1.5, size=2000)
    fim = FisherInformationManifold()
    res = fim.learn_distribution('p', data, np.array([0.0, 1.0]))
    assert res['uncertainty'][0] < 0.1

def test_fim_promotion_ready_large_n():
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 1.0, size=5000)
    fim = FisherInformationManifold()
    fim.learn_distribution('p', data, np.array([0.0, 1.0]))
    assert fim.is_ready_for_promotion('p', threshold=0.1)

def test_fim_promotion_not_ready_small_n():
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 1.0, size=5)
    fim = FisherInformationManifold()
    fim.learn_distribution('p', data, np.array([0.0, 1.0]))
    assert not fim.is_ready_for_promotion('p', threshold=0.01)

def test_fim_promotion_missing_pattern():
    fim = FisherInformationManifold()
    assert not fim.is_ready_for_promotion('nonexistent')

def test_fim_most_informative_order():
    rng = np.random.default_rng(3)
    data = rng.normal(5.0, 0.5, size=500)
    fim = FisherInformationManifold()
    fim.learn_distribution('p', data, np.array([5.0, 0.5]))
    order = fim.get_most_informative_parameters('p')
    assert len(order) == 2

def test_fim_stores_data_size():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, size=200)
    fim = FisherInformationManifold()
    res = fim.learn_distribution('p', data, np.array([0.0, 1.0]))
    assert res['data_size'] == 200


# ── Task 9.2: IsometricFunctionManifold ───────────────────────────────────────

@SKIP_TORCH
def test_irmf_learn_and_reconstruct():
    x = np.linspace(0, 5, 50)
    y = np.exp(-x / 2.0)
    data = list(zip(x.tolist(), y.tolist()))
    irmf = IsometricFunctionManifold(latent_dim=32, hidden_dim=64)
    z = irmf.learn_function('exp_decay', data, n_epochs=300)
    assert z is not None
    assert z.shape == (32,)
    x_test = np.linspace(0, 5, 20)
    y_pred = irmf.generate_function_values('exp_decay', x_test)
    y_true = np.exp(-x_test / 2.0)
    assert np.mean((y_pred - y_true) ** 2) < 0.15

@SKIP_TORCH
def test_irmf_isometric_loss_finite():
    import torch
    irmf = IsometricFunctionManifold(latent_dim=16, hidden_dim=32)
    vecs = torch.randn(5, 16)
    loss = irmf.compute_isometric_loss(vecs, n_pairs=10)
    assert float(loss) >= 0

@SKIP_TORCH
def test_irmf_single_vector_loss_zero():
    import torch
    irmf = IsometricFunctionManifold(latent_dim=16, hidden_dim=32)
    vecs = torch.randn(1, 16)
    loss = irmf.compute_isometric_loss(vecs, n_pairs=5)
    assert float(loss) == 0.0

def test_irmf_raises_without_torch(monkeypatch):
    irmf = IsometricFunctionManifold.__new__(IsometricFunctionManifold)
    irmf.latent_dim = 16
    irmf.hidden_dim = 32
    irmf.patterns = {}
    irmf._decoder = None
    irmf._torch = None
    irmf._nn = None
    with pytest.raises(RuntimeError, match="torch not available"):
        irmf.learn_function('x', [(0, 0)], n_epochs=1)


# ── Task 9.3: DualGeometrySystem ──────────────────────────────────────────────

def test_classify_statistical_high_variance():
    dual = DualGeometrySystem()
    pat = {'observations': np.random.randn(100) + 5, 'domains': ['ece'], 'conserved_quantity': False}
    assert dual.classify_pattern(pat) == 'statistical'

def test_classify_statistical_few_domains():
    dual = DualGeometrySystem()
    pat = {'observations': np.ones(100) * 5, 'domains': ['ece', 'biology'], 'conserved_quantity': True}
    assert dual.classify_pattern(pat) == 'statistical'

def test_classify_deterministic():
    dual = DualGeometrySystem()
    rng = np.random.default_rng(0)
    pat = {
        'observations': 5.0 + rng.normal(0, 0.01, 100),
        'domains': ['ece', 'biology', 'finance'],
        'conserved_quantity': True,
    }
    assert dual.classify_pattern(pat) == 'deterministic'

def test_learn_pattern_statistical_routes_to_fim():
    dual = DualGeometrySystem()
    rng = np.random.default_rng(5)
    data = rng.normal(2.0, 0.5, 300)
    pat = {'observations': data.tolist(), 'domains': ['ece'], 'conserved_quantity': False}
    dual.learn_pattern('p1', pat, data)
    assert 'p1' in dual.fisher.patterns

def test_promote_pattern_not_ready():
    dual = DualGeometrySystem()
    rng = np.random.default_rng(9)
    data = rng.normal(0, 1, 5)
    dual.fisher.learn_distribution('p', data, np.array([0.0, 1.0]))
    assert not dual.promote_pattern('p')

def test_promote_pattern_ready():
    dual = DualGeometrySystem()
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1, 5000)
    dual.fisher.learn_distribution('p', data, np.array([0.0, 1.0]))
    # uncertainty ~0.014 with n=5000; use threshold=0.1 (Cramér-Rao near bound)
    result = dual.promote_pattern('p', threshold=0.1)
    assert result is True

def test_promote_missing_pattern():
    dual = DualGeometrySystem()
    assert not dual.promote_pattern('ghost')
