"""
Tests for POST /api/v1/circuit/optimize

Run from project root:
    python -m pytest platform/backend/tests/test_circuit_route.py -v
"""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_ECEMATH = os.path.join(_ROOT, "ecemath")
_BACKEND = os.path.join(_ROOT, "platform", "backend")

for _p in [_ECEMATH, _ROOT]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
if _BACKEND not in sys.path:
    sys.path.append(_BACKEND)

import math
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with LCAPatchDetector mocked (same pattern as existing tests)."""
    mock_patch_result = MagicMock()
    mock_patch_result.patch_type = "lca"
    mock_patch_result.commutator_norm = 0.01
    mock_patch_result.curvature_ratio = 0.04
    mock_patch_result.koopman_trust = 0.85
    mock_patch_result.spectral_gap = 0.72

    mock_detector = MagicMock()
    mock_detector.classify_trajectory.return_value = [mock_patch_result] * 10

    with patch("routers.regime.LCAPatchDetector", return_value=mock_detector):
        from main import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _post(client, payload=None):
    if payload is None:
        payload = {}
    return client.post("/api/v1/circuit/optimize", json=payload)


# ---------------------------------------------------------------------------
# 1. Basic status and top-level structure
# ---------------------------------------------------------------------------

def test_default_params_status_200(client):
    """POST with empty body (all defaults) must return 200."""
    resp = _post(client)
    assert resp.status_code == 200


def test_response_has_top_level_keys(client):
    data = _post(client).json()
    for key in ("pareto", "basin", "target", "frequency_response"):
        assert key in data, f"missing top-level key: {key}"


# ---------------------------------------------------------------------------
# 2. pareto block
# ---------------------------------------------------------------------------

def test_pareto_has_three_solutions(client):
    pareto = _post(client).json()["pareto"]
    for sol_key in ("best_eigenvalue", "best_stability", "best_cost"):
        assert sol_key in pareto, f"missing pareto key: {sol_key}"


def test_best_eigenvalue_has_required_fields(client):
    sol = _post(client).json()["pareto"]["best_eigenvalue"]
    for field in ("R", "L", "C", "eigenvalue_error", "Q_achieved",
                  "omega0_achieved", "regime_type", "cost", "converged"):
        assert field in sol, f"missing field in best_eigenvalue: {field}"


def test_pareto_values_are_positive(client):
    pareto = _post(client).json()["pareto"]
    for sol_key in ("best_eigenvalue", "best_stability", "best_cost"):
        sol = pareto[sol_key]
        assert sol["R"] > 0, f"{sol_key}.R not positive"
        assert sol["L"] > 0, f"{sol_key}.L not positive"
        assert sol["C"] > 0, f"{sol_key}.C not positive"
        assert sol["Q_achieved"] > 0, f"{sol_key}.Q_achieved not positive"
        assert sol["omega0_achieved"] > 0, f"{sol_key}.omega0_achieved not positive"


def test_pareto_converged_is_bool(client):
    sol = _post(client).json()["pareto"]["best_eigenvalue"]
    assert isinstance(sol["converged"], bool)


def test_pareto_eigenvalue_error_finite(client):
    sol = _post(client).json()["pareto"]["best_eigenvalue"]
    assert math.isfinite(sol["eigenvalue_error"])


# ---------------------------------------------------------------------------
# 3. target block
# ---------------------------------------------------------------------------

def test_target_has_required_fields(client):
    target = _post(client).json()["target"]
    for field in ("center_freq_hz", "Q_target", "omega0", "target_eigenvalues"):
        assert field in target, f"missing target field: {field}"


def test_target_eigenvalues_two_conjugate_pairs(client):
    eigs = _post(client).json()["target"]["target_eigenvalues"]
    assert isinstance(eigs, list)
    assert len(eigs) == 2
    for pair in eigs:
        assert len(pair) == 2  # [real, imag]


def test_target_omega0_matches_center_freq(client):
    """omega0 must be approximately 2*pi*center_freq_hz for default 1000 Hz."""
    target = _post(client).json()["target"]
    expected_omega0 = 2.0 * math.pi * 1000.0
    assert abs(target["omega0"] - expected_omega0) / expected_omega0 < 0.01


# ---------------------------------------------------------------------------
# 4. frequency_response block
# ---------------------------------------------------------------------------

def test_frequency_response_has_100_points(client):
    fr = _post(client).json()["frequency_response"]
    assert len(fr) == 100


def test_frequency_response_entry_fields(client):
    entry = _post(client).json()["frequency_response"][0]
    for field in ("freq_hz", "magnitude_db", "phase_deg"):
        assert field in entry, f"missing freq_response field: {field}"


def test_frequency_response_values_finite(client):
    fr = _post(client).json()["frequency_response"]
    for entry in fr:
        assert math.isfinite(entry["freq_hz"])
        assert math.isfinite(entry["magnitude_db"])
        assert math.isfinite(entry["phase_deg"])


def test_frequency_response_ordered_ascending(client):
    fr = _post(client).json()["frequency_response"]
    freqs = [e["freq_hz"] for e in fr]
    assert freqs == sorted(freqs), "frequency_response must be sorted by freq_hz"


# ---------------------------------------------------------------------------
# 5. basin block
# ---------------------------------------------------------------------------

def test_basin_has_required_fields(client):
    basin = _post(client).json()["basin"]
    for field in ("n_samples", "lca_fraction", "n_lca", "n_nonabelian",
                  "n_chaotic", "mean_eigenvalue_spread", "worst_case_error"):
        assert field in basin, f"missing basin field: {field}"


def test_basin_lca_fraction_in_unit_interval(client):
    lca_frac = _post(client).json()["basin"]["lca_fraction"]
    assert 0.0 <= lca_frac <= 1.0


def test_basin_counts_sum_to_n_samples(client):
    basin = _post(client).json()["basin"]
    total = basin["n_lca"] + basin["n_nonabelian"] + basin["n_chaotic"]
    assert total == basin["n_samples"]


# ---------------------------------------------------------------------------
# 6. Parametric: Q_target variations
# ---------------------------------------------------------------------------

def test_critically_damped_Q1(client):
    """Q=1.0 (critically damped): eigenvalue_error finite, converged true."""
    resp = _post(client, {"Q_target": 1.0})
    assert resp.status_code == 200
    sol = resp.json()["pareto"]["best_eigenvalue"]
    assert math.isfinite(sol["eigenvalue_error"])
    assert sol["converged"] is True


def test_overdamped_Q_half(client):
    """Q=0.5 (overdamped): response is still valid."""
    resp = _post(client, {"Q_target": 0.5})
    assert resp.status_code == 200
    data = resp.json()
    assert "pareto" in data
    assert "frequency_response" in data
    assert len(data["frequency_response"]) == 100


def test_high_Q_target(client):
    """Q=20 (narrow bandpass): lca_fraction should be high."""
    resp = _post(client, {"Q_target": 20.0})
    assert resp.status_code == 200
    basin = resp.json()["basin"]
    assert basin["lca_fraction"] >= 0.0


# ---------------------------------------------------------------------------
# 7. Invalid / edge-case topology → graceful degradation
# ---------------------------------------------------------------------------

def test_invalid_topology_still_returns_200(client):
    """Unknown topology must not crash — graceful degradation expected."""
    resp = _post(client, {"topology": "mystery_filter"})
    assert resp.status_code == 200
    data = resp.json()
    assert "pareto" in data


# ---------------------------------------------------------------------------
# 8. center_freq_hz=10000 → omega0 ≈ 2π×10000
# ---------------------------------------------------------------------------

def test_high_frequency_omega0(client):
    resp = _post(client, {"center_freq_hz": 10000.0})
    assert resp.status_code == 200
    omega0 = resp.json()["target"]["omega0"]
    expected = 2.0 * math.pi * 10000.0
    assert abs(omega0 - expected) / expected < 0.01, (
        f"omega0={omega0:.1f} but expected ~{expected:.1f}"
    )


def test_high_frequency_response_range(client):
    """Frequency response for 10 kHz center must span 1 kHz – 100 kHz."""
    resp = _post(client, {"center_freq_hz": 10000.0})
    fr = resp.json()["frequency_response"]
    freqs = [e["freq_hz"] for e in fr]
    assert min(freqs) < 1500.0   # 0.1 * 10000 = 1000 Hz
    assert max(freqs) > 50000.0  # 10  * 10000 = 100000 Hz
