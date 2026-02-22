"""
Backend API route tests for the Unified Tensor System platform.

All heavy model calls are mocked to keep tests fast and deterministic.
Run from project root:
    python -m pytest platform/backend/tests/test_api_routes.py -q
"""
import sys
import os

# Ensure project root is on sys.path FIRST, then ecemath, then backend.
# This ordering prevents ecemath/src/optimization from shadowing project/optimization.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_ECEMATH = os.path.join(_ROOT, "ecemath")
_BACKEND = os.path.join(_ROOT, "platform", "backend")

for _p in [_ECEMATH, _ROOT]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
if _BACKEND not in sys.path:
    sys.path.append(_BACKEND)

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient with LCAPatchDetector mocked."""
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
# Health / root
# ---------------------------------------------------------------------------

def test_root_returns_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Regime router
# ---------------------------------------------------------------------------

def test_regime_status_keys(client):
    r = client.get("/api/v1/regime/status")
    assert r.status_code == 200
    data = r.json()
    for key in ("patch_type", "commutator_norm", "curvature_ratio", "koopman_trust", "spectral_gap"):
        assert key in data, f"Missing key: {key}"


def test_regime_status_patch_type_is_string(client):
    r = client.get("/api/v1/regime/status")
    assert isinstance(r.json()["patch_type"], str)


def test_regime_status_metrics_are_floats(client):
    data = client.get("/api/v1/regime/status").json()
    for key in ("commutator_norm", "curvature_ratio", "koopman_trust", "spectral_gap"):
        assert isinstance(data[key], (int, float)), f"{key} not numeric"


# ---------------------------------------------------------------------------
# Calendar router
# ---------------------------------------------------------------------------

def test_calendar_phase_default_date(client):
    r = client.get("/api/v1/calendar/phase")
    assert r.status_code == 200


def test_calendar_phase_specific_date(client):
    r = client.get("/api/v1/calendar/phase?date=2026-02-22")
    assert r.status_code == 200
    data = r.json()
    assert data["date"] == "2026-02-22"


def test_calendar_phase_keys(client):
    r = client.get("/api/v1/calendar/phase?date=2026-02-22")
    data = r.json()
    for key in ("channels", "amplitudes", "active_events", "dominant_cycle",
                "regime_label", "resonance_detected", "vol_multiplier", "date"):
        assert key in data, f"Missing key: {key}"


def test_calendar_phase_amplitudes_length(client):
    r = client.get("/api/v1/calendar/phase?date=2026-02-22")
    data = r.json()
    assert len(data["amplitudes"]) == 5
    assert len(data["channels"]) == 5


def test_calendar_phase_invalid_date(client):
    r = client.get("/api/v1/calendar/phase?date=not-a-date")
    assert r.status_code == 422


def test_calendar_range_returns_series(client):
    r = client.get("/api/v1/calendar/range?start=2026-02-01&end=2026-02-07")
    assert r.status_code == 200
    data = r.json()
    assert "dates" in data and "series" in data
    assert len(data["dates"]) == 7
    assert len(data["series"]) == 7


def test_calendar_range_bad_order(client):
    r = client.get("/api/v1/calendar/range?start=2026-02-10&end=2026-02-01")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# CodeGen router
# ---------------------------------------------------------------------------

def test_codegen_templates_returns_list(client):
    r = client.get("/api/v1/codegen/templates")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_codegen_templates_keys(client):
    r = client.get("/api/v1/codegen/templates")
    tpl = r.json()[0]
    for key in ("name", "domain", "operation", "borrow_profile", "e_borrow", "description"):
        assert key in tpl, f"Missing key: {key}"


def test_codegen_generate_returns_expected_keys(client):
    payload = {"domain": "numeric", "operation": "sma", "parameters": {"window": 20}}
    r = client.post("/api/v1/codegen/generate", json=payload)
    assert r.status_code == 200
    data = r.json()
    for key in ("rust_source", "e_borrow", "predicted_compiles", "template_name",
                "borrow_vector", "probability", "success"):
        assert key in data, f"Missing key: {key}"


def test_codegen_generate_rust_source_nonempty(client):
    payload = {"domain": "numeric", "operation": "sma", "parameters": {}}
    r = client.post("/api/v1/codegen/generate", json=payload)
    data = r.json()
    assert len(data["rust_source"]) > 0


def test_codegen_generate_e_borrow_is_float(client):
    payload = {"domain": "numeric", "operation": "sma", "parameters": {}}
    r = client.post("/api/v1/codegen/generate", json=payload)
    data = r.json()
    assert isinstance(data["e_borrow"], float)
    assert 0.0 <= data["e_borrow"] <= 1.0


def test_codegen_generate_borrow_vector_length(client):
    payload = {"domain": "numeric", "operation": "sma", "parameters": {}}
    r = client.post("/api/v1/codegen/generate", json=payload)
    data = r.json()
    assert len(data["borrow_vector"]) == 6


# ---------------------------------------------------------------------------
# HDV router
# ---------------------------------------------------------------------------

def test_hdv_encode_returns_expected_keys(client):
    payload = {"text": "neural network gradient descent", "domain": "math"}
    r = client.post("/api/v1/hdv/encode", json=payload)
    assert r.status_code == 200
    data = r.json()
    for key in ("pca_2d", "domain", "norm", "dim"):
        assert key in data, f"Missing key: {key}"


def test_hdv_encode_pca_2d_shape(client):
    payload = {"text": "circuit resonance oscillation", "domain": "physics"}
    r = client.post("/api/v1/hdv/encode", json=payload)
    data = r.json()
    assert len(data["pca_2d"]) == 2


def test_hdv_encode_norm_positive(client):
    payload = {"text": "optimization loop convergence", "domain": "behavioral"}
    r = client.post("/api/v1/hdv/encode", json=payload)
    data = r.json()
    assert data["norm"] > 0.0


def test_hdv_encode_invalid_domain(client):
    payload = {"text": "test", "domain": "nonexistent_domain"}
    r = client.post("/api/v1/hdv/encode", json=payload)
    assert r.status_code == 422


def test_hdv_encode_empty_text(client):
    payload = {"text": "   ", "domain": "math"}
    r = client.post("/api/v1/hdv/encode", json=payload)
    assert r.status_code == 422


def test_hdv_universals_returns_expected_keys(client):
    r = client.get("/api/v1/hdv/universals")
    assert r.status_code == 200
    data = r.json()
    for key in ("universals", "count", "domains_active"):
        assert key in data, f"Missing key: {key}"


def test_hdv_universals_count_matches_list(client):
    r = client.get("/api/v1/hdv/universals")
    data = r.json()
    assert data["count"] == len(data["universals"])


# ---------------------------------------------------------------------------
# Physics router
# ---------------------------------------------------------------------------

def test_physics_simulate_rlc_keys(client):
    payload = {"system_type": "rlc", "params": {"R": 10.0, "L": 0.01, "C": 1e-6}, "n_steps": 50}
    r = client.post("/api/v1/physics/simulate", json=payload)
    assert r.status_code == 200
    data = r.json()
    for key in ("trajectory", "koopman_trust", "regime_type", "omega0", "Q", "system_type"):
        assert key in data, f"Missing key: {key}"


def test_physics_simulate_rlc_trajectory_nonempty(client):
    payload = {"system_type": "rlc", "params": {}, "n_steps": 30}
    r = client.post("/api/v1/physics/simulate", json=payload)
    data = r.json()
    assert len(data["trajectory"]) > 0


def test_physics_simulate_duffing_keys(client):
    payload = {
        "system_type": "duffing",
        "params": {"alpha": 1.0, "beta": 0.1, "delta": 0.3, "x0": 1.0, "v0": 0.0},
        "n_steps": 40,
    }
    r = client.post("/api/v1/physics/simulate", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["system_type"] == "duffing"
    assert "omega0" in data and "Q" in data


def test_physics_simulate_harmonic_keys(client):
    payload = {
        "system_type": "harmonic",
        "params": {"omega0": 6.283, "zeta": 0.1, "x0": 1.0},
        "n_steps": 40,
    }
    r = client.post("/api/v1/physics/simulate", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["regime_type"] in ("lca", "overdamped", "abelian", "nonabelian", "near_separatrix")


def test_physics_simulate_trajectory_has_t_x_v(client):
    payload = {"system_type": "harmonic", "params": {}, "n_steps": 20}
    r = client.post("/api/v1/physics/simulate", json=payload)
    data = r.json()
    pt = data["trajectory"][0]
    assert "t" in pt and "x" in pt and "v" in pt


def test_physics_simulate_invalid_system_type(client):
    payload = {"system_type": "quantum", "params": {}, "n_steps": 10}
    r = client.post("/api/v1/physics/simulate", json=payload)
    assert r.status_code == 422


def test_physics_simulate_n_steps_too_large(client):
    payload = {"system_type": "harmonic", "params": {}, "n_steps": 9999}
    r = client.post("/api/v1/physics/simulate", json=payload)
    assert r.status_code == 422
