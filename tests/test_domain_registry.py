"""Tests for tensor/domain_registry.py — 150-domain expansion registry."""

import json
import tempfile
from pathlib import Path

import pytest

from tensor.domain_registry import (
    DOMAIN_MANIFEST,
    DomainRegistry,
    _ALL_IDS,
    _ID_TO_ENTRY,
    _IDX_TO_ENTRY,
)


# ── Manifest integrity ─────────────────────────────────────────────────────────

def test_manifest_has_150_entries():
    assert len(DOMAIN_MANIFEST) == 150


def test_manifest_indices_are_0_to_149():
    indices = [e[0] for e in DOMAIN_MANIFEST]
    assert indices == list(range(150))


def test_manifest_ids_are_unique():
    ids = [e[1] for e in DOMAIN_MANIFEST]
    assert len(ids) == len(set(ids)), "Duplicate domain IDs found"


def test_manifest_each_has_nonempty_keywords():
    for idx, domain_id, display, keywords in DOMAIN_MANIFEST:
        assert len(keywords) >= 3, f"Domain {domain_id} has too few keywords"


def test_lookup_structures_consistent():
    assert len(_ID_TO_ENTRY) == 150
    assert len(_IDX_TO_ENTRY) == 150
    assert len(_ALL_IDS) == 150


# ── Classification ─────────────────────────────────────────────────────────────

class TestClassification:
    def test_pde_paper_matches_nonlinear_pde(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_classify.json")
        result = reg.best_domain("physics-informed neural network for solving partial differential equation")
        assert result == "nonlinear_pde"

    def test_protein_text_matches_protein_structure(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_classify.json")
        result = reg.best_domain("protein folding prediction using neural network")
        assert result == "protein_structure"

    def test_fraud_matches_fraud_detection(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_classify.json")
        result = reg.best_domain("credit card fraud detection anomaly")
        assert result in ("fraud_detection", "telecom_fraud")

    def test_unknown_gibberish_returns_unknown(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_classify.json")
        result = reg.best_domain("zzz qqqq xyzfoo")
        assert result == "unknown"

    def test_top_k_returns_multiple(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_classify.json")
        results = reg.classify_domain("neural network learning optimization", top_k=3)
        assert len(results) >= 1
        assert len(results) <= 3

    def test_weather_matches_weather_domain(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_classify.json")
        result = reg.best_domain("weather forecast atmospheric precipitation wind pressure")
        assert result == "weather_forecasting"

    def test_language_model_matches_llm(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_classify.json")
        result = reg.best_domain("large language model transformer attention token embedding bert")
        assert result == "large_language_model"


# ── Activation ─────────────────────────────────────────────────────────────────

class TestActivation:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.reg = DomainRegistry(persist_path=self.tmp.name)

    def test_activate_returns_head_index(self):
        idx = self.reg.activate_domain("nonlinear_pde")
        assert idx == 0

    def test_activate_adds_to_active_set(self):
        assert not self.reg.is_active("nonlinear_pde")
        self.reg.activate_domain("nonlinear_pde")
        assert self.reg.is_active("nonlinear_pde")

    def test_activate_unknown_returns_minus_one(self):
        idx = self.reg.activate_domain("not_a_real_domain")
        assert idx == -1

    def test_n_active_increments(self):
        before = self.reg.n_active()
        self.reg.activate_domain("protein_structure")
        assert self.reg.n_active() == before + 1

    def test_activate_same_domain_twice_does_not_double_count(self):
        self.reg.activate_domain("protein_structure")
        self.reg.activate_domain("protein_structure")
        assert self.reg.n_active() == 1

    def test_inactive_domains_shrinks_on_activation(self):
        before = self.reg.n_inactive()
        self.reg.activate_domain("quantum_systems")
        assert self.reg.n_inactive() == before - 1

    def test_activate_for_text_classifies_and_activates(self):
        domain_id, head_idx = self.reg.activate_for_text(
            "quantum entanglement qubit hamiltonian variational"
        )
        assert domain_id == "quantum_systems"
        assert head_idx == 7
        assert self.reg.is_active("quantum_systems")

    def test_activate_for_unknown_text_returns_unknown(self):
        domain_id, head_idx = self.reg.activate_for_text("zzzzz qqqq")
        assert domain_id == "unknown"
        assert head_idx == -1


# ── Persistence ────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_persists_and_reloads_active_domains(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        reg1 = DomainRegistry(persist_path=path)
        reg1.activate_domain("power_grid")
        reg1.activate_domain("battery_degradation")

        reg2 = DomainRegistry(persist_path=path)
        assert reg2.is_active("power_grid")
        assert reg2.is_active("battery_degradation")
        assert reg2.n_active() == 2

    def test_persist_file_is_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        reg = DomainRegistry(persist_path=path)
        reg.activate_domain("supply_chain")

        with open(path) as fh:
            data = json.load(fh)
        assert "active" in data
        assert "supply_chain" in data["active"]


# ── Status ─────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_shows_correct_counts(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = DomainRegistry(persist_path=path)
        reg.activate_domain("nonlinear_pde")
        reg.activate_domain("weather_forecasting")

        s = reg.status()
        assert s["total_domains"] == 150
        assert s["active"] == 2
        assert s["inactive"] == 148
        assert s["coverage_pct"] == round(100 * 2 / 150, 1)

    def test_get_head_idx_correct(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_status.json")
        assert reg.get_head_idx("nonlinear_pde") == 0
        assert reg.get_head_idx("global_systems_modeling") == 149

    def test_get_display_name(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_status.json")
        assert "PDE" in reg.get_display_name("nonlinear_pde")

    def test_get_keywords(self):
        reg = DomainRegistry(persist_path="/tmp/test_dr_status.json")
        kws = reg.get_keywords("nonlinear_pde")
        assert "pde" in kws
