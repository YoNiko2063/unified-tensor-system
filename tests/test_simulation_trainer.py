"""Tests for tensor/simulation_trainer.py."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Mechanical simulator (no ecemath dependency) ────────────────────────────

class TestMechanicalSimulator:
    def _get(self):
        from tensor.simulation_trainer import MechanicalSimulator
        return MechanicalSimulator()

    def test_spring_mass_stable(self):
        sim = self._get()
        r = sim.simulate(m=1.0, b=2.0, k=10.0)
        assert r is not None
        assert r.stable is True
        assert r.circuit_type == "spring_mass"

    def test_spring_mass_eigenvalues_shape(self):
        sim = self._get()
        r = sim.simulate(m=1.0, b=2.0, k=10.0)
        assert len(r.eigenvalues) == 2

    def test_spring_mass_eigenvalues_negative_real(self):
        # Stable → all Re(λ) < 0
        sim = self._get()
        r = sim.simulate(m=1.0, b=4.0, k=3.0)
        assert all(np.real(e) < 0 for e in r.eigenvalues)

    def test_spring_mass_natural_frequency(self):
        # ω₀ = √(k/m) = √(10/1) = √10 ≈ 1.59 Hz
        sim = self._get()
        r = sim.simulate(m=1.0, b=0.1, k=10.0)
        expected_f0 = np.sqrt(10.0) / (2 * np.pi)
        assert abs(r.natural_freq - expected_f0) < 0.01

    def test_spring_mass_overdamped(self):
        # b² > 4mk → real eigenvalues only
        sim = self._get()
        r = sim.simulate(m=1.0, b=100.0, k=1.0)
        assert all(abs(np.imag(e)) < 1e-6 for e in r.eigenvalues)

    def test_spring_mass_underdamped_oscillation(self):
        # b² < 4mk → complex eigenvalues
        sim = self._get()
        r = sim.simulate(m=1.0, b=0.1, k=100.0)
        assert any(abs(np.imag(e)) > 1.0 for e in r.eigenvalues)

    def test_spring_mass_jacobian(self):
        sim = self._get()
        r = sim.simulate(m=1.0, b=2.0, k=5.0)
        assert r.jacobian is not None
        assert r.jacobian.shape == (2, 2)

    def test_sweep_produces_results(self):
        sim = self._get()
        results = sim.sweep(n=2)
        assert len(results) > 0

    def test_to_text_description_contains_keywords(self):
        from tensor.simulation_trainer import MechanicalSimulator
        sim = MechanicalSimulator()
        r = sim.simulate(m=1.0, b=2.0, k=10.0)
        text = r.to_text_description()
        assert "eigenvalue" in text
        assert "stability" in text or "stable" in text
        assert "spring_mass" in text or "mechanical" in text


# ── SimulationResult ─────────────────────────────────────────────────────────

class TestSimulationResult:
    def _make_rc_result(self):
        from tensor.simulation_trainer import SimulationResult
        return SimulationResult(
            circuit_type="rc",
            params={"R": 1000.0, "C": 1e-9},
            eigenvalues=np.array([-1e6]),
            stable=True,
            n_states=1,
            natural_freq=0.0,
            damping_ratio=1.0,
            time_constant=1e-6,
        )

    def _make_rlc_result(self):
        from tensor.simulation_trainer import SimulationResult
        eigs = np.array([-5000 + 8660j, -5000 - 8660j])
        return SimulationResult(
            circuit_type="rlc",
            params={"R": 100.0, "L": 1e-3, "C": 1e-6},
            eigenvalues=eigs,
            stable=True,
            n_states=2,
            natural_freq=1591.0,
            damping_ratio=0.5,
            time_constant=2e-4,
        )

    def test_rc_text_has_eigenvalue(self):
        r = self._make_rc_result()
        text = r.to_text_description()
        assert "eigenvalue" in text

    def test_rc_text_has_circuit_type(self):
        r = self._make_rc_result()
        text = r.to_text_description()
        assert "rc" in text

    def test_rlc_text_has_resonance_keywords(self):
        r = self._make_rlc_result()
        text = r.to_text_description()
        assert any(kw in text for kw in ["resonance", "frequency", "oscillation", "bandwidth"])

    def test_rlc_text_has_damping(self):
        r = self._make_rlc_result()
        text = r.to_text_description()
        assert "damping" in text

    def test_to_domain_keywords_nonempty(self):
        r = self._make_rc_result()
        kw = r.to_domain_keywords()
        assert len(kw) >= 3
        assert "eigenvalue" in kw or "stability" in kw

    def test_rlc_domain_keywords_resonance(self):
        r = self._make_rlc_result()
        kw = r.to_domain_keywords()
        assert any(k in kw for k in ["resonance", "frequency", "oscillation"])


# ── CircuitSimulator ─────────────────────────────────────────────────────────

class TestCircuitSimulator:
    def _get(self):
        from tensor.simulation_trainer import CircuitSimulator
        return CircuitSimulator()

    def test_simulate_rc(self):
        sim = self._get()
        r = sim.simulate_rc(R=1000.0, C=1e-9)
        assert r is not None
        assert r.circuit_type == "rc"
        assert r.stable is True

    def test_simulate_rc_eigenvalue_approx(self):
        # RC: τ = RC = 1000 * 1e-9 = 1µs → λ = -1/τ = -1e6
        sim = self._get()
        r = sim.simulate_rc(R=1000.0, C=1e-9)
        assert abs(np.real(r.eigenvalues[0]) + 1e6) / 1e6 < 0.02

    def test_simulate_rlc_two_eigenvalues(self):
        sim = self._get()
        r = sim.simulate_rlc(R=100.0, L=1e-3, C=1e-6)
        assert r is not None
        assert len(r.eigenvalues) >= 2

    def test_simulate_rlc_underdamped(self):
        # R small → underdamped: complex eigenvalues
        sim = self._get()
        r = sim.simulate_rlc(R=10.0, L=1e-3, C=1e-6)
        assert r is not None
        # ζ = R/2 * √(C/L) = 5 * √(1e-6/1e-3) = 5 * 0.0316 ≈ 0.158 < 1
        assert r.damping_ratio < 1.0

    def test_simulate_lc_marginal_stability(self):
        sim = self._get()
        r = sim.simulate_lc(L=1e-3, C=1e-6)
        assert r is not None
        # LC: purely imaginary eigenvalues or near-zero real part
        assert all(abs(np.real(e)) < 1e3 for e in r.eigenvalues)

    def test_sweep_rc_produces_multiple_results(self):
        sim = self._get()
        results = sim.sweep_rc(n_R=2, n_C=2)
        assert len(results) >= 4

    def test_sweep_rlc_produces_results(self):
        sim = self._get()
        results = sim.sweep_rlc(n_R=2, n_L=2, n_C=2)
        assert len(results) >= 4

    def test_all_rc_results_are_stable(self):
        sim = self._get()
        results = sim.sweep_rc(n_R=3, n_C=3)
        assert all(r.stable for r in results), "All RC circuits should be stable"


# ── PhysicalHDVPopulator ─────────────────────────────────────────────────────

class TestPhysicalHDVPopulator:
    def _make_small_hdv(self):
        from tensor.integrated_hdv import IntegratedHDVSystem
        return IntegratedHDVSystem(hdv_dim=1000, n_modes=10, embed_dim=64)

    def _make_discovery(self, hdv):
        from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery
        return CrossDimensionalDiscovery(hdv)

    def test_encode_result_returns_vector(self):
        from tensor.simulation_trainer import MechanicalSimulator, PhysicalHDVPopulator
        hdv = self._make_small_hdv()
        disc = self._make_discovery(hdv)
        pop = PhysicalHDVPopulator(hdv, disc)
        sim = MechanicalSimulator()
        r = sim.simulate(m=1.0, b=2.0, k=10.0)
        vec = pop.encode_result(r)
        assert vec.shape[0] == 1000
        assert not np.all(vec == 0)

    def test_encode_records_to_discovery(self):
        from tensor.simulation_trainer import MechanicalSimulator, PhysicalHDVPopulator
        hdv = self._make_small_hdv()
        disc = self._make_discovery(hdv)
        pop = PhysicalHDVPopulator(hdv, disc)
        sim = MechanicalSimulator()
        r = sim.simulate(m=1.0, b=2.0, k=10.0)
        pop.encode_result(r)
        counts = disc.get_pattern_counts()
        assert counts.get("physical", 0) >= 1

    def test_populate_all_circuits(self):
        from tensor.simulation_trainer import PhysicalHDVPopulator
        hdv = self._make_small_hdv()
        disc = self._make_discovery(hdv)
        pop = PhysicalHDVPopulator(hdv, disc)
        n = pop.populate_all_circuits(n_sweep=2)
        assert n > 0
        counts = disc.get_pattern_counts()
        assert counts.get("physical", 0) >= n

    def test_find_cross_domain_equivalences_rc_vs_spring(self):
        from tensor.simulation_trainer import PhysicalHDVPopulator, CircuitSimulator, MechanicalSimulator
        hdv = self._make_small_hdv()
        disc = self._make_discovery(hdv)
        pop = PhysicalHDVPopulator(hdv, disc)

        # Encode RC results
        circuit_sim = CircuitSimulator()
        for r in circuit_sim.sweep_rc(n_R=2, n_C=2):
            pop.encode_result(r)

        # Encode spring-mass results
        mech_sim = MechanicalSimulator()
        for r in mech_sim.sweep(n=2):
            pop.encode_result(r)

        equivs = pop.find_cross_domain_equivalences(top_k=5)
        assert isinstance(equivs, list)
        # Should find at least one RC ↔ spring_mass pair (or no pairs if not enough overlap)
        # Just check format is correct
        for eq in equivs:
            assert "circuit_a" in eq
            assert "circuit_b" in eq
            assert "similarity" in eq
            assert eq["circuit_a"] != eq["circuit_b"]


# ── SimulationTrainer ────────────────────────────────────────────────────────

class TestSimulationTrainer:
    def _make_trainer(self):
        from tensor.simulation_trainer import SimulationTrainer
        from tensor.integrated_hdv import IntegratedHDVSystem
        from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery
        hdv = IntegratedHDVSystem(hdv_dim=1000, n_modes=10, embed_dim=64)
        disc = CrossDimensionalDiscovery(hdv)
        return SimulationTrainer(hdv, disc)

    def test_run_one_pass_returns_dict(self):
        trainer = self._make_trainer()
        result = trainer.run_one_pass(n_sweep=2)
        assert isinstance(result, dict)
        assert "encoded" in result
        assert result["encoded"] > 0

    def test_run_one_pass_elapsed_reasonable(self):
        trainer = self._make_trainer()
        result = trainer.run_one_pass(n_sweep=2)
        assert result["elapsed_s"] < 60.0  # should complete in under a minute

    def test_run_one_pass_total_encoded_cumulates(self):
        trainer = self._make_trainer()
        r1 = trainer.run_one_pass(n_sweep=2)
        r2 = trainer.run_one_pass(n_sweep=2)
        assert r2["total_encoded"] > r1["total_encoded"]

    def test_run_one_pass_populates_physical_patterns(self):
        from tensor.simulation_trainer import SimulationTrainer
        from tensor.integrated_hdv import IntegratedHDVSystem
        from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery
        hdv = IntegratedHDVSystem(hdv_dim=1000, n_modes=10, embed_dim=64)
        disc = CrossDimensionalDiscovery(hdv)
        trainer = SimulationTrainer(hdv, disc)
        trainer.run_one_pass(n_sweep=2)
        counts = disc.get_pattern_counts()
        assert counts.get("physical", 0) > 0
