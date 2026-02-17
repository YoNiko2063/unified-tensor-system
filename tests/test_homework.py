"""Tests for HomeworkSolver: DC circuit solving with ECEMath."""
import os
import sys
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'ecemath', 'examples'))

from homework_solver import HomeworkSolver


@pytest.fixture
def circuit_solver():
    """Build the homework circuit: V1=15V, R1=2.2kΩ, R2=3.3kΩ, R3=150Ω."""
    solver = HomeworkSolver()
    solver.add_voltage_source('V1', 'n1', 'gnd', 15.0)
    solver.add_resistor('R1', 'n1', 'gnd', 2200.0)
    solver.add_resistor('R2', 'n1', 'n2', 3300.0)
    solver.add_resistor('R3', 'n2', 'gnd', 150.0)
    return solver


# ─── Test 1: solve_dc matches expected values ───
def test_solve_dc(circuit_solver):
    result = circuit_solver.solve_dc()

    v_n1 = result['node_voltages']['n1']
    v_n2 = result['node_voltages']['n2']
    i_r1 = result['component_currents']['R1']
    i_r2 = result['component_currents']['R2']
    i_r3 = result['component_currents']['R3']

    assert abs(v_n1 - 15.0) < 1e-6, f"V(n1)={v_n1}, expected 15.0"
    assert abs(v_n2 - 0.652174) < 1e-3, f"V(n2)={v_n2}, expected ~0.652"
    assert abs(abs(i_r1) - 6.818182e-3) < 1e-5, f"I(R1)={i_r1}"
    assert abs(abs(i_r2) - 4.347826e-3) < 1e-5, f"I(R2)={i_r2}"
    assert abs(abs(i_r3) - 4.347826e-3) < 1e-5, f"I(R3)={i_r3}"
    # I(R2) == I(R3) since they're in series
    assert abs(abs(i_r2) - abs(i_r3)) < 1e-10


# ─── Test 2: KCL verification passes ───
def test_kcl_verify(circuit_solver):
    result = circuit_solver.solve_dc()
    kcl = circuit_solver.verify_kcl(result)

    assert kcl['pass'] is True
    for node, residual in kcl['residuals'].items():
        assert abs(residual) < 1e-6, f"KCL fail at {node}: {residual}"


# ─── Test 3: power conservation ───
def test_power_check(circuit_solver):
    result = circuit_solver.solve_dc()
    power = circuit_solver.power_check(result)

    assert power['balanced'] is True
    assert power['p_delivered'] > 0
    assert power['p_dissipated'] > 0
    assert abs(power['balance_error']) < 1e-6


# ─── Test 4: manifold lookup returns valid signature ───
def test_manifold_lookup(circuit_solver):
    result = circuit_solver.solve_dc()
    manifold = circuit_solver.manifold_lookup(result)

    assert 'consonance_score' in manifold
    assert 0.0 <= manifold['consonance_score'] <= 1.0
    assert 'dominant_interval' in manifold
    assert 'stability_verdict' in manifold
    assert manifold['n_nodes'] >= 2
    assert manifold['n_branches'] >= 1  # V1 has a branch variable


# ─── Test 5: format_solution produces readable output ───
def test_format_solution(circuit_solver):
    result = circuit_solver.solve_dc()
    output = circuit_solver.format_solution(result, title="Test Circuit")

    assert "Test Circuit" in output
    assert "V(n1)" in output
    assert "V(n2)" in output
    assert "I(R1)" in output
    assert "KCL Verification: PASS" in output
    assert "Balance: OK" in output
